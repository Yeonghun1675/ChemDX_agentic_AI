from typing import Optional, List
import os
import pandas as pd
import difflib

from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger


def xyY_to_hex(x: float, y: float, Y: float) -> str:
    """Convert CIE 1931 xyY to sRGB hex string (e.g., #RRGGBB)."""
    try:
        if y == 0:
            return "Error: y must be non-zero to convert xyY to XYZ."

        # xyY -> XYZ
        X = (x * Y) / y
        Z = ((1.0 - x - y) * Y) / y

        return xyz_to_hex(X, Y, Z)
    except Exception as exc:
        return f"Error converting xyY to hex: {exc}"


def xyz_to_hex(X: float, Y: float, Z: float) -> str:
    """Convert CIE 1931 XYZ (D65, 2°) to sRGB hex string."""
    # XYZ -> linear sRGB
    R_lin = 3.2406 * X + (-1.5372) * Y + (-0.4986) * Z
    G_lin = (-0.9689) * X + 1.8758 * Y + 0.0415 * Z
    B_lin = 0.0557 * X + (-0.2040) * Y + 1.0570 * Z

    def gamma_encode(channel: float) -> float:
        if channel <= 0.0031308:
            return 12.92 * channel
        return 1.055 * (channel ** (1.0 / 2.4)) - 0.055

    # Clamp after gamma encoding bounds handling
    R = max(0.0, min(1.0, gamma_encode(R_lin)))
    G = max(0.0, min(1.0, gamma_encode(G_lin)))
    B = max(0.0, min(1.0, gamma_encode(B_lin)))

    r_255 = int(round(R * 255.0))
    g_255 = int(round(G * 255.0))
    b_255 = int(round(B * 255.0))

    return f"#{r_255:02X}{g_255:02X}{b_255:02X}"


name = "PhosphorLookupAgent"
role = "Look up phosphor data (emission, decay, color) from Inorganic_Phosphor CSV DB"
context = "Use provided tools. DB path: PHOSPHOR_DB_PATH env var or ./data/Inorganic_Phosphor.csv"

system_prompt = f"""You are {name}. {role}. {context}"""

phosphor_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={"temperature": 0.0, "parallel_tool_calls": False},
    system_prompt=system_prompt,
)


# Global cache
_df_cache: Optional[pd.DataFrame] = None
_path_cache: Optional[str] = None


def _resolve_path(file_path: Optional[str]) -> Optional[str]:
    """Resolve DB path from args, env var, or defaults"""
    candidates = []
    if file_path:
        candidates.append(file_path)
    if env_path := os.getenv("PHOSPHOR_DB_PATH"):
        candidates.append(env_path)
    candidates.extend([
        "data/Inorganic_Phosphor.csv", "Inorganic_Phosphor.csv",
        "data/Inorganic_Phosphor.xlsx", "Inorganic_Phosphor.xlsx",  # fallback for Excel
        "data/Inorganic_Phosphor.xls", "Inorganic_Phosphor.xls"
    ])
    return next((c for c in candidates if c and os.path.exists(c)), None)


def _find_col(columns: List[str], keywords: List[str]) -> Optional[str]:
    """Find column by keywords (case-insensitive contains)"""
    lower_cols = {c.lower(): c for c in columns}
    for keyword in keywords:
        for col in lower_cols:
            if keyword.lower() in col:
                return lower_cols[col]
    return None


def _get_row(df: pd.DataFrame, formula: str) -> Optional[pd.Series]:
    """Get row by formula (case-insensitive exact match)"""
    formula_col = _find_col(list(df.columns), ["formula", "compound", "name"])
    if not formula_col:
        return None
    
    target = str(formula).strip().lower()
    try:
        match = df[df[formula_col].astype(str).str.strip().str.lower() == target]
        return match.iloc[0] if not match.empty else None
    except Exception:
        return None


def _get_numeric(row: pd.Series, col: str) -> Optional[float]:
    """Extract numeric value from row column"""
    try:
        return float(pd.to_numeric(row[col])) if col in row and pd.notna(row[col]) else None
    except Exception:
        return None


@phosphor_agent.tool_plain
def load_phosphor_db(file_path: Optional[str] = None) -> str:
    """Load Inorganic_Phosphor CSV database into memory"""
    global _df_cache, _path_cache
    
    resolved = _resolve_path(file_path)
    if not resolved:
        return "Error: Could not resolve DB path. Set PHOSPHOR_DB_PATH or place at ./data/Inorganic_Phosphor.csv"
    
    try:
        if resolved.endswith('.csv'):
            _df_cache = pd.read_csv(resolved)
        else:
            _df_cache = pd.read_excel(resolved)
        _path_cache = resolved
        return f"Loaded {len(_df_cache)} rows from '{resolved}'"
    except Exception as exc:
        return f"Error loading '{resolved}': {exc}"


@phosphor_agent.tool_plain
def lookup_by_formula(formula: str, file_path: Optional[str] = None) -> str:
    """Find emission max and decay time for formula (exact match)"""
    global _df_cache
    
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status
    
    if _df_cache is None:
        return "Error: Database not loaded"
    
    if row := _get_row(_df_cache, formula):
        columns = list(_df_cache.columns)
        emission_col = _find_col(columns, ["emission", "em_max", "emission_max", "lambda_em"])
        decay_col = _find_col(columns, ["decay", "lifetime", "tau"])
        
        emission = _get_numeric(row, emission_col) if emission_col else None
        decay = _get_numeric(row, decay_col) if decay_col else None
        
        return f"Formula: {formula}; Emission: {emission or 'N/A'}; Decay: {decay or 'N/A'}"
    
    return f"Not found: '{formula}'"


@phosphor_agent.tool_plain
def similar_formulas(formula: str, top_k: int = 5, file_path: Optional[str] = None) -> str:
    """Find top_k most similar formulas with emission/decay data"""
    global _df_cache
    
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status
    
    if _df_cache is None:
        return "Error: Database not loaded"
    
    formula_col = _find_col(list(_df_cache.columns), ["formula", "compound", "name"])
    if not formula_col:
        return "Error: No formula column found"
    
    # Get unique formulas and calculate similarities
    formulas = _df_cache[formula_col].astype(str).dropna().unique()
    target = str(formula).strip().lower()
    
    scored = [(f, difflib.SequenceMatcher(None, target, str(f).lower()).ratio()) 
              for f in formulas if str(f).strip()]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    if not scored:
        return "No candidates found"
    
    # Build result lines
    lines = [f"Query: {formula} | top_k={top_k}"]
    columns = list(_df_cache.columns)
    emission_col = _find_col(columns, ["emission", "em_max", "emission_max", "lambda_em"])
    decay_col = _find_col(columns, ["decay", "lifetime", "tau"])
    
    for cand, score in scored[:top_k]:
        row = _df_cache[_df_cache[formula_col] == cand].iloc[0]
        emission = _get_numeric(row, emission_col) if emission_col else None
        decay = _get_numeric(row, decay_col) if decay_col else None
        lines.append(f"{cand} | similarity={score:.3f} | emission={emission or 'N/A'} | decay={decay or 'N/A'}")
    
    return "\n".join(lines)


@phosphor_agent.tool_plain
def formula_to_hex_color(formula: str, file_path: Optional[str] = None) -> str:
    """Convert formula to hex color using CIE values from DB"""
    global _df_cache
    
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status
    
    if _df_cache is None:
        return "Error: Database not loaded"
    
    if not (row := _get_row(_df_cache, formula)):
        return f"Not found: '{formula}'"
    
    columns = list(_df_cache.columns)
    
    # Try xyY first, then XYZ
    x_col = _find_col(columns, ["x", "cie_x", "chromaticity x"])
    y_col = _find_col(columns, ["y", "cie_y", "chromaticity y"]) 
    Y_col = _find_col(columns, ["Y", "cie_y", "luminance", "intensity"])
    
    if all(col for col in [x_col, y_col, Y_col]):
        x_val = _get_numeric(row, x_col)
        y_val = _get_numeric(row, y_col)
        Y_val = _get_numeric(row, Y_col)
        if all(v is not None for v in [x_val, y_val, Y_val]):
            hex_str = xyY_to_hex(x_val, y_val, Y_val)
            if not hex_str.startswith("Error"):
                return f"Formula: {formula}; Color: {hex_str}; Source: xyY({x_val}, {y_val}, {Y_val})"
    
    # Fallback to XYZ
    X_col = _find_col(columns, ["X", "cie_x", "tristimulus x"])
    Z_col = _find_col(columns, ["Z", "cie_z", "tristimulus z"])
    
    if all(col for col in [X_col, Y_col, Z_col]):
        X_val = _get_numeric(row, X_col)
        Y_val = _get_numeric(row, Y_col)
        Z_val = _get_numeric(row, Z_col)
        if all(v is not None for v in [X_val, Y_val, Z_val]):
            hex_str = xyz_to_hex(X_val, Y_val, Z_val)
            if not hex_str.startswith("Error"):
                return f"Formula: {formula}; Color: {hex_str}; Source: XYZ({X_val}, {Y_val}, {Z_val})"
    
    return "No CIE data found for formula"


async def call_phosphor_lookup_agent(ctx: RunContext[AgentState], message2agent: str):
    """Call phosphor lookup agent"""
    deps = ctx.deps
    logger.info(f"[{name}] Message2Agent: {message2agent}")
    result = await phosphor_agent.run(message2agent, deps=deps)
    output = result.output
    logger.info(f"[{name}] Action: {output.action}")
    logger.info(f"[{name}] Result: {output.result}")
    return output


