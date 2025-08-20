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

working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

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
        "Inorganic_Phosphor_Optical_Properties_DB.csv",  # Use actual database file
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
    formula_col = _find_col(list(df.columns), ["inorganic phosphor", "formula", "compound", "name"])
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
        emission_col = _find_col(columns, ["emission", "em_max", "emission_max", "lambda_em", "emission max. (nm)"])
        decay_col = _find_col(columns, ["decay", "lifetime", "tau", "decay time (ns)"])
        
        emission = _get_numeric(row, emission_col) if emission_col else None
        decay = _get_numeric(row, decay_col) if decay_col else None
        
        return f"Formula: {formula}; Emission: {emission or 'N/A'}; Decay: {decay or 'N/A'}"
    
    # If exact match not found, find similar formulas
    formula_col = _find_col(list(_df_cache.columns), ["formula", "compound", "name"])
    if not formula_col:
        return f"Not found: '{formula}' and no formula column available"
    
    # Get unique formulas and calculate similarities
    formulas = _df_cache[formula_col].astype(str).dropna().unique()
    target = str(formula).strip().lower()
    
    scored = [(f, difflib.SequenceMatcher(None, target, str(f).lower()).ratio()) 
              for f in formulas if str(f).strip()]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    if not scored:
        return f"Not found: '{formula}'"
    
    # Get top 3 similar formulas
    top_similar = scored[:3]
    columns = list(_df_cache.columns)
    emission_col = _find_col(columns, ["emission", "em_max", "emission_max", "lambda_em"])
    decay_col = _find_col(columns, ["decay", "lifetime", "tau"])
    
    lines = [f"Exact match not found for '{formula}'. Here are similar formulas:"]
    
    for cand, score in top_similar:
        if score > 0.3:  # Only show if similarity > 30%
            row = _df_cache[_df_cache[formula_col] == cand].iloc[0]
            emission = _get_numeric(row, emission_col) if emission_col else None
            decay = _get_numeric(row, decay_col) if decay_col else None
            lines.append(f"- {cand} (similarity: {score:.2f}) | Emission: {emission or 'N/A'}; Decay: {decay or 'N/A'}")
    
    if len(lines) == 1:  # Only header line
        return f"Not found: '{formula}' (no similar formulas found)"
    
    return "\n".join(lines)


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
    
    formula_col = _find_col(list(_df_cache.columns), ["inorganic phosphor", "formula", "compound", "name"])
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
    x_col = _find_col(columns, ["CIE x coordinate", "x", "cie_x", "chromaticity x", "cie x"])
    y_col = _find_col(columns, ["CIE y coordinate", "y", "cie_y", "chromaticity y", "cie y"]) 
    Y_col = _find_col(columns, ["Y", "cie_y", "luminance", "intensity"]) 
    
    if x_col and y_col:
        x_val = _get_numeric(row, x_col)
        y_val = _get_numeric(row, y_col)
        Y_val = _get_numeric(row, Y_col) if Y_col else 1.0
        if x_val is not None and y_val is not None and Y_val is not None:
            hex_str = xyY_to_hex(x_val, y_val, Y_val)
            if not hex_str.startswith("Error"):
                return f"Formula: {formula}; Color: {hex_str}; Source: xyY(x={x_val}, y={y_val}, Y={Y_val})"
    
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


@phosphor_agent.tool_plain
def debug_formula_search(formula: str, file_path: Optional[str] = None) -> str:
    """Debug function to check what formulas are in the database and find matches"""
    global _df_cache
    
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status
    
    if _df_cache is None:
        return "Error: Database not loaded"
    
    formula_col = _find_col(list(_df_cache.columns), ["inorganic phosphor", "formula", "compound", "name"])
    if not formula_col:
        return "Error: No formula column found"
    
    # Show all formulas in database
    all_formulas = _df_cache[formula_col].astype(str).dropna().unique()
    
    target = str(formula).strip().lower()
    
    # Try different matching strategies
    exact_matches = []
    contains_matches = []
    similar_matches = []
    
    for f in all_formulas:
        f_str = str(f).strip()
        f_lower = f_str.lower()
        
        # Exact match (case-insensitive)
        if f_lower == target:
            exact_matches.append(f_str)
        
        # Contains match
        if target in f_lower or f_lower in target:
            contains_matches.append(f_str)
        
        # Similarity match
        similarity = difflib.SequenceMatcher(None, target, f_lower).ratio()
        if similarity > 0.7:
            similar_matches.append((f_str, similarity))
    
    # Sort similar matches by similarity
    similar_matches.sort(key=lambda x: x[1], reverse=True)
    
    lines = [f"Debug search for: '{formula}'"]
    lines.append(f"Total formulas in DB: {len(all_formulas)}")
    lines.append("")
    
    if exact_matches:
        lines.append("EXACT MATCHES:")
        for match in exact_matches:
            lines.append(f"  ✓ {match}")
    else:
        lines.append("NO EXACT MATCHES")
    
    if contains_matches:
        lines.append("")
        lines.append("CONTAINS MATCHES:")
        for match in contains_matches:
            lines.append(f"  ~ {match}")
    
    if similar_matches:
        lines.append("")
        lines.append("SIMILAR MATCHES (similarity > 0.7):")
        for match, sim in similar_matches[:5]:
            lines.append(f"  ≈ {match} (similarity: {sim:.3f})")
    
    # Show first 10 formulas in DB for reference
    lines.append("")
    lines.append("FIRST 10 FORMULAS IN DATABASE:")
    for i, f in enumerate(all_formulas[:10]):
        lines.append(f"  {i+1}. {f}")
    
    return "\n".join(lines)


@phosphor_agent.tool_plain
def debug_database_info(file_path: Optional[str] = None) -> str:
    """Debug function to check database structure and content"""
    global _df_cache
    
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status
    
    if _df_cache is None:
        return "Error: Database not loaded"
    
    lines = ["DATABASE DEBUG INFO:"]
    lines.append(f"Total rows: {len(_df_cache)}")
    lines.append(f"Total columns: {len(_df_cache.columns)}")
    lines.append("")
    
    lines.append("COLUMNS:")
    for i, col in enumerate(_df_cache.columns):
        lines.append(f"  {i+1}. {col}")
    
    lines.append("")
    lines.append("FIRST 5 ROWS:")
    for i, (_, row) in enumerate(_df_cache.head().iterrows()):
        lines.append(f"Row {i+1}:")
        for col in _df_cache.columns:
            lines.append(f"  {col}: {row[col]}")
        lines.append("")
    
    # Check for formula column specifically
    formula_col = _find_col(list(_df_cache.columns), ["inorganic phosphor", "formula", "compound", "name"])
    if formula_col:
        lines.append(f"FORMULA COLUMN FOUND: '{formula_col}'")
        lines.append("FIRST 10 FORMULAS:")
        formulas = _df_cache[formula_col].astype(str).dropna().unique()
        for i, f in enumerate(formulas[:10]):
            lines.append(f"  {i+1}. {f}")
    else:
        lines.append("NO FORMULA COLUMN FOUND!")
    
    return "\n".join(lines)


async def call_phosphor_lookup_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""Call phosphor lookup agent to execute the task: {role}

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do. 
        
        The phosphor lookup agent can:
        - load_phosphor_db: Load CSV/Excel database from path or auto-resolve from env var PHOSPHOR_DB_PATH
        - lookup_by_formula: Find exact formula match and return emission max and decay time
        - similar_formulas: Find top-k most similar formulas with similarity scores and emission/decay data
        - formula_to_hex_color: Convert formula to hex color using CIE xyY/XYZ values from database
        
        Example messages:
        - "Load the phosphor database from './data/Inorganic_Phosphor.csv'"
        - "Find emission and decay for formula 'SrAl2O4:Eu2+'"
        - "Find 5 most similar formulas to 'YAG:Ce' with their emission and decay data"
        - "Convert formula 'SrAl2O4:Eu2+' to hex color using CIE values from the database"
    """
    agent_name = "PhosphorLookupAgent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    result = await phosphor_agent.run(message2agent, deps=deps)
    output = result.output
    
    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")
    
    return output
    