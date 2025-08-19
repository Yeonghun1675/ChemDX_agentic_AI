from typing import Optional, List
import os
import pandas as pd

from pydantic_ai import Agent, RunContext

from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger
from chemdx_agent.agents.cie_to_color_agent.subagent import (
    xyY_to_hex as convert_xyY_to_hex,
    xyz_to_hex as convert_XYZ_to_hex,
)


name = "PhosphorLookupAgent"
role = (
    "Given a chemical formula, look up emission maximum and decay (lifetime) from an Inorganic_Phosphor Excel database."
)
context = (
    "Use ONLY the provided tools. The database is an Excel file. If a file path is not provided, use the PHOSPHOR_DB_PATH environment variable or try default locations like 'data/Inorganic_Phosphor.xlsx' or './Inorganic_Phosphor.xlsx'."
)


system_prompt = f"""You are the {name}. You can use available tools or request help from specialized sub-agents that perform specific tasks. You must only carry out the role assigned to you. If a request is outside your capabilities, you should ask for support from the appropriate agent instead of trying to handle it yourself.

Your Currunt Role: {role}
Important Context: {context}
"""


phosphor_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt=system_prompt,
)


_phosphor_df_cache: Optional[pd.DataFrame] = None
_phosphor_db_path_cache: Optional[str] = None


def _resolve_db_path(file_path: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if file_path:
        candidates.append(file_path)
    env_path = os.getenv("PHOSPHOR_DB_PATH")
    if env_path:
        candidates.append(env_path)
    # Defaults
    candidates.extend([
        os.path.join("data", "Inorganic_Phosphor.xlsx"),
        os.path.join("data", "Inorganic_Phosphor.xls"),
        "Inorganic_Phosphor.xlsx",
        "Inorganic_Phosphor.xls",
    ])

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


@phosphor_agent.tool_plain
def load_phosphor_db(file_path: Optional[str] = None) -> str:
    """Load the Inorganic_Phosphor Excel database into memory.
    args:
        file_path: (Optional[str]) Path to Excel database. If omitted, uses env var PHOSPHOR_DB_PATH or common defaults.
    output:
        (str) Human-readable load status, including resolved path and row count.
    """
    global _phosphor_df_cache, _phosphor_db_path_cache

    resolved = _resolve_db_path(file_path)
    if not resolved:
        return (
            "Error: Could not resolve database path. Provide 'file_path' or set PHOSPHOR_DB_PATH, "
            "or place the file at './data/Inorganic_Phosphor.xlsx'."
        )

    try:
        df = pd.read_excel(resolved)
    except Exception as exc:  # pragma: no cover
        return f"Error loading Excel file at '{resolved}': {exc}"

    _phosphor_df_cache = df
    _phosphor_db_path_cache = resolved
    return f"Loaded {len(df)} rows from '{resolved}'."


def _find_column(candidates: List[str], columns: List[str]) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in columns}
    lower_columns = list(lower_to_original.keys())
    for keyword in candidates:
        for col in lower_columns:
            if keyword in col:
                return lower_to_original[col]
    return None


@phosphor_agent.tool_plain
def lookup_by_formula(formula: str, file_path: Optional[str] = None) -> str:
    """Find emission maximum and decay (lifetime) for a given formula in the Excel DB.
    args:
        formula: (str) Chemical formula to look up (case-insensitive exact match).
        file_path: (Optional[str]) Excel database path; if omitted, use cached/auto-resolved path.
    output:
        (str) A concise string summarizing emission max and decay time if available, or an error/not-found message.
    """
    global _phosphor_df_cache, _phosphor_db_path_cache

    if _phosphor_df_cache is None or (file_path and file_path != _phosphor_db_path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status

    df = _phosphor_df_cache
    if df is None:
        return "Error: Database is not loaded."

    # Identify likely columns
    columns = list(df.columns)
    formula_col = _find_column(["formula", "compound", "name"], columns)
    emission_col = _find_column([
        "emission", "em_max", "emission_max", "lambda_em", "em-wavelength", "em wl",
    ], columns)
    decay_col = _find_column(["decay", "lifetime", "tau"], columns)

    if not formula_col:
        return "Error: Could not find a 'formula' column in the database."

    # Normalize and match formula (case-insensitive, trim spaces)
    target = str(formula).strip().lower()
    try:
        match_df = df[df[formula_col].astype(str).str.strip().str.lower() == target]
    except Exception:
        return "Error: Failed to process the 'formula' column."

    if match_df.empty:
        return f"Not found: No entries for formula '{formula}'."

    # Use first match; could be extended to aggregate
    row = match_df.iloc[0]

    emission_value = None
    if emission_col and emission_col in row and pd.notna(row[emission_col]):
        emission_value = str(row[emission_col])

    decay_value = None
    if decay_col and decay_col in row and pd.notna(row[decay_col]):
        decay_value = str(row[decay_col])

    pieces: List[str] = [f"Formula: {formula}"]
    if emission_value is not None:
        pieces.append(f"Emission max: {emission_value}")
    else:
        pieces.append("Emission max: N/A")
    if decay_value is not None:
        pieces.append(f"Decay time: {decay_value}")
    else:
        pieces.append("Decay time: N/A")

    return "; ".join(pieces)


# Helper for robust CIE column detection
def _find_col_exact_then_contains(
    columns: List[str], exact_candidates: List[str], contains_candidates: List[str]
) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in columns}
    # exact first
    for key in exact_candidates:
        k = key.lower()
        if k in lower_to_original:
            return lower_to_original[k]
    # then contains
    for key in contains_candidates:
        k = key.lower()
        for col in lower_to_original:
            if k in col:
                return lower_to_original[col]
    return None


@phosphor_agent.tool_plain
def formula_to_hex_color(formula: str, file_path: Optional[str] = None) -> str:
    """Resolve a formula to CIE values in the DB and convert to sRGB hex.
    Prefers xyY if present, otherwise falls back to XYZ.

    args:
        formula: (str) Chemical formula (exact match, case-insensitive).
        file_path: (Optional[str]) Excel DB path; if omitted, uses cached/auto-resolved path.
    output:
        (str) Hex color string with provenance (xyY/XYZ) or a clear error message.
    """
    global _phosphor_df_cache, _phosphor_db_path_cache

    if _phosphor_df_cache is None or (file_path and file_path != _phosphor_db_path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status

    df = _phosphor_df_cache
    if df is None:
        return "Error: Database is not loaded."

    columns = list(df.columns)
    # Identify formula column
    formula_col = _find_column(["formula", "compound", "name"], columns)
    if not formula_col:
        return "Error: Could not find a 'formula' column in the database."

    # Find row by exact match (case-insensitive)
    target = str(formula).strip().lower()
    try:
        match_df = df[df[formula_col].astype(str).str.strip().str.lower() == target]
    except Exception:
        return "Error: Failed to process the 'formula' column."

    if match_df.empty:
        return f"Not found: No entries for formula '{formula}'."

    row = match_df.iloc[0]

    # Detect xyY columns
    x_col = _find_col_exact_then_contains(
        columns,
        exact_candidates=["x", "cie x"],
        contains_candidates=["cie_x", "chromaticity x", "cie x", "x (cie)"]
    )
    y_col = _find_col_exact_then_contains(
        columns,
        exact_candidates=["y", "cie y"],
        contains_candidates=["cie_y", "chromaticity y", "cie y", "y (cie)"]
    )
    Y_col = _find_col_exact_then_contains(
        columns,
        exact_candidates=["Y", "cie Y", "luminance", "intensity"],
        contains_candidates=["cie_y_luminance", "luminance", "relative luminance", "cie Y"]
    )

    def _get_num(v) -> Optional[float]:
        try:
            return float(pd.to_numeric(v))
        except Exception:
            return None

    # Try xyY first
    if x_col and y_col and Y_col and x_col in row and y_col in row and Y_col in row:
        x_val = _get_num(row[x_col])
        y_val = _get_num(row[y_col])
        Y_val = _get_num(row[Y_col])
        if x_val is not None and y_val is not None and Y_val is not None:
            hex_str = convert_xyY_to_hex(float(x_val), float(y_val), float(Y_val))
            if hex_str.startswith("Error"):
                return hex_str
            return (
                f"Formula: {formula}; Color: {hex_str}; Source: xyY(x={x_val}, y={y_val}, Y={Y_val})"
            )

    # Detect XYZ columns
    X_col = _find_col_exact_then_contains(
        columns,
        exact_candidates=["X", "cie X"],
        contains_candidates=["cie_x_tristimulus", "tristimulus x", "X (cie)"]
    )
    Y2_col = _find_col_exact_then_contains(
        columns,
        exact_candidates=["Y", "cie Y", "luminance"],
        contains_candidates=["cie_y_luminance", "relative luminance", "Y (cie)"]
    )
    Z_col = _find_col_exact_then_contains(
        columns,
        exact_candidates=["Z", "cie Z"],
        contains_candidates=["cie_z_tristimulus", "tristimulus z", "Z (cie)"]
    )

    if X_col and Y2_col and Z_col and X_col in row and Y2_col in row and Z_col in row:
        X_val = _get_num(row[X_col])
        Y_val = _get_num(row[Y2_col])
        Z_val = _get_num(row[Z_col])
        if X_val is not None and Y_val is not None and Z_val is not None:
            hex_str = convert_XYZ_to_hex(float(X_val), float(Y_val), float(Z_val))
            if hex_str.startswith("Error"):
                return hex_str
            return (
                f"Formula: {formula}; Color: {hex_str}; Source: XYZ(X={X_val}, Y={Y_val}, Z={Z_val})"
            )

    return (
        "Not found: CIE columns (xyY or XYZ) were not detected or contained non-numeric values for the matched row."
    )

# Similarity search for near-matching formulas
@phosphor_agent.tool_plain
def similar_formulas(formula: str, top_k: int = 5, file_path: Optional[str] = None) -> str:
    """Return up to top_k most similar formulas with their emission and decay values when available.
    args:
        formula: (str) Query formula to compare against database entries (case-insensitive).
        top_k: (int) Number of candidates to return (default 5).
        file_path: (Optional[str]) Excel database path; if omitted, use cached/auto-resolved path.
    output:
        (str) A multi-line string listing candidates ranked by similarity with emission/decay fields when present.
    """
    import difflib

    global _phosphor_df_cache, _phosphor_db_path_cache

    if _phosphor_df_cache is None or (file_path and file_path != _phosphor_db_path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status

    df = _phosphor_df_cache
    if df is None:
        return "Error: Database is not loaded."

    columns = list(df.columns)
    formula_col = _find_column(["formula", "compound", "name"], columns)
    emission_col = _find_column([
        "emission", "em_max", "emission_max", "lambda_em", "em-wavelength", "em wl",
    ], columns)
    decay_col = _find_column(["decay", "lifetime", "tau"], columns)

    if not formula_col:
        return "Error: Could not find a 'formula' column in the database."

    # Prepare candidates set
    try:
        series = df[formula_col].astype(str).fillna("")
    except Exception:
        return "Error: Failed to process the 'formula' column."

    unique_formulas = series.dropna().unique().tolist()
    target = str(formula).strip().lower()

    def sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    scored = []
    for cand in unique_formulas:
        cand_norm = str(cand).strip().lower()
        if not cand_norm:
            continue
        scored.append((cand, sim(target, cand_norm)))

    if not scored:
        return "Not found: database has no formula candidates."

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: max(1, int(top_k))]

    lines: List[str] = []
    for cand, score in top:
        row = df[df[formula_col].astype(str) == str(cand)].iloc[0]
        emission_value = None
        if emission_col and emission_col in row and pd.notna(row[emission_col]):
            emission_value = str(row[emission_col])
        decay_value = None
        if decay_col and decay_col in row and pd.notna(row[decay_col]):
            decay_value = str(row[decay_col])
        em_text = emission_value if emission_value is not None else "N/A"
        dc_text = decay_value if decay_value is not None else "N/A"
        lines.append(f"{cand} | similarity={score:.3f} | emission={em_text} | decay={dc_text}")

    header = f"Query: {formula} | top_k={top_k}"
    return "\n".join([header] + lines)

# call agent function
async def call_phosphor_lookup_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""Call phosphor lookup agent to execute the task: {role}

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do.
    """
    agent_name = name
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    result = await phosphor_agent.run(message2agent, deps=deps)
    output = result.output

    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")

    return output


