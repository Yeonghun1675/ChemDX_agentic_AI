from typing import Optional, List
import os
import pandas as pd
import difflib

from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger
from chemdx_agent.utils import make_tool_message


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

@phosphor_agent.system_prompt(dynamic=True)
def dynamic_system_prompt(ctx: RunContext[AgentState]) -> str:
    deps = ctx.deps
    return working_memory_prompt.format(
        main_goal = deps.main_task,
        working_memory = deps.working_memory_description,
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
        "databases/Inorganic_Phosphor_Optical_Properties_DB.csv",  # Use actual database file
        "databases/Inorganic_Phosphor.csv", "Inorganic_Phosphor.csv",
        "databases/Inorganic_Phosphor.xlsx", "Inorganic_Phosphor.xlsx",  # fallback for Excel
        "databases/Inorganic_Phosphor.xls", "Inorganic_Phosphor.xls"
    ])
    return next((c for c in candidates if c and os.path.exists(c)), None)


def _split_paths(text: str) -> List[str]:
    for sep in [";", ",", ":"]:
        text = text.replace(sep, "|")
    return [p.strip() for p in text.split("|") if p.strip()]


def _resolve_paths(file_path: Optional[str]) -> List[str]:
    candidates: List[str] = []
    if file_path:
        candidates.extend(_split_paths(file_path))
    if env_path := os.getenv("PHOSPHOR_DB_PATH"):
        candidates.extend(_split_paths(env_path))
    candidates.extend([
        "databases/Inorganic_Phosphor_Optical_Properties_DB.csv",
        "databases/Inorganic_Phosphor.csv", "Inorganic_Phosphor.csv",
        "databases/Inorganic_Phosphor.xlsx", "Inorganic_Phosphor.xlsx",
        "databases/Inorganic_Phosphor.xls", "Inorganic_Phosphor.xls",
        "databases/estm.csv", "databases/MatDX_EF.csv",
    ])
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return [c for c in uniq if os.path.exists(c)]


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
        # Filter out NaN values in formula column
        df_clean = df.dropna(subset=[formula_col])
        df_clean = df_clean[df_clean[formula_col].astype(str).str.strip() != '']
        match = df_clean[df_clean[formula_col].astype(str).str.strip().str.lower() == target]
        return match.iloc[0] if not match.empty else None
    except Exception:
        return None


def _get_numeric(row: pd.Series, col: str) -> Optional[float]:
    """Extract numeric value from row column"""
    try:
        return float(pd.to_numeric(row[col])) if col in row and pd.notna(row[col]) else None
    except Exception:
        return None


def _get_numeric_value(val) -> Optional[float]:
    """Extract numeric from a scalar, handling percentage strings like '85%'"""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        text = str(val).strip()
        if text.endswith('%'):
            return float(pd.to_numeric(text[:-1]))
        return float(pd.to_numeric(text))
    except Exception:
        return None


@phosphor_agent.tool_plain
def load_phosphor_db(file_path: Optional[str] = None) -> str:
    """Load and combine CSV/Excel databases into memory for lookup agent."""
    global _df_cache, _path_cache
    paths = _resolve_paths(file_path)
    if not paths:
        return "Error: Could not resolve any DB file"
    frames: List[pd.DataFrame] = []
    names: List[str] = []
    for p in paths:
        try:
            df = pd.read_csv(p) if p.lower().endswith('.csv') else pd.read_excel(p)
            df["source"] = os.path.basename(p)
            frames.append(df)
            names.append(os.path.basename(p))
        except Exception:
            continue
    if not frames:
        return "Error: Resolved files exist but none could be loaded"
    _df_cache = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    
    # Clean the database: remove unnamed columns and filter out problematic rows
    if _df_cache is not None:
        # Remove columns that start with "Unnamed"
        unnamed_cols = [col for col in _df_cache.columns if str(col).startswith('Unnamed')]
        if unnamed_cols:
            _df_cache = _df_cache.drop(columns=unnamed_cols)
        
        # Remove rows where key columns are all NaN
        key_cols = ['Inorganic phosphor', 'Host', '1st dopant', 'Emission max. (nm)', 'Decay time (ns)']
        existing_key_cols = [col for col in key_cols if col in _df_cache.columns]
        if existing_key_cols:
            _df_cache = _df_cache.dropna(subset=existing_key_cols, how='all')
        
        # Remove rows where formula is NaN or contains only whitespace
        formula_col = _find_col(list(_df_cache.columns), ["inorganic phosphor", "formula", "compound", "name"])
        if formula_col:
            _df_cache = _df_cache.dropna(subset=[formula_col])
            _df_cache = _df_cache[_df_cache[formula_col].astype(str).str.strip() != '']
        
        # Remove rows that have too many NaN values (more than 80% NaN)
        if len(_df_cache.columns) > 0:
            nan_threshold = len(_df_cache.columns) * 0.8
            _df_cache = _df_cache.dropna(thresh=len(_df_cache.columns) - nan_threshold)
        
        # Clean up string values that are 'nan' or 'NaN'
        for col in _df_cache.columns:
            if _df_cache[col].dtype == 'object':
                _df_cache[col] = _df_cache[col].replace(['nan', 'NaN', ''], pd.NA)
    
    _path_cache = ";".join(paths)
    return f"Loaded {len(_df_cache)} rows from {len(frames)} sources ({', '.join(names)})"


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
    
    row = _get_row(_df_cache, formula)
    if row is not None:
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
    # Filter out NaN and empty values
    df_clean = _df_cache.dropna(subset=[formula_col])
    df_clean = df_clean[df_clean[formula_col].astype(str).str.strip() != '']
    formulas = df_clean[formula_col].astype(str).unique()
    target = str(formula).strip().lower()
    
    scored = [(f, difflib.SequenceMatcher(None, target, str(f).lower()).ratio()) 
              for f in formulas if str(f).strip() and str(f).lower() not in ['nan', '']]
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
    # Filter out NaN and empty values
    df_clean = _df_cache.dropna(subset=[formula_col])
    df_clean = df_clean[df_clean[formula_col].astype(str).str.strip() != '']
    formulas = df_clean[formula_col].astype(str).unique()
    target = str(formula).strip().lower()
    
    scored = [(f, difflib.SequenceMatcher(None, target, str(f).lower()).ratio()) 
              for f in formulas if str(f).strip() and str(f).lower() not in ['nan', '']]
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
def search_candidates(min_emission_nm: float, max_emission_nm: float, max_decay_ns: float, min_iqe_percent: float, file_path: Optional[str] = None) -> str:
    """Search phosphor candidates that satisfy emission range [min,max] (nm), decay time <= max_decay_ns, and internal QE >= min_iqe_percent.

    Returns a ranked list with host, dopant, emission, decay, IQE, CIE/hex color if available.
    """
    global _df_cache

    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status
    if _df_cache is None:
        return "Error: Database not loaded"

    df = _df_cache
    cols = list(df.columns)
    host_col = _find_col(cols, ["host", "matrix"]) or _find_col(cols, ["inorganic phosphor", "formula", "compound"])  # fallback
    dopant_col = _find_col(cols, ["1st dopant", "dopant", "activator"])  # activator element
    doping_conc_col = _find_col(cols, ["1st doping concentration", "doping concentration", "concentration"])  # doping concentration
    emission_col = _find_col(cols, ["Emission max. (nm)", "emission max", "emission", "lambda_em"])  # nm
    decay_col = _find_col(cols, ["decay time (ns)", "decay", "lifetime", "tau"])  # ns
    iqe_col = _find_col(cols, ["internal quantum efficiency", "int. qe", "iqe", "quantum efficiency"])  # % or fraction
    x_col = _find_col(cols, ["CIE x coordinate", "chromaticity x", "cie x", "x"])  # cie x
    y_col = _find_col(cols, ["CIE y coordinate", "chromaticity y", "cie y", "y"])  # cie y
    Y_col = _find_col(cols, ["Y", "luminance", "intensity"])  # luminance optional

    if not emission_col:
        return "Error: Emission column not found in DB"
    if not decay_col:
        return "Error: Decay time column not found in DB"
    if not iqe_col:
        return "Error: Internal QE column not found in DB"

    candidates = []
    for _, row in df.iterrows():
        # Skip rows with problematic values
        if host_col and (pd.isna(row.get(host_col)) or str(row.get(host_col)).strip() in ['nan', 'NaN', '']):
            continue
        if dopant_col and (pd.isna(row.get(dopant_col)) or str(row.get(dopant_col)).strip() in ['nan', 'NaN', '']):
            continue
            
        em = _get_numeric_value(row.get(emission_col))
        dt_ns = _get_numeric_value(row.get(decay_col))
        iqe = _get_numeric_value(row.get(iqe_col))
        if em is None or dt_ns is None or iqe is None:
            continue
        if not (min_emission_nm <= em <= max_emission_nm):
            continue
        if not (dt_ns <= max_decay_ns):
            continue
        # IQE might be 0-1 or 0-100; normalize heuristically
        iqe_norm = iqe * 100.0 if iqe <= 1.0 else iqe
        if iqe_norm < min_iqe_percent:
            continue

        # color via CIE if available
        hex_color = None
        color_name = None
        try:
            x = _get_numeric_value(row.get(x_col)) if x_col else None
            y = _get_numeric_value(row.get(y_col)) if y_col else None
            Y = _get_numeric_value(row.get(Y_col)) if Y_col else 1.0
            if x is not None and y is not None and Y is not None:
                hex_color = xyY_to_hex(x, y, Y)
        except Exception:
            pass

        host = str(row.get(host_col)).strip() if host_col and not pd.isna(row.get(host_col)) else "N/A"
        dopant = str(row.get(dopant_col)).strip() if dopant_col and not pd.isna(row.get(dopant_col)) else "N/A"
        doping_conc = str(row.get(doping_conc_col)).strip() if doping_conc_col and not pd.isna(row.get(doping_conc_col)) else "N/A"
        
        # Skip if host or dopant are still problematic
        if host in ['nan', 'NaN', ''] or dopant in ['nan', 'NaN', '']:
            continue
            
        score = (min(max_emission_nm - em, em - min_emission_nm, key=abs) if min_emission_nm <= em <= max_emission_nm else 0) + (min(100.0, iqe_norm) / 100.0)
        candidates.append({
            "host": host,
            "dopant": dopant,
            "doping_conc": doping_conc,
            "emission_nm": em,
            "decay_ns": dt_ns,
            "iqe_percent": iqe_norm,
            "hex": hex_color,
            "score": score,
        })

    if not candidates:
        return "No candidates matched all constraints"

    candidates.sort(key=lambda x: x["score"], reverse=True)
    lines: List[str] = [
        f"Candidates for {min_emission_nm}-{max_emission_nm} nm, decay <= {max_decay_ns} ns, IQE >= {min_iqe_percent}%:",
    ]
    for i, c in enumerate(candidates[:10], 1):
        hex_str = f", hex={c['hex']}" if c.get('hex') else ""
        doping_str = f", conc={c['doping_conc']}" if c['doping_conc'] != "N/A" else ""
        lines.append(
            f"{i}. {c['host']}:{c['dopant']}{doping_str} | emission={c['emission_nm']:.0f} nm | decay={c['decay_ns']:.0f} ns | IQE={c['iqe_percent']:.0f}%{hex_str}"
        )
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
    
    row = _get_row(_df_cache, formula)
    if row is None:
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
    # Filter out NaN and empty values
    df_clean = _df_cache.dropna(subset=[formula_col])
    df_clean = df_clean[df_clean[formula_col].astype(str).str.strip() != '']
    all_formulas = df_clean[formula_col].astype(str).unique()
    
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
    valid_formulas = [f for f in all_formulas if str(f).strip() and str(f).lower() not in ['nan', '']]
    for i, f in enumerate(valid_formulas[:10]):
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
    
    # Filter out unnamed columns for display
    valid_cols = [col for col in _df_cache.columns if not str(col).startswith('Unnamed')]
    lines.append(f"Valid columns (excluding unnamed): {len(valid_cols)}")
    lines.append("")
    
    lines.append("COLUMNS:")
    for i, col in enumerate(valid_cols):
        lines.append(f"  {i+1}. {col}")
    
    lines.append("")
    lines.append("FIRST 5 ROWS (cleaned):")
    for i, (_, row) in enumerate(_df_cache.head().iterrows()):
        lines.append(f"Row {i+1}:")
        for col in valid_cols:
            val = row[col]
            # Skip NaN values and show only meaningful data
            if pd.notna(val) and str(val).strip() not in ['nan', 'NaN', '']:
                lines.append(f"  {col}: {val}")
        lines.append("")
    
    # Check for formula column specifically
    formula_col = _find_col(list(_df_cache.columns), ["inorganic phosphor", "formula", "compound", "name"])
    if formula_col:
        lines.append(f"FORMULA COLUMN FOUND: '{formula_col}'")
        lines.append("FIRST 10 FORMULAS:")
        # Filter out NaN and empty values
        df_clean = _df_cache.dropna(subset=[formula_col])
        df_clean = df_clean[df_clean[formula_col].astype(str).str.strip() != '']
        formulas = df_clean[formula_col].astype(str).unique()
        valid_formulas = [f for f in formulas if str(f).strip() and str(f).lower() not in ['nan', '']]
        for i, f in enumerate(valid_formulas[:10]):
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
        - "Load the phosphor database from Inorganic_Phosphor_Optical_Properties_DB.csv"
        - "Find emission and decay for formula 'SrAl2O4:Eu2+'"
        - "Find 5 most similar formulas to 'YAG:Ce' with their emission and decay data"
        - "Convert formula 'SrAl2O4:Eu2+' to hex color using CIE values from the database"
    """
    agent_name = "PhosphorLookupAgent"
    deps = ctx.deps or AgentState()

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}ㅤ")
    
    user_prompt = f"Current Task of your role: {message2agent}"
    result = await phosphor_agent.run(user_prompt, deps=deps)
    output = result.output
    if hasattr(deps, "add_working_memory"):
        deps.add_working_memory(agent_name, message2agent)
    if hasattr(deps, "increment_step"):
        deps.increment_step()
    logger.info(f"[{agent_name}] Action: {output.action}ㅤ")
    
    list_tool_log = make_tool_message(result)
    for log in list_tool_log:
        logger.info(log)
    
    logger.info(f"[{agent_name}] Result: {output.result}ㅤ")
    
    return output
    