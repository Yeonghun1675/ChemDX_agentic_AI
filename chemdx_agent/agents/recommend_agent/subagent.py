from typing import Optional, List, Tuple
import os
import pandas as pd
import difflib
import colorsys

from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger
from chemdx_agent.utils import make_tool_message


def hex_to_hsv(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color to HSV values"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return colorsys.rgb_to_hsv(r, g, b)


def color_similarity(hex1: str, hex2: str) -> float:
    """Calculate color similarity between two hex colors (0-1, higher is more similar)"""
    try:
        hsv1 = hex_to_hsv(hex1)
        hsv2 = hex_to_hsv(hex2)
        
        # Calculate HSV distance (weighted)
        h_diff = min(abs(hsv1[0] - hsv2[0]), 1 - abs(hsv1[0] - hsv2[0]))  # Hue is circular
        s_diff = abs(hsv1[1] - hsv2[1])
        v_diff = abs(hsv1[2] - hsv2[2])
        
        # Weighted distance (hue is most important)
        distance = (h_diff * 0.6 + s_diff * 0.2 + v_diff * 0.2)
        return 1.0 - distance  # Convert to similarity
    except Exception:
        return 0.0


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


def _fmt_float(value: Optional[float], digits: int = 3) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


name = "RecommendAgent"
role = "Recommend phosphor materials with long decay times based on desired color"
context = "Use provided tools. DB path: PHOSPHOR_DB_PATH env var or ./data/Inorganic_Phosphor.csv"

working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

system_prompt = f"""You are {name}. {role}. {context}"""

recommend_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={"temperature": 0.0, "parallel_tool_calls": False},
    system_prompt=system_prompt,
)

@recommend_agent.system_prompt(dynamic=True)
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
    candidates = []
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
    
    # First try exact matches
    for keyword in keywords:
        k = keyword.lower()
        if k in lower_cols:
            return lower_cols[k]
    
    # Then try contains matches
    for keyword in keywords:
        k = keyword.lower()
        for col in lower_cols:
            if k in col:
                return lower_cols[col]
    
    # If still not found, try more flexible matching
    for col in lower_cols:
        col_lower = col.lower()
        for keyword in keywords:
            if keyword.lower() in col_lower or col_lower in keyword.lower():
                return lower_cols[col]
    
    return None


def _get_numeric(row: pd.Series, col: str) -> Optional[float]:
    """Extract numeric value from row column"""
    try:
        return float(pd.to_numeric(row[col])) if col in row and pd.notna(row[col]) else None
    except Exception:
        return None


@recommend_agent.tool_plain
def load_phosphor_db(file_path: Optional[str] = None) -> str:
    """Load and combine CSV/Excel databases into memory for recommendation tasks."""
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
    _path_cache = ";".join(paths)
    return f"Loaded {len(_df_cache)} rows from {len(frames)} sources ({', '.join(names)})"


@recommend_agent.tool_plain
def recommend_by_color(desired_color: str, min_decay_ms: float = 1.0, top_k: int = 5, file_path: Optional[str] = None) -> str:
    """Recommend phosphor materials with long decay times based on desired color using CIE coordinates
    
    args:
        desired_color: (str) Desired color name (e.g., 'blue', 'red', 'green', 'yellow', 'white', etc.)
        min_decay_ms: (float) Minimum decay time in milliseconds (default 1.0ms)
        top_k: (int) Number of recommendations to return (default 5)
        file_path: (Optional[str]) Database path
    output:
        (str) Ranked list of recommendations with formula, CIE coordinates, decay time, and similarity score
    """
    global _df_cache
    
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status
    
    if _df_cache is None:
        return "Error: Database not loaded"
    
    # Define CIE coordinate ranges for different colors based on actual database
    color_ranges = {
        'blue': {'x_min': 0.15, 'x_max': 0.18, 'y_min': 0.08, 'y_max': 0.13},
        'cyan': {'x_min': 0.18, 'x_max': 0.30, 'y_min': 0.25, 'y_max': 0.40},
        'red': {'x_min': 0.60, 'x_max': 0.70, 'y_min': 0.30, 'y_max': 0.40},
        'green': {'x_min': 0.20, 'x_max': 0.30, 'y_min': 0.40, 'y_max': 0.55},
        'yellow': {'x_min': 0.40, 'x_max': 0.50, 'y_min': 0.45, 'y_max': 0.55},
        'white': {'x_min': 0.25, 'x_max': 0.35, 'y_min': 0.25, 'y_max': 0.35},
        'magenta': {'x_min': 0.35, 'x_max': 0.45, 'y_min': 0.15, 'y_max': 0.25},
        'orange': {'x_min': 0.55, 'x_max': 0.65, 'y_min': 0.35, 'y_max': 0.45},
        'purple': {'x_min': 0.25, 'x_max': 0.35, 'y_min': 0.15, 'y_max': 0.25}
    }
    
    target_color = desired_color.lower().strip()
    if target_color not in color_ranges:
        return f"Error: Color '{desired_color}' not supported. Available colors: {', '.join(color_ranges.keys())}"
    
    color_range = color_ranges[target_color]
    
    # Find required columns
    columns = list(_df_cache.columns)
    formula_col = _find_col(columns, ["inorganic phosphor", "formula", "compound", "name", "chemical", "material", "composition"])
    decay_col = _find_col(columns, ["decay time (ns)", "decay", "lifetime", "tau", "decay_time", "lifetime_ms"])
    x_col = _find_col(columns, ["CIE x coordinate", "x", "cie_x", "chromaticity x", "cie x"])
    y_col = _find_col(columns, ["CIE y coordinate", "y", "cie_y", "chromaticity y", "cie y"]) 
    Y_col = _find_col(columns, ["Y", "cie_y", "luminance", "intensity", "cie Y"])
    
    # Debug output
    debug_info = f"Found columns - Formula: {formula_col}, Decay: {decay_col}, x: {x_col}, y: {y_col}, Y: {Y_col}"
    
    if not formula_col:
        return f"Error: No formula column found in database. {debug_info}"
    
    if not decay_col:
        return f"Error: No decay/lifetime column found in database. {debug_info}"
    
    if not all([x_col, y_col]):
        return f"Error: CIE x,y coordinates not found in database. {debug_info}"
    
    # Process each row and calculate scores
    recommendations = []
    debug_info = []
    total_rows = len(_df_cache)
    rows_with_cie = 0
    rows_with_decay = 0
    
    for idx, (_, row) in enumerate(_df_cache.iterrows()):
        # Get decay time
        decay_ns = _get_numeric(row, decay_col)
        if decay_ns is not None:
            rows_with_decay += 1
        if decay_ns is None:
            continue

        # convert ns → ms
        decay_ms = decay_ns / 1_000_000.0
        if decay_ms < min_decay_ms:
            continue

        # Get CIE coordinates
        x_val = _get_numeric(row, x_col)
        y_val = _get_numeric(row, y_col)
        Y_val = _get_numeric(row, Y_col) if Y_col else 1.0

        if x_val is not None and y_val is not None:
            rows_with_cie += 1

        if x_val is None or y_val is None:
            continue

        # Check if coordinates are within color range
        in_range = (color_range['x_min'] <= x_val <= color_range['x_max'] and 
               color_range['y_min'] <= y_val <= color_range['y_max'])

        # Debug: Track first few materials
        if idx < 10:
            debug_info.append(f"Row {idx}: x={x_val:.3f}, y={y_val:.3f}, decay={decay_ms:.3f}ms, in_range={in_range}")

        if not in_range:
            continue

        # Calculate distance from center of color range
        center_x = (color_range['x_min'] + color_range['x_max']) / 2
        center_y = (color_range['y_min'] + color_range['y_max']) / 2
        distance = ((x_val - center_x) ** 2 + (y_val - center_y) ** 2) ** 0.5

        # Calculate combined score (color proximity + decay time bonus)
        color_score = 1.0 - min(distance * 10, 1.0)  # Normalize distance to 0-1
        decay_bonus = min(decay_ms / 100.0, 0.3)  # Cap decay bonus at 30%
        combined_score = color_score + decay_bonus

        recommendations.append({
            'formula': str(row[formula_col]),
            'decay_ms': decay_ms,
            'x': x_val,
            'y': y_val,
            'Y': Y_val,
            'color_score': color_score,
            'combined_score': combined_score
        })
    
    if not recommendations:
        debug_output = "\n".join(debug_info[:10])  # Show first 10 debug entries
        return f"No materials found for color '{desired_color}' with decay time >= {min_decay_ms}ms\n\nDEBUG INFO:\nTotal rows: {total_rows}\nRows with decay data: {rows_with_decay}\nRows with CIE data: {rows_with_cie}\n\nFirst 10 rows:\n{debug_output}"
    
    # Sort by combined score (color proximity + decay bonus)
    recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Format output
    lines = [f"Recommendations for {desired_color} color (min decay: {min_decay_ms}ms):"]
    
    for i, rec in enumerate(recommendations[:top_k], 1):
        lines.append(
            f"{i}. {rec['formula']} | "
            f"Decay: {_fmt_float(rec['decay_ms'], 1)}ms | "
            f"CIE(x={_fmt_float(rec['x'])}, y={_fmt_float(rec['y'])}, Y={_fmt_float(rec['Y'])}) | "
            f"Color score: {_fmt_float(rec['color_score'])} | "
            f"Total score: {_fmt_float(rec['combined_score'])}"
        )
    
    return "\n".join(lines)


@recommend_agent.tool_plain
def recommend_by_color_range(color_range: str, min_decay_ms: float = 1.0, top_k: int = 5, file_path: Optional[str] = None) -> str:
    """Recommend phosphor materials for a color range (e.g., 'warm white', 'cool blue')
    
    args:
        color_range: (str) Color range description (e.g., 'warm white', 'cool blue', 'red-orange')
        min_decay_ms: (float) Minimum decay time in milliseconds (default 1.0ms)
        top_k: (int) Number of recommendations to return (default 5)
        file_path: (Optional[str]) Database path
    output:
        (str) Ranked list of recommendations for the color range
    """
    # Map color ranges to target colors
    range_map = {
        'warm white': 'white',
        'cool white': 'white', 
        'warm yellow': 'yellow',
        'cool blue': 'blue',
        'red-orange': 'orange',
        'green-yellow': 'green',
        'purple-blue': 'purple'
    }
    
    target_color = range_map.get(color_range.lower(), color_range)
    return recommend_by_color(target_color, min_decay_ms, top_k, file_path)


@recommend_agent.tool_plain
def analyze_color_distribution(file_path: Optional[str] = None) -> str:
    """Analyze the color distribution in the phosphor database"""
    global _df_cache
    
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status
    
    if _df_cache is None:
        return "Error: Database not loaded"
    
    # Find required columns
    columns = list(_df_cache.columns)
    x_col = _find_col(columns, ["CIE x coordinate", "x", "cie_x", "chromaticity x", "cie x"])
    y_col = _find_col(columns, ["CIE y coordinate", "y", "cie_y", "chromaticity y", "cie y"])
    
    if not all([x_col, y_col]):
        return "Error: CIE x,y coordinates not found in database"
    
    # Define color ranges
    color_ranges = {
        'blue': {'x_min': 0.10, 'x_max': 0.30, 'y_min': 0.02, 'y_max': 0.20},
        'red': {'x_min': 0.55, 'x_max': 0.80, 'y_min': 0.20, 'y_max': 0.40},
        'green': {'x_min': 0.20, 'x_max': 0.40, 'y_min': 0.50, 'y_max': 0.70},
        'yellow': {'x_min': 0.40, 'x_max': 0.60, 'y_min': 0.40, 'y_max': 0.60},
        'white': {'x_min': 0.25, 'x_max': 0.45, 'y_min': 0.25, 'y_max': 0.45}
    }
    
    lines = ["COLOR DISTRIBUTION ANALYSIS:"]
    
    for color_name, range_def in color_ranges.items():
        count = 0
        for _, row in _df_cache.iterrows():
            x_val = _get_numeric(row, x_col)
            y_val = _get_numeric(row, y_col)
            
            if x_val is not None and y_val is not None:
                in_range = (range_def['x_min'] <= x_val <= range_def['x_max'] and 
                           range_def['y_min'] <= y_val <= range_def['y_max'])
                if in_range:
                    count += 1
        
        lines.append(f"{color_name.capitalize()}: {count} materials")
    
    lines.append("")
    lines.append("SAMPLE CIE COORDINATES (first 10 rows):")
    for i, (_, row) in enumerate(_df_cache.head(10).iterrows()):
        x_val = _get_numeric(row, x_col)
        y_val = _get_numeric(row, y_col)
        if x_val is not None and y_val is not None:
            lines.append(f"  Row {i+1}: x={x_val:.3f}, y={y_val:.3f}")
    
    return "\n".join(lines)


@recommend_agent.tool_plain
def check_database_structure(file_path: Optional[str] = None) -> str:
    """Check the database structure and show available columns"""
    global _df_cache
    
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status
    
    if _df_cache is None:
        return "Error: Database not loaded"
    
    lines = ["DATABASE STRUCTURE CHECK:"]
    lines.append(f"Total rows: {len(_df_cache)}")
    lines.append(f"Total columns: {len(_df_cache.columns)}")
    lines.append("")
    
    lines.append("ALL COLUMNS:")
    for i, col in enumerate(_df_cache.columns):
        lines.append(f"  {i+1}. '{col}'")
    
    lines.append("")
    lines.append("FIRST 3 ROWS:")
    for i, (_, row) in enumerate(_df_cache.head(3).iterrows()):
        lines.append(f"Row {i+1}:")
        for col in _df_cache.columns:
            lines.append(f"  {col}: {row[col]}")
        lines.append("")
    
    return "\n".join(lines)


async def call_recommend_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""Call recommend agent to execute the task: {role}

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do. 
        
        The recommend agent can:
        - load_phosphor_db: Load CSV/Excel database from path or auto-resolve from env var PHOSPHOR_DB_PATH
        - recommend_by_color: Recommend materials based on specific color (hex or name) with long decay times
        - recommend_by_color_range: Recommend materials for color ranges (e.g., 'warm white', 'cool blue')
        - recommend_by_emission_decay_qe: Recommend materials filtered by Emission max (nm) range, maximum Decay time (ns), and minimum Quantum Efficiency (%)
        
        Example messages:
        - "Load the phosphor database from './data/Inorganic_Phosphor.csv'"
        - "Recommend 5 materials with decay time >= 10ms for color #FF0000 (red)"
        - "Find materials with decay >= 5ms for warm white color range"
        - "Recommend 3 phosphors for blue color with minimum 2ms decay time"
        - "Recommend blue phosphors with Emission max in [360, 420] nm, Decay time <= 100 ns, and QE >= 80%"
    """

    agent_name = "RecommendAgent"
    deps = ctx.deps or AgentState()

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    
    user_prompt = f"Current Task of your role: {message2agent}"
    result = await recommend_agent.run(user_prompt, deps=deps)
    output = result.output
    if hasattr(deps, "add_working_memory"):
        deps.add_working_memory(agent_name, message2agent)
    if hasattr(deps, "increment_step"):
        deps.increment_step()
    logger.info(f"[{agent_name}] Action: {output.action}")
    
    list_tool_log = make_tool_message(result)
    for log in list_tool_log:
        logger.info(log)
    
    logger.info(f"[{agent_name}] Result: {output.result}")
    
    return output


@recommend_agent.tool_plain
def recommend_by_emission_decay_qe(
    emission_min_nm: float,
    emission_max_nm: float,
    decay_max_ns: float,
    qe_min_percent: float = 0.0,
    top_k: int = 10,
    file_path: Optional[str] = None,
) -> str:
    """Recommend phosphor materials using Emission max (nm), Decay time (ns), and Quantum Efficiency (%) filters.

    args:
        emission_min_nm: (float) Minimum Emission max in nm (inclusive)
        emission_max_nm: (float) Maximum Emission max in nm (inclusive)
        decay_max_ns: (float) Maximum Decay time in ns (inclusive)
        qe_min_percent: (float) Minimum Quantum Efficiency in % (consider Int. or Ext.), default 0
        top_k: (int) Number of recommendations to return (default 10)
        file_path: (Optional[str]) Database path

    output:
        (str) Ranked list with host, dopant, concentration, Emission max (nm), Decay time (ns), and QE (%)
    """
    global _df_cache

    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status

    if _df_cache is None:
        return "Error: Database not loaded"

    columns = list(_df_cache.columns)

    # Column resolution
    emission_col = _find_col(columns, [
        "Emission max. (nm)", "Emission max (nm)", "Emission max", "emission_max", "emission (nm)", "emission"
    ])
    decay_col = _find_col(columns, [
        "Decay time (ns)", "decay", "lifetime", "tau", "decay_time"
    ])
    iqe_col = _find_col(columns, [
        "Int. quantum efficiency (%)", "Internal quantum efficiency", "IQE", "int qe", "internal qe"
    ])
    eqe_col = _find_col(columns, [
        "Ext. quantum efficiency (%)", "External quantum efficiency", "EQE", "ext qe", "external qe"
    ])
    host_col = _find_col(columns, [
        "Host", "host"
    ])
    dopant1_col = _find_col(columns, [
        "1st dopant", "dopant", "activator", "1st activator"
    ])
    dopant1_conc_col = _find_col(columns, [
        "1st doping concentration", "dopant concentration", "activator concentration", "1st dopant concentration"
    ])
    formula_col = _find_col(columns, [
        "Inorganic phosphor", "formula", "compound", "name", "chemical", "material", "composition"
    ])

    debug_cols = (
        f"emission={emission_col}, decay={decay_col}, IQE={iqe_col}, EQE={eqe_col}, "
        f"host={host_col}, dopant={dopant1_col}, conc={dopant1_conc_col}, formula={formula_col}"
    )

    if not emission_col:
        return f"Error: Emission max column not found. Columns: {debug_cols}"
    if not decay_col:
        return f"Error: Decay time (ns) column not found. Columns: {debug_cols}"

    results: List[dict] = []

    for _, row in _df_cache.iterrows():
        emission_nm = _get_numeric(row, emission_col)
        decay_ns = _get_numeric(row, decay_col)
        iqe_val = _get_numeric(row, iqe_col) if iqe_col else None
        eqe_val = _get_numeric(row, eqe_col) if eqe_col else None

        if emission_nm is None or decay_ns is None:
            continue

        # Apply numeric filters
        if not (emission_min_nm <= emission_nm <= emission_max_nm):
            continue
        if not (decay_ns <= decay_max_ns):
            continue

        # QE filter (accept if either IQE or EQE meets threshold)
        if qe_min_percent > 0:
            has_qe = False
            if iqe_val is not None and iqe_val >= qe_min_percent:
                has_qe = True
            if eqe_val is not None and eqe_val >= qe_min_percent:
                has_qe = True
            if not has_qe:
                continue

        result_qe_val: Optional[float] = None
        result_qe_label = ""
        if iqe_val is not None and (eqe_val is None or iqe_val >= (eqe_val or -1)):
            result_qe_val = iqe_val
            result_qe_label = "IQE"
        elif eqe_val is not None:
            result_qe_val = eqe_val
            result_qe_label = "EQE"

        results.append({
            "host": str(row[host_col]) if host_col and host_col in row else "N/A",
            "dopant": str(row[dopant1_col]) if dopant1_col and dopant1_col in row else "N/A",
            "conc": _fmt_float(_get_numeric(row, dopant1_conc_col), 3) if dopant1_conc_col else "N/A",
            "emission_nm": emission_nm,
            "decay_ns": decay_ns,
            "qe_val": result_qe_val,
            "qe_label": result_qe_label,
            "formula": str(row[formula_col]) if formula_col and formula_col in row else ""
        })

    if not results:
        return (
            "No materials found for the specified filters.\n" 
            f"Filters: Emission in [{emission_min_nm}, {emission_max_nm}] nm; "
            f"Decay <= {decay_max_ns} ns; QE >= {qe_min_percent}%\n"
            f"Resolved columns: {debug_cols}"
        )

    # Sort by QE (desc, preferring entries with QE), then by shorter decay
    def sort_key(item: dict):
        qe_score = item["qe_val"] if item["qe_val"] is not None else -1
        return (-qe_score, item["decay_ns"])

    results.sort(key=sort_key)

    lines: List[str] = [
        (
            f"Recommendations (Emission {emission_min_nm}-{emission_max_nm} nm, "
            f"Decay <= {decay_max_ns} ns, QE >= {qe_min_percent}%):"
        )
    ]

    for i, rec in enumerate(results[:top_k], 1):
        qe_str = f"{rec['qe_label']}={_fmt_float(rec['qe_val'], 1)}%" if rec["qe_val"] is not None else "QE=N/A"
        host_str = rec["host"] if rec["host"] and rec["host"] != "nan" else "N/A"
        dopant_str = rec["dopant"] if rec["dopant"] and rec["dopant"] != "nan" else "N/A"
        lines.append(
            f"{i}. Host: {host_str} | Dopant: {dopant_str} | Conc: {rec['conc']} | "
            f"Emission: {_fmt_float(rec['emission_nm'], 1)} nm | Decay: {_fmt_float(rec['decay_ns'], 1)} ns | {qe_str}"
        )

    return "\n".join(lines)
