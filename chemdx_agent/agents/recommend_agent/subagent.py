from typing import Optional, List, Tuple
import os
import pandas as pd
import difflib
import colorsys

from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger


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
            f"Decay: {rec['decay_ms']:.1f}ms | "
            f"CIE(x={rec['x']:.3f}, y={rec['y']:.3f}, Y={rec['Y']:.3f}) | "
            f"Color score: {rec['color_score']:.3f} | "
            f"Total score: {rec['combined_score']:.3f}"
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
        
        Example messages:
        - "Load the phosphor database from './data/Inorganic_Phosphor.csv'"
        - "Recommend 5 materials with decay time >= 10ms for color #FF0000 (red)"
        - "Find materials with decay >= 5ms for warm white color range"
        - "Recommend 3 phosphors for blue color with minimum 2ms decay time"
    """

    agent_name = "RecommendAgent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    result = await recommend_agent.run(message2agent, deps=deps)
    output = result.output
    
    # Removed working_memory/increment_step since not guaranteed in deps
    # deps.add_working_memory(agent_name, message2agent)
    # deps.increment_step()

    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")
    
    return output
