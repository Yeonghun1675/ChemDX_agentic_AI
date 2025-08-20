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
    for keyword in keywords:
        for col in lower_cols:
            if keyword.lower() in col:
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
    """Recommend phosphor materials with long decay times based on desired color
    
    args:
        desired_color: (str) Desired color (hex like #FF0000, or color name like 'red', 'blue', etc.)
        min_decay_ms: (float) Minimum decay time in milliseconds (default 1.0ms)
        top_k: (int) Number of recommendations to return (default 5)
        file_path: (Optional[str]) Database path
    output:
        (str) Ranked list of recommendations with formula, color, decay time, and similarity score
    """
    global _df_cache
    
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status
    
    if _df_cache is None:
        return "Error: Database not loaded"
    
    # Normalize color input
    target_hex = desired_color.strip()
    if not target_hex.startswith('#'):
        # Try to convert color name to hex (basic mapping)
        color_map = {
            'red': '#FF0000', 'green': '#00FF00', 'blue': '#0000FF',
            'yellow': '#FFFF00', 'cyan': '#00FFFF', 'magenta': '#FF00FF',
            'white': '#FFFFFF', 'black': '#000000', 'orange': '#FFA500',
            'purple': '#800080', 'pink': '#FFC0CB', 'brown': '#A52A2A'
        }
        target_hex = color_map.get(target_hex.lower(), target_hex)
    
    if not target_hex.startswith('#'):
        return "Error: Invalid color format. Use hex (#RRGGBB) or color name."
    
    # Find required columns
    columns = list(_df_cache.columns)
    formula_col = _find_col(columns, ["formula", "compound", "name"])
    decay_col = _find_col(columns, ["decay", "lifetime", "tau"])
    x_col = _find_col(columns, ["x", "cie_x", "chromaticity x"])
    y_col = _find_col(columns, ["y", "cie_y", "chromaticity y"]) 
    Y_col = _find_col(columns, ["Y", "cie_y", "luminance", "intensity"])
    
    if not formula_col:
        return "Error: No formula column found in database"
    
    if not decay_col:
        return "Error: No decay/lifetime column found in database"
    
    # Process each row and calculate scores
    recommendations = []
    
    for _, row in _df_cache.iterrows():
        # Get decay time
        decay_val = _get_numeric(row, decay_col)
        if decay_val is None or decay_val < min_decay_ms:
            continue
        
        # Calculate color similarity if CIE data available
        color_sim = 0.0
        material_hex = "N/A"
        
        if all(col for col in [x_col, y_col, Y_col]):
            x_val = _get_numeric(row, x_col)
            y_val = _get_numeric(row, y_col)
            Y_val = _get_numeric(row, Y_col)
            
            if all(v is not None for v in [x_val, y_val, Y_val]):
                material_hex = xyY_to_hex(x_val, y_val, Y_val)
                if not material_hex.startswith("Error"):
                    color_sim = color_similarity(target_hex, material_hex)
        
        # Calculate combined score (color similarity + decay time bonus)
        decay_bonus = min(decay_val / 100.0, 0.3)  # Cap decay bonus at 30%
        combined_score = color_sim + decay_bonus
        
        recommendations.append({
            'formula': str(row[formula_col]),
            'decay_ms': decay_val,
            'color_hex': material_hex,
            'color_similarity': color_sim,
            'combined_score': combined_score
        })
    
    if not recommendations:
        return f"No materials found with decay time >= {min_decay_ms}ms"
    
    # Sort by combined score (color similarity + decay bonus)
    recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Format output
    lines = [f"Recommendations for color {desired_color} (min decay: {min_decay_ms}ms):"]
    
    for i, rec in enumerate(recommendations[:top_k], 1):
        lines.append(
            f"{i}. {rec['formula']} | "
            f"Decay: {rec['decay_ms']:.1f}ms | "
            f"Color: {rec['color_hex']} | "
            f"Color similarity: {rec['color_similarity']:.3f} | "
            f"Score: {rec['combined_score']:.3f}"
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
        'warm white': '#FFF8DC',
        'cool white': '#F0F8FF', 
        'warm yellow': '#FFD700',
        'cool blue': '#4169E1',
        'red-orange': '#FF4500',
        'green-yellow': '#ADFF2F',
        'purple-blue': '#9370DB'
    }
    
    target_color = range_map.get(color_range.lower(), color_range)
    return recommend_by_color(target_color, min_decay_ms, top_k, file_path)


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
    
    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")
    
    return output
