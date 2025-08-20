from typing import Optional, List, Tuple
import os
import difflib
import pandas as pd

from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger


name = "ColorTrendAgent"
role = "Analyze how emission/color changes with dopant ratio for a given host and dopant using the phosphor DB"
context = (
    "Use the tools to load the DB, select rows for a specific host and dopant, sort by dopant ratio, "
    "and summarize color/emission trends. DB path can be provided or auto-resolved."
)

system_prompt = f"""You are the {name}. {role}. {context}

Guidelines:
- Prefer exact host matching if possible; otherwise use case-insensitive contains matching.
- The dopant is typically given in the '1st dopant' column.
- Use '1st doping concentration' as the ratio when available.
- If CIE xy data exist, compute an sRGB hex for readability; otherwise report emission wavelength trend.
- Provide a brief natural language summary about red/blue shift and chromaticity movement vs. ratio.
"""

color_trend_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={"temperature": 0.0, "parallel_tool_calls": False},
    system_prompt=system_prompt,
)


# -----------------------
# Helpers and DB handling
# -----------------------
_df_cache: Optional[pd.DataFrame] = None
_path_cache: Optional[str] = None


def _resolve_path(file_path: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if file_path:
        candidates.append(file_path)
    env_path = os.getenv("PHOSPHOR_DB_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        [
            "Inorganic_Phosphor_Optical_Properties_DB.csv",
            "data/Inorganic_Phosphor.csv",
            "Inorganic_Phosphor.csv",
            "data/Inorganic_Phosphor.xlsx",
            "Inorganic_Phosphor.xlsx",
            "data/Inorganic_Phosphor.xls",
            "Inorganic_Phosphor.xls",
        ]
    )
    return next((c for c in candidates if c and os.path.exists(c)), None)


def _split_paths(text: str) -> List[str]:
    for sep in [";", ",", ":"]:
        text = text.replace(sep, "|")
    return [p.strip() for p in text.split("|") if p.strip()]


def _resolve_paths(file_path: Optional[str]) -> List[str]:
    candidates: List[str] = []
    if file_path:
        candidates.extend(_split_paths(file_path))
    env_path = os.getenv("PHOSPHOR_DB_PATH")
    if env_path:
        candidates.extend(_split_paths(env_path))
    candidates.extend([
        "Inorganic_Phosphor_Optical_Properties_DB.csv",
        "data/Inorganic_Phosphor.csv",
        "Inorganic_Phosphor.csv",
        "data/Inorganic_Phosphor.xlsx",
        "Inorganic_Phosphor.xlsx",
        "data/Inorganic_Phosphor.xls",
        "Inorganic_Phosphor.xls",
        "estm.csv",
        "MatDX_EF.csv",
    ])
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return [c for c in uniq if os.path.exists(c)]


def _find_col(columns: List[str], keywords: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for kw in keywords:
        kw_l = kw.lower()
        for lc, orig in lower_map.items():
            if kw_l in lc:
                return orig
    return None


def _get_numeric(val) -> Optional[float]:
    try:
        if pd.isna(val):
            return None
        return float(pd.to_numeric(val))
    except Exception:
        return None


def _xyY_to_hex(x: float, y: float, Y: float) -> str:
    try:
        if y == 0:
            return "Error: y must be non-zero"
        X = (x * Y) / y
        Z = ((1.0 - x - y) * Y) / y
        return _xyz_to_hex(X, Y, Z)
    except Exception as exc:
        return f"Error converting xyY: {exc}"


def _xyz_to_hex(X: float, Y: float, Z: float) -> str:
    R_lin = 3.2406 * X + (-1.5372) * Y + (-0.4986) * Z
    G_lin = (-0.9689) * X + 1.8758 * Y + 0.0415 * Z
    B_lin = 0.0557 * X + (-0.2040) * Y + 1.0570 * Z

    def gamma(c: float) -> float:
        if c <= 0.0031308:
            return 12.92 * c
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055

    R = max(0.0, min(1.0, gamma(R_lin)))
    G = max(0.0, min(1.0, gamma(G_lin)))
    B = max(0.0, min(1.0, gamma(B_lin)))

    r = int(round(R * 255))
    g = int(round(G * 255))
    b = int(round(B * 255))
    return f"#{r:02X}{g:02X}{b:02X}"


# -----------------------
# Color naming helpers
# -----------------------
def _wavelength_to_color_name(nm: float) -> Optional[str]:
    try:
        if nm < 380 or nm > 780:
            return None
        # Coarse mapping tuned so ~610+ nm is recognized as red
        if nm >= 610:
            return "red"
        if nm >= 590:
            return "orange"
        if nm >= 570:
            return "yellow"
        if nm >= 495:
            return "green"
        if nm >= 450:
            return "blue"
        return "violet"
    except Exception:
        return None


def _hex_to_color_name(hex_str: str) -> Optional[str]:
    try:
        hs = hex_str.lstrip('#')
        r = int(hs[0:2], 16)
        g = int(hs[2:4], 16)
        b = int(hs[4:6], 16)
        # Simple rule-of-thumb
        if r > 200 and g < 100 and b < 100:
            return "red"
        if r > 200 and g > 150 and b < 80:
            return "orange"
        if r > 200 and g > 200 and b < 120:
            return "yellow"
        if g > 160 and r < 120 and b < 120:
            return "green"
        if b > 150 and r < 120 and g < 160:
            return "blue"
        if r > 160 and g < 120 and b > 160:
            return "magenta"
        # Fallbacks
        if r > g and r > b:
            return "red"
        if g > r and g > b:
            return "green"
        if b > r and b > g:
            return "blue"
        return None
    except Exception:
        return None


# -----------------------
# Tools
# -----------------------
@color_trend_agent.tool_plain
def load_phosphor_db(file_path: Optional[str] = None) -> str:
    """Load and combine CSV/Excel databases within this agent.

    Supports multi-path input via file_path/PHOSPHOR_DB_PATH; falls back to optical DB + estm + MatDX_EF.
    """
    global _df_cache, _path_cache
    paths = _resolve_paths(file_path)
    if not paths:
        return "Error: Could not resolve any DB file"
    frames: List[pd.DataFrame] = []
    names: List[str] = []
    for p in paths:
        try:
            df = pd.read_csv(p) if p.lower().endswith(".csv") else pd.read_excel(p)
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


@color_trend_agent.tool_plain
def trend_by_ratio(host: str, dopant: str, file_path: Optional[str] = None) -> str:
    """Analyze color/emission change as a function of dopant ratio for a given host and dopant.

    args:
        host: Host composition (e.g., 'Ba2V3O11', 'CaLaB7O13', 'YAG').
        dopant: Activator element (e.g., 'Eu', 'Tb', 'Ce').
        file_path: Optional DB path; if omitted, auto-resolve.

    output:
        Human-readable summary with a ratio→color/emission table and trend description.
    """
    global _df_cache

    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status

    if _df_cache is None:
        return "Error: Database not loaded"

    cols = list(_df_cache.columns)
    host_col = _find_col(cols, ["host"]) or _find_col(cols, ["matrix"])  # primary key
    dopant_col = _find_col(cols, ["1st dopant", "dopant", "activator"])  # activator
    ratio_col = _find_col(cols, ["1st doping concentration", "doping concentration", "ratio", "x"])  # ratio
    x_col = _find_col(cols, ["CIE x coordinate", "chromaticity x", "cie x", "x"])  # cie x
    y_col = _find_col(cols, ["CIE y coordinate", "chromaticity y", "cie y", "y"])  # cie y
    Y_col = _find_col(cols, ["Y", "luminance", "intensity"])  # luminance (optional)
    em_col = _find_col(cols, ["Emission max. (nm)", "emission max", "emission", "lambda_em"])  # emission

    if not host_col or not dopant_col:
        return "Error: Required columns ('Host' and '1st dopant') not found in DB"

    # Filter rows by host and dopant
    host_norm = str(host).strip().lower()
    dopant_norm = str(dopant).strip().lower()

    def _host_match(s: str) -> bool:
        s_norm = str(s).strip().lower()
        return (s_norm == host_norm) or (host_norm in s_norm) or (s_norm in host_norm)

    df = _df_cache
    df_f = df[df[dopant_col].astype(str).str.strip().str.lower() == dopant_norm]
    if host_col in df_f:
        df_f = df_f[df_f[host_col].apply(_host_match)]

    if df_f.empty:
        # Try fuzzy host match if nothing found
        host_vals = df[host_col].astype(str).dropna().unique() if host_col in df else []
        matches = sorted(
            [(h, difflib.SequenceMatcher(None, host_norm, str(h).lower()).ratio()) for h in host_vals],
            key=lambda x: x[1],
            reverse=True,
        )
        hint = f"; similar hosts: {[m[0] for m in matches[:3]]}" if matches else ""
        return f"No records for host='{host}' and dopant='{dopant}'{hint}"

    # Prepare rows with extracted fields
    records: List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[str], Optional[str]]] = []
    for _, row in df_f.iterrows():
        ratio = _get_numeric(row.get(ratio_col)) if ratio_col else None
        em = _get_numeric(row.get(em_col)) if em_col else None
        x = _get_numeric(row.get(x_col)) if x_col else None
        y = _get_numeric(row.get(y_col)) if y_col else None
        Y = _get_numeric(row.get(Y_col)) if Y_col else 1.0

        hex_color: Optional[str] = None
        if x is not None and y is not None and Y is not None:
            hex_color = _xyY_to_hex(x, y, Y)
            if hex_color.startswith("Error"):
                hex_color = None
        color_name: Optional[str] = None
        if hex_color:
            color_name = _hex_to_color_name(hex_color)
        if not color_name and em is not None:
            color_name = _wavelength_to_color_name(em)

        records.append((ratio, em, x, y, hex_color, color_name))

    # Sort by ratio when present; else keep original order
    records.sort(key=lambda r: (float("inf") if r[0] is None else r[0]))

    if not records:
        return f"Found rows but no usable data fields for host='{host}', dopant='{dopant}'"

    # Build lines
    lines: List[str] = []
    lines.append(f"Host: {host} | Dopant: {dopant}")
    lines.append("ratio -> color/emission")
    for ratio, em, x, y, hex_color, color_name in records:
        ratio_str = "N/A" if ratio is None else f"{ratio:g}"
        cie_str = (
            ""
            if (x is None or y is None)
            else f"; CIE(x={x:.3f}, y={y:.3f})"
        )
        hex_str = "" if not hex_color else f"; hex={hex_color}"
        em_str = "" if em is None else f"; emission={em:.0f} nm"
        color_str = "" if not color_name else f"; color={color_name}"
        lines.append(f"{ratio_str} ->{cie_str}{hex_str}{em_str}{color_str}")

    # Trend summary (based on first/last with valid values)
    def first_last(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
        seq = [v for v in vals if v is not None]
        if not seq:
            return None, None
        return seq[0], seq[-1]

    ratios = [r for r, *_ in records]
    ems = [em for _, em, *_ in records]
    xs = [x for *_, x, y, _, _ in [(r, em, x, y, h, c) for r, em, x, y, h, c in records]]
    ys = [y for *_, y, _, _ in [(r, em, x, y, h, c) for r, em, x, y, h, c in records]]
    colors = [c for *_, c in records]

    em0, emN = first_last(ems)
    x0, xN = first_last(xs)
    y0, yN = first_last(ys)

    trend_bits: List[str] = []
    if em0 is not None and emN is not None:
        shift = emN - em0
        if abs(shift) < 5:
            trend_bits.append("emission peak stable")
        elif shift > 0:
            trend_bits.append("red-shift with higher ratio")
        else:
            trend_bits.append("blue-shift with higher ratio")
    if x0 is not None and xN is not None and y0 is not None and yN is not None:
        dx = xN - x0
        dy = yN - y0
        move = []
        if abs(dx) >= 0.005:
            move.append(f"x {'↑' if dx > 0 else '↓'} {abs(dx):.3f}")
        if abs(dy) >= 0.005:
            move.append(f"y {'↑' if dy > 0 else '↓'} {abs(dy):.3f}")
        if move:
            trend_bits.append("chromaticity moves: " + ", ".join(move))

    if trend_bits:
        lines.append("")
        lines.append("Trend: " + "; ".join(trend_bits))

    # Color name trend summary
    color_seq = [c for c in colors if c]
    if color_seq:
        first_color = color_seq[0]
        last_color = color_seq[-1]
        if first_color == last_color:
            lines.append(f"Summary: color remains '{first_color}' across ratios.")
        else:
            lines.append(f"Summary: color changes '{first_color}' → '{last_color}'")

    return "\n".join(lines)


# -----------------------
# Agent caller
# -----------------------
async def call_color_trend_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""[Scope] Use ONLY for host+dopant specific color/emission change vs dopant ratio.

    The agent can:
    - load_phosphor_db(file_path?): Load the DB
    - trend_by_ratio(host, dopant, file_path?): Summarize color/emission vs ratio for a given host and activator

    Example messages:
    - "For host 'Ba2V3O11' doped with 'Eu', how does color change with ratio?"
    - "I'm going to dope element Tb on Sr8ZnScP7O28; show color change by ratio."

    Do NOT use this agent for generic cross-dataset feature–feature trends (e.g.,
    "emission max vs color across the DB"). For generic X–Y trends, use TrendAgent instead.
    """
    agent_name = name
    deps = ctx.deps
    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    result = await color_trend_agent.run(message2agent, deps=deps)
    output = result.output
    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")
    return output


