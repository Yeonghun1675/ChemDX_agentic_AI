from typing import Optional, List, Tuple, Dict
import os
import difflib
import math
import pandas as pd

from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger


name = "TrendAgent"
role = (
    "Analyze data-driven trends between two features (X vs Y) in the phosphor DB, "
    "summarize them clearly (favoring color wording where applicable), and provide a brief domain-based explanation."
)
context = (
    "You can load the DB, automatically resolve column names by fuzzy matching, "
    "coerce values to numeric or ordinal (e.g., map emission wavelengths to color ranks), and compute correlations or grouped summaries. "
    "Use this agent for generic or cross-dataset X–Y questions, e.g., 'emission max vs color', 'intensity vs temperature'."
)


system_prompt = f"""You are the {name}. {role}

{context}

Guidelines:
- Always attempt to answer in color words when color can be inferred (user prefers color wording over raw wavelengths).
- Use robust statistics: Spearman correlation for monotonicity; show a compact evidence snippet.
- If a feature is categorical (e.g., color), map it to an ordinal scale when sensible (violet < blue < green < yellow < orange < red).
- Provide a short, plausible "because ..." explanation grounded in phosphor physics (energy-wavelength relation, activator transitions, crystal field, concentration quenching, thermal quenching).
- Keep answers concise and directly actionable.
"""


trend_agent = Agent(
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


def _find_best_column(columns: List[str], feature_name: str) -> Optional[str]:
    # Try direct/contains match (case-insensitive)
    feat = feature_name.strip().lower()
    lower_map = {c.lower(): c for c in columns}
    if feat in lower_map:
        return lower_map[feat]
    # contains match over tokens
    for lc, orig in lower_map.items():
        if feat in lc or lc in feat:
            return orig
    # difflib best ratio
    best = None
    best_score = 0.0
    for c in columns:
        score = difflib.SequenceMatcher(None, feature_name.lower(), c.lower()).ratio()
        if score > best_score:
            best = c
            best_score = score
    return best if best_score >= 0.45 else None


def _get_numeric(val) -> Optional[float]:
    try:
        if pd.isna(val):
            return None
        text = str(val).strip()
        if text.endswith("%"):
            return float(text[:-1])
        return float(pd.to_numeric(val))
    except Exception:
        return None


def _wavelength_to_color_name(nm: float) -> Optional[str]:
    try:
        if nm < 380 or nm > 780:
            return None
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


def _color_rank(name: Optional[str]) -> Optional[int]:
    order: Dict[str, int] = {
        "violet": 0,
        "blue": 1,
        "green": 2,
        "yellow": 3,
        "orange": 4,
        "red": 5,
    }
    if not name:
        return None
    return order.get(name.lower())


def _keyword(name: str) -> str:
    return name.strip().lower()


def _infer_numeric_series(df: pd.DataFrame, col_name: str, all_cols: List[str]) -> Tuple[pd.Series, Optional[str], Optional[str]]:
    """
    Returns (series_numeric, value_note, color_pref_name)
    - value_note: description of how values were derived
    - color_pref_name: color wording if applicable (for summaries)
    """
    k = _keyword(col_name)
    s = df[col_name]

    # If emission wavelength: numeric directly
    if any(tag in k for tag in ["emission", "λ", "lambda", "peak nm", "nm"]):
        series = s.apply(_get_numeric)
        return series, None, None

    # If color/Chromaticity: map via emission wavelength if available
    if any(tag in k for tag in ["color", "chromaticity", "cie x", "cie y", "cie"]):
        # Try to find an emission column to convert to color name
        em_col = None
        for cand in all_cols:
            if any(t in cand.lower() for t in ["emission max", "emission", "lambda", "nm"]):
                em_col = cand
                break
        if em_col is not None:
            em_series = df[em_col].apply(_get_numeric)
            color_name_series = em_series.apply(lambda v: _wavelength_to_color_name(v) if v is not None else None)
            series = color_name_series.apply(_color_rank)
            return series, f"mapped color from '{em_col}'", "color"
        # Fallback: attempt numeric coercion
        series = s.apply(_get_numeric)
        return series, "numeric-coerced color field (fallback)", "color"

    # Default: try numeric coercion
    series = s.apply(_get_numeric)
    return series, None, None


def _apply_filters(df: pd.DataFrame, filters: Optional[str]) -> pd.DataFrame:
    if not filters:
        return df
    # parse "key1=value1; key2=value2"
    work = df
    try:
        parts = [p.strip() for p in filters.split(";") if p.strip()]
        for part in parts:
            if "=" not in part:
                continue
            key, value = [x.strip() for x in part.split("=", 1)]
            if not key or not value:
                continue
            # fuzzy match column
            col = _find_best_column(list(work.columns), key)
            if not col:
                continue
            val_norm = value.lower()
            work = work[work[col].astype(str).str.lower().str.contains(val_norm)]
        return work
    except Exception:
        return df


# -----------------------
# Tools
# -----------------------
@trend_agent.tool_plain
def load_phosphor_db(file_path: Optional[str] = None) -> str:
    """Load the phosphor database (CSV/Excel). Auto-resolves path if not provided.

    args:
        file_path: Optional explicit path.
    output:
        Status string describing rows loaded or error.
    """
    global _df_cache, _path_cache
    resolved = _resolve_path(file_path)
    if not resolved:
        return (
            "Error: Could not resolve DB path. Set PHOSPHOR_DB_PATH or place the file at "
            "./Inorganic_Phosphor_Optical_Properties_DB.csv or ./data/Inorganic_Phosphor.csv"
        )
    try:
        if resolved.endswith(".csv"):
            _df_cache = pd.read_csv(resolved)
        else:
            _df_cache = pd.read_excel(resolved)
        _path_cache = resolved
        return f"Loaded {len(_df_cache)} rows from '{resolved}'"
    except Exception as exc:
        return f"Error loading '{resolved}': {exc}"


@trend_agent.tool_plain
def analyze_trend(x_feature: str, y_feature: str, file_path: Optional[str] = None, filters: Optional[str] = None) -> str:
    """Analyze the trend of Y vs X using the phosphor DB.

    args:
        x_feature: Column name or description for X (e.g., '1st doping concentration').
        y_feature: Column name or description for Y (e.g., 'Emission max. (nm)' or 'color').
        file_path: Optional DB path; if omitted, auto-resolve.
        filters: Optional filter string like "host=YAG; 1st dopant=Ce".

    output:
        Human-readable summary: trend statement, evidence (correlation or grouped statistics), and a brief explanation.
    """
    global _df_cache

    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status

    if _df_cache is None:
        return "Error: Database not loaded"

    df = _apply_filters(_df_cache, filters)
    if df.empty:
        return "No rows left after applying filters"

    cols = list(df.columns)
    x_col = _find_best_column(cols, x_feature)
    y_col = _find_best_column(cols, y_feature)
    if not x_col or not y_col:
        return f"Error: Could not resolve columns for X='{x_feature}' and Y='{y_feature}'"

    x_series, x_note, x_pref = _infer_numeric_series(df, x_col, cols)
    y_series, y_note, y_pref = _infer_numeric_series(df, y_col, cols)

    work = pd.DataFrame({"X": x_series, "Y": y_series}).dropna()
    if work.empty:
        return "Error: Not enough numeric/ordinal data to analyze"

    # Correlations
    try:
        spearman = work[["X", "Y"]].corr(method="spearman").loc["X", "Y"]
    except Exception:
        spearman = float("nan")
    try:
        pearson = work[["X", "Y"]].corr(method="pearson").loc["X", "Y"]
    except Exception:
        pearson = float("nan")

    n = len(work)
    trend_phrase = ""
    if not math.isnan(spearman):
        if spearman >= 0.2:
            trend_phrase = "↑ Y with ↑ X (positive monotonic trend)"
        elif spearman <= -0.2:
            trend_phrase = "↓ Y with ↑ X (negative monotonic trend)"
        else:
            trend_phrase = "weak or no monotonic trend"
    else:
        trend_phrase = "trend undetermined"

    # Grouped evidence: bin X into quantiles and show mean Y
    try:
        work["X_bin"] = pd.qcut(work["X"], q=min(5, max(2, min(10, n))), duplicates="drop")
        grouped = work.groupby("X_bin")["Y"].agg(["count", "mean"]).reset_index()
    except Exception:
        grouped = None

    # Build lines
    lines: List[str] = []
    lines.append(f"Trend of '{y_feature}' vs '{x_feature}'")
    if filters:
        lines.append(f"Filters: {filters}")
    if x_note:
        lines.append(f"X derived: {x_note}")
    if y_note:
        lines.append(f"Y derived: {y_note}")

    # Color preference wording when applicable
    if y_pref == "color":
        if spearman >= 0.2:
            lines.append("Summary: As X increases, color shifts toward red (warmer).")
        elif spearman <= -0.2:
            lines.append("Summary: As X increases, color shifts toward violet/blue (cooler).")
        else:
            lines.append("Summary: Color remains roughly stable across X.")
    else:
        lines.append(f"Summary: {trend_phrase}")

    lines.append(
        f"Evidence: n={n}, Spearman={spearman:.2f}{'' if math.isnan(pearson) else f', Pearson={pearson:.2f}'}"
    )

    if grouped is not None and not grouped.empty:
        lines.append("")
        lines.append("X-bin → count, mean(Y)")
        for _, row in grouped.iterrows():
            lines.append(f"{row['X_bin']} → {int(row['count'])}, {row['mean']:.3g}")

    # Brief domain explanation heuristic
    xk = _keyword(x_col)
    yk = _keyword(y_col)
    expl: str = ""
    if any(t in yk for t in ["emission", "nm", "lambda"]) and (y_pref == "color" or any(t in y_feature.lower() for t in ["color", "chromaticity"])):
        expl = (
            "Because longer wavelength corresponds to lower photon energy (E = hc/λ), and variations in activator sites "
            "and crystal field splitting shift transition energies; increasing X often stabilizes lower-energy states, yielding a redder color."
        )
    elif any(t in xk for t in ["concentration", "doping"]) and any(t in yk for t in ["emission", "intensity", "color", "nm"]):
        expl = (
            "Because doping concentration alters activator–activator interactions and energy transfer; beyond an optimal level, "
            "cross-relaxation and concentration quenching change emission pathways, often causing red-shift or intensity drop."
        )
    elif any(t in xk for t in ["temperature", "temp"]) and any(t in yk for t in ["intensity", "emission", "color"]):
        expl = (
            "Because thermal quenching increases non-radiative phonon-assisted transitions; with higher temperature, intensity decreases and spectra may red-shift/broaden."
        )
    elif any(t in xk for t in ["host", "crystal", "bandgap"]) and any(t in yk for t in ["emission", "color", "nm"]):
        expl = (
            "Because host crystal field and covalency (nephelauxetic effect) tune the activator's 5d/4f levels, shifting emission toward longer (redder) or shorter (bluer) wavelengths."
        )
    else:
        expl = "Because the underlying physical/chemical mechanism alters the electronic transition energies or non-radiative rates affecting the observed trend."

    lines.append("")
    lines.append(f"Reason: {expl}")

    return "\n".join(lines)


# -----------------------
# Agent caller
# -----------------------
async def call_trend_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""General trend analysis agent for arbitrary X vs Y relationships.

    Tools:
    - load_phosphor_db(file_path?)
    - analyze_trend(x_feature, y_feature, file_path?, filters?)
    """
    agent_name = name
    deps = ctx.deps
    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    result = await trend_agent.run(message2agent, deps=deps)
    output = result.output
    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")
    return output


