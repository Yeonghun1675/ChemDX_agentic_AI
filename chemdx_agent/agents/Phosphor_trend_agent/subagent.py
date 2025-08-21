from typing import Optional, List
import os
import difflib
import math
import pandas as pd

from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger


name = "PhosphorTrendAgent"
role = "Analyze optical phosphor trends in Inorganic_Phosphor_Optical_Properties_DB.csv"
context = "Columns include: Host, 1st dopant, 1st doping concentration, Emission max. (nm), CIE x/y, Decay time (ns)"


system_prompt = f"""You are the {name}. {role}.

{context}
- Fuzzy-match features; coerce numbers; prefer color wording when applicable.
- Provide concise evidence-backed trend summaries.
"""


phosphor_trend_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={"temperature": 0.0, "parallel_tool_calls": False},
    system_prompt=system_prompt,
)

working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

_df_cache: Optional[pd.DataFrame] = None
_path_cache: Optional[str] = None


def _find_best_column(columns: List[str], feature_name: str) -> Optional[str]:
    feat = feature_name.strip().lower()
    lower_map = {c.lower(): c for c in columns}
    if feat in lower_map:
        return lower_map[feat]
    for lc, orig in lower_map.items():
        if feat in lc or lc in feat:
            return orig
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


@phosphor_trend_agent.tool_plain
def load_phosphor_db(file_path: Optional[str] = None) -> str:
    global _df_cache, _path_cache
    path = file_path or "Inorganic_Phosphor_Optical_Properties_DB.csv"
    if not os.path.exists(path):
        return f"Error: Not found: '{path}'"
    try:
        _df_cache = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
        _path_cache = path
        return f"Loaded {len(_df_cache)} rows from '{path}'"
    except Exception as exc:
        return f"Error loading '{path}': {exc}"


@phosphor_trend_agent.tool_plain
def analyze_trend(x_feature: str, y_feature: str, file_path: Optional[str] = None, filters: Optional[str] = None) -> str:
    global _df_cache
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_phosphor_db(file_path)
        if status.startswith("Error"):
            return status
    if _df_cache is None:
        return "Error: Database not loaded"

    df = _df_cache
    if filters:
        f = filters.strip().lower()
        cols = [c for c in df.columns if isinstance(c, str)]
        mask = pd.Series(False, index=df.index)
        for c in cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(f)
        df = df[mask]
    if df.empty:
        return "No rows left after applying filters"

    cols = list(df.columns)
    x_col = _find_best_column(cols, x_feature)
    y_col = _find_best_column(cols, y_feature)
    if not x_col or not y_col:
        return f"Error: Could not resolve columns for X='{x_feature}' and Y='{y_feature}'"

    x_series = df[x_col].apply(_get_numeric)
    y_series = df[y_col].apply(_get_numeric)
    work = pd.DataFrame({"X": x_series, "Y": y_series}).dropna()
    if work.empty:
        return "Error: Not enough numeric data to analyze"

    try:
        spearman = work[["X", "Y"]].corr(method="spearman").loc["X", "Y"]
    except Exception:
        spearman = float("nan")
    try:
        pearson = work[["X", "Y"]].corr(method="pearson").loc["X", "Y"]
    except Exception:
        pearson = float("nan")

    n = len(work)
    if not math.isnan(spearman):
        if spearman >= 0.2:
            trend_phrase = "↑ Y with ↑ X (positive monotonic trend)"
        elif spearman <= -0.2:
            trend_phrase = "↓ Y with ↑ X (negative monotonic trend)"
        else:
            trend_phrase = "weak or no monotonic trend"
    else:
        trend_phrase = "trend undetermined"

    lines: List[str] = []
    lines.append(f"Phosphor DB trend of '{y_feature}' vs '{x_feature}'")
    if filters:
        lines.append(f"Filters: {filters}")
    lines.append(f"Summary: {trend_phrase}")
    lines.append(f"Evidence: n={n}, Spearman={spearman:.2f}{'' if math.isnan(pearson) else f', Pearson={pearson:.2f}'}")
    return "\n".join(lines)


async def call_phosphor_trend_agent(ctx: RunContext[AgentState], message2agent: str):
    deps = ctx.deps
    logger.info(f"[PhosphorTrendAgent] Message2Agent: {message2agent}")
    result = await phosphor_trend_agent.run(message2agent, deps=deps)
    output = result.output
    deps.add_working_memory("PhosphorTrendAgent", message2agent)
    deps.increment_step()
    deps.add_working_memory(name, message2agent)
    deps.increment_step()
    logger.info(f"[PhosphorTrendAgent] Action: {output.action}")
    logger.info(f"[PhosphorTrendAgent] Result: {output.result}")
    return output


