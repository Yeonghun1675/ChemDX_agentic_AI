from typing import Optional, List
import os
import difflib
import math
import pandas as pd

from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger
from chemdx_agent.utils import make_tool_message


name = "ESTMTrendAgent"
role = "Analyze trends (X vs Y) in thermoelectric dataset 'estm.csv'"
context = "Columns include: Formula, temperature(K), seebeck_coefficient(μV/K), electrical_conductivity(S/m), thermal_conductivity(W/mK), power_factor(W/mK2), ZT, reference"

working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

system_prompt = f"""You are the {name}. {role}.

{context}
- Robustly fuzzy-match feature names; handle numeric coercion.
- Provide concise, evidence-backed trend summaries.
"""


estm_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={"temperature": 0.0, "parallel_tool_calls": False},
    system_prompt=system_prompt,
)

@estm_agent.system_prompt(dynamic=True)
def dynamic_system_prompt(ctx: RunContext[AgentState]) -> str:
    deps = ctx.deps
    return working_memory_prompt.format(
        main_goal = deps.main_task,
        working_memory = deps.working_memory_description,
    )

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
        return float(pd.to_numeric(val))
    except Exception:
        return None


@estm_agent.tool_plain
def load_estm_db(file_path: Optional[str] = None) -> str:
    global _df_cache, _path_cache
    path = file_path or "estm.csv"
    if not os.path.exists(path):
        return f"Error: Not found: '{path}'"
    try:
        _df_cache = pd.read_csv(path)
        _path_cache = path
        return f"Loaded {len(_df_cache)} rows from '{path}'"
    except Exception as exc:
        return f"Error loading '{path}': {exc}"


@estm_agent.tool_plain
def analyze_trend(x_feature: str, y_feature: str, file_path: Optional[str] = None, filters: Optional[str] = None) -> str:
    global _df_cache
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_estm_db(file_path)
        if status.startswith("Error"):
            return status
    if _df_cache is None:
        return "Error: Database not loaded"

    df = _df_cache
    if filters:
        df = df[df["Formula"].astype(str).str.lower().str.contains(filters.strip().lower())]
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
    lines.append(f"ESTM trend of '{y_feature}' vs '{x_feature}'")
    lines.append(f"Summary: {trend_phrase}")
    lines.append(f"Evidence: n={n}, Spearman={spearman:.2f}{'' if math.isnan(pearson) else f', Pearson={pearson:.2f}'}")
    return "\n".join(lines)


async def call_estm_trend_agent(ctx: RunContext[AgentState], message2agent: str):
    deps = ctx.deps
    logger.info(f"[ESTMTrendAgent] Message2Agent: {message2agent}")
    
    user_prompt = f"Current Task of your role: {message2agent}"
    result = await estm_agent.run(user_prompt, deps=deps)
    output = result.output
    deps.add_working_memory(name, message2agent)
    deps.increment_step()
    logger.info(f"[ESTMTrendAgent] Action: {output.action}")
    
    list_tool_log = make_tool_message(result)
    for log in list_tool_log:
        logger.info(log)
    
    logger.info(f"[ESTMTrendAgent] Result: {output.result}")
    return output


