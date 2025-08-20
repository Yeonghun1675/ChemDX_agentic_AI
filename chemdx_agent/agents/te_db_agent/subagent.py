from typing import Optional, List
import os
import pandas as pd
import difflib

from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger


name = "ThermoelectricLookupAgent"
role = "Look up thermoelectric properties (Seebeck, conductivity, ZT, etc.) from LitDX_TE CSV DB"
context = "Use provided tools. DB path: TE_DB_PATH env var or ./databases/tme_db_litdx.csv"

system_prompt = f"""You are {name}. {role}. {context}
Rules:
- For any question about materials in the database, FIRST call a tool to fetch rows.
- If a tool returns an error or no rows, report that result verbatim; do not infer values.
"""

te_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={"temperature": 0.0, "parallel_tool_calls": False},
    system_prompt=system_prompt,
)


_df_cache: Optional[pd.DataFrame] = None
_path_cache: Optional[str] = None


def _resolve_path(file_path: Optional[str]) -> Optional[str]:
    """Resolve DB path from args, env var, or defaults."""
    candidates: List[str] = []
    if file_path:
        candidates.append(file_path)
    if env := os.getenv("TE_DB_PATH"):
        candidates.append(env)
    candidates.extend([
        "databases/tme_db_litdx.csv",
        "tme_db_litdx.csv",
        "databases/tme_db_litdx.xlsx",
        "tme_db_litdx.xlsx",
    ])
    return next((p for p in candidates if p and os.path.exists(p)), None)

def _find_col(columns: List[str], keywords: List[str]) -> Optional[str]:
    """Find a column whose name contains any keyword (case-insensitive)."""
    lowered = {c.lower(): c for c in columns}
    for kw in keywords:
        kwl = kw.lower()
        for lc, orig in lowered.items():
            if kwl in lc:
                return orig
    return None

def _get_numeric(row: pd.Series, col: Optional[str]) -> Optional[float]:
    if not col or col not in row or pd.isna(row[col]):
        return None
    try:
        return float(pd.to_numeric(row[col], errors="coerce"))
    except Exception:
        return None

def _fmt(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:g}"

# tools
@te_agent.tool_plain
def load_te_db(file_path: Optional[str] = None) -> str:
    """Load LitDX_TE database into memory (CSV or Excel)."""
    global _df_cache, _path_cache
    resolved = _resolve_path(file_path)
    if not resolved:
        return "Error: Could not resolve DB path. Set TE_DB_PATH or place at ./databases/tme_db_litdx.csv"

    try:
        if resolved.lower().endswith(".csv"):
            _df_cache = pd.read_csv(resolved)
        else:
            _df_cache = pd.read_excel(resolved)
        _path_cache = resolved
        return f"Loaded {len(_df_cache)} rows from '{resolved}'"
    except Exception as exc:
        return f"Error loading '{resolved}': {exc}"

@te_agent.tool_plain
def lookup_by_formula(
    formula: str,
    temperature_K: Optional[float] = None,
    window_K: float = 25.0,
    top_n: int = 5,
    file_path: Optional[str] = None,
) -> str:
    """Find thermoelectric rows for an EXACT formula match (case-insensitive). Optionally filter within ±window_K around temperature_K.

    Returns a concise text table with: T, Seebeck, σ, κ, PF, ZT, reference.
    """
    global _df_cache, _path_cache

    # Ensure DB loaded
    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_te_db(file_path)
        if status.startswith("Error"):
            return status

    if _df_cache is None:
        return "Error: Database not loaded"

    cols = list(_df_cache.columns)
    col_formula = _find_col(cols, ["formula", "compound", "name", "composition"])
    if not col_formula:
        return "Error: No 'Formula' column found"

    # exact, case-insensitive match
    target = str(formula).strip().lower()
    try:
        df = _df_cache[_df_cache[col_formula].astype(str).str.strip().str.lower() == target].copy()
    except Exception:
        return "Error: Failed during formula match"

    if df.empty:
        return f"Not found: '{formula}'"

    # resolve property columns
    col_T   = _find_col(cols, ["temperature", "temperature(", "temp"])
    col_S   = _find_col(cols, ["seebeck"])
    col_sig = _find_col(cols, ["electrical_conductivity", "(s/m)", "sigma", "σ"])
    col_k   = _find_col(cols, ["thermal_conductivity", "(w/mk)", "k (w/mk)"])
    col_PF  = _find_col(cols, ["power_factor", "pf"])
    col_ZT  = _find_col(cols, ["zt"])
    col_ref = _find_col(cols, ["reference", "doi"])

    # optional temperature window
    if temperature_K is not None and col_T is not None:
        t = pd.to_numeric(df[col_T], errors="coerce")
        df = df[(t - float(temperature_K)).abs() <= float(window_K)]

    if df.empty:
        if temperature_K is not None:
            return f"Found '{formula}' but no rows within {temperature_K}±{window_K} K"
        return f"Not found: '{formula}'"  # should not happen unless all rows NaN

    # sort by closeness to requested T (if given) or by ZT desc
    if temperature_K is not None and col_T is not None:
        t = pd.to_numeric(df[col_T], errors="coerce")
        df = df.assign(_dt=(t - float(temperature_K)).abs()).sort_values("_dt", kind="mergesort")
    elif col_ZT and col_ZT in df.columns:
        z = pd.to_numeric(df[col_ZT], errors="coerce")
        df = df.assign(_zt=z).sort_values("_zt", ascending=False, na_position="last", kind="mergesort")

    # build lines
    lines = []
    lines.append(f"Formula: {formula}" + (f" | T≈{temperature_K}K (±{window_K}K)" if temperature_K is not None else ""))
    lines.append("T(K) | Seebeck(μV/K) | σ(S/m) | κ(W/mK) | PF(W/mK^2) | ZT | ref")
    lines.append("-" * 78)

    count = 0
    for _, r in df.iterrows():
        T   = _get_numeric(r, col_T)
        S   = _get_numeric(r, col_S)
        SIG = _get_numeric(r, col_sig)
        K   = _get_numeric(r, col_k)
        PF  = _get_numeric(r, col_PF)
        ZT  = _get_numeric(r, col_ZT)
        REF = (str(r[col_ref]).strip() if col_ref and pd.notna(r.get(col_ref)) else "N/A")

        lines.append(f"{_fmt(T):>5} | {_fmt(S):>14} | {_fmt(SIG):>6} | {_fmt(K):>7} | {_fmt(PF):>10} | {_fmt(ZT):>3} | {REF}")
        count += 1
        if count >= int(top_n):
            break

    return "\n".join(lines)

@te_agent.tool_plain
def similar_formulas(
    formula: str,
    top_k: int = 5,
    file_path: Optional[str] = None,
) -> str:
    """Find top_k most similar formulas (string similarity)."""
    global _df_cache, _path_cache

    if _df_cache is None or (file_path and file_path != _path_cache):
        status = load_te_db(file_path)
        if status.startswith("Error"):
            return status

    if _df_cache is None:
        return "Error: Database not loaded"

    col_formula = _find_col(list(_df_cache.columns), ["formula", "compound", "name", "composition"])
    if not col_formula:
        return "Error: No 'Formula' column found"

    formulas = _df_cache[col_formula].astype(str).dropna().unique()
    target = str(formula).strip().lower()

    scored = [(f, difflib.SequenceMatcher(None, target, str(f).lower()).ratio())
              for f in formulas if str(f).strip()]
    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored:
        return "No candidates found"

    lines = [f"Similar to '{formula}' | top_k={top_k}"]
    for cand, score in scored[:int(top_k)]:
        lines.append(f"{cand} | similarity={score:.3f}")
    return "\n".join(lines)

# 

async def call_sample_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""Call {name} to execute a task.

    args:
        message2agent: (str) Detailed instruction to the agent.
        Tools:
        - load_te_db(file_path?): Load CSV/Excel DB from arg, env TE_DB_PATH, or defaults.
        - lookup_by_formula(formula, temperature_K?, window_K?, top_n?): Exact formula; returns rows with T, S, σ, κ, PF, ZT, ref.
        - similar_formulas(formula, top_k?): Suggest close formula strings to help find exact matches.

    examples:
        - "Load the TE database from 'databases/tme_db_litdx.csv'"
        - "For formula 'BiSb(Se0.94Br0.06)3' list properties near 600 K (±25 K), top 5 rows"
        - "Show 5 formulas similar to 'Bi2Te3'"
    """
    agent_name = "SampleAgent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    result = await sample_agent.run(
        message2agent, deps=deps
    )
    output = result.output

    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")

    return output