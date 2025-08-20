# chemdx_agent/viz_agent.py
from typing import List, Dict, Any, Optional, Literal
from pydantic_ai import Agent, RunContext

from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger

import os
import uuid
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless (no GUI)
import matplotlib.pyplot as plt


name = "DataVisAgent"
role = "Create scientific plots from structured rows (dict records) using matplotlib."
context = """You receive structured tabular data as List[dict] (e.g., from DatabaseAgent).
Use pandas for table handling and matplotlib for plotting (line, scatter, bar).
Never invent data; only plot what is provided. Always save a PNG and return its path."""

system_prompt = f"""You are the {name}.
You strictly create plots from structured rows provided to you.
If the requested axes/columns are missing, respond with a clear error message.
Your Current Role: {role}
Important Context: {context}
"""

working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

viz_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt=system_prompt,
)

@viz_agent.system_prompt(dynamic=True)
def dynamic_system_prompt(ctx: RunContext[AgentState]) -> str:
    deps = ctx.deps
    return working_memory_prompt.format(
        main_goal=deps.main_task,
        working_memory=deps.working_memory_description,
    )

def _to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list-of-dicts to DataFrame and gently coerce numeric columns."""
    df = pd.DataFrame(rows)
    # soft numeric coercion (ignore errors to keep strings when needed)
    for col in df.columns:
        # try only if it looks numeric-ish
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

def _outfile_path(prefix: str = "plot") -> str:
    os.makedirs("plots", exist_ok=True)
    return os.path.join("plots", f"{prefix}_{uuid.uuid4().hex[:8]}.png")

def _require_columns(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return f"Missing columns: {missing}. Available: {list(df.columns)}"
    return None

# -----------
# Tools

@viz_agent.tool_plain
def plot_line(
    rows: List[Dict[str, Any]],
    x: str,
    y: str,
    groupby: str = "Formula",
    title: Optional[str] = None,
    outfile: Optional[str] = None
) -> str:
    """Line plot: y vs x, with one line per group.

    Args:
        rows: List[dict] from the DB agent (structured records).
        x: x-axis column (e.g., 'temperature(K)').
        y: y-axis column (e.g., 'thermal_conductivity(W/mK)').
        groupby: column used to split series (default 'Formula').
        title: optional plot title.
        outfile: optional custom path; if not provided, an auto path is used.

    Returns:
        Path to saved PNG file (str), or error message (str) if invalid input.
    """
    df = _to_dataframe(rows)
    err = _require_columns(df, [x, y])
    if err:
        return f"[plot_line error] {err}"

    if outfile is None:
        outfile = _outfile_path("line")

    plt.figure()
    # sort for nice lines if x is numeric
    if pd.api.types.is_numeric_dtype(df[x]):
        df = df.sort_values([groupby, x], kind="mergesort") if groupby in df.columns else df.sort_values(x)

    if groupby in df.columns:
        for key, g in df.groupby(groupby, dropna=True):
            plt.plot(g[x], g[y], marker="o", label=str(key))
        plt.legend()
    else:
        plt.plot(df[x], df[y], marker="o")

    plt.xlabel(x)
    plt.ylabel(y)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    return outfile

@viz_agent.tool_plain
def plot_scatter(
    rows: List[Dict[str, Any]],
    x: str,
    y: str,
    hue: Optional[str] = "Formula",
    title: Optional[str] = None,
    outfile: Optional[str] = None
) -> str:
    """Scatter plot: y vs x, optionally colored by 'hue' (legend per category)."""
    df = _to_dataframe(rows)
    err = _require_columns(df, [x, y])
    if err:
        return f"[plot_scatter error] {err}"

    if outfile is None:
        outfile = _outfile_path("scatter")

    plt.figure()
    if hue and hue in df.columns:
        for key, g in df.groupby(hue, dropna=True):
            plt.scatter(g[x], g[y], label=str(key))
        plt.legend()
    else:
        plt.scatter(df[x], df[y])

    plt.xlabel(x)
    plt.ylabel(y)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    return outfile

@viz_agent.tool_plain
def plot_bar(
    rows: List[Dict[str, Any]],
    x: str,
    y: str,
    title: Optional[str] = None,
    outfile: Optional[str] = None
) -> str:
    """Bar plot: aggregate-friendly (assumes rows already aggregated if needed)."""
    df = _to_dataframe(rows)
    err = _require_columns(df, [x, y])
    if err:
        return f"[plot_bar error] {err}"

    if outfile is None:
        outfile = _outfile_path("bar")

    plt.figure()
    # ensure x as string labels for bars
    plt.bar(df[x].astype(str), df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    if title:
        plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    return outfile


async def call_viz_agent(ctx: RunContext[AgentState], message2agent: str):
    """Call DataVisAgent to execute a visualization task."""
    agent_name = "DataVisAgent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    user_prompt = f"Current Task of your role: {message2agent}"

    result = await viz_agent.run(user_prompt, deps=deps)

    output = result.output
    deps.add_working_memory(agent_name, message2agent)
    deps.increment_step()

    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")
    return output

