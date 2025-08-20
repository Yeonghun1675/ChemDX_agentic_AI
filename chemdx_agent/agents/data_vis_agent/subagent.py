# chemdx_agent/viz_agent.py
from typing import List, Dict, Any, Optional, Literal
from pydantic_ai import Agent, RunContext

from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger

import os
import uuid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless (no GUI)
import matplotlib.pyplot as plt


name = "DataVisAgent"
role = "Create scientific plots from structured rows (dict records) using matplotlib."
context = """You receive structured tabular data as List[dict] (e.g., from DatabaseAgent).
Use pandas for table handling and matplotlib for plotting.
Never invent data; only plot what is provided. Always save a figure and return its path."""

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
    """Convert list-of-dicts to a DataFrame (no blanket coercion)."""
    return pd.DataFrame(rows)

def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Ensure requested columns are numeric; coerce invalid entries to NaN and drop those rows.
    Does not touch categorical columns like 'Formula' unless they are explicitly in cols.
    """
    present = [c for c in cols if c in df.columns]
    for c in present:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if present:
        df = df.dropna(subset=present)
    return df

def _outfile_path(prefix: str = "plot", ext: Literal["png","svg","pdf"]="png") -> str:
    os.makedirs("plots", exist_ok=True)
    return os.path.join("plots", f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}")

def _require_columns(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return f"Missing columns: {missing}. Available: {list(df.columns)}"
    return None

def _is_categorical(series: pd.Series) -> bool:
    """Heuristic: treat as categorical if non-numeric or few unique values."""
    if series.empty:
        return True
    if pd.api.types.is_numeric_dtype(series):
        return series.nunique(dropna=True) <= 12
    return True


@viz_agent.tool_plain
def plot_property_vs_property(
    rows: List[Dict[str, Any]],
    x: str,
    y: str,
    material_col: str = "Formula",
    plot_type: Literal["auto","line","scatter","bar"] = "auto",
    aggregate: Optional[Literal["mean","median"]] = None,
    error_y: Optional[str] = None,  # e.g., std column name or precomputed error
    xscale: Optional[Literal["linear","log","symlog"]] = None,
    yscale: Optional[Literal["linear","log","symlog"]] = None,
    title: Optional[str] = None,
    outfile_ext: Literal["png","svg","pdf"] = "png",
    outfile: Optional[str] = None,
) -> str:
    """
    Versatile plotter: y vs x. Chooses a reasonable plot type or uses the one you specify.
    - Groups by `material_col` if present.
    - If plot_type='auto':
        * bar if x is categorical (few unique values or non-numeric),
        * line if x is numeric and each group has >1 x-value,
        * else scatter.
    - If `aggregate` provided, reduce duplicates per (material, x) via mean/median.
    - Optional error bars via `error_y` if column exists (line/scatter/bar).
    - Optional axis scales: xscale/yscale ('log', 'symlog', 'linear').

    Returns: path to saved figure (str) or an error string.
    """
    df = _to_dataframe(rows)
    err = _require_columns(df, [x, y])
    if err:
        return f"[plot_property_vs_property error] {err}"

    # Keep only necessary columns
    keep_cols = [c for c in [x, y, material_col, error_y] if c and c in df.columns]
    df = df[keep_cols].copy()

    # numeric axes
    # y must be numeric; x may be numeric or categorical depending on plot
    df = _ensure_numeric(df, [y] + ([x] if plot_type in ("auto","line","scatter") else []))
    if df.empty:
        return f"[plot_property_vs_property error] No plottable rows after numeric checks."

    # apply aggregation if requested (per material & x)
    if aggregate:
        group_keys = [g for g in [material_col if material_col in df.columns else None, x] if g]
        if group_keys:
            agg_map = {y: aggregate}
            if error_y and (error_y in df.columns):
                # if an error column is provided, aggregate it in the same way
                agg_map[error_y] = aggregate
            df = df.groupby(group_keys, dropna=True).agg(agg_map).reset_index()

    # choose plot type automatically if needed
    if plot_type == "auto":
        if x in df.columns and _is_categorical(df[x]):
            plot_type = "bar"
        else:
            # numeric x: decide line vs scatter based on per-group density
            if material_col in df.columns:
                dense = any(g[x].nunique(dropna=True) > 1 for _, g in df.groupby(material_col, dropna=True))
            else:
                dense = df[x].nunique(dropna=True) > 1 if x in df.columns else False
            plot_type = "line" if dense else "scatter"

    # figure out output path
    if outfile is None:
        outfile = _outfile_path("prop_vs_prop", outfile_ext)

    # plotting
    plt.figure()

    # apply axis scales
    if xscale: plt.xscale(xscale)
    if yscale: plt.yscale(yscale)

    # order data for nicer lines
    if plot_type == "line" and x in df.columns and pd.api.types.is_numeric_dtype(df[x]):
        if material_col in df.columns:
            df = df.sort_values([material_col, x], kind="mergesort")
        else:
            df = df.sort_values([x], kind="mergesort")

    if plot_type in ("line", "scatter"):
        if material_col in df.columns:
            for key, g in df.groupby(material_col, dropna=True):
                if plot_type == "line":
                    if error_y and (error_y in g.columns):
                        plt.errorbar(g[x], g[y], yerr=g[error_y], marker="o", label=str(key))
                    else:
                        plt.plot(g[x], g[y], marker="o", label=str(key))
                else:  # scatter
                    if error_y and (error_y in g.columns):
                        plt.errorbar(g[x], g[y], yerr=g[error_y], fmt="o", label=str(key))
                    else:
                        plt.scatter(g[x], g[y], label=str(key))
            plt.legend()
        else:
            if plot_type == "line":
                if error_y and (error_y in df.columns):
                    plt.errorbar(df[x], df[y], yerr=df[error_y], marker="o")
                else:
                    plt.plot(df[x], df[y], marker="o")
            else:
                if error_y and (error_y in df.columns):
                    plt.errorbar(df[x], df[y], yerr=df[error_y], fmt="o")
                else:
                    plt.scatter(df[x], df[y])

    elif plot_type == "bar":
        # grouped bar by category x (string labels) and columns = materials
        if x not in df.columns:
            return "[plot_property_vs_property error] x not found for bar plot."
        cats = [str(v) for v in pd.Series(df[x]).astype(str).unique().tolist()]
        if material_col in df.columns:
            mats = [str(v) for v in df[material_col].astype(str).unique().tolist()]
            # pivot to (cats x mats)
            pivot = df.copy()
            pivot[x] = pivot[x].astype(str)
            pivot[material_col] = pivot[material_col].astype(str)
            # aggregate mean per (cat, material) if duplicates remain
            pivot = pivot.groupby([x, material_col], dropna=True)[y].mean().unstack(material_col).reindex(cats)
            n_m = len(mats)
            idx = np.arange(len(cats))
            width = 0.8 / max(n_m, 1)
            for i, m in enumerate(mats):
                heights = pivot[m].values if m in pivot.columns else np.zeros(len(cats))
                # yerr if available (aggregate std over duplicates for bars)
                yerr = None
                if error_y and (error_y in df.columns):
                    err_pivot = df.copy()
                    err_pivot[x] = err_pivot[x].astype(str)
                    err_pivot = err_pivot.groupby([x, material_col], dropna=True)[error_y].mean().unstack(material_col).reindex(cats)
                    yerr = err_pivot[m].values if m in err_pivot.columns else None
                plt.bar(idx + (i - (n_m - 1) / 2) * width, heights, width=width, label=m, yerr=yerr)
            plt.xticks(idx, cats, rotation=45, ha="right")
            plt.legend()
        else:
            # single series bar by category
            agg = df.copy()
            agg[x] = agg[x].astype(str)
            agg = agg.groupby(x, dropna=True)[y].mean().reindex(cats)
            yerr = None
            if error_y and (error_y in df.columns):
                err = df.copy()
                err[x] = err[x].astype(str)
                yerr = err.groupby(x, dropna=True)[error_y].mean().reindex(cats).values
            plt.bar(cats, agg.values, yerr=yerr)
            plt.xticks(rotation=45, ha="right")

    else:
        return f"[plot_property_vs_property error] Unsupported plot_type '{plot_type}'."

    plt.xlabel(x)
    plt.ylabel(y)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    return outfile

@viz_agent.tool_plain
def plot_property_distribution(
    rows: List[Dict[str, Any]],
    property_name: str,
    groupby: Optional[str] = None,
    kind: Literal["hist","box"] = "hist",
    bins: int = 20,
    xscale: Optional[Literal["linear","log","symlog"]] = None,
    title: Optional[str] = None,
    outfile_ext: Literal["png","svg","pdf"] = "png",
    outfile: Optional[str] = None,
) -> str:
    """
    Show a property's distribution.
    - kind='hist': histogram (overlay per group if groupby provided)
    - kind='box': boxplot across groups (requires groupby)

    Returns: path to saved figure (str) or an error string.
    """
    df = _to_dataframe(rows)
    if property_name not in df.columns:
        return f"[plot_property_distribution error] Column '{property_name}' not found. Available: {list(df.columns)}"

    df = _ensure_numeric(df, [property_name])
    if df.empty:
        return f"[plot_property_distribution error] No numeric data for '{property_name}'."

    if outfile is None:
        outfile = _outfile_path("distribution", outfile_ext)

    plt.figure()
    if kind == "hist":
        if groupby and groupby in df.columns:
            for key, g in df.groupby(groupby, dropna=True):
                plt.hist(g[property_name].dropna().values, bins=bins, alpha=0.6, label=str(key))
            plt.legend()
        else:
            plt.hist(df[property_name].dropna().values, bins=bins, alpha=0.8)
        if xscale: plt.xscale(xscale)
        plt.xlabel(property_name)
        plt.ylabel("Count")
    elif kind == "box":
        if not groupby or groupby not in df.columns:
            return "[plot_property_distribution error] 'box' requires a valid 'groupby' column."
        groups = []
        labels = []
        for key, g in df.groupby(groupby, dropna=True):
            vals = g[property_name].dropna().values
            if vals.size > 0:
                groups.append(vals)
                labels.append(str(key))
        if not groups:
            return "[plot_property_distribution error] No data to boxplot after grouping."
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(property_name)
    else:
        return f"[plot_property_distribution error] Unsupported kind '{kind}'."

    if title:
        plt.title(title)
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
