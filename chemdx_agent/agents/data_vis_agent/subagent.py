# --- in chemdx_agent/viz_agent.py ---
from typing import Optional, Dict, Any, List, Literal
from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState
from chemdx_agent.logger import logger

import os, uuid
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

viz_agent = Agent(
    model="openai:gpt-4o",
    output_type=dict,
    deps_type=AgentState,
    system_prompt="You strictly plot from DB rows; never invent data.",
    model_settings={"temperature": 0.0},
)

def _outfile(prefix="plot", ext="png"):
    os.makedirs("plots", exist_ok=True)
    return os.path.join("plots", f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}")

@viz_agent.tool
async def plot_yt_vs_T_for_best(
    ctx: RunContext[AgentState],
    metric: str = "ZT",
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None,
    y: Literal["ZT","power_factor(W/mK2)","seebeck_coefficient(μV/K)","electrical_conductivity(S/m)","thermal_conductivity(W/mK)"]="ZT",
    groupby: str = "Formula",
) -> Dict[str, Any]:
    """
    1) Ask DatabaseAgent for top-1 by `metric` (in T-range).
    2) Fetch all rows for that formula.
    3) Plot y vs temperature(K). Return figure path + data snapshot path.
    """
    # import here to avoid circular import
    from chemdx_agent.agents.tme_db_agent.subagent import database_agent

    # step 1: top-1
    r1 = await database_agent.run(
        {"op":"get_top_performers","property_name":metric,"max_results":1,
         "min_temperature":temp_min,"max_temperature":temp_max},
        deps=ctx.deps, usage=ctx.usage
    )
    top_payload = r1.output.result if isinstance(r1.output.result, dict) else {}
    top_list = top_payload.get("results", [])
    if not top_list:
        return {"ok": False, "error": "No top candidate found."}
    formula = top_list[0].get("formula")
    if not formula:
        return {"ok": False, "error": "Top entry missing 'formula'."}

    # step 2: rows for formula
    r2 = await database_agent.run({"op":"get_material_properties","formula":formula}, deps=ctx.deps, usage=ctx.usage)
    rows_payload = r2.output.result if isinstance(r2.output.result, dict) else {}
    rows: List[Dict[str, Any]] = rows_payload.get("rows", [])
    if not rows:
        return {"ok": False, "error": f"No rows for formula {formula}."}

    df = pd.DataFrame(rows)
    if "temperature(K)" not in df.columns or y not in df.columns:
        return {"ok": False, "error": "Required columns missing."}

    # numeric checks
    df = df.dropna(subset=["temperature(K)", y])
    df = df.sort_values("temperature(K)")

    # step 3: plot
    fig_path = _outfile("yt_vs_T")
    data_path = fig_path.replace(".png", "_data.csv")
    plt.figure()
    plt.plot(df["temperature(K)"], df[y], marker="o", label=formula)
    plt.xlabel("Temperature (K)")
    plt.ylabel(y)
    plt.title(f"{y} vs Temperature for {formula}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    df.to_csv(data_path, index=False)

    return {"ok": True, "figure_path": fig_path, "data_csv_path": data_path, "formula": formula}
