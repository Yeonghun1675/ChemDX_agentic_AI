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

# -----------------------------

name = "DataVisualisationAgent"
role = "You take in data and create graphs or plots using matplotlib and seaborn."
context = "Important Context that agent should be know. For example, you must not use numpy array. instead, use pandas DataFrame."


system_prompt = f"""You are the {name}. You can use available tools or request help from specialized sub-agents that perform specific tasks. You must only carry out the role assigned to you. If a request is outside your capabilities, you should ask for support from the appropriate agent instead of trying to handle it yourself.

Your Currunt Role: {role}
Important Context: {context}
"""

working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

viz_agent = Agent(
    model = "openai:gpt-4o",
    output_type = Result,
    deps_type = AgentState,
    model_settings = {
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt = system_prompt,
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



    # call agent function
async def call_viz_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""Call general agent to execute the task: {role}

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do.

        This DataVisualisationAgent is used to plot the data from the database and return the figure path and the data path.
    """
    agent_name = "VisualisationAgent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")

    user_prompt = "Current Task of your role: {message2agent}"

    result = await viz_agent.run(
        user_prompt, deps=deps
    )

    output = result.output
    deps.add_working_memory(agent_name, message2agent)
    deps.increment_step()

    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")

    return output
