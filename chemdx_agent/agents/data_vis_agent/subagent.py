# --- in chemdx_agent/viz_agent.py ---
from typing import Optional, Dict, Any, List, Literal
from pydantic_ai import Agent, RunContext


from chemdx_agent.schema import AgentState, Result

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

Your Current Role: {role}
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

    # Use the proper call function instead of direct agent import
    from chemdx_agent.agents.tme_db_agent.subagent import call_database_agent

    # step 1: get top performers
    db_query = f"Get top 1 materials by {metric}"
    if temp_min:
        db_query += f" with minimum temperature {temp_min}K"
    if temp_max:
        db_query += f" with maximum temperature {temp_max}K"
    
    r1 = await call_database_agent(ctx, db_query)
    if not hasattr(r1, 'result'):
        return {"ok": False, "error": "Database agent call failed"}
    
    top_payload = r1.result if hasattr(r1, 'result') else {}
    if isinstance(top_payload, str):
        # Try to parse if it's a string
        try:
            import json
            top_payload = json.loads(top_payload)
        except:
            return {"ok": False, "error": "Could not parse database response"}
    
    top_list = top_payload.get("results", [])
    if not top_list:
        return {"ok": False, "error": "No top candidate found."}
    formula = top_list[0].get("formula")
    if not formula:
        return {"ok": False, "error": "Top entry missing 'formula'."}

    # step 2: get rows for formula
    r2 = await call_database_agent(ctx, f"Get material properties for formula {formula}")
    if not hasattr(r2, 'result'):
        return {"ok": False, "error": "Database agent call for material properties failed"}
    
    rows_payload = r2.result if hasattr(r2, 'result') else {}
    if isinstance(rows_payload, str):
        try:
            import json
            rows_payload = json.loads(rows_payload)
        except:
            return {"ok": False, "error": "Could not parse material properties response"}
    
    rows: List[Dict[str, Any]] = rows_payload.get("results", [])
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

@viz_agent.tool_plain
def create_simple_plot(
    x_data: list,
    y_data: list,
    title: str = "Simple Plot",
    x_label: str = "X",
    y_label: str = "Y",
    prefix: str = "simple"
) -> Dict[str, Any]:
    """
    Create a simple plot from provided data.
    
    Args:
        x_data: list of x values
        y_data: list of y values  
        title: plot title
        x_label: x-axis label
        y_label: y-axis label
        prefix: filename prefix
        
    Returns:
        Dict with plot path and data path
    """
    try:
        # Create plot
        fig_path = _outfile(prefix)
        data_path = fig_path.replace(".png", "_data.csv")
        
        plt.figure()
        plt.plot(x_data, y_data, marker="o", linewidth=2, markersize=6)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        # Save data
        import pandas as pd
        df = pd.DataFrame({'x': x_data, 'y': y_data})
        df.to_csv(data_path, index=False)
        
        return {
            "ok": True,
            "figure_path": fig_path,
            "data_csv_path": data_path,
            "title": title
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "figure_path": None,
            "data_csv_path": None
        }



    # call agent function
async def call_viz_agent(ctx: RunContext[AgentState], message2agent: str):
    """Call visualization agent to execute the task: {role}

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do.

        This DataVisualisationAgent is used to plot the data from the database and return the figure path and the data path.
    """
    agent_name = "VisualisationAgent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")

    user_prompt = f"Current Task of your role: {message2agent}"

    result = await viz_agent.run(
        user_prompt, deps=deps
    )

    output = result.output
    deps.add_working_memory(agent_name, message2agent)
    deps.increment_step()

    logger.info(f"[{agent_name}] Action: {getattr(output, 'action', 'N/A')}")
    logger.info(f"[{agent_name}] Result: {getattr(output, 'result', 'N/A')}")

    return output
