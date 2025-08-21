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
    try:
        # Import database agent functions directly
        from chemdx_agent.agents.tme_db_agent.subagent import get_top_performers, get_material_properties
        
        # Step 1: Get top performer by metric
        top_result = get_top_performers(
            property_name=metric, 
            max_results=1, 
            min_temperature=temp_min, 
            max_temperature=temp_max
        )
        
        if not top_result.get("results"):
            return {
                "ok": False,
                "error": f"No materials found for metric '{metric}' in temperature range {temp_min}K - {temp_max}K",
                "figure_path": None,
                "data_csv_path": None
            }
        
        # Get the top material
        top_material = top_result["results"][0]
        formula = top_material["formula"]
        
        # Step 2: Get all temperature data for this material
        material_data = get_material_properties(formula)
        
        if not material_data.get("results"):
            return {
                "ok": False,
                "error": f"No temperature data found for material {formula}",
                "figure_path": None,
                "data_csv_path": None
            }
        
        # Step 3: Create the plot
        data_rows = material_data["results"]
        
        # Extract temperature and y-axis data
        temp_data = [row.get("temperature(K)", 0) for row in data_rows]
        y_data = [row.get(y, 0) for row in data_rows]
        
        # Filter out invalid data points
        valid_data = [(t, y_val) for t, y_val in zip(temp_data, y_data) 
                     if t is not None and y_val is not None and not pd.isna(t) and not pd.isna(y_val)]
        
        if not valid_data:
            return {
                "ok": False,
                "error": f"No valid data points found for {y} vs temperature",
                "figure_path": None,
                "data_csv_path": None
            }
        
        temp_clean, y_clean = zip(*valid_data)
        
        # Create the plot
        title = f"{y} vs Temperature for {formula} (Top {metric} performer)"
        if temp_min or temp_max:
            temp_range = f" ({temp_min or 'min'}K - {temp_max or 'max'}K)"
            title += temp_range
        
        plot_result = create_thermoelectric_plot(
            material_formula=formula,
            temperature_data=list(temp_clean),
            zt_data=list(y_clean),
            title=title,
            prefix=f"top_{metric.lower()}_{formula.replace('(', '').replace(')', '').replace('.', '')}"
        )
        
        if plot_result.get("ok"):
            # Add metadata about the top performer
            plot_result["top_performer_info"] = {
                "formula": formula,
                "metric": metric,
                "avg_value": top_material.get(f"avg_{metric.lower()}", "N/A"),
                "temperature_range": top_material.get("temperature_range", "N/A"),
                "data_points": top_material.get("data_points", 0)
            }
        
        return plot_result
        
    except Exception as e:
        logger.error(f"[DataVisualisationAgent] Error in plot_yt_vs_T_for_best: {e}")
        return {
            "ok": False,
            "error": f"Failed to create plot: {str(e)}",
            "figure_path": None,
            "data_csv_path": None
        }

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

@viz_agent.tool_plain
def create_plot_from_database_data(
    data_rows: List[Dict[str, Any]],
    x_column: str = "temperature(K)",
    y_column: str = "ZT",
    title: str = "Database Data Plot",
    formula: str = "Unknown",
    prefix: str = "db_plot"
) -> Dict[str, Any]:
    """
    Create a plot from database data rows.
    
    Args:
        data_rows: List of dictionaries containing database rows
        x_column: Column name for x-axis (default: "temperature(K)")
        y_column: Column name for y-axis (default: "ZT")
        title: Plot title
        formula: Material formula for labeling
        prefix: Filename prefix
        
    Returns:
        Dict with plot path and data path
    """
    try:
        if not data_rows:
            return {
                "ok": False,
                "error": "No data rows provided",
                "figure_path": None,
                "data_csv_path": None
            }
        
        # Convert to DataFrame
        df = pd.DataFrame(data_rows)
        
        # Check required columns
        if x_column not in df.columns or y_column not in df.columns:
            return {
                "ok": False,
                "error": f"Required columns missing. Available columns: {list(df.columns)}",
                "figure_path": None,
                "data_csv_path": None
            }
        
        # Clean and sort data
        df = df.dropna(subset=[x_column, y_column])
        df = df.sort_values(x_column)
        
        if len(df) == 0:
            return {
                "ok": False,
                "error": "No valid data points after cleaning",
                "figure_path": None,
                "data_csv_path": None
            }
        
        # Create plot
        fig_path = _outfile(prefix)
        data_path = fig_path.replace(".png", "_data.csv")
        
        plt.figure(figsize=(10, 6))
        plt.plot(df[x_column], df[y_column], marker="o", linewidth=2, markersize=6, label=formula)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        # Save data
        df.to_csv(data_path, index=False)
        
        return {
            "ok": True,
            "figure_path": fig_path,
            "data_csv_path": data_path,
            "title": title,
            "formula": formula,
            "data_points": len(df),
            "x_range": f"{df[x_column].min():.1f} - {df[x_column].max():.1f}",
            "y_range": f"{df[y_column].min():.3f} - {df[y_column].max():.3f}"
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "figure_path": None,
            "data_csv_path": None
        }

@viz_agent.tool_plain
def create_thermoelectric_plot(
    material_formula: str,
    temperature_data: List[float],
    zt_data: List[float],
    title: str = None,
    prefix: str = "thermoelectric"
) -> Dict[str, Any]:
    """
    Create a thermoelectric plot showing ZT vs Temperature for a specific material.
    
    Args:
        material_formula: Chemical formula of the material
        temperature_data: List of temperature values in K
        zt_data: List of ZT values
        title: Optional plot title
        prefix: Filename prefix
        
    Returns:
        Dict with plot path and data path
    """
    try:
        if len(temperature_data) != len(zt_data):
            return {
                "ok": False,
                "error": "Temperature and ZT data lists must have the same length",
                "figure_path": None,
                "data_csv_path": None
            }
        
        if not temperature_data or not zt_data:
            return {
                "ok": False,
                "error": "No data provided",
                "figure_path": None,
                "data_csv_path": None
            }
        
        # Create plot
        fig_path = _outfile(prefix)
        data_path = fig_path.replace(".png", "_data.csv")
        
        # Use provided title or generate one
        plot_title = title or f"ZT vs Temperature for {material_formula}"
        
        plt.figure(figsize=(10, 6))
        plt.plot(temperature_data, zt_data, marker="o", linewidth=2, markersize=6, label=material_formula)
        plt.xlabel("Temperature (K)")
        plt.ylabel("ZT")
        plt.title(plot_title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        # Save data
        import pandas as pd
        df = pd.DataFrame({
            'temperature_K': temperature_data,
            'ZT': zt_data
        })
        df.to_csv(data_path, index=False)
        
        # Find optimal temperature
        max_zt_idx = zt_data.index(max(zt_data))
        optimal_temp = temperature_data[max_zt_idx]
        optimal_zt = zt_data[max_zt_idx]
        
        return {
            "ok": True,
            "figure_path": fig_path,
            "data_csv_path": data_path,
            "title": plot_title,
            "formula": material_formula,
            "data_points": len(temperature_data),
            "temperature_range": f"{min(temperature_data):.0f}K - {max(temperature_data):.0f}K",
            "optimal_temperature": optimal_temp,
            "optimal_zt": optimal_zt,
            "zt_range": f"{min(zt_data):.3f} - {max(zt_data):.3f}"
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

        This DataVisualisationAgent is used to visualise the desired data and return the figure path and the data path.
    """
    agent_name = "VisualisationAgent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}ㅤ")

    user_prompt = f"Current Task of your role: {message2agent}"

    result = await viz_agent.run(
        user_prompt, deps=deps
    )

    output = result.output
    deps.add_working_memory(agent_name, message2agent)
    deps.increment_step()

    logger.info(f"[{agent_name}] Action: {getattr(output, 'action', 'N/A')}ㅤ")
    logger.info(f"[{agent_name}] Result: {getattr(output, 'result', 'N/A')}ㅤ")

    return output
