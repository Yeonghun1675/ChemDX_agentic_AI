from typing import List
from pydantic_ai import Agent, RunContext
import pandas as pd

from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger

name = "ThermoelectricDBAgent"
role = "an agent that can search and answer questions about thermoelectric materials based on the LitDX_TE database."
context = "You are the agent of the thermoelectric materials database. you have access to the database of thermoelectric materials at different composition ratios and doping. you also have access to their seebeck_coefficient(μV/K),electrical_conductivity(S/m),thermal_conductivity(W/mK),power_factor(W/mK2) and ZT data at differnet temperatures (K)."

file_path = r'databases/tme_db_litdx.csv'

system_prompt = f"""You are the {name}. You can use available tools or request help from specialized sub-agents that perform specific tasks. You must only carry out the role assigned to you. If a request is outside your capabilities, you should ask for support from the appropriate agent instead of trying to handle it yourself.

Your Currunt Role: {role}
Important Context: {context}
"""


sample_agent = Agent(
    model = "openai:gpt-4o",
    output_type = Result,
    deps_type = AgentState,
    model_settings = {
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt = system_prompt,
)

#Tool setting 
@sample_agent.tool_plain
def read_tme_csv(file_path: str) -> str:
    """Reads a CSV file and returns information about its content."""
    try:
        df = pd.read_csv(file_path)
        return f"Successfully read data from {file_path}. DataFrame shape: {df.shape}, Columns: {df.columns.tolist()}"
    except Exception as e:
        return f"Error reading file: {e}"

# call agent function
async def call_sample_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""Call general agent to execute the task: {role}

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do.
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
