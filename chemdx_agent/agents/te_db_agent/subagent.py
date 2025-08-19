<<<<<<< Updated upstream
from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result


system_prompt = "You are the agent of the thermoelectric materials database. you have access to the database of thermoelectric materials at different composition ratios and doping. you also have access to their seebeck_coefficient(μV/K),electrical_conductivity(S/m),thermal_conductivity(W/mK),power_factor(W/mK2) and ZT data at differnet temperatures (K)."

general_agent = Agent(
=======
from typing import List
from pydantic_ai import Agent, RunContext

from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger

name = "ThermoelectricDBAgent"
role = "an agent that can search and answer questions about thermoelectric materials based on the LitDX_TE database."
context = "You are the agent of the thermoelectric materials database. you have access to the database of thermoelectric materials at different composition ratios and doping. you also have access to their seebeck_coefficient(μV/K),electrical_conductivity(S/m),thermal_conductivity(W/mK),power_factor(W/mK2) and ZT data at differnet temperatures (K)."


system_prompt = f"""You are the {name}. You can use available tools or request help from specialized sub-agents that perform specific tasks. You must only carry out the role assigned to you. If a request is outside your capabilities, you should ask for support from the appropriate agent instead of trying to handle it yourself.

Your Currunt Role: {role}
Important Context: {context}
"""


TE_DB_agent = Agent(
>>>>>>> Stashed changes
    model = "openai:gpt-4o",
    output_type = Result,
    deps_type = AgentState,
    model_settings = {
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
<<<<<<< Updated upstream
    system_propmpt = system_prompt,
)


# Tool setting
@general_agent.tool_plain
def material_property_search(name: str) -> str:
    """Search the desired property of the material
    args:
        name: (str) The name of the material to search
        property: (str) The property value of the material to search
    output:
        (str) The 
=======
    system_prompt = system_prompt,
)

# Tool setting
@TE_DB_agent.tool_plain
def material_properties_at_temp(args_1: str, args_2: List[str]) -> str:
    """Sample function
    args:
        Name: (str) Material name
        Temp: (str) Target temperature (K)
    output:
        (str) The result of the function
>>>>>>> Stashed changes
    """
    return "FEJWOE"


<<<<<<< Updated upstream

# call agent function
async def call_general_agent(ctx: RunContext[AgentState], message2agent: str):
    """Call general agent to execute the task: general agnet can ~~
    To call draw agent, numpy array for data is needed.
=======
# call agent function
async def call_TE_DB_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""Call general agent to execute the task: {role}
>>>>>>> Stashed changes

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do.
    """
<<<<<<< Updated upstream
    deps = ctx.deps
    return await general_agent.run(
        message2agent, deps=deps
    )
=======
    agent_name = "SampleAgent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    result = await TE_DB_agent.run(
        message2agent, deps=deps
    )
    output = result.output

    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")

    return output
>>>>>>> Stashed changes
