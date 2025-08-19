from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result


system_prompt = "You are the agent of the thermoelectric materials database. you have access to the database of thermoelectric materials at different composition ratios and doping. you also have access to their seebeck_coefficient(μV/K),electrical_conductivity(S/m),thermal_conductivity(W/mK),power_factor(W/mK2) and ZT data at differnet temperatures (K)."

general_agent = Agent(
    model = "openai:gpt-4o",
    output_type = Result,
    deps_type = AgentState,
    model_settings = {
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
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
    """
    return "FEJWOE"



# call agent function
async def call_general_agent(ctx: RunContext[AgentState], message2agent: str):
    """Call general agent to execute the task: general agnet can ~~
    To call draw agent, numpy array for data is needed.

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do.
    """
    deps = ctx.deps
    return await general_agent.run(
        message2agent, deps=deps
    )
