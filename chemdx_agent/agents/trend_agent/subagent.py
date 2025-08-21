from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger
from chemdx_agent.utils import make_tool_message
from chemdx_agent.agents.MatDX_trend_agent.subagent import call_matdx_trend_agent
from chemdx_agent.agents.estm_trend_agent.subagent import call_estm_trend_agent
from chemdx_agent.agents.Phosphor_trend_agent.subagent import call_phosphor_trend_agent


name = "TrendAgent"
role = "Route user requests to the correct dataset-specific trend subagent (MatDX/ESTM/Phosphor)"
context = "Use the registered tools to dispatch exactly one subagent and return its result"

router_guidelines = (
    "You are the Trend router agent for ChemDX.\n"
    "Route the user's request to one of the following subagents based on dataset/intent:\n"
    "- MatDXTrendAgent: Trends on formation energy/materials in MatDX_EF.csv.\n"
    "- ESTMTrendAgent: Thermoelectric trends (Seebeck, conductivity, thermal conductivity, power factor, ZT).\n"
    "- PhosphorTrendAgent: Optical phosphor dataset-level trends (emission, CIE, decay, dopant, host).\n"
    "Guidelines:\n"
    "- If the user mentions ZT, Seebeck, power factor or TE properties, use ESTMTrendAgent.\n"
    "- If the user asks about formation energy or MatDX_EF columns, use MatDXTrendAgent.\n"
    "- If the user asks cross-material optical trends (not a single host+dopant ratio), use PhosphorTrendAgent.\n"
)

system_prompt = f"You are the {name}. {role}. {context}\n\n{router_guidelines}"


trend_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt=system_prompt,
)

@trend_agent.system_prompt(dynamic=True)
def dynamic_system_prompt(ctx: RunContext[AgentState]) -> str:
    deps = ctx.deps
    return working_memory_prompt.format(
        main_goal = deps.main_task,
        working_memory = deps.working_memory_description,
    )

working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

# Register downstream subagents
trend_agent.tool(call_matdx_trend_agent)
trend_agent.tool(call_estm_trend_agent)
trend_agent.tool(call_phosphor_trend_agent)


async def call_trend_agent(ctx: RunContext[AgentState], message2agent: str):
    """Call the Trend router agent.

    Accepts a user query and dispatches exactly one of the MatDX/ESTM/Phosphor
    trend subagents. It explicitly supports property-to-property trend comparison
    requests (e.g., Seebeck vs. electrical conductivity, emission wavelength vs.
    CIE x/y, decay time vs. dopant concentration) and forwards the full context
    to the appropriate dataset domain (MatDX_EF, ESTM, phosphor optical DB).

    Behavior:
    - Logs the input and passes the agent state (deps).
    - Invokes exactly one chosen subagent.
    - Logs and returns the action and result.

    Returns:
    - Result model containing the selected action and the final text/structured output.
    """
    agent_name = "TrendAgent"
    deps = ctx.deps or AgentState()
    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    
    user_prompt = f"Current Task of your role: {message2agent}"
    result = await trend_agent.run(user_prompt, deps=deps)
    output = result.output
    if hasattr(deps, "add_working_memory"):
        deps.add_working_memory(agent_name, message2agent)
    if hasattr(deps, "increment_step"):
        deps.increment_step()
    logger.info(f"[{agent_name}] Action: {output.action}")
    
    list_tool_log = make_tool_message(result)
    for log in list_tool_log:
        logger.info(log)
    
    logger.info(f"[{agent_name}] Result: {output.result}")
    return output


