from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger
from chemdx_agent.agents.phosphor_lookup_agent.subagent import call_phosphor_lookup_agent
from chemdx_agent.agents.recommend_agent.subagent import call_recommend_agent
from chemdx_agent.agents.color_trend_agent.subagent import call_color_trend_agent


system_prompt = (
    "You are the phosphors router agent for ChemDX.\n"
    "Route the user's request to exactly one of the following subagents based on intent:\n"
    "- PhosphorLookupAgent: Look up optical properties by formula (emission, decay, CIE, hex color).\n"
    "  - Use 'search_candidates' when the user provides constraints like emission range, decay limit, and/or IQE threshold.\n"
    "- RecommendAgent: Recommend phosphor materials given a desired color and minimum decay time.\n"
    "- ColorTrendAgent: Analyze color/emission trend vs dopant ratio for a specific host and dopant.\n"
    "Guidelines:\n"
    "- If the user asks to find data for a specific composition/formula, use PhosphorLookupAgent.\n"
    "- If the user presents hard constraints (e.g., emission 360–420 nm, decay <= 100 ns, IQE >= 80%), use PhosphorLookupAgent.search_candidates first, then pass the top 1-2 to ColorTrendAgent for ratio trend analysis.\n"
    "- If the user asks for recommendations by color (e.g., 'red', 'blue', 'warm white') with constraints like decay, use RecommendAgent.\n"
    "- If the user provides a host and a dopant and wants the trend vs concentration/ratio, use ColorTrendAgent.\n"
)


phosphor_data_research_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt=system_prompt,
)

# Register downstream subagents as tools of the router
phosphor_data_research_agent.tool(call_phosphor_lookup_agent)
phosphor_data_research_agent.tool(call_recommend_agent)
phosphor_data_research_agent.tool(call_color_trend_agent)


async def call_phosphor_data_research_agent(ctx: RunContext[AgentState], message2agent: str):
    """Call the PhosphorDataResearchAgent router. It will decide which subagent to use based on the message."""
    agent_name = "PhosphorDataResearchAgent"
    deps = ctx.deps
    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    result = await phosphor_data_research_agent.run(message2agent, deps=deps)
    output = result.output
    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")
    return output


