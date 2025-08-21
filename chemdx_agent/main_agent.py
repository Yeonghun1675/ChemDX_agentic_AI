from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from chemdx_agent.schema import AgentState, AgentInput, FinalAnswer
from chemdx_agent.logger import logger
from chemdx_agent.agents import *


system_prompt = (
    "You are the Main Agent of ChemDX Agentic AI. Your role is to efficiently solve complex tasks by coordinating sub-agents rather than handling problems directly. Break down the main task into smaller, well-defined subproblems, and delegate each to the most suitable sub-agent. Always value efficiency: avoid redundant steps and reuse results when possible. Your responsibility is to integrate the sub-agents’ outputs, resolve conflicts if their results differ, and decide the next step. You do not perform the detailed work of solving subproblems; you orchestrate, monitor progress, and ensure the final solution is coherent and complete.\n"
    "Route user requests to the appropriate router: PhosphorDataResearchAgent (lookup/recommend/color-trend) or TrendAgent (dataset-level trends).\n"
)

tools = []

main_agent = Agent(
    model = "openai:gpt-4o",
    output_type = FinalAnswer,
    deps_type = AgentState,
    model_settings = {
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt = system_prompt,
    tools = tools
)

# connect main agent with subagent
#main_agent.tool(call_sample_agent)
# Connect to routers
main_agent.tool(call_phosphor_data_research_agent)
main_agent.tool(call_trend_agent)


async def run_main_agent(message: str):
    logger.info(f"[Question] {message}")
    result = await main_agent.run(
        message,
        deps=AgentState(main_task=message),
        usage_limits=UsageLimits(
            request_limit=None,
            input_tokens_limit=None,
            output_tokens_limit=None,
        ),
    )
    output = result.output
    logger.info(f"[Final Answer] {output.final_answer}")
    logger.info(f"[Evaluation] {output.evaluation}")
    return output