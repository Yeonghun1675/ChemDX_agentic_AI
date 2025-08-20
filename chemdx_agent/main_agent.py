from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_graph import BaseNode
from dataclasses import dataclass
from typing import Optional

from chemdx_agent.schema import AgentState, AgentInput, FinalAnswer
from chemdx_agent.logger import logger
from chemdx_agent.agents import *


system_prompt = (
    "You are the main agent of the ChemDX database.\n"
    "Routing rules for tools/subagents:\n"
    "- Use TrendAgent for generic cross-dataset X–Y trend questions (e.g., 'emission max vs color', 'intensity vs temperature').\n"
    "- Use ColorTrendAgent ONLY for host+dopant specific color/emission vs dopant ratio analysis (requires host and dopant).\n"
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
# Register tools in an order that biases generic trend questions to TrendAgent first
main_agent.tool(call_trend_agent)
main_agent.tool(call_color_trend_agent)
main_agent.tool(call_phosphor_lookup_agent)
main_agent.tool(call_recommend_agent)


async def run_main_agent(message: str):
    logger.info(f"[Question] {message}")
    result = await main_agent.run(
        message,
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