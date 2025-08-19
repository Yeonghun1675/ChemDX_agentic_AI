from pydantic_ai import Agent
from pydantic_graph import BaseNode
from dataclasses import dataclass

from chemdx_agent.schema import AgentState, AgentInput, FinalAnswer
from chemdx_agent.logger import logger
from chemdx_agent.agents import *


system_prompt = "You are the main agent of the ChemDX database."

tools = []

main_agent = Agent(
    model = "openai:gpt-4o",
    output_type = FinalAnswer,
    deps_type = AgentState,
    model_settings = {
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_propmpt = system_prompt,
    tools = tools
)

# connect main agent with subagent
main_agent.tool(call_sample_agent)


async def run_main_agent(message: str):
    logger.info(f"[Question] {message}")
    result = await main_agent.run(message)
    output = result.output
    logger.info(f"[Final Answer] {output.final_answer}")
    logger.info(f"[Evaluation] {output.evaluation}")
    return output