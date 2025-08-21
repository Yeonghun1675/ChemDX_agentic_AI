
from pydantic_graph import BaseNode
from dataclasses import dataclass
from typing import Optional

from chemdx_agent.schema import AgentState, FinalAnswer
from chemdx_agent.logger import logger
from chemdx_agent.agents import *


system_prompt = "You are the Main Agent of ChemDX Agentic AI. Your role is to efficiently solve complex tasks by coordinating sub-agents rather than handling problems directly. Break down the main task into smaller, well-defined subproblems, and delegate each to the most suitable sub-agent. Always value efficiency: avoid redundant steps and reuse results when possible. Your responsibility is to integrate the sub-agents’ outputs, resolve conflicts if their results differ, and decide the next step. You do not perform the detailed work of solving subproblems; you orchestrate, monitor progress, and ensure the final solution is coherent and complete. You MUST call on several agents to complete the task. If the user asks to generate a POSCAR file you MUST call on the dft poscar agent and the mat proj lookup agent and the databse agent."

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
main_agent.tool(call_database_agent)
main_agent.tool(call_dft_poscar_agent)
main_agent.tool(call_recommend_agent)
main_agent.tool(call_viz_agent)
main_agent.tool(call_materials_project_agent)
main_agent.tool(call_phosphor_lookup_agent)
main_agent.tool(call_sample_agent)
main_agent.tool(call_general_agent)


async def run_main_agent(message: str, deps: Optional[AgentState] = None):
    if deps is None:
        deps = AgentState()

    logger.info(f"[Question] {message}")
    deps.main_task = message
    result = await main_agent.run(message, deps=deps)
    output = result.output
    logger.info(f"[Final Answer] {output.final_answer}")
    logger.info(f"[Evaluation] {output.evaluation}")
    return output