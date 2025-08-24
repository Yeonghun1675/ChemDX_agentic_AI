from pathlib import Path
import os

from pydantic_ai import Agent
from typing import Optional

from chemdx_agent.schema import AgentState, FinalAnswer
from chemdx_agent.logger import logger
from chemdx_agent.utils import make_tool_message
from chemdx_agent.agents import *


system_prompt = "You are the Main Agent of ChemDX Agentic AI. Your role is to efficiently solve complex tasks by coordinating sub-agents rather than handling problems directly. Break down the main task into smaller, well-defined subproblems, and delegate each to the most suitable sub-agent. Always value efficiency: avoid redundant steps and reuse results when possible. Your responsibility is to integrate the sub-agents’ outputs, resolve conflicts if their results differ, and decide the next step. You do not perform the detailed work of solving subproblems; you orchestrate, monitor progress, and ensure the final solution is coherent and complete."

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
main_agent.tool(call_thermoelectric_database_agent)
main_agent.tool(call_poscar_agent)
main_agent.tool(call_viz_agent)
main_agent.tool(call_mp_agent)
main_agent.tool(call_phosphor_data_research_agent)
main_agent.tool(call_trend_agent)
main_agent.tool(call_matdx_trend_agent)
main_agent.tool(call_MatDX_EF_DB_agent)
main_agent.tool(call_MatDX_Struc_Gen_agent)

async def run_main_agent(message: str, deps: Optional[AgentState] = None):
    if deps is None:
        deps = AgentState()

    # Clear log file contents only (keep the file)
    try:
        with open("log.txt", "r+") as f:
            f.truncate(0)  # Clear file contents 
    except FileNotFoundError:
        # Create new file if it doesn't exist
        with open("log.txt", "w") as f:
            f.write("")
    except Exception as e:
        logger.error(f"Failed to clear log file: {e}")

    deps.main_task = message

    logger.info(f"[Question] {message}ㅤ")
    result = await main_agent.run(
        message,
        deps=deps,
    )
    
    output = result.output
    list_tool_log = make_tool_message(result)
    for log in list_tool_log:
        logger.info(log)

    logger.info(f"[Final Answer] {output.final_answer}ㅤ")
    logger.info(f"[Evaluation] {output.evaluation}ㅤ")
    return output
