from pydantic_ai import Agent
from pydantic_graph import BaseNode
from dataclasses import dataclass

from chemdx_agent.schema import AgentState, AgentInput, Result
from chemdx_agent.agents import *


system_prompt = "You are the main agent of the ChemDX database."

tools = []

main_agent = Agent(
    model = "openai:gpt-4o",
    output_type = Result,
    deps_type = AgentState,
    model_settings = {
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_propmpt = system_prompt,
    tools = tools
)

main_agent.tool(call_general_agent)