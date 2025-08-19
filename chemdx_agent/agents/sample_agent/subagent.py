from typing import List
from pydantic_ai import Agent, RunContext

from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger

name = "SampleAgent"
role = "description of sample"
context = "Important Context that agent should be know. For example, you must not use numpy array. instead, use pandas DataFrame."


system_prompt = f"""You are the {name}. You can use available tools or request help from specialized sub-agents that perform specific tasks. You must only carry out the role assigned to you. If a request is outside your capabilities, you should ask for support from the appropriate agent instead of trying to handle it yourself.

Your Currunt Role: {role}
Important Context: {context}
"""


sample_agent = Agent(
    model = "openai:gpt-4o",
    output_type = Result,
    deps_type = AgentState,
    model_settings = {
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt = system_prompt,
)

# Tool setting
@sample_agent.tool_plain
def sample_function(args_1: str, args_2: List[str]) -> str:
    """Sample function
    args:
        args_1: (str) The first argument
        args_2: (List[str]) The second argument
    output:
        (str) The result of the function
    """
    return "FEJWOE"


# call agent function
async def call_sample_agent(ctx: RunContext[AgentState], message2agent: str):
    f"""Call general agent to execute the task: {role}

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do.
    """
    agent_name = "SampleAgent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")
    result = await sample_agent.run(
        message2agent, deps=deps
    )
    output = result.output

    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")

    return output
