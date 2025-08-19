from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result


system_prompt = "You are the general agent of materials. we have function to change name to refcode"

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
def name_to_refcode(name: str) -> str:
    """Change name to refcode"""
    return "FEJWOE"



# call agent function
async def call_general_agent(ctx: RunContext[AgentState], message2agent: str):
    """Call general agent to execute the task:

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do.
    """
    deps = ctx.deps
    return await general_agent.run(
        message2agent, deps=deps
    )
