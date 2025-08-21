from pydantic_ai import Agent, RunContext
from chemdx_agent.schema import AgentState, Result


system_prompt = "You are the general agent of materials. we have function to change name to refcode "


working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""


general_agent = Agent(
    model = "openai:gpt-4o",
    output_type = Result,
    deps_type = AgentState,
    model_settings = {
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt = system_prompt,
)


@general_agent.system_prompt(dynamic=True)
def dynamic_system_prompt(ctx: RunContext[AgentState]) -> str:
    deps = ctx.deps
    return working_memory_prompt.format(
        main_goal=deps.main_task,
        working_memory=deps.working_memory_description,
    )


# Tool setting
@general_agent.tool_plain
def name_to_refcode(name: str) -> str:
    """Change name to refcode
    args:
        name: (str) The name of the material to change to refcode
    output:
        (str) The refcode of the materials
    """
    return "FEJWOE"


@general_agent.tool_plain
def name_to_smiles(name: str) -> str:
    """Change name to refcode
    args:
        name: (str) The name of the material to change to refcode
    output:
        (str) The refcode of the materials
    """
    return "FEJWOE"



# call agent function
async def call_general_agent(ctx: RunContext[AgentState], message2agent: str):
    """Call general agent to execute the task: general agnet can ~~
    To call draw agent, numpy array for data is needed.

    args:
        message2agent: (str) A message to pass to the agent. Since you're talking to another AGENT, you must describe in detail and specifically what you need to do.
    """
    deps = ctx.deps
    result = await general_agent.run(message2agent, deps=deps)
    output = result.output
    deps.add_working_memory("DrawPlotAgent", message2agent)
    deps.increment_step()
    return output
