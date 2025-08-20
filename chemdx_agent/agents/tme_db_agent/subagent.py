from typing import List
from pydantic_ai import Agent, RunContext

from chemdx_agent.schema import AgentState, Result
from chemdx_agent.logger import logger

import pandas as pd
df = pd.read_csv("thermoelectrics.csv")

name = "DatabaseAgent"
role = "Query thermoelectric materials database"
context = """You are connected to a thermoelectric materials database. You can search for material properties, return results at specific temperatures, and compare entries. You have access to data on 
Formula	temperature(K), seebeck_coefficient(μV/K),	electrical_conductivity(S/m),	thermal_conductivity(W/mK),	power_factor(W/mK2),	ZT	and reference study. 
Always clarify if multiple candidate compositions are found.
"""

system_prompt = f"""You are the {name}. 
You can use available tools or request help from specialized sub-agents (e.g., VisualizationAgent, MaterialsProjectAgent). You must only carry out the role assigned to you. 
If a request is outside your capabilities, ask for support from the appropriate agent instead of handling it yourself.

Your Current Role: {role}
Important Context: {context}
"""

working_memory_prompt = """Main Goal: {main_goal}
Working Memory: {working_memory}
"""

database_agent = Agent(
    model="openai:gpt-4o",
    output_type=Result,
    deps_type=AgentState,
    model_settings={
        "temperature": 0.0,
        "parallel_tool_calls": False,
    },
    system_prompt=system_prompt,
)

@database_agent.system_prompt(dynamic=True)
def dynamic_system_prompt(ctx: RunContext[AgentState]) -> str:
    deps = ctx.deps
    return working_memory_prompt.format(
        main_goal=deps.main_task,
        working_memory=deps.working_memory_description,
    )

#-----------

#Tools

@database_agent.tool_plain
def read_database_schema() -> List[str]:
    """Return the list of available columns in the thermoelectric database."""
    return df.columns.tolist()

@database_agent.tool_plain
def find_material_variants(material_hint: str) -> List[str]:
    """Find all material formulas in the database that contain the given hint (e.g. element symbol or name).
    Args:
        material_hint: (str) a partial string such as 'Bi', 'Sb', or 'bismuth'
    Output:
        (List[str]) all matching formulas
    """
    matches = df[df["Formula"].str.contains(material_hint, case=False, regex=False)]
    unique_formulas = matches["Formula"].unique().tolist()
    return unique_formulas


#----------

async def call_database_agent(ctx: RunContext[AgentState], message2agent: str):
    """Call DatabaseAgent to execute a query on thermoelectric materials DB."""
    agent_name = "DatabaseAgent"
    deps = ctx.deps

    logger.info(f"[{agent_name}] Message2Agent: {message2agent}")

    user_prompt = f"Current Task of your role: {message2agent}"

    result = await database_agent.run(user_prompt, deps=deps)

    output = result.output
    deps.add_working_memory(agent_name, message2agent)
    deps.increment_step()

    logger.info(f"[{agent_name}] Action: {output.action}")
    logger.info(f"[{agent_name}] Result: {output.result}")

    return output
