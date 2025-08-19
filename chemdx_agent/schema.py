from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Callable


class AgentArgument(BaseModel):
    name: str = Field(description="Name of the agent")
    description: str = Field(description="Description of the agent")
    context: str = Field(description="Context of the agent")
    connected_nodes: List[str] = Field(description="List of nodes that you can go to")
    list_of_tools: Optional[List[Callable]] = Field(description="List of tools that you can use")


class ResultTool(BaseModel):
    tool_name: str = Field(description="Name of the tool that was used")
    tool_result: str = Field(description="Result of the tool")


class Result(BaseModel):
    task: str = Field(description="Clear description of the task that was executed")
    action: str = Field(description="Detailed action of the task")
    result: str = Field(description="Result of the task")


class AgentInput(BaseModel):
    main_task: str = Field(description="Clear description of the main task")
    current_task: str = Field(description="Clear description of the current task")


class MoveNode(BaseModel):
    current_task: str = Field(description="Current task of this agent")
    action_in_this_node: List[Result] = Field(description="List of actions that were executed in this node")
    summary_result: str = Field(description="Summary of the result")
    next_agent_name: str = Field(description="Name of the agent to move")
    next_action_to_take: str = Field(description="Action of the next node")


class AgentState(BaseModel):
    main_task: str = Field(description="Clear description of the main task")
    working_memory: str = Field(description="Working memory of the agent", default="")
    current_step: int = 0
    results: List[ResultTool] = Field(description="List of results", default=[])