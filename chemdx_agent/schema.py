from pydantic import BaseModel, Field
from typing import List


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


class AgentState(BaseModel):
    main_task: str = Field(description="Clear description of the main task")
    working_memory: str = Field(description="Working memory of the agent", default="")
    current_step: int = 0
    results: List[ResultTool] = Field(description="List of results", default=[])


class FinalAnswer(BaseModel):
    task: str = Field(description="Clear description of the task that was executed")
    final_answer: str = Field(description="Final answer of the task")
    evaluation: str = Field(description="Evaluation whether the final_answer solved the task well or not")