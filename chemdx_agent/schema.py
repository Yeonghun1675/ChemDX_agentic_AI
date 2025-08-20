from pydantic import BaseModel, Field
from typing import List, Any
import threading


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
    main_task: str = Field(description="Clear description of the main task", default="")
    working_memory: List[str] = Field(description="Working memory of the agent", default=[])
    current_step: int = Field(description="Current step of the agent", default=0)
    lock: Any = Field(default_factory=threading.Lock, exclude=True)

    def add_working_memory(self, agent_name: str, memory: str):
        with self.lock:
            self.working_memory.append(f"[{agent_name}] {memory}")

    def increment_step(self) -> int:
        """Thread-safe하게 current_step을 1씩 증가시키고 증가된 값을 반환합니다."""
        with self.lock:
            self.current_step += 1
            return self.current_step

    @property
    def working_memory_description(self) -> str:
        return "\n".join(self.working_memory)


class FinalAnswer(BaseModel):
    task: str = Field(description="Clear description of the task that was executed")
    final_answer: str = Field(description="Final answer of the task")
    evaluation: str = Field(description="Evaluation whether the final_answer solved the task well or not")