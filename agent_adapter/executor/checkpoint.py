from typing import Any
from agent_framework import Executor, WorkflowContext, handler


class CheckpointExecutor(Executor):
    """Base class for executors with checkpoint saving and restoring behavior."""

    def __init__(self, id: str) -> None:
        super().__init__(id=id)
        self._messages: list[str] = []

    async def on_checkpoint_save(self) -> dict[str, Any]:
        return {"messages": self._messages}

    async def on_checkpoint_restore(self, state: dict[str, Any]) -> None:
        self._messages = state.get("messages", [])
