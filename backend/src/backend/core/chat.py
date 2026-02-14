from enum import StrEnum
from typing import Protocol, TypedDict

from .reply import Reply


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(TypedDict):
    role: Role
    content: str


class Chat(Protocol):
    model: str

    def ask(
        self,
        messages: list[Message],
        tools: dict[str, callable] | None = None,
    ) -> Reply: ...

    def query(
        self,
        messages: list[Message],
        schema: dict,
    ) -> Reply: ...
