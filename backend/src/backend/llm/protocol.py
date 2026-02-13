from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, TypedDict

from pydantic import BaseModel


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(TypedDict):
    role: Role
    content: str


@dataclass
class TokenUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Response:
    text: str
    tokens: TokenUsage = field(default_factory=lambda: TokenUsage(model=""))
    tools_used: list[str] = field(default_factory=list)


class Chat(Protocol):
    model: str

    def chat(self, messages: list[Message], schema: type[BaseModel] | None = None, tools: dict[str, callable] | None = None) -> Response: ...


class Embeddings(Protocol):
    model: str

    def embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        ...
