from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Protocol, TypedDict


@dataclass
class UserContext:
    city: str | None = None
    timezone: str | None = None
    local_time: str | None = None

    def __bool__(self) -> bool:
        return any(v is not None for v in asdict(self).values())

    def __str__(self) -> str:
        return f"user_context={asdict(self)}"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(TypedDict):
    role: Role
    content: str


@dataclass
class Tokens:
    input_tokens: int = 0
    output_tokens: int = 0

    def __str__(self) -> str:
        return f"tokens=[{self.input_tokens}/{self.output_tokens}]"


@dataclass
class Reply:
    text: str
    model: str = ""
    tokens: Tokens = field(default_factory=Tokens)
    tools_used: list[str] = field(default_factory=list)
    success: bool = True

    def __str__(self) -> str:
        parts = [f"[{self.model}] {self.tokens}: {self.text}"]
        if self.tools_used:
            parts.append(f"tools={self.tools_used}")
        if not self.success:
            parts.append("FAILED")
        return " ".join(parts)


class Chat(Protocol):
    model: str

    async def ask(
        self,
        messages: list[Message],
        tools: dict[str, callable] | None = None,
    ) -> Reply: ...

    async def query(
        self,
        messages: list[Message],
        schema: dict,
    ) -> Reply: ...


Embedding = list[float]


class Embeddings(Protocol):
    model: str

    def embed(self, texts: list[str] | str) -> Embedding | list[Embedding]: ...
