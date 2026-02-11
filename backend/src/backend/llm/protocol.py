from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Response:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    tools_used: list[str] = field(default_factory=list)


class Chat(Protocol):
    model: str

    def chat(self, messages: list[dict[str, str]], schema: type | None = None, tools: dict[str, callable] | None = None) -> Response: ...


class Embeddings(Protocol):
    model: str

    def embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        ...
