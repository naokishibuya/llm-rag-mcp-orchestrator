from dataclasses import dataclass
from typing import Protocol


@dataclass
class Response:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class Chat(Protocol):
    model: str

    def chat(self, messages: list[dict[str, str]], schema: type | None = None) -> Response: ...


class Embeddings(Protocol):
    model: str

    def embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        ...
