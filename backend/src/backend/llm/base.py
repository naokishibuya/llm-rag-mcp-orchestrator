from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    raw: Any = None


@dataclass
class EmbeddingResponse:
    embeddings: list[list[float]]
    input_tokens: int = 0
    model: str = ""


class BaseChat(ABC):
    def __init__(self, model: str, temperature: float = 0.5, **kwargs):
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs

    @abstractmethod
    def complete(self, prompt: str) -> LLMResponse:
        pass

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        pass


class BaseEmbeddings(ABC):
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def embed(self, texts: list[str]) -> EmbeddingResponse:
        pass

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text]).embeddings[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts).embeddings
