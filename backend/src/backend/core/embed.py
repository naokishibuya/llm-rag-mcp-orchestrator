from typing import Protocol


Embedding = list[float]


class Embeddings(Protocol):
    model: str

    def embed(self, texts: list[str] | str) -> Embedding | list[Embedding]: ...
