from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..llm import Embeddings


@dataclass
class RAGResult:
    response: str = ""
    model: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    result: str | dict = ""
    success: bool = True
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Document:
    content: str
    source: str = ""
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    document: Document
    score: float


class RAGClient:
    DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data"

    def __init__(self):
        self._documents: list[Document] = []
        self._embeddings: np.ndarray | None = None
        self._embedder_model: str | None = None

    def search(
        self, query: str, embedder: Embeddings, top_k: int = 3
    ) -> list[SearchResult]:
        self._ensure_indexed(embedder)

        if not self._documents or self._embeddings is None:
            return []

        query_vec = np.array(embedder.embed(query))

        scores = self._embeddings @ query_vec / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-9
        )

        indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self._documents[i], float(scores[i])) for i in indices]

    def _ensure_indexed(self, embedder: Embeddings) -> None:
        if self._embedder_model == embedder.model and self._embeddings is not None:
            return

        if not self._documents:
            self._documents = self._load_documents()

        if not self._documents:
            self._embeddings = None
            self._embedder_model = embedder.model
            return

        texts = [d.content for d in self._documents]
        vectors = embedder.embed(texts)

        for doc, vec in zip(self._documents, vectors):
            doc.embedding = vec

        self._embeddings = np.array(vectors)
        self._embedder_model = embedder.model

    def _load_documents(self) -> list[Document]:
        docs = []
        if self.DATA_DIR.exists():
            for file in self.DATA_DIR.glob("*"):
                if file.is_file() and file.suffix in (".txt", ".md"):
                    try:
                        content = file.read_text(encoding="utf-8")
                        docs.append(Document(content=content, source=file.name))
                    except Exception:
                        continue
        return docs
