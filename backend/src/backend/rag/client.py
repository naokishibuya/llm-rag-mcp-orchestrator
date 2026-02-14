from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core import Embeddings


@dataclass
class Document:
    content: str
    source: str = ""


@dataclass
class SearchResult:
    document: Document
    score: float


class RAGClient:
    DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "data"

    def __init__(self, embedder: Embeddings):
        self._embedder = embedder
        self._documents: list[Document] = []
        self._embeddings: np.ndarray | None = None

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        self._ensure_indexed()

        if not self._documents or self._embeddings is None:
            return []

        query_vec = np.array(self._embedder.embed(query))

        scores = self._embeddings @ query_vec / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-9
        )

        indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self._documents[i], float(scores[i])) for i in indices]

    def _ensure_indexed(self) -> None:
        if self._embeddings is not None:
            return

        self._documents = self._load_documents()
        if not self._documents:
            return

        texts = [d.content for d in self._documents]
        self._embeddings = np.array(self._embedder.embed(texts))

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
