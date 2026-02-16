from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..types import Embeddings


KNOWLEDGE_DIR = Path(__file__).parent.parent.parent.parent.parent / "knowledge"


@dataclass
class Document:
    content: str
    source: str = ""
    topic: str = ""


@dataclass
class SearchResult:
    document: Document
    score: float


class RAGClient:
    def __init__(self, embedder: Embeddings, topics: list[dict] | None = None):
        self._embedder = embedder
        self._topics = topics or []
        self._documents: list[Document] = []
        self._embeddings: np.ndarray | None = None

    def search(self, query: str, top_k: int = 3, topic: str = "") -> list[SearchResult]:
        self._ensure_indexed()

        if not self._documents or self._embeddings is None:
            return []

        query_vec = np.array(self._embedder.embed(query))

        scores = self._embeddings @ query_vec / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-9
        )

        # Filter by topic if specified
        if topic:
            topic_lower = topic.lower()
            mask = np.array([d.topic.lower() == topic_lower for d in self._documents])
            scores = np.where(mask, scores, -1.0)

        indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self._documents[i], float(scores[i])) for i in indices]

    def topic_names(self) -> list[str]:
        return [t["name"] for t in self._topics]

    def topic_descriptions(self) -> list[str]:
        return [f"{t['name']}: {t['description']}" for t in self._topics]

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
        if not self._topics:
            docs.extend(self._scan_dir(KNOWLEDGE_DIR))
        else:
            for topic in self._topics:
                topic_dir = KNOWLEDGE_DIR / topic["directory"]
                docs.extend(self._scan_dir(topic_dir, topic["name"]))
        return docs

    @staticmethod
    def _scan_dir(directory: Path, topic: str = "") -> list[Document]:
        docs = []
        if not directory.exists():
            return docs
        for file in directory.glob("**/*"):
            if file.is_file() and file.suffix in (".txt", ".md"):
                try:
                    content = file.read_text(encoding="utf-8")
                    source = f"[{topic}] {file.name}" if topic else file.name
                    docs.append(Document(content=content, source=source, topic=topic))
                except Exception:
                    continue
        return docs
