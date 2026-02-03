from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass
class Document:
    content: str
    metadata: dict = None
    embedding: list[float] = None


@dataclass
class SearchResult:
    document: Document
    score: float


class VectorIndex:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents: list[Document] = []
        self._vectors: np.ndarray | None = None

    def add_documents(self, documents: list[Document]):
        texts = [d.content for d in documents]
        resp = self.embeddings.embed(texts)

        for doc, emb in zip(documents, resp.embeddings):
            doc.embedding = emb
            self.documents.append(doc)

        self._rebuild_index()

    def _rebuild_index(self):
        if not self.documents:
            self._vectors = None
            return
        self._vectors = np.array([d.embedding for d in self.documents])

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not self.documents:
            return []

        query_emb = np.array(self.embeddings.embed_query(query))
        scores = self._vectors @ query_emb / (
            np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query_emb) + 1e-9
        )

        indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self.documents[i], float(scores[i])) for i in indices]


def load_documents_from_dir(path: Path) -> list[Document]:
    documents = []
    for file in path.glob("*"):
        if file.is_file() and file.suffix in (".txt", ".md", ".pdf"):
            try:
                content = file.read_text(encoding="utf-8")
                documents.append(Document(content=content, metadata={"source": str(file)}))
            except Exception:
                continue
    return documents


_index: VectorIndex | None = None


def get_index() -> VectorIndex:
    global _index
    if _index is None:
        from ..llm import get_embeddings
        _index = VectorIndex(get_embeddings())

        data_dir = Path(__file__).parent.parent.parent.parent.parent / "data"
        if data_dir.exists():
            docs = load_documents_from_dir(data_dir)
            if docs:
                _index.add_documents(docs)

    return _index
