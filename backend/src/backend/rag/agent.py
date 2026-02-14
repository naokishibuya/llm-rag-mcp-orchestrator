import json
import logging

from pydantic import BaseModel

from ..core import Chat, Message, Reply, Role, UserContext
from .client import RAGClient


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context from the knowledge base.

Use the context to answer the user's question. If the context doesn't contain relevant information, set relevant to false — do not answer from conversation history or general knowledge.
Be concise and accurate. Cite the source if available.

Respond as JSON: {"answer": "...", "relevant": true/false}"""


class RAGResponse(BaseModel):
    answer: str
    relevant: bool


class RAGAgent:
    def __init__(self, model: Chat, index: RAGClient, top_k: int = 3):
        self._model = model
        self._index = index
        self._top_k = top_k

    async def act(
        self,
        *,
        query: str,
        context: UserContext,
        params: dict = None,
        **_,
    ) -> Reply:
        params = params or {}
        search_query = params.get("query", query)
        logger.info(f"RAG search query: {search_query!r}")

        knowledge = self._retrieve(search_query)
        if knowledge is None:
            return Reply(
                text="I couldn't find relevant information in the knowledge base for your question.",
                model=self._model.model,
                success=False,
            )

        system = f"{SYSTEM_PROMPT}\n\nContext from knowledge base:\n\n{knowledge}"
        if context:
            system += f"\n\n{context}"

        messages = [
            Message(role=Role.SYSTEM, content=system),
            Message(role=Role.USER, content=query),
        ]

        reply = self._model.query(messages, RAGResponse.model_json_schema())
        if not reply.success:
            return reply

        try:
            parsed = RAGResponse(**json.loads(reply.text))
            answer = parsed.answer
            found_relevant = parsed.relevant
        except Exception:
            answer = reply.text
            found_relevant = True  # Benefit of the doubt if parsing fails

        logger.info(f"RAG {reply} relevant={found_relevant}")

        return Reply(
            text=answer,
            model=reply.model,
            success=found_relevant,
            tokens=reply.tokens,
        )

    def _retrieve(self, query: str) -> str | None:
        docs = self._index.search(query, self._top_k)
        for i, doc in enumerate(docs or []):
            logger.info(f"  doc[{i}]: score={doc.score:.3f} source={doc.document.source!r} content={doc.document.content[:80]!r}")

        if not docs or docs[0].score < 0.3:
            logger.info("RAG: no relevant docs found (score < 0.3 or empty)")
            return None

        chunks = []
        for doc in docs:
            if doc.score >= 0.3:
                source = f" (Source: {doc.document.source})" if doc.document.source else ""
                chunks.append(f"{doc.document.content[:500]}{source}")

        return "\n\n---\n\n".join(chunks)
