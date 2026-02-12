import json
import logging

from pydantic import BaseModel

from ..llm import Chat, Embeddings
from .client import RAGClient, RAGResult


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context from the knowledge base.

Use the context to answer the user's question. If the context doesn't contain relevant information, set relevant to false — do not answer from conversation history or general knowledge.
Be concise and accurate. Cite the source if available.

Respond as JSON: {"answer": "...", "relevant": true/false}"""


class RAGResponse(BaseModel):
    answer: str
    relevant: bool


class RAGAgent:
    def __init__(self, embedder: Embeddings):
        self.index = RAGClient()
        self._embedder = embedder

    async def execute(
        self,
        *,
        model: Chat,
        query: str,
        params: dict = None,
        **_,
    ) -> RAGResult:
        params = params or {}
        search_query = params.get("query", query)
        logger.info(f"RAG search query: {search_query!r}")

        docs = self.index.search(search_query, self._embedder, 3)
        for i, doc in enumerate(docs or []):
            logger.info(f"  doc[{i}]: score={doc.score:.3f} source={doc.document.source!r} content={doc.document.content[:80]!r}")

        if not docs or (docs and docs[0].score < 0.3):
            logger.info("RAG: no relevant docs found (score < 0.3 or empty)")
            return RAGResult(
                response="I couldn't find relevant information in the knowledge base for your question.",
                tool="search_knowledge_base",
                args={"query": search_query},
                result={"docs_found": 0},
                success=False,
            )

        doc_contexts = []
        for doc in docs:
            if doc.score >= 0.3:
                source = f" (Source: {doc.document.source})" if doc.document.source else ""
                doc_contexts.append(f"{doc.document.content[:500]}{source}")

        context_text = "\n\n---\n\n".join(doc_contexts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Context from knowledge base:\n\n{context_text}"},
            {"role": "user", "content": query},
        ]

        response = model.chat(messages, schema=RAGResponse)

        try:
            parsed = RAGResponse(**json.loads(response.text))
            answer = parsed.answer
            found_relevant = parsed.relevant
        except Exception:
            answer = response.text
            found_relevant = True  # Benefit of the doubt if parsing fails

        logger.info(
            f"RAG response: relevant={found_relevant} tokens=[{response.input_tokens}/{response.output_tokens}] "
            f"answer={answer[:120]!r}"
        )

        return RAGResult(
            response=answer,
            tool="search_knowledge_base",
            args={"query": search_query},
            result={
                "docs_found": len([d for d in docs if d.score >= 0.3]),
                "top_score": docs[0].score if docs else 0,
            },
            success=found_relevant,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
