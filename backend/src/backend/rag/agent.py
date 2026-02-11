from ..llm import Chat, Embeddings
from .client import RAGClient, RAGResult


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context from the knowledge base.

Use the context to answer the user's question. If the context doesn't contain relevant information, say so clearly — do not answer from conversation history or general knowledge.
Be concise and accurate. Cite the source if available."""


class RAGAgent:
    def __init__(self):
        self.index = RAGClient()

    async def execute(
        self,
        *,
        model: Chat,
        query: str,
        history: list[dict],
        embedder: Embeddings,
        params: dict = None,
        **_,
    ) -> RAGResult:
        params = params or {}
        search_query = params.get("query", query)

        docs = self.index.search(search_query, embedder, 3)

        if not docs or (docs and docs[0].score < 0.3):
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
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": query})

        response = model.chat(messages)

        return RAGResult(
            response=response.text,
            tool="search_knowledge_base",
            args={"query": search_query},
            result={
                "docs_found": len([d for d in docs if d.score >= 0.3]),
                "top_score": docs[0].score if docs else 0,
            },
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
