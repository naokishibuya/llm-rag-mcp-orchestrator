import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..config import calc_cost
from ..llm import get_chat, LLMResponse
from ..metrics import get_tracker
from ..rag import get_index
from ..safety import Intent, IntentResult, ModerationResult, classify_intent, run_moderation


@dataclass
class AgentResponse:
    answer: str
    intent: Intent
    moderation: ModerationResult
    rationale: str | None = None
    model: str = ""
    embedding_model: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "intent": self.intent.value,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "moderation": {
                "verdict": self.moderation.verdict.value,
                "severity": self.moderation.severity.value,
                "categories": list(self.moderation.categories),
                "rationale": self.moderation.rationale,
            },
            "routing_rationale": self.rationale,
            "metrics": self.metrics,
        }


class BaseAgent(ABC):
    def __init__(self, model: str | None = None, embedding_model: str | None = None):
        self.chat = get_chat(model)
        self.index = get_index(embedding_model)
        self.tracker = get_tracker()
        self.model = self.chat.model
        self.embedding_model = self.index.embeddings.model

    def analyze(self, text: str) -> tuple[IntentResult, ModerationResult]:
        moderation = run_moderation(text, self.chat)
        if moderation.is_blocked:
            return IntentResult(Intent.BAD, moderation.rationale), moderation

        intent = classify_intent(text, self.chat)
        return intent, moderation

    def should_refuse(self, intent: IntentResult, moderation: ModerationResult) -> bool:
        return moderation.is_blocked or intent.intent == Intent.BAD

    def should_escalate(self, intent: IntentResult) -> bool:
        return intent.intent == Intent.ESCALATE

    def _build_metrics(self, response: LLMResponse) -> dict[str, Any]:
        cost = calc_cost(response.model, response.input_tokens, response.output_tokens)
        return {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost": round(cost, 6),
        }

    def generate_with_context(self, query: str, top_k: int = 3) -> LLMResponse:
        results = self.index.search(query, top_k=top_k)
        context = "\n\n".join(r.document.content for r in results)

        prompt = f"""Answer based on context. Be concise (1-2 sentences).

Context:
{context}

Question: {query}

Answer:"""

        start = time.time()
        response = self.chat.complete(prompt)
        latency = (time.time() - start) * 1000

        self.tracker.record(
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=latency,
            operation="rag_query",
        )

        return response

    def generate_small_talk(self, text: str) -> LLMResponse:
        prompt = f"""Respond warmly and concisely (max 2 sentences):
\"\"\"{text}\"\"\""""

        start = time.time()
        response = self.chat.complete(prompt)
        latency = (time.time() - start) * 1000

        self.tracker.record(
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=latency,
            operation="small_talk",
        )

        return response

    @abstractmethod
    async def run(self, *args, **kwargs) -> AgentResponse:
        pass
