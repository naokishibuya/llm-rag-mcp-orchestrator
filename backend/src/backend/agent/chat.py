import time
from dataclasses import dataclass

from ..safety import Intent
from ..tools import get_finance_quote
from ..llm.base import LLMResponse
from .base import BaseAgent, AgentResponse


@dataclass
class Message:
    role: str
    content: str


SYSTEM_INSTRUCTION = """You are a helpful assistant. For greetings, respond naturally.
For factual questions, answer clearly and concisely (1-2 sentences)."""


class ChatAgent(BaseAgent):
    async def run(self, messages: list[Message]) -> AgentResponse:
        if not messages or messages[-1].role != "user":
            return AgentResponse(
                answer="Error: Last message must be from user.",
                intent=Intent.BAD,
                moderation=None,
                model=self.model,
                embedding_model=self.embedding_model,
            )

        last_message = messages[-1].content
        history = messages[:-1]

        intent, moderation = self.analyze(last_message)

        if self.should_refuse(intent, moderation):
            return AgentResponse(
                answer="I'm sorry, but I can't assist with that request.",
                intent=intent.intent,
                moderation=moderation,
                rationale=intent.rationale,
                model=self.model,
                embedding_model=self.embedding_model,
            )

        if self.should_escalate(intent):
            return AgentResponse(
                answer="This request may require a human assistant.",
                intent=intent.intent,
                moderation=moderation,
                rationale=intent.rationale,
                model=self.model,
                embedding_model=self.embedding_model,
            )

        if intent.intent == Intent.SMALL_TALK:
            response = self.generate_small_talk(last_message)
            return AgentResponse(
                answer=response.text,
                intent=intent.intent,
                moderation=moderation,
                rationale=intent.rationale,
                model=self.model,
                embedding_model=self.embedding_model,
                metrics=self._build_metrics(response),
            )

        if intent.intent == Intent.FINANCE_QUOTE:
            answer = await get_finance_quote(last_message)
            return AgentResponse(
                answer=answer,
                intent=intent.intent,
                moderation=moderation,
                rationale=intent.rationale,
                model=self.model,
                embedding_model=self.embedding_model,
                metrics={"tool": "mcp:finance_quote"},
            )

        if intent.intent == Intent.MEMORY_WRITE:
            return AgentResponse(
                answer="Got it. I'll remember that once persistent memory is configured.",
                intent=intent.intent,
                moderation=moderation,
                rationale=intent.rationale,
                model=self.model,
                embedding_model=self.embedding_model,
            )

        response = self._chat_with_context(last_message, history)
        return AgentResponse(
            answer=response.text,
            intent=intent.intent,
            moderation=moderation,
            rationale=intent.rationale,
            model=self.model,
                embedding_model=self.embedding_model,
            metrics=self._build_metrics(response),
        )

    def _chat_with_context(self, query: str, history: list[Message]) -> LLMResponse:
        results = self.index.search(query, top_k=3)
        context = "\n\n".join(r.document.content for r in results)

        chat_messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

        if context:
            chat_messages.append({"role": "system", "content": f"Context:\n{context}"})

        for msg in history:
            chat_messages.append({"role": msg.role, "content": msg.content})

        chat_messages.append({"role": "user", "content": query})

        start = time.time()
        response = self.chat.chat(chat_messages)
        latency = (time.time() - start) * 1000

        self.tracker.record(
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=latency,
            operation="chat_rag",
        )

        return response
