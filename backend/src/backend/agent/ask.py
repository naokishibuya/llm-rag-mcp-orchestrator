from ..safety import Intent
from ..tools import get_finance_quote
from .base import BaseAgent, AgentResponse


class AskAgent(BaseAgent):
    async def run(self, question: str) -> AgentResponse:
        intent, moderation = self.analyze(question)

        if self.should_refuse(intent, moderation):
            return AgentResponse(
                answer="I'm sorry, but I can't assist with that request.",
                intent=intent.intent,
                moderation=moderation,
                rationale=intent.rationale,
            )

        if self.should_escalate(intent):
            return AgentResponse(
                answer="This request may require a human assistant.",
                intent=intent.intent,
                moderation=moderation,
                rationale=intent.rationale,
            )

        if intent.intent == Intent.SMALL_TALK:
            response = self.generate_small_talk(question)
            return AgentResponse(
                answer=response.text,
                intent=intent.intent,
                moderation=moderation,
                rationale=intent.rationale,
                metrics={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
            )

        if intent.intent == Intent.FINANCE_QUOTE:
            answer = await get_finance_quote(question)
            return AgentResponse(
                answer=answer,
                intent=intent.intent,
                moderation=moderation,
                rationale=intent.rationale,
            )

        if intent.intent == Intent.MEMORY_WRITE:
            return AgentResponse(
                answer="I'll remember that for later once memory storage is enabled.",
                intent=intent.intent,
                moderation=moderation,
                rationale=intent.rationale,
            )

        response = self.generate_with_context(question)
        return AgentResponse(
            answer=response.text,
            intent=intent.intent,
            moderation=moderation,
            rationale=intent.rationale,
            metrics={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
        )
