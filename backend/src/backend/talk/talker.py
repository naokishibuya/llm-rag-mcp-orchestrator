import logging
from dataclasses import dataclass

from ..llm import Chat

logger = logging.getLogger(__name__)


@dataclass
class TalkerResult:
    response: str
    success: bool = True
    input_tokens: int = 0
    output_tokens: int = 0


SYSTEM_PROMPT = """You are a helpful, friendly assistant.
Engage in natural conversation, answer questions, and provide assistance.

Be concise but informative. If you don't know something, say so rather than making things up."""


class TalkerAgent:
    async def execute(
        self, *, model: Chat, query: str, history: list[dict], **_
    ) -> TalkerResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})

        response = model.chat(messages)
        logger.info(f"Talker response ({response.input_tokens}+{response.output_tokens} tokens): {response.text}")

        return TalkerResult(
            response=response.text,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
