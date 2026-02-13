import logging
from dataclasses import dataclass, field

from ..llm import Chat, Message, Role
from .tools import TOOLS


logger = logging.getLogger(__name__)


@dataclass
class TalkResult:
    response: str
    model: str = ""
    success: bool = True
    input_tokens: int = 0
    output_tokens: int = 0
    tools_used: list[str] = field(default_factory=list)


SYSTEM_PROMPT = """You are a helpful, friendly assistant.
Engage in natural conversation, answer questions, and provide assistance.

Be concise but informative. If you don't know something, say so rather than making things up.

CRITICAL MATH RULES:
- Use $...$ for inline math.
- Use $$...$$ for block math.
- NEVER use \( \) or \[ \]. If you use them, the math will be invisible.
- Format all equations, fractions, and symbols using these delimiters."""


class TalkAgent:
    async def execute(
        self, *, model: Chat, query: str, history: list[dict], **_
    ) -> TalkResult:
        messages = [Message(role=Role.SYSTEM, content=SYSTEM_PROMPT)]
        messages.extend(history)
        messages.append(Message(role=Role.USER, content=query))

        response = model.chat(messages, tools=TOOLS)
        logger.info(f"Talker [{response.tokens.model}] tokens=[{response.tokens.input_tokens}/{response.tokens.output_tokens}]: {response.text}")

        return TalkResult(
            response=response.text,
            model=response.tokens.model,
            input_tokens=response.tokens.input_tokens,
            output_tokens=response.tokens.output_tokens,
            tools_used=response.tools_used,
        )
