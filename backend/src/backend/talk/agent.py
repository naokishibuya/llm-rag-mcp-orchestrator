import logging

from ..core import Chat, Message, Reply, Role
from ..core.agent import UserContext

from .tools import TOOLS


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful, friendly assistant.
Engage in natural conversation, answer questions, and provide assistance.

Be concise but informative. If you don't know something, say so rather than making things up.

CRITICAL MATH RULES:
- Use $...$ for inline math.
- Use $$...$$ for block math.
- NEVER use \( \) or \[ \]. If you use them, the math will be invisible.
- Format all equations, fractions, and symbols using these delimiters."""


class TalkAgent:
    async def act(
        self, *, model: Chat, query: str, history: list[dict],
        context: UserContext, **_,
    ) -> Reply:
        system = SYSTEM_PROMPT
        if context:
            system += f"\n\n{context}"

        messages = [Message(role=Role.SYSTEM, content=system)]
        messages.extend(history)
        messages.append(Message(role=Role.USER, content=query))

        reply = model.ask(messages, tools=TOOLS)
        logger.info(f"Talker {reply}")
        return reply
