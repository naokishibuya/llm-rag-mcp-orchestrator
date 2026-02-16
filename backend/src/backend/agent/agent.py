import logging

from .types import Chat, Message, Reply, Role, UserContext


logger = logging.getLogger(__name__)


_COMMON_RULES = """
TOOL USAGE:
- Only call a tool when the user's question DIRECTLY asks for it.
- Do NOT call tools speculatively or for background context.

MATH & CURRENCY RULES:
- Use $...$ ONLY for LaTeX math (equations, variables, formulas).
- Use $$...$$ ONLY for block math.
- NEVER use \\( \\) or \\[ \\].
""".strip()


class Agent:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

    async def act(
        self,
        *,
        model: Chat,
        query: str,
        history: list[Message],
        tools: dict[str, callable] | None = None,
        context: UserContext | None = None,
    ) -> Reply:
        system = f"{self.system_prompt}\n\n{_COMMON_RULES}"
        if context:
            system += f"\n\n{context}"

        messages = [Message(role=Role.SYSTEM, content=system)]
        messages.extend(history)
        messages.append(Message(role=Role.USER, content=query))

        logger.info("Agent[%s] query=%r", self.name, query)
        reply = await model.ask(messages, tools=tools)
        logger.info("Agent[%s] %s", self.name, reply)
        return reply
