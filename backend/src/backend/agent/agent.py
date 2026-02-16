import logging
import re

from .types import Chat, Message, Reply, Role, UserContext


logger = logging.getLogger(__name__)


_COMMON_RULES = R"""
TOOL USAGE:
- Only call a tool when the user's question DIRECTLY asks for it.
- Do NOT call tools speculatively or for background context.
""".strip()


# Forbidden delimiters
_FORBIDDEN_INLINE_RE = re.compile(r"\\\(\s*(.*?)\s*\\\)", re.DOTALL)
_FORBIDDEN_BLOCK_RE = re.compile(r"\\\[\s*(.*?)\s*\\\]", re.DOTALL)


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
        user_content = f"{query}\n\n{context}" if context else query

        messages = [Message(role=Role.SYSTEM, content=system)]
        messages.extend(history)
        messages.append(Message(role=Role.USER, content=user_content))

        logger.info("Agent[%s] query=%r", self.name, query)
        reply = await model.ask(messages, tools=tools)

        # Hard guardrail: enforce supported math delimiters regardless of model behavior.
        reply.text = _normalize_math_delimiters(reply.text)

        logger.info("Agent[%s] %s", self.name, reply)
        return reply


def _normalize_math_delimiters(text: str) -> str:
    """Deterministically rewrite forbidden math delimiters to supported ones."""
    def to_block(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        if "\n" in body:
            return f"$$\n{body}\n$$"
        return f"$${body}$$"

    def to_inline(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        return f"${body}$"

    # Replace block first, then inline
    out = _FORBIDDEN_BLOCK_RE.sub(to_block, text)
    out = _FORBIDDEN_INLINE_RE.sub(to_inline, out)
    return out
