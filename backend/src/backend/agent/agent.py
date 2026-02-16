import logging
import re

from .types import Chat, Message, Reply, Role, UserContext


logger = logging.getLogger(__name__)


_COMMON_RULES = R"""
TOOL USAGE:
- Only call a tool when the user's question DIRECTLY asks for it.
- Do NOT call tools speculatively or for background context.

MATH FORMATTING:
- Use \( ... \) for inline math and \[ ... \] for display math.
- NEVER use $...$ or $$...$$ as math delimiters.
""".strip()

_BLOCK_RE = re.compile(r"\\\[([\s\S]*?)\\\]")
_INLINE_RE = re.compile(r"\\\(([\s\S]*?)\\\)")


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
        reply.text = _normalize_math(reply.text)

        logger.info("Agent[%s] %s", self.name, reply)
        return reply


def _normalize_math(text: str) -> str:
    """Convert LaTeX delimiters to $/$$ and escape bare $ signs."""
    # 1. Extract math blocks into placeholders
    inlines: list[str] = []
    blocks: list[str] = []

    def _save_block(m: re.Match[str]) -> str:
        blocks.append(m.group(1).strip())
        return f"%%BLOCK{len(blocks) - 1}%%"

    def _save_inline(m: re.Match[str]) -> str:
        inlines.append(m.group(1).strip())
        return f"%%INLINE{len(inlines) - 1}%%"

    text = _BLOCK_RE.sub(_save_block, text)
    text = _INLINE_RE.sub(_save_inline, text)

    # 2. Escape bare $ so currency signs are never treated as math
    text = re.sub(r"(?<!\\)\$", r"\$", text)

    # 3. Restore math with $/$$ delimiters
    def _restore_block(m: re.Match[str]) -> str:
        body = blocks[int(m.group(1))]
        return f"$$\n{body}\n$$" if "\n" in body else f"$${body}$$"

    text = re.sub(r"%%BLOCK(\d+)%%", _restore_block, text)
    text = re.sub(r"%%INLINE(\d+)%%", lambda m: f"${inlines[int(m.group(1))]}$", text)

    return text
