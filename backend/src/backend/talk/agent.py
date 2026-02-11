import logging
from dataclasses import dataclass, field

from ..llm import Chat
from .tools import TOOLS


logger = logging.getLogger(__name__)


@dataclass
class TalkResult:
    response: str
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
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        
        # We append a direct instruction to the query to reinforce the math rules,
        # as models often 'forget' system instructions after multiple tool rounds.
        prompt = f"{query}\n\n(IMPORTANT: Use $...$ for all math. NEVER use \\( or \\[.)"
        messages.append({"role": "user", "content": prompt})

        response = model.chat(messages, tools=TOOLS)
        logger.info(f"Talker response ({response.input_tokens}+{response.output_tokens} tokens): {response.text}")

        return TalkResult(
            response=response.text,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            tools_used=response.tools_used,
        )
