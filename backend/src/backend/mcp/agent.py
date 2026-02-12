import json
import logging
from dataclasses import dataclass
from math import log

from ..llm import Chat
from .client import MCPService
from .handler import MCPHandler


logger = logging.getLogger(__name__)


FORMAT_PROMPT = """Format the following raw tool data into a clear, concise response for the user.
Only present the data below — do not add disclaimers about missing data or suggest other sources.

Tool: {tool_name}
Raw data:
{raw_data}

{format_instruction}"""

DEFAULT_FORMAT = "Respond with a well-formatted answer. Be concise."


@dataclass
class MCPResult:
    response: str
    model: str = ""
    success: bool = True
    input_tokens: int = 0
    output_tokens: int = 0


class MCPAgent:
    def __init__(self, handler: MCPHandler, service: MCPService, model: Chat):
        self._handler = handler
        self._model = model
        self._format_hint = service.format_hint or DEFAULT_FORMAT

    async def execute(self, *, query: str, params: dict = None, **_) -> MCPResult:
        data = await self._handler.handle(**(params or {}))

        if isinstance(data, dict) and data.get("unavailable"):
            name = self._handler.service.name
            return MCPResult(
                response=f"The **{name}** service is currently unavailable. You can:\n"
                f"- Try again in a moment\n"
                f"- Ask me something else in the meantime",
                success=False,
            )

        if isinstance(data, dict) and "error" in data:
            logger.error(f"MCPAgent received error from tool {self._handler.service.name}: {data['error']}")
            return MCPResult(response=str(data["error"]), success=False)

        raw_data = json.dumps(data, indent=2, ensure_ascii=False) if not isinstance(data, str) else data

        try:
            prompt = FORMAT_PROMPT.format(
                query=query,
                tool_name=self._handler.service.name,
                raw_data=raw_data,
                format_instruction=self._format_hint,
            )
            messages = [{"role": "user", "content": prompt}]
            response = self._model.chat(messages)
            logger.info(f"MCPAgent response for {self._handler.service.name}: {response.text[:100]}...")
            return MCPResult(
                response=response.text,
                model=self._model.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        except Exception as e:
            logger.error(f"LLM formatting failed for {self._handler.service.name}: {e}")
            return MCPResult(response=raw_data)
