import json
import logging

from ..core import Chat, Message, Reply, Role, UserContext
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


class MCPAgent:
    def __init__(self, handler: MCPHandler, service: MCPService, model: Chat):
        self._handler = handler
        self._model = model
        self._format_hint = service.format_hint or DEFAULT_FORMAT

    async def act(self, *, query: str, context: UserContext, params: dict = None, **_) -> Reply:
        try:
            raw_data = await self._invoke(params or {})
        except RuntimeError as e:
            return Reply(text=str(e), success=False)

        try:
            prompt = FORMAT_PROMPT.format(
                query=query,
                tool_name=self._handler.service.name,
                raw_data=raw_data,
                format_instruction=self._format_hint,
            )
            if context:
                prompt += f"\n\n{context}"
            messages = [Message(role=Role.USER, content=prompt)]
            reply = self._model.ask(messages)
            logger.info(f"MCPAgent reply for {self._handler.service.name}: {reply.text[:100]}...")
            return reply
        except Exception as e:
            logger.error(f"LLM formatting failed for {self._handler.service.name}: {e}")
            return Reply(text=raw_data)

    async def _invoke(self, params: dict) -> str:
        data = await self._handler.handle(**params)

        if isinstance(data, dict) and data.get("unavailable"):
            name = self._handler.service.name
            raise RuntimeError(
                f"The **{name}** service is currently unavailable. You can:\n"
                f"- Try again in a moment\n"
                f"- Ask me something else in the meantime"
            )

        if isinstance(data, dict) and "error" in data:
            name = self._handler.service.name
            logger.error(f"MCPAgent received error from tool {name}: {data['error']}")
            raise RuntimeError(str(data["error"]))

        return json.dumps(data, indent=2, ensure_ascii=False) if not isinstance(data, str) else data
