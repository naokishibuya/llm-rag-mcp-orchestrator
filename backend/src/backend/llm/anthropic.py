import json
import logging
import os

import anthropic

from ..core import Message, Reply, Role, Tokens


logger = logging.getLogger(__name__)


_ROLE_MAP = {Role.USER: "user", Role.ASSISTANT: "assistant"}


class AnthropicChat:
    def __init__(self, model: str, api_key_env: str = "", params: dict = None, max_tool_rounds: int = 3):
        self.model = model
        self.params = params or {}
        self.max_tool_rounds = max_tool_rounds
        self._client = anthropic.Anthropic(api_key=_resolve_api_key(api_key_env))

    def ask(self, messages: list[Message], tools: dict[str, callable] | None = None) -> Reply:
        system, messages = _map_messages(messages)
        if tools:
            return self._with_tools(system, messages, tools)
        return self._plain(system, messages)

    def query(self, messages: list[Message], schema: dict) -> Reply:
        system, messages = _map_messages(messages)
        return self._plain(system, messages, schema)

    def _plain(self, system: str | None, messages: list[dict], schema: dict | None = None) -> Reply:
        kwargs: dict = {}
        if schema is not None:
            kwargs["tools"] = [{
                "name": "structured_output",
                "description": "Return structured data matching the schema.",
                "input_schema": schema,
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

        try:
            response = self._create(messages, system, **kwargs)
        except anthropic.APIError as e:
            logger.warning("Anthropic API error: %s", e.message)
            return Reply(text=f"[Anthropic error: {e.message}]", model=self.model, success=False)

        tokens = Tokens(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)

        if schema is not None:
            for block in response.content:
                if block.type == "tool_use" and block.name == "structured_output":
                    return Reply(text=json.dumps(block.input), model=self.model, tokens=tokens)

        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return Reply(text=text, model=self.model, tokens=tokens)

    def _with_tools(self, system: str | None, messages: list[dict], tools: dict[str, callable]) -> Reply:
        messages = list(messages)
        tool_schemas = [fn.tool_schema for fn in tools.values()]
        input_tokens = 0
        output_tokens = 0
        tools_used: list[str] = []

        for _ in range(self.max_tool_rounds):
            try:
                response = self._create(messages, system, tools=tool_schemas)
            except anthropic.APIError as e:
                logger.warning("Anthropic API error: %s", e.message)
                return Reply(text=f"[Anthropic error: {e.message}]", model=self.model, success=False)

            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                break

            messages.append({"role": _ROLE_MAP[Role.ASSISTANT], "content": response.content})
            tool_results = []
            for block in tool_use_blocks:
                fn = tools.get(block.name)
                if fn is None:
                    result = f"Unknown tool: {block.name}"
                else:
                    try:
                        result = str(fn(**block.input))
                    except Exception as e:
                        result = f"Error: {e}"
                    tools_used.append(f"{block.name}({block.input})")
                logger.info("Tool %s(%s) -> %s", block.name, block.input, result)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
            messages.append({"role": _ROLE_MAP[Role.USER], "content": tool_results})

        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return Reply(
            text=text,
            model=self.model,
            tokens=Tokens(input_tokens=input_tokens, output_tokens=output_tokens),
            tools_used=tools_used,
        )

    def _create(self, messages, system, **kwargs):
        params = {"model": self.model, "max_tokens": 8192, "messages": messages}
        params.update(self.params)
        if system:
            params["system"] = system
        params.update(kwargs)
        return self._client.messages.create(**params)


def _map_messages(messages: list[Message]) -> tuple[str | None, list[dict]]:
    system_parts = []
    mapped = []
    for msg in messages:
        role = msg["role"]
        if role == Role.SYSTEM:
            system_parts.append(msg["content"])
        else:
            mapped.append({"role": _ROLE_MAP[role], "content": msg["content"]})
    return "\n\n".join(system_parts) or None, mapped


def _resolve_api_key(api_key_env: str = "") -> str:
    env_var = api_key_env or "ANTHROPIC_API_KEY"
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(f"{env_var} is not set")
    return api_key
