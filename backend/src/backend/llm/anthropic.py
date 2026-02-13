import json
import logging
import os

import anthropic
from pydantic import BaseModel

from .protocol import Message, Response, Role, TokenUsage


logger = logging.getLogger(__name__)


class AnthropicChat:
    def __init__(self, model: str, temperature: float = 0.5, api_key_env: str = "", max_tool_rounds: int = 3):
        self.model = model
        self.temperature = temperature
        self.max_tool_rounds = max_tool_rounds
        self._client = anthropic.Anthropic(api_key=_resolve_api_key(api_key_env))

    def chat(
        self, messages: list[Message], schema: type[BaseModel] | None = None, tools: dict[str, callable] | None = None
    ) -> Response:
        messages, system = _split_messages(messages)
        use_tools = tools and schema is None

        kwargs = {}
        if schema is not None:
            tool_def = {
                "name": "structured_output",
                "description": "Return structured data matching the schema.",
                "input_schema": schema.model_json_schema(),
            }
            kwargs["tools"] = [tool_def]
            kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}
        elif use_tools:
            kwargs["tools"] = [fn.tool_schema for fn in tools.values()]  # fn must be decorated with @tool

        input_tokens = 0
        output_tokens = 0
        tools_used: list[str] = []
        max_rounds = self.max_tool_rounds if use_tools else 1

        for _ in range(max_rounds):
            try:
                response = self._create(messages, system, **kwargs)
            except anthropic.APIError as e:
                logger.warning("Anthropic API error: %s", e.message)
                return Response(text=f"[Anthropic error: {e.message}]", tokens=TokenUsage(model=self.model))

            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens

            if schema is not None:
                for block in response.content:
                    if block.type == "tool_use" and block.name == "structured_output":
                        return Response(
                            text=json.dumps(block.input),
                            tokens=TokenUsage(model=self.model, input_tokens=input_tokens, output_tokens=output_tokens),
                        )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                break

            messages.append({"role": "assistant", "content": response.content})
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
                    tools_used.append(block.name)
                logger.info("Tool %s(%s) -> %s", block.name, block.input, result)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
            messages.append({"role": "user", "content": tool_results})

        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return Response(
            text=text,
            tokens=TokenUsage(model=self.model, input_tokens=input_tokens, output_tokens=output_tokens),
            tools_used=tools_used,
        )

    def _create(self, messages, system, **kwargs):
        params = {"model": self.model, "max_tokens": 8192, "temperature": self.temperature, "messages": messages}
        if system:
            params["system"] = system
        params.update(kwargs)
        return self._client.messages.create(**params)


def _split_messages(messages: list[Message]) -> tuple[list[dict], str | None]:
    result = []
    system = None
    for msg in messages:
        if msg["role"] == Role.SYSTEM:
            system = msg["content"]
        else:
            result.append({"role": msg["role"], "content": msg["content"]})
    return result, system


def _resolve_api_key(api_key_env: str = "") -> str:
    env_var = api_key_env or "ANTHROPIC_API_KEY"
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(f"{env_var} is not set")
    return api_key
