import json
import logging
import os

import openai
from pydantic import BaseModel

from .protocol import Message, Response, Role, TokenUsage


logger = logging.getLogger(__name__)


class OpenAIChat:
    def __init__(self, model: str, temperature: float = 0.5, api_key_env: str = "", max_tool_rounds: int = 3):
        self.model = model
        self.temperature = temperature
        self.max_tool_rounds = max_tool_rounds
        self._client = openai.OpenAI(api_key=_resolve_api_key(api_key_env))

    def chat(
        self, messages: list[Message], schema: type[BaseModel] | None = None, tools: dict[str, callable] | None = None
    ) -> Response:
        msgs = _build_messages(messages)
        use_tools = tools and schema is None

        kwargs: dict = {}
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_output", "strict": False, "schema": schema.model_json_schema()},
            }
        elif use_tools:
            kwargs["tools"] = [_to_openai_tool(fn) for fn in tools.values()]

        input_tokens = 0
        output_tokens = 0
        tools_used: list[str] = []
        max_rounds = self.max_tool_rounds if use_tools else 1

        for _ in range(max_rounds):
            try:
                response = self._client.chat.completions.create(
                    model=self.model, temperature=self.temperature, messages=msgs, **kwargs
                )
            except openai.APIError as e:
                logger.warning("OpenAI API error: %s", e.message)
                return Response(text=f"[OpenAI error: {e.message}]", tokens=TokenUsage(model=self.model))

            input_tokens += response.usage.prompt_tokens
            output_tokens += response.usage.completion_tokens

            choice = response.choices[0]

            if schema is not None:
                return Response(
                    text=choice.message.content or "",
                    tokens=TokenUsage(model=self.model, input_tokens=input_tokens, output_tokens=output_tokens),
                )

            tool_calls = choice.message.tool_calls
            if not tool_calls:
                break

            msgs.append(choice.message)
            for tc in tool_calls:
                fn = tools.get(tc.function.name)
                if fn is None:
                    result = f"Unknown tool: {tc.function.name}"
                else:
                    try:
                        args = json.loads(tc.function.arguments)
                        result = str(fn(**args))
                    except Exception as e:
                        result = f"Error: {e}"
                    tools_used.append(tc.function.name)
                logger.info("Tool %s(%s) -> %s", tc.function.name, tc.function.arguments, result)
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        text = choice.message.content or ""
        return Response(
            text=text,
            tokens=TokenUsage(model=self.model, input_tokens=input_tokens, output_tokens=output_tokens),
            tools_used=tools_used,
        )


def _build_messages(messages: list[Message]) -> list[dict]:
    return [{"role": msg["role"], "content": msg["content"]} for msg in messages]


def _to_openai_tool(fn) -> dict:
    schema = fn.tool_schema
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["input_schema"],
        },
    }


def _resolve_api_key(api_key_env: str = "") -> str:
    env_var = api_key_env or "OPENAI_API_KEY"
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(f"{env_var} is not set")
    return api_key
