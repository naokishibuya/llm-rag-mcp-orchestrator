import json
import logging
import os

import openai

from ..core import Message, Reply, Role, Tokens


logger = logging.getLogger(__name__)


_ROLE_MAP = {Role.SYSTEM: "system", Role.USER: "user", Role.ASSISTANT: "assistant"}


class OpenAIChat:
    def __init__(self, model: str, api_key_env: str = "", params: dict = None, max_tool_rounds: int = 3):
        self.model = model
        self.params = params or {}
        self.max_tool_rounds = max_tool_rounds
        self._client = openai.OpenAI(api_key=_resolve_api_key(api_key_env))

    def ask(self, messages: list[Message], tools: dict[str, callable] | None = None) -> Reply:
        messages = _map_messages(messages)
        if tools:
            return self._with_tools(messages, tools)
        return self._plain(messages)

    def query(self, messages: list[Message], schema: dict) -> Reply:
        return self._plain(_map_messages(messages), schema)

    def _plain(self, messages: list[dict], schema: dict | None = None) -> Reply:
        kwargs: dict = {}
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_output", "strict": False, "schema": schema},
            }
        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=messages, **self.params, **kwargs
            )
        except openai.APIError as e:
            logger.warning("OpenAI API error: %s", e.message)
            return Reply(text=f"[OpenAI error: {e.message}]", model=self.model, success=False)

        return Reply(
            text=response.choices[0].message.content or "",
            model=self.model,
            tokens=Tokens(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def _with_tools(self, messages: list[dict], tools: dict[str, callable]) -> Reply:
        msgs = list(messages)
        openai_tools = [_to_openai_tool(fn) for fn in tools.values()]
        input_tokens = 0
        output_tokens = 0
        tools_used: list[str] = []

        for _ in range(self.max_tool_rounds):
            try:
                response = self._client.chat.completions.create(
                    model=self.model, messages=msgs, tools=openai_tools, **self.params,
                )
            except openai.APIError as e:
                logger.warning("OpenAI API error: %s", e.message)
                return Reply(text=f"[OpenAI error: {e.message}]", model=self.model, success=False)

            input_tokens += response.usage.prompt_tokens
            output_tokens += response.usage.completion_tokens

            choice = response.choices[0]
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
                    tools_used.append(f"{tc.function.name}({tc.function.arguments})")
                logger.info("Tool %s(%s) -> %s", tc.function.name, tc.function.arguments, result)
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        text = choice.message.content or ""
        return Reply(
            text=text,
            model=self.model,
            tokens=Tokens(input_tokens=input_tokens, output_tokens=output_tokens),
            tools_used=tools_used,
        )


def _map_messages(messages: list[Message]) -> list[dict]:
    return [{"role": _ROLE_MAP[msg["role"]], "content": msg["content"]} for msg in messages]


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
