import asyncio
import logging
import os

from google import genai
from google.genai import types
from google.genai.errors import APIError

from ..types import Embedding, Message, Reply, Role, Tokens


logger = logging.getLogger(__name__)


_ROLE_MAP = {Role.USER: "user", Role.ASSISTANT: "model"}


class GeminiChat:
    def __init__(self, model: str, api_key_env: str = "", params: dict = None, max_tool_rounds: int = 3):
        self.model = model
        self.params = params or {}
        self.max_tool_rounds = max_tool_rounds
        self._client = genai.Client(api_key=_resolve_api_key(api_key_env))

    async def ask(self, messages: list[Message], tools: dict[str, callable] | None = None) -> Reply:
        system, messages = _map_messages(messages)
        if tools:
            return await self._with_tools(system, messages, tools)
        return await self._plain(system, messages)

    async def query(self, messages: list[Message], schema: dict) -> Reply:
        system, messages = _map_messages(messages)
        return await self._plain(system, messages, schema)

    async def _plain(self, system: str | None, messages: list[types.Content], schema: dict | None = None) -> Reply:
        config = types.GenerateContentConfig(
            **self.params,
            system_instruction=system,
        )
        if schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = schema
        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self.model, contents=messages, config=config,
            )
        except APIError as e:
            logger.warning(f"Gemini API error: {e.code} {e.message}")
            return Reply(text=f"[Gemini error: {e.message}]", model=self.model, success=False)
        except Exception as e:
            logger.warning(f"Gemini unexpected error: {e}")
            return Reply(text=f"[Gemini error: {e}]", model=self.model, success=False)
        return self._to_reply(response)

    async def _with_tools(self, system: str | None, messages: list[types.Content], tools: dict[str, callable]) -> Reply:
        declarations = [_to_gemini_declaration(fn) for fn in tools.values()]
        config = types.GenerateContentConfig(
            **self.params,
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)],
        )

        contents = list(messages)
        input_tokens = 0
        output_tokens = 0
        tools_used: list[str] = []

        for _ in range(self.max_tool_rounds):
            try:
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self.model, contents=contents, config=config,
                )
            except APIError as e:
                logger.warning(f"Gemini API error: {e.code} {e.message}")
                return Reply(text=f"[Gemini error: {e.message}]", model=self.model, success=False)
            except Exception as e:
                logger.warning(f"Gemini unexpected error: {e}")
                return Reply(text=f"[Gemini error: {e}]", model=self.model, success=False)

            usage = getattr(response, "usage_metadata", None)
            input_tokens += usage.prompt_token_count if usage else 0
            output_tokens += usage.candidates_token_count if usage else 0

            # Check for function calls
            parts = response.candidates[0].content.parts if response.candidates else []
            function_calls = [p for p in parts if p.function_call and p.function_call.name]
            if not function_calls:
                break

            # Add model response to contents
            contents.append(response.candidates[0].content)

            # Execute tools and collect responses
            response_parts = []
            for part in function_calls:
                fc = part.function_call
                fn = tools.get(fc.name)
                if fn is None:
                    result = {"error": f"Unknown tool: {fc.name}"}
                else:
                    try:
                        args = dict(fc.args) if fc.args else {}
                        if asyncio.iscoroutinefunction(fn):
                            result_val = await fn(**args)
                        else:
                            result_val = fn(**args)
                        result = {"result": str(result_val)}
                        tools_used.append(f"{fc.name}({args})")
                    except Exception as e:
                        result = {"error": str(e)}
                logger.info("Tool %s(%s) -> %s", fc.name, fc.args, result)
                response_parts.append(types.Part.from_function_response(name=fc.name, response=result))

            contents.append(types.Content(role="user", parts=response_parts))

        text = response.text or "" if response.candidates else ""
        return Reply(
            text=text,
            model=self.model,
            tokens=Tokens(input_tokens=input_tokens, output_tokens=output_tokens),
            tools_used=tools_used,
        )

    def _to_reply(self, response) -> Reply:
        usage_metadata = getattr(response, "usage_metadata", None)
        return Reply(
            text=response.text or "",
            model=self.model,
            tokens=Tokens(
                input_tokens=usage_metadata.prompt_token_count if usage_metadata else 0,
                output_tokens=usage_metadata.candidates_token_count if usage_metadata else 0,
            ),
        )


def _to_gemini_declaration(fn) -> types.FunctionDeclaration:
    schema = fn.tool_schema
    input_schema = schema["input_schema"]
    # Convert JSON Schema to Gemini's Schema format
    properties = {}
    for name, prop in input_schema.get("properties", {}).items():
        properties[name] = types.Schema(
            type=_json_type_to_gemini(prop.get("type", "string")),
            description=prop.get("description", ""),
        )
    return types.FunctionDeclaration(
        name=schema["name"],
        description=schema["description"],
        parameters=types.Schema(
            type="OBJECT",
            properties=properties,
            required=input_schema.get("required", []),
        ) if properties else None,
    )


def _json_type_to_gemini(json_type: str) -> str:
    return {
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
    }.get(json_type, "STRING")


def _map_messages(messages: list[Message]) -> tuple[str | None, list[types.Content]]:
    system_parts = []
    mapped = []
    for msg in messages:
        role = msg["role"]
        if role == Role.SYSTEM:
            system_parts.append(msg["content"])
        else:
            mapped.append(types.Content(role=_ROLE_MAP[role], parts=[types.Part(text=msg["content"])]))
    return "\n\n".join(system_parts) or None, mapped


def _resolve_api_key(api_key_env: str = "") -> str:
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} is not set")
        return api_key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    return api_key


class GeminiEmbeddings:
    def __init__(self, model: str = "text-embedding-004", api_key_env: str = ""):
        self.model = model
        self._client = genai.Client(api_key=_resolve_api_key(api_key_env))

    def embed(self, texts: list[str] | str) -> Embedding | list[Embedding]:
        if isinstance(texts, str):
            texts = [texts]
        result = self._client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        embeddings = [e.values for e in result.embeddings]
        return embeddings[0] if len(embeddings) == 1 else embeddings
