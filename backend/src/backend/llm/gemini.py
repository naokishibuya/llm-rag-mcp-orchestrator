import logging
import os

from google import genai
from google.genai import types
from google.genai.errors import APIError

from ..core import Embedding, Message, Reply, Role, Tokens


logger = logging.getLogger(__name__)


_ROLE_MAP = {Role.USER: "user", Role.ASSISTANT: "model"}


class GeminiChat:
    def __init__(self, model: str, api_key_env: str = "", params: dict = None, max_tool_rounds: int = 3):
        self.model = model
        self.params = params or {}
        self.max_tool_rounds = max_tool_rounds
        self._client = genai.Client(api_key=_resolve_api_key(api_key_env))

    def ask(self, messages: list[Message], tools: dict[str, callable] | None = None) -> Reply:
        system, messages = _map_messages(messages)
        if tools:
            return self._with_tools(system, messages, tools)
        return self._plain(system, messages)

    def query(self, messages: list[Message], schema: dict) -> Reply:
        system, messages = _map_messages(messages)
        return self._plain(system, messages, schema)

    def _plain(self, system: str | None, messages: list[types.Content], schema: dict | None = None) -> Reply:
        config = types.GenerateContentConfig(
            **self.params,
            system_instruction=system,
        )
        if schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = schema
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=messages, config=config,
            )
        except APIError as e:
            logger.warning(f"Gemini API error: {e.code} {e.message}")
            return Reply(text=f"[Gemini error: {e.message}]", model=self.model, success=False)
        except Exception as e:
            logger.warning(f"Gemini unexpected error: {e}")
            return Reply(text=f"[Gemini error: {e}]", model=self.model, success=False)
        return self._to_reply(response)

    def _with_tools(self, system: str | None, messages: list[types.Content], tools: dict[str, callable]) -> Reply:
        config = types.GenerateContentConfig(
            **self.params,
            system_instruction=system,
            tools=list(tools.values()),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=self.max_tool_rounds,
            ),
        )
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=messages, config=config,
            )
        except APIError as e:
            logger.warning(f"Gemini API error: {e.code} {e.message}")
            return Reply(text=f"[Gemini error: {e.message}]", model=self.model, success=False)
        except Exception as e:
            logger.warning(f"Gemini unexpected error: {e}")
            return Reply(text=f"[Gemini error: {e}]", model=self.model, success=False)
        resp = self._to_reply(response)
        resp.tools_used = self._extract_tool_calls(response)
        return resp

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

    @staticmethod
    def _extract_tool_calls(response) -> list[str]:
        history = getattr(response, "automatic_function_calling_history", None)
        if not history:
            return []
        return [
            f"{part.function_call.name}({dict(part.function_call.args)})"
            for content in history
            for part in (content.parts or [])
            if getattr(part, "function_call", None)
        ]


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
