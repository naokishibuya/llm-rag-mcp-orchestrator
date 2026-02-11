import logging
import os

from google import genai
from google.genai import types
from google.genai.errors import APIError

from .protocol import Response


logger = logging.getLogger(__name__)


class GeminiChat:
    def __init__(self, model: str, temperature: float = 0.5, api_key_env: str = ""):
        self.model = model
        self.temperature = temperature
        self._client = genai.Client(api_key=_resolve_api_key(api_key_env))

    def chat(self, messages: list[dict[str, str]], schema: type | None = None) -> Response:
        contents, system_instruction = self._build_contents(messages)
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_instruction,
        )
        if schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = schema.model_json_schema()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except APIError as e:
            logger.warning(f"Gemini API error: {e.code} {e.message}")
            return Response(text=f"[Gemini error: {e.message}]", model=self.model)
        except Exception as e:
            logger.warning(f"Gemini unexpected error: {e}")
            return Response(text=f"[Gemini error: {e}]", model=self.model)
        return self._to_response(response)

    def _build_contents(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[types.Content], str | None]:
        contents = []
        system_instruction = None
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            else:
                contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
        return contents, system_instruction

    def _to_response(self, response) -> Response:
        return Response(
            text=response.text or "",
            input_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
            output_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
            model=self.model,
        )


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

    def embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        if isinstance(texts, str):
            texts = [texts]
        result = self._client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        embeddings = [e.values for e in result.embeddings]
        return embeddings[0] if len(embeddings) == 1 else embeddings
