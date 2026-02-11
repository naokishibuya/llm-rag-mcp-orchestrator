import os

from google import genai
from google.genai import types

from .protocol import Response


class GeminiChat:
    def __init__(self, model: str, temperature: float = 0.5):
        self.model = model
        self.temperature = temperature
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required")
        self._client = genai.Client(api_key=api_key)

    def chat(self, messages: list[dict[str, str]], schema: type | None = None) -> Response:
        contents, system_instruction = self._build_contents(messages)
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_instruction,
        )
        if schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = schema
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
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
                contents.append(types.Content(role="user", parts=[types.Part.from_text(content)]))
            else:
                contents.append(types.Content(role="model", parts=[types.Part.from_text(content)]))
        return contents, system_instruction

    def _to_response(self, response) -> Response:
        return Response(
            text=response.text or "",
            input_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
            output_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
            model=self.model,
        )


class GeminiEmbeddings:
    def __init__(self, model: str = "text-embedding-004"):
        self.model = model
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required")
        self._client = genai.Client(api_key=api_key)

    def embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        if isinstance(texts, str):
            texts = [texts]
        result = self._client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        embeddings = [e.values for e in result.embeddings]
        return embeddings[0] if len(embeddings) == 1 else embeddings
