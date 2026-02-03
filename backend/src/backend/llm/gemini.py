import os
from dotenv import load_dotenv
from google import genai
from .base import BaseChat, BaseEmbeddings, LLMResponse, EmbeddingResponse


def _get_client() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Add to .env or export it.")
    return genai.Client(api_key=api_key)


class GeminiChat(BaseChat):
    def __init__(self, model: str, temperature: float = 0.5, **kwargs):
        super().__init__(model, temperature, **kwargs)
        self._client = _get_client()

    def complete(self, prompt: str) -> LLMResponse:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": self.temperature},
        )
        usage = resp.usage_metadata

        return LLMResponse(
            text=resp.text,
            input_tokens=getattr(usage, "prompt_token_count", 0),
            output_tokens=getattr(usage, "candidates_token_count", 0),
            model=self.model,
            raw=resp,
        )

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        chat = self._client.chats.create(model=self.model)

        # Send all messages except the last one to build history
        for msg in messages[:-1]:
            if msg["role"] == "user":
                chat.send_message(message=msg["content"])

        # Send the last message and get the response
        last_msg = messages[-1]["content"] if messages else ""
        resp = chat.send_message(message=last_msg)
        usage = resp.usage_metadata

        return LLMResponse(
            text=resp.text,
            input_tokens=getattr(usage, "prompt_token_count", 0),
            output_tokens=getattr(usage, "candidates_token_count", 0),
            model=self.model,
            raw=resp,
        )


class GeminiEmbeddings(BaseEmbeddings):
    def __init__(self, model: str = "text-embedding-004", **kwargs):
        super().__init__(model, **kwargs)
        self._client = _get_client()

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        result = self._client.models.embed_content(
            model=self.model,
            contents=texts,
        )

        # Handle both single and batch embeddings
        embeddings = result.embeddings
        if embeddings and hasattr(embeddings[0], "values"):
            embedding_list = [e.values for e in embeddings]
        else:
            embedding_list = embeddings

        return EmbeddingResponse(
            embeddings=embedding_list,
            input_tokens=0,
            model=self.model,
        )
