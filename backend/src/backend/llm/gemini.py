import os
from dotenv import load_dotenv
import google.generativeai as genai
from .base import BaseChat, BaseEmbeddings, LLMResponse, EmbeddingResponse


def _configure_genai():
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Add to .env or export it.")
    genai.configure(api_key=api_key)


class GeminiChat(BaseChat):
    def __init__(self, model: str, temperature: float = 0.5, **kwargs):
        super().__init__(model, temperature, **kwargs)
        _configure_genai()
        self._model = genai.GenerativeModel(
            model_name=model,
            generation_config={"temperature": temperature},
        )

    def complete(self, prompt: str) -> LLMResponse:
        resp = self._model.generate_content(prompt)
        usage = resp.usage_metadata

        return LLMResponse(
            text=resp.text,
            input_tokens=getattr(usage, "prompt_token_count", 0),
            output_tokens=getattr(usage, "candidates_token_count", 0),
            model=self.model,
            raw=resp,
        )

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        chat = self._model.start_chat(history=[])
        for msg in messages[:-1]:
            if msg["role"] == "user":
                chat.send_message(msg["content"])

        last_msg = messages[-1]["content"] if messages else ""
        resp = chat.send_message(last_msg)
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
        _configure_genai()

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        result = genai.embed_content(model=f"models/{self.model}", content=texts)

        return EmbeddingResponse(
            embeddings=result["embedding"] if isinstance(result["embedding"][0], list) else [result["embedding"]],
            input_tokens=0,
            model=self.model,
        )
