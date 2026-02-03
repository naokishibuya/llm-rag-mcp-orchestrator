import logging
from ollama import Client
from tqdm import tqdm
from .base import BaseChat, BaseEmbeddings, LLMResponse, EmbeddingResponse


logger = logging.getLogger(__name__)


class OllamaChat(BaseChat):
    def __init__(
        self,
        model: str,
        temperature: float = 0.5,
        num_ctx: int = 16384,
        host: str | None = None,
        pull_if_missing: bool = True,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.client = Client(host=host)
        self.num_ctx = num_ctx

        if pull_if_missing:
            _ensure_model(self.client, model)

    def complete(self, prompt: str) -> LLMResponse:
        return self.chat([{"role": "user", "content": prompt}])

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "num_ctx": self.num_ctx},
        )

        msg = response.get("message", {})
        return LLMResponse(
            text=msg.get("content", ""),
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            model=self.model,
            raw=response,
        )


class OllamaEmbeddings(BaseEmbeddings):
    def __init__(
        self,
        model: str,
        host: str | None = None,
        pull_if_missing: bool = True,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.client = Client(host=host)

        if pull_if_missing:
            _ensure_model(self.client, model)

    def embed(self, texts: list[str]) -> EmbeddingResponse:
        response = self.client.embed(model=self.model, input=texts)

        return EmbeddingResponse(
            embeddings=response.get("embeddings", []),
            input_tokens=response.get("prompt_eval_count", 0),
            model=self.model,
        )


def _ensure_model(client: Client, model: str) -> None:
    try:
        client.show(model)
    except Exception:
        logger.info(f"Model {model} not found locally. Pulling...")
        _pull_model(client, model)


def _pull_model(client: Client, model: str) -> None:
    try:
        with tqdm(total=0, unit="B", unit_scale=True, desc=f"Pulling {model}") as pbar:
            current = 0
            for progress in client.pull(model, stream=True):
                if "total" in progress:
                    pbar.total = progress["total"]
                if "completed" in progress:
                    pbar.update(progress["completed"] - current)
                    current = progress["completed"]
        logger.info(f"Model {model} pulled successfully.")
    except Exception as e:
        logger.error(f"Failed to pull model {model}: {e}")
        raise
