import logging

import ollama
from tqdm import tqdm

from .protocol import Response

logger = logging.getLogger(__name__)


class OllamaChat:
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
        try:
            ollama.show(model)
        except ollama.ResponseError:
            _pull_model(model)

    def chat(self, messages: list[dict[str, str]], schema: type | None = None) -> Response:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": self.temperature},
        }
        if schema is not None:
            kwargs["format"] = schema.model_json_schema()
        response = ollama.chat(**kwargs)
        return Response(
            text=response["message"]["content"],
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            model=self.model,
        )


class OllamaEmbeddings:
    def __init__(self, model: str):
        self.model = model
        try:
            ollama.show(model)
        except ollama.ResponseError:
            _pull_model(model)

    def embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = [ollama.embed(model=self.model, input=t)["embeddings"][0] for t in texts]
        return embeddings[0] if len(embeddings) == 1 else embeddings


def _pull_model(model: str):
    """Pull an Ollama model with tqdm progress bar."""
    logger.info("Pulling model %s ...", model)
    with tqdm(total=0, unit="B", unit_scale=True, desc=model) as bar:
        current = 0
        for progress in ollama.pull(model, stream=True):
            if "total" in progress:
                bar.total = progress["total"]
            if "completed" in progress:
                bar.update(progress["completed"] - current)
                current = progress["completed"]
    logger.info("Model %s ready", model)
