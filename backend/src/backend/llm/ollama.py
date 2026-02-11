import ollama

from .protocol import Response


class OllamaChat:
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
        try:
            ollama.show(model)
        except ollama.ResponseError:
            ollama.pull(model)

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
            ollama.pull(model)

    def embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = [ollama.embed(model=self.model, input=t)["embeddings"][0] for t in texts]
        return embeddings[0] if len(embeddings) == 1 else embeddings
