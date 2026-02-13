import logging

import ollama
from pydantic import BaseModel
from tqdm import tqdm

from .protocol import Message, Response, TokenUsage

logger = logging.getLogger(__name__)


class OllamaChat:
    def __init__(self, model: str, temperature: float, max_tool_rounds: int = 3):
        self.model = model
        self.temperature = temperature
        self.max_tool_rounds = max_tool_rounds
        try:
            ollama.show(model)
        except ollama.ResponseError:
            _pull_model(model)

    def chat(self, messages: list[Message], schema: type[BaseModel] | None = None, tools: dict[str, callable] | None = None) -> Response:
        use_tools = tools and schema is None
        max_rounds = self.max_tool_rounds if use_tools else 1

        kwargs = {
            "model": self.model,
            "messages": list(messages),  # Make a copy to avoid modifying the original in tool calls
            "options": {"temperature": self.temperature},
        }
        if schema is not None:
            kwargs["format"] = schema.model_json_schema()
        if use_tools:
            kwargs["tools"] = list(tools.values())

        input_tokens = 0
        output_tokens = 0
        tools_used: list[str] = []

        for _ in range(max_rounds):
            response = ollama.chat(**kwargs)
            input_tokens += response.get("prompt_eval_count", 0)
            output_tokens += response.get("eval_count", 0)

            if not response.message.tool_calls:
                break

            kwargs["messages"].append(response.message.model_copy())
            for tool_call in response.message.tool_calls:
                fn = tools.get(tool_call.function.name)
                if fn is None:
                    result = f"Unknown tool: {tool_call.function.name}"
                else:
                    try:
                        result = str(fn(**tool_call.function.arguments))
                    except Exception as e:
                        result = f"Error: {e}"
                    tools_used.append(tool_call.function.name)
                logger.info("Tool %s(%s) -> %s", tool_call.function.name, tool_call.function.arguments, result)
                kwargs["messages"].append({"role": "tool", "content": result})

        return Response(
            text=response.message.content or "",
            tokens=TokenUsage(
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            tools_used=tools_used,
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
