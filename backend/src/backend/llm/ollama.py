import logging

import ollama
from tqdm import tqdm

from ..core import Embedding, Message, Reply, Role, Tokens


logger = logging.getLogger(__name__)


_ROLE_MAP = {Role.SYSTEM: "system", Role.USER: "user", Role.ASSISTANT: "assistant"}


class OllamaChat:
    def __init__(self, model: str, params: dict = None, max_tool_rounds: int = 3):
        self.model = model
        self.params = params or {}
        self.max_tool_rounds = max_tool_rounds
        _ensure_model(model)

    def ask(self, messages: list[Message], tools: dict[str, callable] | None = None) -> Reply:
        messages = _map_messages(messages)
        if tools:
            return self._with_tools(messages, tools)
        return self._plain(messages)

    def query(self, messages: list[Message], schema: dict) -> Reply:
        return self._plain(_map_messages(messages), schema)

    def _plain(self, messages: list[dict], schema: dict | None = None) -> Reply:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "options": self.params,
        }
        if schema is not None:
            kwargs["format"] = schema
        response = ollama.chat(**kwargs)
        return Reply(
            text=response.message.content or "",
            model=self.model,
            tokens=Tokens(
                input_tokens=response.get("prompt_eval_count", 0),
                output_tokens=response.get("eval_count", 0),
            ),
        )

    def _with_tools(self, messages: list[dict], tools: dict[str, callable]) -> Reply:
        messages = list(messages)
        input_tokens = 0
        output_tokens = 0
        tools_used: list[str] = []

        for _ in range(self.max_tool_rounds):
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options=self.params,
                tools=list(tools.values()),
            )
            input_tokens += response.get("prompt_eval_count", 0)
            output_tokens += response.get("eval_count", 0)

            if not response.message.tool_calls:
                break

            messages.append(response.message.model_copy())
            for tool_call in response.message.tool_calls:
                fn = tools.get(tool_call.function.name)
                if fn is None:
                    result = f"Unknown tool: {tool_call.function.name}"
                else:
                    try:
                        result = str(fn(**tool_call.function.arguments))
                    except Exception as e:
                        result = f"Error: {e}"
                    tools_used.append(f"{tool_call.function.name}({tool_call.function.arguments})")
                logger.info("Tool %s(%s) -> %s", tool_call.function.name, tool_call.function.arguments, result)
                messages.append({"role": "tool", "content": result})

        return Reply(
            text=response.message.content or "",
            model=self.model,
            tokens=Tokens(input_tokens=input_tokens, output_tokens=output_tokens),
            tools_used=tools_used,
        )


class OllamaEmbeddings:
    def __init__(self, model: str):
        self.model = model
        _ensure_model(model)

    def embed(self, texts: list[str] | str) -> Embedding | list[Embedding]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = [ollama.embed(model=self.model, input=t)["embeddings"][0] for t in texts]
        return embeddings[0] if len(embeddings) == 1 else embeddings


def _map_messages(messages: list[Message]) -> list[dict]:
    mapped = [{"role": _ROLE_MAP[m["role"]], "content": m["content"]} for m in messages]
    logger.debug("Ollama messages: %s", [{"role": m["role"], "content": m["content"][:100]} for m in mapped])
    return mapped


def _ensure_model(model: str):
    try:
        ollama.show(model)
    except ollama.ResponseError:
        _pull_model(model)


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
