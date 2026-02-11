import importlib

from ..config import Config
from .protocol import Chat, Embeddings


class Registry:
    def __init__(self, config: Config):
        self._config = config
        self._chat_model_cache: dict[str, Chat] = {}
        self._embeddings_cache: dict[str, Embeddings] = {}

    def get_chat_model(self, model: str) -> Chat:
        if model in self._chat_model_cache:
            return self._chat_model_cache[model]

        cfg = self._config.find_chat_model(model)
        if not cfg:
            raise ValueError(f"Unknown chat model: {model}")

        llm = _load_class(cfg)
        self._chat_model_cache[model] = llm
        return llm

    def get_embeddings(self, model: str) -> Embeddings:
        if model in self._embeddings_cache:
            return self._embeddings_cache[model]

        cfg = self._config.find_embedding_model(model)
        if not cfg:
            raise ValueError(f"Unknown embedding model: {model}")

        embeddings = _load_class(cfg)
        self._embeddings_cache[model] = embeddings
        return embeddings


_CONFIG_ONLY_KEYS = {"class"}


def _load_class(cfg: dict) -> type:
    class_path = cfg["class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**{k: v for k, v in cfg.items() if k not in _CONFIG_ONLY_KEYS})
