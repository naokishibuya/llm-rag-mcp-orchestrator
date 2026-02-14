import importlib

from ..config import Config
from ..core import Chat, Embeddings


class Registry:
    def __init__(self, config: Config):
        self._config = config
        self._chat_model_cache: dict[str, Chat] = {}
        self._embeddings_cache: dict[str, Embeddings] = {}

    def get_talk_model(self, model: str) -> Chat:
        if model in self._chat_model_cache:
            return self._chat_model_cache[model]

        cfg = self._config.find_talk_model(model)
        if not cfg:
            raise ValueError(f"Unknown talk model: {model}")

        llm = _load_class(cfg)
        self._chat_model_cache[model] = llm
        return llm

    def resolve_model(self, key: str) -> Chat:
        """Resolve a pipeline model by role key (e.g. 'mcp', 'rag', 'orchestrator')."""
        if key in self._chat_model_cache:
            return self._chat_model_cache[key]

        cfg = self._config.get_pipeline_model(key)
        llm = _load_class(cfg)
        self._chat_model_cache[key] = llm
        return llm

    def resolve_embeddings(self) -> Embeddings:
        """Resolve the fixed embedding model from config."""
        key = "_embedding"
        if key in self._embeddings_cache:
            return self._embeddings_cache[key]

        cfg = self._config.get_embedding_config()
        embeddings = _load_class(cfg)
        self._embeddings_cache[key] = embeddings
        return embeddings


_CONFIG_ONLY_KEYS = {"class"}


def _load_class(cfg: dict) -> type:
    class_path = cfg["class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**{k: v for k, v in cfg.items() if k not in _CONFIG_ONLY_KEYS})
