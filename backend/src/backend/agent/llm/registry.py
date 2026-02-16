import importlib

from ..types import Chat, Embeddings


class Registry:
    def __init__(self, llm_configs: list[dict], embedding_config: dict):
        self._llm_configs = llm_configs
        self._embedding_config = embedding_config
        self._chat_model_cache: dict[str, Chat] = {}
        self._embeddings_cache: dict[str, Embeddings] = {}

    def get_talk_model(self, model: str) -> Chat:
        if model in self._chat_model_cache:
            return self._chat_model_cache[model]

        cfg = self._find_llm_config(model)
        if not cfg:
            raise ValueError(f"Unknown model: {model}")

        llm = _load_class(cfg)
        self._chat_model_cache[model] = llm
        return llm

    def resolve_embeddings(self) -> Embeddings:
        key = "_embedding"
        if key in self._embeddings_cache:
            return self._embeddings_cache[key]

        embeddings = _load_class(self._embedding_config)
        self._embeddings_cache[key] = embeddings
        return embeddings

    def _find_llm_config(self, model: str) -> dict | None:
        for cfg in self._llm_configs:
            if cfg.get("model") == model:
                return cfg
        return None


_CONFIG_ONLY_KEYS = {"class"}


def _load_class(cfg: dict) -> type:
    class_path = cfg["class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**{k: v for k, v in cfg.items() if k not in _CONFIG_ONLY_KEYS})
