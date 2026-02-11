import os
import yaml
from pathlib import Path


class Config:
    def __init__(self, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        self._chat = _dict_to_list(data["chat"])
        self._embeddings = _dict_to_list(data["embeddings"])
        self._pricing = data["pricing"]
        self._mcp_services = data["mcp_services"]
        self._metrics = data["metrics"]

    @property
    def chat(self) -> list[dict]:
        return self._chat

    @property
    def embeddings(self) -> list[dict]:
        return self._embeddings

    @property
    def pricing(self) -> dict:
        return self._pricing

    @property
    def mcp_services(self) -> dict:
        return self._mcp_services

    @property
    def metrics(self) -> dict:
        return self._metrics

    def find_chat_model(self, model: str) -> dict | None:
        for cfg in self._chat:
            if cfg.get("model") == model:
                return cfg
        return None

    def find_embedding_model(self, model: str) -> dict | None:
        for cfg in self._embeddings:
            if cfg.get("model") == model:
                return cfg
        return None

    def list_chat_models(self) -> list[str]:
        return [cfg["model"] for cfg in self._chat if _is_available(cfg)]

    def list_embedding_models(self) -> list[str]:
        return [cfg["model"] for cfg in self._embeddings if _is_available(cfg)]


def _dict_to_list(cfg) -> list[dict]:
    if isinstance(cfg, dict):
        return [cfg]
    return cfg

def _is_available(cfg: dict) -> bool:
    api_key_env = cfg.get("api_key_env")
    return not api_key_env or os.environ.get(api_key_env) is not None
