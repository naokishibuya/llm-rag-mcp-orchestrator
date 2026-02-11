import os
import yaml
from pathlib import Path


class Config:
    def __init__(self, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        self._talk = _dict_to_list(data["talk"])
        self._embeddings = _dict_to_list(data["embeddings"])
        self._pipeline_models = {
            k: data[k] for k in ("mcp", "rag", "orchestrator")
        }
        self._pricing = data["pricing"]
        self._mcp_services = data["mcp_services"]
        self._metrics = data["metrics"]

    # --- Talk (user-facing) models ---

    @property
    def talk(self) -> list[dict]:
        return self._talk

    def find_talk_model(self, model: str) -> dict | None:
        for cfg in self._talk:
            if cfg.get("model") == model:
                return cfg
        return None

    def list_talk_models(self) -> list[str]:
        return [cfg["model"] for cfg in self._talk if _is_available(cfg)]

    # --- Pipeline models ---

    def get_pipeline_model(self, key: str) -> dict:
        return self._pipeline_models[key]

    # --- Embeddings ---

    @property
    def embeddings(self) -> list[dict]:
        return self._embeddings

    def find_embedding_model(self, model: str) -> dict | None:
        for cfg in self._embeddings:
            if cfg.get("model") == model:
                return cfg
        return None

    def list_embedding_models(self) -> list[str]:
        return [cfg["model"] for cfg in self._embeddings if _is_available(cfg)]

    # --- Other ---

    @property
    def pricing(self) -> dict:
        return self._pricing

    @property
    def mcp_services(self) -> dict:
        return self._mcp_services

    @property
    def metrics(self) -> dict:
        return self._metrics


def _dict_to_list(cfg) -> list[dict]:
    if isinstance(cfg, dict):
        return [cfg]
    return cfg

def _is_available(cfg: dict) -> bool:
    api_key_env = cfg.get("api_key_env")
    return not api_key_env or os.environ.get(api_key_env) is not None
