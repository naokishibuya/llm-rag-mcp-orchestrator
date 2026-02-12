import os
import yaml
from pathlib import Path


class Config:
    def __init__(self, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        self._talk = _dict_to_list(data["talk"])
        self._embedding = data["embedding"]

        # Separate orchestrator behavior settings from model config
        orch_cfg = dict(data["orchestrator"])
        self._max_reflections: int = orch_cfg.pop("max_reflections", 2)
        self._pipeline_models = {
            "mcp": data["mcp"],
            "rag": data["rag"],
            "orchestrator": orch_cfg,
        }

        self._pricing = data["pricing"]
        self._mcp_services = data["mcp_services"]
        self._metrics = data["metrics"]

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

    def default_talk_model(self) -> str:
        models = self.list_talk_models()
        if not models:
            raise RuntimeError("No talk models available")
        return models[0]

    def get_pipeline_model(self, key: str) -> dict:
        return self._pipeline_models[key]

    def get_embedding_config(self) -> dict:
        return self._embedding

    @property
    def max_reflections(self) -> int:
        return self._max_reflections

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
