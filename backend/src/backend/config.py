import os
import yaml
from pathlib import Path


class Config:
    def __init__(self, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        self._llm = _dict_to_list(data["llm"])
        self._embedding = data["rag"]["embeddings"]
        self._rag_params = data["rag"].get("params", {})
        self._pricing = data.get("pricing", {})
        self._mcp_services = data.get("mcp", {}).get("services", {})
        self._agents = data.get("agents", {})
        self._workflow = data.get("workflow", {})

    @property
    def llm(self) -> list[dict]:
        return self._llm

    @property
    def embedding(self) -> dict:
        return self._embedding

    def list_models(self) -> list[str]:
        return [cfg["model"] for cfg in self._llm if _is_available(cfg)]

    def default_model(self) -> str:
        models = self.list_models()
        if not models:
            raise RuntimeError("No models available")
        return models[0]

    @property
    def rag_params(self) -> dict:
        return self._rag_params

    @property
    def pricing(self) -> dict:
        return self._pricing

    @property
    def mcp_services(self) -> dict:
        return self._mcp_services

    @property
    def agents(self) -> dict:
        return self._agents

    @property
    def workflow(self) -> dict:
        return self._workflow


def _dict_to_list(cfg) -> list[dict]:
    if isinstance(cfg, dict):
        return [cfg]
    return cfg

def _is_available(cfg: dict) -> bool:
    api_key_env = cfg.get("api_key_env")
    return not api_key_env or os.environ.get(api_key_env) is not None
