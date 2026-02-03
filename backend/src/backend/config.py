import importlib
from pathlib import Path
from typing import Any
import yaml


CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    global _config
    if _config is None:
        _config = _load_yaml(CONFIG_DIR / "config.yaml")
    return _config


def get_chat_configs() -> list[dict[str, Any]]:
    return get_config().get("chat", [])


def get_embeddings_configs() -> list[dict[str, Any]]:
    configs = get_config().get("embeddings", [])
    # Handle old single-config format for backwards compatibility
    if isinstance(configs, dict):
        configs = [configs]
    return configs


def get_pricing_config() -> dict[str, Any]:
    return get_config().get("pricing", {})


def get_metrics_config() -> dict[str, Any]:
    return get_config().get("metrics", {})


def get_pricing(model: str) -> dict[str, float]:
    pricing = get_pricing_config()
    return pricing.get(model, {"input": 0.0, "output": 0.0})


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = get_pricing(model)
    input_cost = (input_tokens / 1_000_000) * pricing.get("input", 0.0)
    output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0.0)
    return input_cost + output_cost


def load_class(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
