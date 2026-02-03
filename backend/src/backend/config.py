import importlib
from pathlib import Path
from typing import Any
import yaml


CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config() -> dict[str, Any]:
    main = _load_yaml(CONFIG_DIR / "config.yaml")

    if llm_profile := main.get("llm", {}).get("profile"):
        main["llm"] = _load_yaml(CONFIG_DIR / llm_profile)

    return main


def load_class(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_llm_config() -> dict[str, Any]:
    return get_config().get("llm", {})


def get_metrics_config() -> dict[str, Any]:
    return get_config().get("metrics", {})
