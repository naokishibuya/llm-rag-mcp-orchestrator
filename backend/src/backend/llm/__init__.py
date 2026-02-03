import logging

from .base import BaseChat, BaseEmbeddings, LLMResponse, EmbeddingResponse
from ..config import get_chat_configs, get_embeddings_configs, load_class


logger = logging.getLogger(__name__)

_chat_instances: dict[str, BaseChat] = {}
_chat_unavailable: set[str] = set()
_embeddings_instances: dict[str, BaseEmbeddings] = {}
_embeddings_unavailable: set[str] = set()


def _try_create_chat(cfg: dict) -> BaseChat | None:
    """Try to create a chat instance, return None if it fails."""
    model = cfg.get("model")
    try:
        cfg = dict(cfg)
        cls = load_class(cfg.pop("class"))
        return cls(**cfg)
    except Exception as e:
        logger.warning(f"Chat model {model} unavailable: {e}")
        return None


def _try_create_embeddings(cfg: dict) -> BaseEmbeddings | None:
    """Try to create an embeddings instance, return None if it fails."""
    model = cfg.get("model")
    try:
        cfg = dict(cfg)
        cls = load_class(cfg.pop("class"))
        return cls(**cfg)
    except Exception as e:
        logger.warning(f"Embedding model {model} unavailable: {e}")
        return None


def get_chat(model: str | None = None) -> BaseChat:
    available = list_models()
    if not available:
        raise ValueError("No chat models available")

    if model is None:
        model = available[0]

    if model in _chat_unavailable:
        raise ValueError(f"Model {model} is not available")

    if model not in _chat_instances:
        cfg = next((c for c in get_chat_configs() if c.get("model") == model), None)
        if cfg is None:
            raise ValueError(f"Model {model} not found in config")
        instance = _try_create_chat(cfg)
        if instance is None:
            _chat_unavailable.add(model)
            raise ValueError(f"Model {model} is not available")
        _chat_instances[model] = instance

    return _chat_instances[model]


def get_embeddings(model: str | None = None) -> BaseEmbeddings:
    available = list_embeddings()
    if not available:
        raise ValueError("No embedding models available")

    if model is None:
        model = available[0]

    if model in _embeddings_unavailable:
        raise ValueError(f"Embedding model {model} is not available")

    if model not in _embeddings_instances:
        cfg = next((c for c in get_embeddings_configs() if c.get("model") == model), None)
        if cfg is None:
            raise ValueError(f"Embedding model {model} not found in config")
        instance = _try_create_embeddings(cfg)
        if instance is None:
            _embeddings_unavailable.add(model)
            raise ValueError(f"Embedding model {model} is not available")
        _embeddings_instances[model] = instance

    return _embeddings_instances[model]


def list_models() -> list[str]:
    """List available chat models, validating each on first call."""
    configs = get_chat_configs()
    available = []
    for cfg in configs:
        model = cfg.get("model")
        if model in _chat_unavailable:
            continue
        if model in _chat_instances:
            available.append(model)
            continue
        # Try to instantiate to check availability
        instance = _try_create_chat(cfg)
        if instance:
            _chat_instances[model] = instance
            available.append(model)
        else:
            _chat_unavailable.add(model)
    return available


def list_embeddings() -> list[str]:
    """List available embedding models, validating each on first call."""
    configs = get_embeddings_configs()
    available = []
    for cfg in configs:
        model = cfg.get("model")
        if model in _embeddings_unavailable:
            continue
        if model in _embeddings_instances:
            available.append(model)
            continue
        # Try to instantiate to check availability
        instance = _try_create_embeddings(cfg)
        if instance:
            _embeddings_instances[model] = instance
            available.append(model)
        else:
            _embeddings_unavailable.add(model)
    return available


__all__ = [
    "BaseChat",
    "BaseEmbeddings",
    "LLMResponse",
    "EmbeddingResponse",
    "get_chat",
    "get_embeddings",
    "list_models",
    "list_embeddings",
]
