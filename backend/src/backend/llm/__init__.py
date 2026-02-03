from typing import Any
from .base import BaseChat, BaseEmbeddings, LLMResponse, EmbeddingResponse
from ..config import get_llm_config, load_class


_chat_instance: BaseChat | None = None
_embeddings_instance: BaseEmbeddings | None = None


def get_chat(**overrides) -> BaseChat:
    global _chat_instance
    if _chat_instance is None:
        cfg = get_llm_config().get("chat", {})
        cfg = {**cfg, **overrides}
        cls = load_class(cfg.pop("class"))
        _chat_instance = cls(**cfg)
    return _chat_instance


def get_embeddings(**overrides) -> BaseEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        cfg = get_llm_config().get("embeddings", {})
        cfg = {**cfg, **overrides}
        cls = load_class(cfg.pop("class"))
        _embeddings_instance = cls(**cfg)
    return _embeddings_instance


def create_chat(cfg: dict[str, Any]) -> BaseChat:
    cfg = dict(cfg)
    cls = load_class(cfg.pop("class"))
    return cls(**cfg)


def create_embeddings(cfg: dict[str, Any]) -> BaseEmbeddings:
    cfg = dict(cfg)
    cls = load_class(cfg.pop("class"))
    return cls(**cfg)


__all__ = [
    "BaseChat",
    "BaseEmbeddings",
    "LLMResponse",
    "EmbeddingResponse",
    "get_chat",
    "get_embeddings",
    "create_chat",
    "create_embeddings",
]
