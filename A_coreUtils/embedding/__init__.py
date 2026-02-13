"""Embedding subpackage exports."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "EmbeddingModelProcessor": (".embedding_model", "EmbeddingModelProcessor"),
    "VectorizerThread": (".embedding_model", "VectorizerThread"),
    "SemanticSearchEngine": (".embedding_model", "SemanticSearchEngine"),
    "create_video_index": (".embedding_model", "create_video_index"),
    "batch_create_video_index": (".embedding_model", "batch_create_video_index"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
