"""Public entry for A_coreUtils.

Keep package import light by lazy-loading subpackages only when accessed.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "embedding",
    "prompt",
    "qwen_models",
    "search",
    "video_processing",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
