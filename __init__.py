"""Auto-scenes-extraction package marker."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["A_coreUtils", "path_resolver"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
