"""Prompt subpackage exports.

Note: legacy logic_Prompt.py is no longer present in this project.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "PromptVectorCache": (".prompt_vector_cache", "PromptVectorCache"),
    "PromptVectorBatchIterator": (".prompt_vector_cache", "PromptVectorBatchIterator"),
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
