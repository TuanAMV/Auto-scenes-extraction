# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存

"""Package entry point for Auto-scenes-extraction."""

from __future__ import annotations

import importlib
from types import ModuleType

_LAZY_EXPORTS = {
    "A_coreUtils": ".A_coreUtils",
    "path_resolver": ".path_resolver",
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> ModuleType:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(target, __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
