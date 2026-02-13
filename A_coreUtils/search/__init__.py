"""Search subpackage exports."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "AutoSceneSearcher": (".auto_scene_search", "AutoSceneSearcher"),
    "PromptGenerator": (".auto_scene_search", "PromptGenerator"),
    "extract_model_name_from_pkl": (".auto_scene_search", "extract_model_name_from_pkl"),
    "detect_model_type_from_name": (".auto_scene_search", "detect_model_type_from_name"),
    "run_interactive_search": (".auto_scene_search", "run_interactive_search"),
    "BatchTextSearchEngine": (".batch_text_search", "BatchTextSearchEngine"),
    "run_cloze_fill_search": (".cloze_fill_search", "run_cloze_fill_search"),
    "run_label_traverse_search": (".label_traverse_search", "run_label_traverse_search"),
    "RerankerFrameExtractor": (".reranker_frame_extractor", "RerankerFrameExtractor"),
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
