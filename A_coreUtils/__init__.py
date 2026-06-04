# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存

"""Core utility package exports for Auto-scenes-extraction."""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType

# Keep `path_resolver` importable as a top-level module for legacy modules.
_current_file = os.path.abspath(__file__)
_a_core_utils_dir = os.path.dirname(_current_file)
_cut_detect_scene_dir = os.path.dirname(_a_core_utils_dir)
if _cut_detect_scene_dir not in sys.path:
    sys.path.insert(0, _cut_detect_scene_dir)

_SUBPACKAGE_EXPORTS = {
    "aftertreatment": ".aftertreatment",
    "embedding": ".embedding",
    "qwen_models": ".qwen_models",
    "search": ".search",
    "video_processing": ".video_processing",
    "prompt": ".prompt",
    "lance_index_io": ".lance_index_io",
}

# Backward compatibility for old flat module import paths.
# Only includes modules that actually exist in this project.
_LEGACY_MODULE_ALIASES = {
    "embedding_model": ".embedding.embedding_model",
    "qwen3_vl_reranker": ".qwen_models.qwen3_vl_reranker",
    "auto_scene_search": ".search.auto_scene_search",
    "batch_text_search": ".search.batch_text_search",
    "reranker_frame_extractor": ".search.reranker_frame_extractor",
    "ffmpeg_precision_cutter": ".video_processing.ffmpeg_precision_cutter",
    "video_utils": ".video_processing.video_utils",
    "Video_Scene_Merger": ".video_processing.Video_Scene_Merger",
    "Video_Scene_Analyzer": ".video_processing.Video_Scene_Analyzer",
    "video_name_parser": ".video_processing.video_name_parser",
    "prompt_vector_cache": ".prompt.prompt_vector_cache",
    "label_verifier": ".aftertreatment.label_verifier",
    "optical_flow_analyzer": ".aftertreatment.optical_flow_analyzer",
    "shot_type_classifier": ".aftertreatment.shot_type_classifier",
    "shot_analyzer": ".aftertreatment.shot_analyzer",
}

__all__ = sorted(set(_SUBPACKAGE_EXPORTS) | set(_LEGACY_MODULE_ALIASES))


def __getattr__(name: str) -> ModuleType:
    target = _SUBPACKAGE_EXPORTS.get(name) or _LEGACY_MODULE_ALIASES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(target, __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
