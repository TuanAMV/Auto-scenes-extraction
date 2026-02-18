# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存

"""Search utilities for scene retrieval workflows."""

from __future__ import annotations

import os
import sys

# Keep `path_resolver` importable as a top-level module.
_current_file = os.path.abspath(__file__)
_search_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_search_dir)
_project_root_dir = os.path.dirname(_a_core_utils_dir)
if _project_root_dir not in sys.path:
    sys.path.insert(0, _project_root_dir)

from .auto_scene_search import (  # noqa: E402
    AutoSceneSearcher,
    PromptGenerator,
    detect_model_type_from_name,
    extract_model_name_from_index,
)
from .batch_text_search import BatchTextSearchEngine  # noqa: E402

# Backward-compatible alias for old API name.
extract_model_name_from_pkl = extract_model_name_from_index

__all__ = [
    "AutoSceneSearcher",
    "PromptGenerator",
    "extract_model_name_from_index",
    "extract_model_name_from_pkl",
    "detect_model_type_from_name",
    "BatchTextSearchEngine",
]
