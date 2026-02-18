# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存

"""Qwen model adapters used by this project."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys

# Keep `path_resolver` importable as a top-level module.
_current_file = os.path.abspath(__file__)
_qwen_models_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_qwen_models_dir)
_project_root_dir = os.path.dirname(_a_core_utils_dir)
if _project_root_dir not in sys.path:
    sys.path.insert(0, _project_root_dir)

from .qwen3_vl_reranker import Qwen3VLReranker  # noqa: E402

__all__ = ["Qwen3VLReranker"]

# Optional export: only load if the embedding module exists in this repo.
_embedding_module_name = f"{__name__}.qwen3_vl_embedding"
if importlib.util.find_spec(_embedding_module_name) is not None:
    _embedding_module = importlib.import_module(".qwen3_vl_embedding", __name__)
    Qwen3VLForEmbedding = _embedding_module.Qwen3VLForEmbedding
    Qwen3VLProcessor = _embedding_module.Qwen3VLProcessor
    Qwen3VLEmbedder = _embedding_module.Qwen3VLEmbedder
    sample_frames = _embedding_module.sample_frames

    __all__.extend(
        [
            "Qwen3VLForEmbedding",
            "Qwen3VLProcessor",
            "Qwen3VLEmbedder",
            "sample_frames",
        ]
    )
