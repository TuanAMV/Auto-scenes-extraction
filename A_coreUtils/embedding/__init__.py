# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
# embedding - CLIP系列嵌入模型
"""
CLIP系列嵌入模型模块，包含：
- embedding_model.py: 通用嵌入模型处理器（支持CLIP、FG-CLIP2等）
"""

import os
import sys

# ============================================================
#  路径设置 - 确保能找到项目根目录的模块
# ============================================================
_current_file = os.path.abspath(__file__)
_embedding_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_embedding_dir)
_cut_detect_scene_dir = os.path.dirname(_a_core_utils_dir)
if _cut_detect_scene_dir not in sys.path:
    sys.path.insert(0, _cut_detect_scene_dir)

from .embedding_model import (
    EmbeddingModelProcessor,
)

__all__ = [
    'EmbeddingModelProcessor',
]
