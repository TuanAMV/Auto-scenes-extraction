# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
# prompt - Prompt生成模块
"""
Prompt生成模块，包含：
- prompt_vector_cache.py: Prompt向量缓存器（预计算归一化向量）
"""

import os
import sys

# ============================================================
#  路径设置 - 确保能找到项目根目录的模块
# ============================================================
_current_file = os.path.abspath(__file__)
_prompt_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_prompt_dir)
_project_root_dir = os.path.dirname(_a_core_utils_dir)
if _project_root_dir not in sys.path:
    sys.path.insert(0, _project_root_dir)

from .prompt_vector_cache import (
    PromptVectorCache,
    PromptVectorBatchIterator
)

__all__ = [
    'PromptVectorCache',
    'PromptVectorBatchIterator',
]
