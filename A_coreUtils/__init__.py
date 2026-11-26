# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存

"""Core utility package for Cut_DetectScene."""

from __future__ import annotations

import os
import sys

_current_file = os.path.abspath(__file__)
_a_core_utils_dir = os.path.dirname(_current_file)
_cut_detect_scene_dir = os.path.dirname(_a_core_utils_dir)
if _cut_detect_scene_dir not in sys.path:
    sys.path.insert(0, _cut_detect_scene_dir)
