# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
# video_processing - 视频处理模块
"""
视频处理模块，包含：
- ffmpeg_precision_cutter.py: FFmpeg高精度视频处理器
- video_utils.py: 视频处理工具类
- Video_Scene_Merger.py: 视频场景合并器
- Video_Scene_Analyzer.py: 视频场景分析器
- video_name_parser.py: 视频文件名解析器
"""

import os
import sys

# ============================================================
#  路径设置 - 确保能找到项目根目录的模块
# ============================================================
_current_file = os.path.abspath(__file__)
_video_processing_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_video_processing_dir)
_cut_detect_scene_dir = os.path.dirname(_a_core_utils_dir)
if _cut_detect_scene_dir not in sys.path:
    sys.path.insert(0, _cut_detect_scene_dir)

from .video_utils import (
    VideoMetaHelper,
    FrameExtractorThread,
    SceneBoundaryDetector,
    SceneFeatureExtractor,
    cleanup_temp_folder,
    sanitize_name,
    extract_single_frame,
    extract_single_frame_rerank,
    TEMP_DIR,
    DEVICE,
)
from .ffmpeg_precision_cutter import FFmpegPrecisionCutter, VideoInfo
from .video_name_parser import VideoNameParser

__all__ = [
    'VideoMetaHelper',
    'FrameExtractorThread',
    'SceneBoundaryDetector',
    'SceneFeatureExtractor',
    'cleanup_temp_folder',
    'sanitize_name',
    'extract_single_frame',
    'extract_single_frame_rerank',
    'TEMP_DIR',
    'DEVICE',
    'FFmpegPrecisionCutter',
    'VideoInfo',
    'VideoNameParser',
]
