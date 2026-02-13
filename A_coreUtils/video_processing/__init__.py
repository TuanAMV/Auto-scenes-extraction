"""Video processing subpackage exports."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "VideoMetaHelper": (".video_utils", "VideoMetaHelper"),
    "FrameExtractorThread": (".video_utils", "FrameExtractorThread"),
    "SceneBoundaryDetector": (".video_utils", "SceneBoundaryDetector"),
    "SceneFeatureExtractor": (".video_utils", "SceneFeatureExtractor"),
    "AsyncWriter": (".video_utils", "AsyncWriter"),
    "cleanup_temp_folder": (".video_utils", "cleanup_temp_folder"),
    "ensure_temp_folder_clean": (".video_utils", "ensure_temp_folder_clean"),
    "sanitize_name": (".video_utils", "sanitize_name"),
    "resolve_path": (".video_utils", "resolve_path"),
    "extract_single_frame": (".video_utils", "extract_single_frame"),
    "extract_single_frame_rerank": (".video_utils", "extract_single_frame_rerank"),
    "TEMP_DIR": (".video_utils", "TEMP_DIR"),
    "DEVICE": (".video_utils", "DEVICE"),
    "FFmpegPrecisionCutter": (".ffmpeg_precision_cutter", "FFmpegPrecisionCutter"),
    "VideoInfo": (".ffmpeg_precision_cutter", "VideoInfo"),
    "export_video_clip": (".ffmpeg_precision_cutter", "export_video_clip"),
    "VideoNameParser": (".video_name_parser", "VideoNameParser"),
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
