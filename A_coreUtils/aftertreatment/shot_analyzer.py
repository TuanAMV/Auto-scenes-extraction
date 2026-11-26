# -*- coding: utf-8 -*-
"""
MiniCPM-V-4-6 模型加载封装，供 LabelVerifier / MiniRerank 使用。

Usage:
    from A_coreUtils.aftertreatment.shot_analyzer import ShotAnalyzer

    analyzer = ShotAnalyzer()
    analyzer.load_model()
    # 使用 analyzer._model / analyzer._processor 做单帧验证
"""

import os
import sys
from typing import Any

import torch

_current_file = os.path.abspath(__file__)
_aftertreatment_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_aftertreatment_dir)
_project_root = os.path.dirname(_a_core_utils_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# torchcodec 需要 FFmpeg 共享库；项目自带在 models/ffmpeg/
_shared_ffmpeg_bin = os.path.join(
    _project_root, "models", "ffmpeg", "bin"
)
if os.path.isdir(_shared_ffmpeg_bin) and _shared_ffmpeg_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _shared_ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")


class ShotAnalyzer:
    """视频景别和镜头运动分析器"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "MiniCPM-V-4-6", device: str = None):
        if self._initialized:
            return
        self._initialized = True

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Any = None
        self._processor: Any = None
        self.model_loaded = False


    @property
    def model_path(self):
        return os.path.join(_project_root, "models", self.model_name)

    def load_model(self):
        """加载 MiniCPM-V 模型（单例，只加载一次）"""
        if self.model_loaded:
            return

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"模型目录不存在: {self.model_path}")

        print(f"[ShotAnalyzer] 加载模型: {self.model_name}")
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True,
            )

            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)

            self._model.eval()
            self.model_loaded = True
            print(f"[ShotAnalyzer] 模型加载完成，设备: {self.device}")
        except Exception as e:
            raise RuntimeError(f"MiniCPM-V-4-6 模型加载失败: {e}")

    def unload_model(self):
        """释放模型显存"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self.model_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
