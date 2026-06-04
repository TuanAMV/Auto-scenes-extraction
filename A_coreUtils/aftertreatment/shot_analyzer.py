# -*- coding: utf-8 -*-
"""
基于 MiniCPM-V-4-6 原生视频模式的景别和镜头运动分析模块

Usage:
    from A_coreUtils.aftertreatment.shot_analyzer import ShotAnalyzer

    analyzer = ShotAnalyzer()
    result = analyzer.analyze("path/to/video.mp4")
    # {"景别": "中景", "镜头运动": "推镜头"}
"""

import os
import sys
import json
from typing import List

import torch

_current_file = os.path.abspath(__file__)
_aftertreatment_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_aftertreatment_dir)
_project_root = os.path.dirname(_a_core_utils_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class ShotAnalyzer:
    """视频景别和镜头运动分析器"""

    SHOT_TYPES = ["远景","全景","中景", "近景", "特写"]
    CAMERA_MOVEMENTS = ["推镜头", "拉镜头", "摇镜头", "移镜头", "跟镜头", "升镜头", "降镜头", "固定镜头"]

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
        self._model = None
        self._processor = None
        self.model_loaded = False

        # 视觉处理参数
        self.downsample_mode = "4x"  # "16x"(快) / "4x"(精细)
        self.max_slice_nums = 1       # 视频推荐 1
        self.max_num_frames = 128     # 最大采样帧数
        self.stack_frames = 4         # 每秒采样点数（长视频可设 3-5）
        self.use_image_id = False     # 视频模式应为 False

        # 生成参数
        self.max_new_tokens = 128
        self.do_sample = False
        self.temperature = None
        self.top_p = None
        self.top_k = None
        self.repetition_penalty = None

    @property
    def model_path(self):
        return os.path.join(_project_root, "models", self.model_name)

    def load_model(self):
        """加载 MiniCPM-V-4-6 模型（单例，只加载一次）"""
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

    def _build_prompt(self):
        shot_types_str = "、".join(self.SHOT_TYPES)
        movements_str = "、".join(self.CAMERA_MOVEMENTS)
        return (
            f"分析这段视频：\n"
            f"1. 景别（可选: {shot_types_str}）\n"
            f"2. 镜头运动（可选: {movements_str}）\n\n"
            f'只输出一行JSON，以{{开头，不要解释:\n'
            f'{{"景别": "...", "镜头运动": "..."}}'
        )

    def _parse_response(self, text: str) -> dict:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                parsed = json.loads(text[start:end + 1])
                return {
                    "景别": self._match(parsed.get("景别", ""), self.SHOT_TYPES),
                    "镜头运动": self._match(parsed.get("镜头运动", ""), self.CAMERA_MOVEMENTS),
                }
            except json.JSONDecodeError:
                pass

        return {
            "景别": self._match(text, self.SHOT_TYPES),
            "镜头运动": self._match(text, self.CAMERA_MOVEMENTS),
        }

    def _match(self, text: str, candidates: List[str]):
        text = str(text or "")
        for c in candidates:
            if c in text or text in c:
                return c
        return None


    def _run_inference(self, video_path: str, prompt: str) -> str:
        """单次推理，返回模型输出文本"""
        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ]},
        ]

        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            downsample_mode=self.downsample_mode,
            max_num_frames=self.max_num_frames,
            stack_frames=self.stack_frames,
            max_slice_nums=self.max_slice_nums,
            use_image_id=self.use_image_id,
        ).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            eos_token_id=self._processor.tokenizer.eos_token_id,
            downsample_mode=self.downsample_mode,
        )
        if self.do_sample:
            if self.temperature is not None:
                gen_kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                gen_kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                gen_kwargs["top_k"] = self.top_k
            if self.repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = self.repetition_penalty

        generated_ids = self._model.generate(**inputs, **gen_kwargs)
        response_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
        return self._processor.tokenizer.decode(
            response_ids, skip_special_tokens=True
        )

    @torch.no_grad()
    def analyze(self, video_path: str) -> dict:
        """分析视频的景别、镜头运动和美学评分

        Returns:
            {"景别": "中景", "镜头运动": "推镜头"}
        """
        if not self.model_loaded:
            self.load_model()

        video_path = os.path.abspath(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        print(f"[ShotAnalyzer] 分析: {os.path.basename(video_path)}")

        text = self._run_inference(video_path, self._build_prompt())
        print(f"[ShotAnalyzer] 输出: {text}")
        return self._parse_response(text)


_global_analyzer = None


def analyze(video_path: str) -> dict:
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = ShotAnalyzer()
    return _global_analyzer.analyze(video_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="视频景别和镜头运动分析")
    parser.add_argument("video", help="视频文件路径")
    args = parser.parse_args()

    analyzer = ShotAnalyzer()
    result = analyzer.analyze(args.video)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    analyzer.unload_model()
