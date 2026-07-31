# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存

"""
景别（Shot Type）分类模块 — DinoV2。

Usage:
    from A_coreUtils.aftertreatment.shot_type_classifier import DinoV2ShotClassifier
    clf = DinoV2ShotClassifier()
    result = clf.analyze("video.mp4")   # {"景别": "中景", "confidence": 0.85}
"""

import os
import sys
from typing import Tuple, List

import cv2
import numpy as np

_current_file = os.path.abspath(__file__)
_aftertreatment_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_aftertreatment_dir)
_project_root = os.path.dirname(_a_core_utils_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


DINOV2_LABELS_CN = {
    "extreme_close_up": "大特写",
    "close_up":         "特写",
    "medium":           "中景",
    "full":             "全景",
    "wide":             "远景",
}


class DinoV2ShotClassifier:
    """基于 DinoV2 with Registers 的景别分类器。

    模型: models/aslakey_shot_scale/
    类别: 特写 / 大特写 / 中景 / 全景 / 远景
    """

    MODEL_NAME = "DinoV2"
    _HF_BACKBONE = "facebook/dinov2-with-registers-large"

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(_project_root, "models", "aslakey_shot_scale")
        self._model_path = model_path
        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        if torch.cuda.is_available():
            try:
                _test = torch.zeros(1, device="cuda")
                self._device = torch.device("cuda")
            except RuntimeError:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device("cpu")
        print(f"[{self.MODEL_NAME}] 加载: {self._model_path}")

        if not os.path.isdir(self._model_path):
            raise FileNotFoundError(f"模型目录不存在: {self._model_path}")

        self._processor = AutoImageProcessor.from_pretrained(
            self._model_path, local_files_only=True
        )
        self._model = AutoModelForImageClassification.from_pretrained(
            self._model_path, local_files_only=True
        ).to(self._device)
        self._model.eval()
        self._loaded = True
        print(f"[{self.MODEL_NAME}] 就绪, device={self._device}")

    def unload(self):
        """释放 GPU 显存。"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _predict_frame(self, frame_bgr: np.ndarray) -> Tuple[str, float]:
        import torch
        from PIL import Image

        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            idx = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0, idx].item())

        label_en = self._model.config.id2label.get(idx, f"unknown({idx})")
        return DINOV2_LABELS_CN.get(label_en, label_en), conf

    def analyze_frame(self, frame_bgr: np.ndarray) -> dict:
        """分析一张已经在内存中的 BGR 中间帧，不生成临时图片文件。"""
        if not self._loaded:
            self.load()
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return {"景别": "未知", "confidence": 0.0}
        label, conf = self._predict_frame(frame_bgr)
        return {"景别": label, "confidence": round(conf, 4)}

    def analyze_frames_batch(self, frames_bgr: List[np.ndarray]) -> List[dict]:
        """批量分析内存中的 BGR 中间帧，返回结果顺序与输入一致。"""
        if not self._loaded:
            self.load()
        if not frames_bgr:
            return []

        import torch
        from PIL import Image

        results = [{"景别": "未知", "confidence": 0.0} for _ in frames_bgr]
        valid_indices = []
        pil_images = []
        for index, frame_bgr in enumerate(frames_bgr):
            if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
                continue
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(img_rgb))
            valid_indices.append(index)

        if not pil_images:
            return results

        inputs = self._processor(images=pil_images, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            indices = torch.argmax(probs, dim=1)
            confidences = probs.gather(1, indices.unsqueeze(1)).squeeze(1)

        for batch_index, result_index in enumerate(valid_indices):
            class_index = int(indices[batch_index].item())
            confidence = float(confidences[batch_index].item())
            label_en = self._model.config.id2label.get(
                class_index, f"unknown({class_index})"
            )
            results[result_index] = {
                "景别": DINOV2_LABELS_CN.get(label_en, label_en),
                "confidence": round(confidence, 4),
            }
        return results

    def _extract_frames(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            cap.release()
            return []
        indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int).tolist()
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames

    def analyze(self, file_path: str) -> dict:
        """分析视频或图片的景别。

        视频：提取 5 帧做加权投票
        图片：直接分类
        """
        if not self._loaded:
            self.load()

        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            frame = cv2.imread(file_path)
            if frame is None:
                return {"景别": "未知", "confidence": 0.0}
            label, conf = self._predict_frame(frame)
            return {"景别": label, "confidence": round(conf, 4)}

        frames = self._extract_frames(file_path, num_frames=5)
        if not frames:
            return {"景别": "未知", "confidence": 0.0}

        counter = {}
        for f in frames:
            label, conf = self._predict_frame(f)
            counter[label] = counter.get(label, 0.0) + conf

        if not counter:
            return {"景别": "未知", "confidence": 0.0}

        winner = max(counter, key=counter.get)
        total = sum(counter.values())
        return {"景别": winner, "confidence": round(counter[winner] / max(total, 1e-8), 4)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DinoV2 景别分类测试")
    parser.add_argument("file", help="图片或视频文件路径")
    args = parser.parse_args()

    clf = DinoV2ShotClassifier()
    result = clf.analyze(args.file)
    print(f"景别: {result['景别']}, 置信度: {result['confidence']:.4f}")
