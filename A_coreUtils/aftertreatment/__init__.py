# -*- coding: utf-8 -*-
# aftertreatment 模块：视频后处理分析（景别、镜头运动、标签验证等）
#
# 子模块：
#   shot_type_classifier   - 基于 DinoV2 的景别分类
#   optical_flow_analyzer  - 基于稠密光流法（Farneback）的镜头推拉摇移跟预测
#   label_verifier         - 基于 MiniCPM 的标签验证
#   shot_analyzer          - 镜头分析（整合景别 + 光流的并发控制器）

from .shot_type_classifier import DinoV2ShotClassifier
from .optical_flow_analyzer import OpticalFlowAnalyzer
from .label_verifier import LabelVerifier
from .shot_analyzer import ShotAnalyzer

__all__ = [
    "DinoV2ShotClassifier",
    "OpticalFlowAnalyzer",
    "LabelVerifier",
    "ShotAnalyzer",
]
