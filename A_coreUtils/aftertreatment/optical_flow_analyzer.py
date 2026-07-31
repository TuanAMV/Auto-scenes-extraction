# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存

"""
基于稠密光流法（Farneback）的镜头推拉摇移预测模块

通过分析连续帧之间的稠密光流场方向与幅值分布来判断镜头运动类型：
  - 推镜头（dolly in / zoom in）：光流从画面中心向外辐射
  - 拉镜头（dolly out / zoom out）：光流向画面中心汇聚
  - 摇镜头（pan）：光流整体呈水平方向
  - 移镜头（tilt / crane）：光流整体呈垂直方向（含升降）
  - 跟镜头（tracking shot）：光流呈主导方向但有明显空间不均匀性（被摄主体区域幅值低、背景区域幅值高）
  - 固定镜头（static / locked-off）：光流幅值很小

Usage:
    from A_coreUtils.aftertreatment.optical_flow_analyzer import OpticalFlowAnalyzer

    analyzer = OpticalFlowAnalyzer()
    result = analyzer.analyze("path/to/video.mp4")
    # {"镜头运动": "推镜头", "景别": "未知", "confidence": 0.82, "details": {...}}
"""

import os
import sys
from typing import Tuple, Optional

import cv2
import numpy as np

_current_file = os.path.abspath(__file__)
_aftertreatment_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_aftertreatment_dir)
_project_root = os.path.dirname(_a_core_utils_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from path_resolver import PathResolver


class OpticalFlowAnalyzer:
    """基于稠密光流法（Farneback）的镜头运动分析器。

    算法原理：
    1. 从视频中按固定间隔采样连续帧对
    2. 对每对帧计算稠密光流（Farneback 算法）
    3. 对光流场做象限分析提取平移（摇/移）与缩放（推/拉）信号
    4. 对光流场做空间方差分析检测跟镜头（主体静止、背景移动）
    5. 对全片帧对的分类结果做加权投票，输出最终运动类型
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        sample_interval: int = 5,
        flow_resize: Tuple[int, int] = (320, 240),
        static_threshold: float = 0.4,
        pan_tilt_ratio: float = 0.40,
        radial_threshold: float = 0.25,
        tracking_threshold: float = 0.55,
        tracking_grid: int = 5,
        min_analysis_pairs: int = 3,
        farneback_pyr_scale: float = 0.5,
        farneback_levels: int = 3,
        farneback_winsize: int = 15,
        farneback_iterations: int = 3,
        farneback_poly_n: int = 5,
        farneback_poly_sigma: float = 1.2,
        push_symmetry_threshold: float = 0.3,
        push_pan_dominance: float = 1.5,
        static_score_multiplier: float = 0.6,
        tracking_score_scale: float = 0.5,
        push_vote_ratio: float = 0.20,
        tracking_vote_ratio: float = 0.25,
        pan_tilt_vote_ratio: float = 0.20,
    ):
        """初始化光流分析器。

        Args:
            sample_interval: 采样帧间隔（每隔 N 帧取一帧参与光流计算）
            flow_resize: 光流计算前的帧缩放尺寸 (width, height)，值越小越快
            static_threshold: 平均光流幅值低于此值（像素）判定为固定镜头
            pan_tilt_ratio: 摇/移判定时，水平或垂直分量占比需超过此比例
            radial_threshold: 推/拉判定时，径向分量绝对值占比需超过此比例
            tracking_threshold: 跟镜头判定时，网格间幅值变异系数需超过此比例
            tracking_grid: 跟镜头检测时的网格划分数（N x N）
            min_analysis_pairs: 最少有效帧对数，不足时降低阈值或直接判为固定
            farneback_pyr_scale: Farneback 金字塔缩放比
            farneback_levels: 金字塔层数
            farneback_winsize: 均值窗口大小
            farneback_iterations: 每层迭代次数
            farneback_poly_n: 多项式展开邻域大小
            farneback_poly_sigma: 多项式展开高斯标准差
        """
        if self._initialized:
            for key in (
                "sample_interval", "flow_resize", "static_threshold",
                "pan_tilt_ratio", "radial_threshold", "tracking_threshold",
                "tracking_grid", "min_analysis_pairs",
                "farneback_pyr_scale", "farneback_levels", "farneback_winsize",
                "farneback_iterations", "farneback_poly_n", "farneback_poly_sigma",
                "push_symmetry_threshold", "push_pan_dominance",
                "static_score_multiplier", "tracking_score_scale",
                "push_vote_ratio", "tracking_vote_ratio", "pan_tilt_vote_ratio",
            ):
                if key in locals():
                    setattr(self, key, locals()[key])
            return

        self.sample_interval = sample_interval
        self.flow_resize = flow_resize
        self.static_threshold = static_threshold
        self.pan_tilt_ratio = pan_tilt_ratio
        self.radial_threshold = radial_threshold
        self.tracking_threshold = tracking_threshold
        self.tracking_grid = tracking_grid
        self.min_analysis_pairs = min_analysis_pairs
        self.push_symmetry_threshold = push_symmetry_threshold
        self.push_pan_dominance = push_pan_dominance
        self.static_score_multiplier = static_score_multiplier
        self.tracking_score_scale = tracking_score_scale
        self.push_vote_ratio = push_vote_ratio
        self.tracking_vote_ratio = tracking_vote_ratio
        self.pan_tilt_vote_ratio = pan_tilt_vote_ratio

        self.farneback_pyr_scale = farneback_pyr_scale
        self.farneback_levels = farneback_levels
        self.farneback_winsize = farneback_winsize
        self.farneback_iterations = farneback_iterations
        self.farneback_poly_n = farneback_poly_n
        self.farneback_poly_sigma = farneback_poly_sigma

        self._resolver = PathResolver(__file__)
        self._initialized = True

    # ---------- 公有 API ----------

    def analyze(self, video_path: str) -> dict:
        """分析视频的镜头运动类型。

        Args:
            video_path: 视频文件路径

        Returns:
            dict: {
                "镜头运动": str,   # 推镜头 / 拉镜头 / 摇镜头 / 移镜头 / 固定镜头
                "景别": "未知",     # 本模块不预测景别，保留键名以兼容上游
                "confidence": float,
                "details": {
                    "pair_votes": [...],         # 每对帧的分类结果
                    "pair_details": [...],       # 每对帧的详细统计
                }
            }
        """
        video_path = os.path.abspath(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 2:
                return self._static_result("视频帧数不足")

            pair_results = []
            prev_gray = None
            frame_idx = 0
            prev_idx = -1

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.sample_interval != 0:
                    frame_idx += 1
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, self.flow_resize, interpolation=cv2.INTER_LINEAR)

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray,
                        None,
                        self.farneback_pyr_scale,
                        self.farneback_levels,
                        self.farneback_winsize,
                        self.farneback_iterations,
                        self.farneback_poly_n,
                        self.farneback_poly_sigma,
                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
                    )
                    result = self._analyze_flow_field(flow)
                    result["frame_pair"] = (prev_idx, frame_idx)
                    pair_results.append(result)

                prev_gray = gray
                prev_idx = frame_idx
                frame_idx += 1

        finally:
            cap.release()

        movement, confidence, details = self._classify_video(pair_results)
        return {
            "镜头运动": movement,
            "景别": "未知",
            "confidence": round(confidence, 4),
            "details": details,
        }

    def analyze_with_direction(self, video_path: str) -> dict:
        """分析镜头运动类型，附带摇/移的方向信息。

        Returns 中额外包含:
            "direction": "left" / "right" / "up" / "down" / None
        """
        result = self.analyze(video_path)
        direction = self._infer_direction(result.get("details", {}))
        result["direction"] = direction
        return result

    def prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """将解码帧立即转换为光流使用的灰度小图，避免长期保留 BGR。"""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] == 1:
            gray = frame[:, :, 0]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] != self.flow_resize[0] or gray.shape[0] != self.flow_resize[1]:
            gray = cv2.resize(gray, self.flow_resize, interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(gray)

    def analyze_frames(self, frames_bgr: list) -> dict:
        """直接从 BGR 或已预处理的灰度小图列表分析镜头运动。

        分析阶段优先接收 ``prepare_frame`` 生成的灰度 ``320x240`` 图，
        这样调用方无需长期保留光流所用的完整 BGR 帧；仍兼容直接传入 BGR。

        Args:
            frames_bgr: list of np.ndarray, each shape (H, W, 3), dtype uint8, BGR

        Returns:
            同 analyze() 方法
        """
        if len(frames_bgr) < 2:
            return self._static_result("帧数不足")

        pair_results = []
        prev_gray = None
        prev_idx = -1

        for frame_idx, frame in enumerate(frames_bgr):
            if frame_idx % self.sample_interval != 0:
                continue

            gray = self.prepare_frame(frame)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    self.farneback_pyr_scale, self.farneback_levels,
                    self.farneback_winsize, self.farneback_iterations,
                    self.farneback_poly_n, self.farneback_poly_sigma,
                    cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
                )
                result = self._analyze_flow_field(flow)
                result["frame_pair"] = (prev_idx, frame_idx)
                pair_results.append(result)

            prev_gray = gray
            prev_idx = frame_idx

        movement, confidence, details = self._classify_video(pair_results)
        return {
            "镜头运动": movement,
            "景别": "未知",
            "confidence": round(confidence, 4),
            "details": details,
        }

    # ---------- 光流场分析 ----------

    def _analyze_flow_field(self, flow: np.ndarray) -> dict:
        """对单张光流场做统计，提取运动信号。

        使用象限分析法分离平移（摇/移）与缩放（推/拉）信号，
        使用网格空间方差法检测跟镜头。

        Args:
            flow: shape (H, W, 2)，dtype float32，flow[..., 0]=dx, flow[..., 1]=dy

        Returns:
            dict: 各类统计量及本帧对的分类投票
        """
        h, w = flow.shape[:2]
        h2, w2 = h // 2, w // 2

        dx = flow[..., 0].astype(np.float64)
        dy = flow[..., 1].astype(np.float64)

        mag = np.sqrt(dx * dx + dy * dy)
        mean_mag = float(np.mean(mag))
        mean_dx = float(np.mean(dx))
        mean_dy = float(np.mean(dy))

        # ---- 象限均值（推拉 + 摇移信号） ----
        top_dx = float(np.mean(dx[:h2, :]))
        bot_dx = float(np.mean(dx[h2:, :]))
        top_dy = float(np.mean(dy[:h2, :]))
        bot_dy = float(np.mean(dy[h2:, :]))

        left_dx = float(np.mean(dx[:, :w2]))
        right_dx = float(np.mean(dx[:, w2:]))
        left_dy = float(np.mean(dy[:, :w2]))
        right_dy = float(np.mean(dy[:, w2:]))

        # 推拉信号（缩放）：左右反向 + 上下反向
        # 推镜头：左(-)右(+)→right-left=正, 上(-)下(+)→bottom-top=正
        # push_score > 0 → 推（扩张），push_score < 0 → 拉（收缩）
        push_h = right_dx - left_dx
        push_v = bot_dy - top_dy
        push_score = (push_h + push_v) * 0.5

        pan_score_h = (left_dx + right_dx) * 0.5
        tilt_score_v = (top_dy + bot_dy) * 0.5

        # ---- 跟镜头信号（网格空间方差） ----
        tracking_cv, edge_mean = self._compute_spatial_cv(mag, h, w)

        # ---- 中心-边缘对比度信号（跟镜头核心特征） ----
        center_edge_ratio, edge_dir_consistency, edge_mag_mean = \
            self._compute_center_edge_contrast(mag, dx, dy, h, w)

        # ---- 归一化比值 ----
        eps = 1e-8
        # 统一使用整幅光流的 RMS 总能量作为归一化分母
        # sqrt(E[dx²] + E[dy²]) 对平移和缩放信号是公平的同一参照系
        total_energy = float(np.sqrt(np.mean(dx * dx) + np.mean(dy * dy)))
        energy_denom = max(total_energy, eps)

        ratio_push = abs(push_score) / energy_denom
        ratio_pan = abs(pan_score_h) / energy_denom
        ratio_tilt = abs(tilt_score_v) / energy_denom

        push_sign = 1 if push_score > 0 else -1
        pan_sign = 1 if pan_score_h > 0 else -1
        tilt_sign = 1 if tilt_score_v > 0 else -1

        # 推拉对称性校验：左右/上下象限幅值比不小于阈值
        # 自然视频中很少完美对称，取 self.push_symmetry_threshold 与 0.15 中的较小者
        eps = 1e-8
        sym_threshold = min(self.push_symmetry_threshold, 0.15)
        sym_h = min(abs(right_dx), abs(left_dx)) / max(max(abs(right_dx), abs(left_dx)), eps)
        sym_v = min(abs(bot_dy), abs(top_dy)) / max(max(abs(bot_dy), abs(top_dy)), eps)
        push_symmetric = sym_h > sym_threshold and sym_v > sym_threshold

        vote, score = self._vote_flow(
            mean_mag,
            ratio_pan, ratio_tilt, ratio_push, push_score,
            push_sign, pan_score_h, tilt_score_v,
            tracking_cv, edge_mean, push_symmetric,
            center_edge_ratio, edge_dir_consistency, edge_mag_mean,
            pan_sign, tilt_sign,
        )

        return {
            "mean_dx": round(mean_dx, 4),
            "mean_dy": round(mean_dy, 4),
            "mean_mag": round(mean_mag, 4),
            "push_score": round(push_score, 4),
            "pan_score": round(pan_score_h, 4),
            "tilt_score": round(tilt_score_v, 4),
            "ratio_push": round(ratio_push, 4),
            "ratio_pan": round(ratio_pan, 4),
            "ratio_tilt": round(ratio_tilt, 4),
            "tracking_cv": round(tracking_cv, 4),
            "edge_mean": round(edge_mean, 4),
            "vote": vote,
            "score": round(score, 4),
        }

    def _compute_spatial_cv(self, mag: np.ndarray, h: int, w: int) -> Tuple[float, float]:
        """计算光流幅值的网格空间变异系数（CV）及边缘运动量。

        将画面划分为 tracking_grid x tracking_grid 的网格，
        计算每个 cell 的平均幅值，然后求这些均值的变异系数。
        跟镜头中主体区域幅值低、背景区域幅值高 → CV 大；
        摇/移/推/拉中幅值较均匀 → CV 小。

        Returns:
            (tracking_cv, edge_mean)
            - tracking_cv: 空间变异系数 (std / mean)
            - edge_mean: 边缘16格的平均幅值（判断是否相机在动）
        """
        g = self.tracking_grid
        cell_h = h // g
        cell_w = w // g
        cell_means = np.empty(g * g, dtype=np.float64)
        idx = 0
        # 记录哪些格子在边缘（第一行、最后一行、第一列、最后一列）
        grid_positions = []  # (i, j, is_edge) tuples
        for i in range(g):
            for j in range(g):
                y1 = i * cell_h
                y2 = (i + 1) * cell_h if i < g - 1 else h
                x1 = j * cell_w
                x2 = (j + 1) * cell_w if j < g - 1 else w
                cell_means[idx] = np.mean(mag[y1:y2, x1:x2])
                is_edge = (i == 0 or i == g - 1 or j == 0 or j == g - 1)
                grid_positions.append(is_edge)
                idx += 1

        cell_mean = np.mean(cell_means)
        if cell_mean < 1e-8:
            return 0.0, 0.0

        cv = float(np.std(cell_means) / cell_mean)
        # 边缘16格的平均幅值（判断相机是否在动）
        edge_mask = np.array([is_edge for is_edge in grid_positions])
        edge_mean = float(np.mean(cell_means[edge_mask]))
        return cv, edge_mean

    def _compute_center_edge_contrast(
        self, mag: np.ndarray, dx: np.ndarray, dy: np.ndarray, h: int, w: int
    ) -> Tuple[float, float, float]:
        """计算中心-边缘光流对比度信号。

        跟镜头的核心特征：中心区域（被跟踪主体）光流小，
        边缘区域（背景被"划过"）光流大且方向高度一致。

        Returns:
            (center_edge_ratio, edge_dir_consistency, edge_mag_mean)
            - center_edge_ratio: 中心幅值/边缘幅值，越小越像跟镜头
            - edge_dir_consistency: 边缘光流方向一致性 [0,1]
            - edge_mag_mean: 边缘区域平均光流幅值
        """
        margin = 0.15  # 边缘定义为外 15%
        cx1, cx2 = int(w * margin), int(w * (1 - margin))
        cy1, cy2 = int(h * margin), int(h * (1 - margin))

        center_mag = np.mean(mag[cy1:cy2, cx1:cx2])

        # 四个边缘条带
        edge_mags = [
            np.mean(mag[:cy1, :]),          # 上边缘
            np.mean(mag[cy2:, :]),          # 下边缘
            np.mean(mag[:, :cx1]),          # 左边缘
            np.mean(mag[:, cx2:]),          # 右边缘
        ]
        edge_mag_mean = float(np.mean(edge_mags))

        # 方向一致性：主方向幅值 / 总幅值
        edge_dx = np.concatenate([
            dx[:cy1, :].ravel(), dx[cy2:, :].ravel(),
            dx[:, :cx1].ravel(), dx[:, cx2:].ravel(),
        ])
        edge_dy = np.concatenate([
            dy[:cy1, :].ravel(), dy[cy2:, :].ravel(),
            dy[:, :cx1].ravel(), dy[:, cx2:].ravel(),
        ])
        mean_ex = float(np.mean(np.abs(edge_dx)))
        mean_ey = float(np.mean(np.abs(edge_dy)))
        if mean_ex + mean_ey < 1e-6:
            edge_dir_consistency = 0.0
        else:
            edge_dir_consistency = max(mean_ex, mean_ey) / (mean_ex + mean_ey)

        # 中心/边缘幅值比：值越小越像跟镜头
        center_edge_ratio = center_mag / max(edge_mag_mean, 1e-8)

        return center_edge_ratio, edge_dir_consistency, edge_mag_mean

    def _vote_flow(
        self,
        mean_mag: float,
        ratio_pan: float,
        ratio_tilt: float,
        ratio_push: float,
        push_score: float,
        push_sign: int,
        pan_score_h: float,
        tilt_score_v: float,
        tracking_cv: float,
        edge_mean: float,
        push_symmetric: bool,
        center_edge_ratio: float,
        edge_dir_consistency: float,
        edge_mag_mean: float,
        pan_sign: int,
        tilt_sign: int,
    ) -> Tuple[str, float]:
        """逐帧正常判断：所有达标信号比较，取最强的。

        Returns:
            (label, confidence_score)
        """
        WEAK_STATIC = 0.15

        # 收集所有达标的候选（固定镜头不再短路，与推拉摇移跟平等竞争）
        candidates = {}

        # 固定镜头信号：用全幅光流幅值反比，归一化到 [0, 1]
        # static_threshold 对应 medium 信号，越接近0信号越强
        static_score = 1.0 / (1.0 + mean_mag / max(self.static_threshold, 0.001))
        static_score = min(static_score * self.static_score_multiplier, 0.98)
        candidates["固定镜头"] = static_score

        if ratio_push >= self.radial_threshold and push_symmetric:
            # 推拉必须比 XY 平移信号显著强，否则只是平移中夹带的径向分量
            push_mag = abs(push_score)
            pan_tilt_max = max(abs(pan_score_h), abs(tilt_score_v))
            if push_mag > pan_tilt_max * self.push_pan_dominance or pan_tilt_max < self.static_threshold:
                candidates["推镜头" if push_sign > 0 else "拉镜头"] = ratio_push

        # 跟镜头：两级检测
        # 模式A（强）：明显中心-边缘分离 → 主体静止、背景快速划过 → 高置信
        # 模式B（弱）：方向一致但中心-边缘差异不明显 → 紧凑跟拍、小视差 → 低分+对摇移折价
        tracking_score = 0.0
        tracking_qualified = (
            edge_dir_consistency > 0.50
            and edge_mag_mean >= self.static_threshold * 0.5
        )
        if tracking_qualified:
            ce_deviation = abs(1.0 - center_edge_ratio)  # 偏离 1.0 的程度
            if center_edge_ratio < 0.75 and tracking_cv >= self.tracking_threshold * 0.4:
                # 模式A：强跟镜头信号
                tracking_score = edge_dir_consistency * ce_deviation * 3.0
            elif edge_dir_consistency > 0.65 and tracking_cv >= self.tracking_threshold * 0.3:
                # 模式B：弱跟镜头信号（紧凑跟拍）
                tracking_score = edge_dir_consistency * max(ce_deviation, 0.08) * 2.0
            if tracking_score > 0.01:
                candidates["跟镜头"] = min(tracking_score, 0.85)
                # 跟镜头与摇/移光流模式重叠，存在跟镜头证据时对摇/移折价
                if tracking_score > 0.20:
                    penalty = min(0.70, 1.0 - tracking_score * 0.5)
                    for label in ("摇镜头", "移镜头"):
                        if label in candidates:
                            candidates[label] *= penalty

        if ratio_pan >= self.pan_tilt_ratio:
            candidates["摇镜头"] = ratio_pan
        if ratio_tilt >= self.pan_tilt_ratio:
            candidates["移镜头"] = ratio_tilt

        if not candidates:
            return "固定镜头", WEAK_STATIC

        # 取最强信号
        winner = max(candidates, key=candidates.get)
        winner_val = candidates[winner]
        total = sum(candidates.values()) + 0.01
        confidence = min(winner_val / total, 0.95)
        return winner, confidence

    # ---------- 全片汇总 ----------

    def _classify_video(self, pair_results: list) -> Tuple[str, float, dict]:
        """对全片所有帧对的分析结果做加权投票，输出最终镜头运动类型。"""
        if not pair_results:
            return "固定镜头", 0.0, {"pair_votes": [], "pair_details": [], "reason": "无有效帧对"}

        votes = []
        vote_scores = []
        pair_votes = []
        pair_details = []

        for r in pair_results:
            vote = r.get("vote", "固定镜头")
            score = r.get("score", 0.5)
            votes.append(vote)
            vote_scores.append(score)
            pair_votes.append(vote)
            pair_details.append({
                "pair": r.get("frame_pair", (-1, -1)),
                "mean_mag": r.get("mean_mag"),
                "ratio_pan": r.get("ratio_pan"),
                "ratio_tilt": r.get("ratio_tilt"),
                "ratio_push": r.get("ratio_push"),
                "tracking_cv": r.get("tracking_cv"),
                "vote": vote,
                "score": score,
            })

        counter = {}
        for v, s in zip(votes, vote_scores):
            counter[v] = counter.get(v, 0.0) + s

        if not counter:
            return "固定镜头", 0.0, {
                "pair_votes": pair_votes,
                "pair_details": pair_details,
                "reason": "投票为空",
            }

        # 全片汇总：加权票数比例过半才胜，否则视为混合/不确定 → 固定镜头
        total_weight = sum(counter.values())
        if total_weight <= 0:
            return "固定镜头", 0.0, {
                "pair_votes": pair_votes,
                "pair_details": pair_details,
                "reason": "投票为空",
            }

        # 计算各类型占比
        push_score = counter.get("推镜头", 0.0) + counter.get("拉镜头", 0.0)
        tracking_score = counter.get("跟镜头", 0.0)
        pan_score = counter.get("摇镜头", 0.0)
        tilt_score = counter.get("移镜头", 0.0)
        static_score = counter.get("固定镜头", 0.0)

        push_ratio = push_score / total_weight
        tracking_ratio = tracking_score / total_weight
        pan_tilt_ratio = (pan_score + tilt_score) / total_weight

        # 所有类型平等竞争，取加权分最高者
        eligible = {}
        if push_ratio >= self.push_vote_ratio:
            eligible["推镜头" if counter.get("推镜头", 0) >= counter.get("拉镜头", 0) else "拉镜头"] = push_ratio
        if tracking_ratio >= self.tracking_vote_ratio:
            eligible["跟镜头"] = tracking_ratio
        if pan_tilt_ratio >= self.pan_tilt_vote_ratio:
            eligible["摇镜头" if pan_score >= tilt_score else "移镜头"] = pan_tilt_ratio
        # 固定镜头：绝对多数（>50%）时直接胜出；否则维持回退逻辑
        static_ratio = static_score / total_weight
        if static_ratio > 0.50:
            eligible["固定镜头"] = static_ratio
        if eligible:
            winner = max(eligible, key=eligible.get)
            confidence = eligible[winner]
        else:
            winner, confidence = "固定镜头", max(static_ratio, 0.5)

        direction_info = ""
        if winner == "摇镜头":
            direction_info = self._pan_direction(pair_results)
        elif winner == "移镜头":
            direction_info = self._tilt_direction(pair_results)
        elif winner == "跟镜头":
            direction_info = self._tracking_direction(pair_results)

        return winner, confidence, {
            "pair_votes": pair_votes,
            "pair_details": pair_details,
            "vote_weights": {k: round(v, 4) for k, v in counter.items()},
            "direction": direction_info or None,
        }

    def _pan_direction(self, pair_results: list) -> Optional[str]:
        """推断摇镜头的方向（左摇 / 右摇）。

        光流 dx>0 表示像素向右移 → 镜头向左摇。
        """
        pan_sum = 0.0
        for r in pair_results:
            pan_sum += r.get("pan_score", 0.0)
        if abs(pan_sum) < 0.1:
            return None
        return "left" if pan_sum > 0 else "right"

    def _tilt_direction(self, pair_results: list) -> Optional[str]:
        """推断移镜头的方向（上升 / 下降）。

        光流 dy>0 表示像素向下移 → 镜头向上升。
        """
        tilt_sum = 0.0
        for r in pair_results:
            tilt_sum += r.get("tilt_score", 0.0)
        if abs(tilt_sum) < 0.1:
            return None
        return "up" if tilt_sum > 0 else "down"

    def _tracking_direction(self, pair_results: list) -> Optional[str]:
        """推断跟镜头的方向（背景流反向 = 镜头跟随方向）。

        跟镜头中，背景运动方向与镜头运动方向相反：
        - 背景向右移 → 镜头向左跟 → "left"
        - 背景向下移 → 镜头向上跟 → "up"

        使用幅值加权避免主体静止区域稀释方向信号。
        """
        pan_sum = 0.0
        tilt_sum = 0.0
        for r in pair_results:
            dx = r.get("mean_dx", 0.0)
            dy = r.get("mean_dy", 0.0)
            mag = r.get("mean_mag", 0.0)
            # 用幅值加权：光流大的帧对方向判断更可信
            weight = max(mag, 0.01)
            pan_sum += dx * weight
            tilt_sum += dy * weight

        if abs(pan_sum) > abs(tilt_sum) and abs(pan_sum) >= 0.1:
            return "left" if pan_sum > 0 else "right"
        elif abs(tilt_sum) >= 0.1:
            return "up" if tilt_sum > 0 else "down"
        return None

    def _infer_direction(self, details: dict) -> Optional[str]:
        """从 details 中提取方向信息。"""
        return details.get("direction", None)

    # ---------- 工具 ----------

    def _static_result(self, reason: str = "") -> dict:
        return {
            "镜头运动": "固定镜头",
            "景别": "未知",
            "confidence": 0.0,
            "details": {"pair_votes": [], "pair_details": [], "reason": reason},
        }

    @property
    def version(self) -> str:
        return "1.1.0"


_global_analyzer = None


def analyze(video_path: str) -> dict:
    """便捷函数：使用全局单例分析视频镜头运动。"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = OpticalFlowAnalyzer()
    return _global_analyzer.analyze(video_path)


# ---------- CLI 测试入口 ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="基于光流法的镜头运动分析")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("--sample-interval", type=int, default=5, help="采样帧间隔（默认 5）")
    parser.add_argument("--with-direction", action="store_true", help="显示摇/移的方向")
    parser.add_argument("--details", action="store_true", help="输出每帧对的详细统计")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    analyzer = OpticalFlowAnalyzer(sample_interval=args.sample_interval)

    if args.with_direction:
        result = analyzer.analyze_with_direction(args.video)
    else:
        result = analyzer.analyze(args.video)

    output = {
        "镜头运动": result["镜头运动"],
        "景别": result["景别"],
        "confidence": result["confidence"],
    }
    if args.with_direction:
        output["direction"] = result.get("direction")

    if args.details:
        output["details"] = result.get("details", {})

    print(json.dumps(output, ensure_ascii=False, indent=2))
