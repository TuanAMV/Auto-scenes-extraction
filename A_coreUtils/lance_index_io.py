# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
# lance_index_io.py
# Lance 索引文件读写工具
# 替代 PKL + NPZ 缓存方案，使用 Lance 列式存储
#
# Lance 索引 Schema (每行一个场景):
#   video_path: string
#   start_frame: int32
#   end_frame: int32
#   fps: float32
#   feature_count: int32
#   features_blob: large_binary  (numpy float16 数组的原始字节)
#
# 元数据 (Lance metadata):
#   format: "scene_index_lance_v1"
#   total_scenes: str(int)
#   feature_dim: str(int)

import os
import threading
import numpy as np
import pyarrow as pa
import lance
from concurrent.futures import ThreadPoolExecutor


def _build_table(scene_rows):
    """从场景行列表构建 PyArrow Table"""
    video_paths = []
    start_frames = []
    end_frames = []
    fps_list = []
    feature_counts = []
    features_blobs = []

    for row in scene_rows:
        video_paths.append(row["video_path"])
        start_frames.append(row["start_frame"])
        end_frames.append(row["end_frame"])
        fps_list.append(row["fps"])
        feature_counts.append(row["feature_count"])
        features_blobs.append(row["features_blob"])

    table = pa.table(
        {
            "video_path": pa.array(video_paths, type=pa.utf8()),
            "start_frame": pa.array(start_frames, type=pa.int32()),
            "end_frame": pa.array(end_frames, type=pa.int32()),
            "fps": pa.array(fps_list, type=pa.float32()),
            "feature_count": pa.array(feature_counts, type=pa.int32()),
            "features_blob": pa.array(features_blobs, type=pa.large_binary()),
        }
    )
    return table


def scenes_to_lance_rows(video_path, scenes, fps):
    """
    将 process_video 返回的场景列表转换为 Lance 行列表

    Args:
        video_path: 视频文件路径
        scenes: process_video 返回的场景列表
            每个场景: {"start_frame": int, "end_frame": int, "features": [tensor, ...]}
        fps: 视频帧率

    Returns:
        list[dict]: Lance 行列表
    """
    import torch

    rows = []
    for scene in scenes:
        feats = scene.get("features", [])
        if not feats:
            continue

        np_feats = []
        for f in feats:
            if isinstance(f, torch.Tensor):
                np_feats.append(f.cpu().numpy().astype(np.float16))
            elif isinstance(f, np.ndarray):
                np_feats.append(f.astype(np.float16))
            else:
                np_feats.append(np.array(f, dtype=np.float16))

        if not np_feats:
            continue

        stacked = np.vstack(np_feats).astype(np.float16)
        rows.append(
            {
                "video_path": video_path,
                "start_frame": int(scene["start_frame"]),
                "end_frame": int(scene["end_frame"]),
                "fps": float(fps),
                "feature_count": stacked.shape[0],
                "features_blob": stacked.tobytes(),
            }
        )
    return rows


def append_lance_index(lance_path, scene_rows):
    """
    向已有 Lance 索引追加场景数据

    Args:
        lance_path: 已有的 .lance 目录路径
        scene_rows: 场景行列表
    """
    if not scene_rows:
        return

    table = _build_table(scene_rows)

    if not os.path.exists(lance_path):
        lance.write_dataset(table, lance_path)
    else:
        lance.write_dataset(table, lance_path, mode="append")


def get_indexed_video_paths(lance_path):
    """
    获取 Lance 索引中已索引的所有视频路径

    Args:
        lance_path: .lance 目录路径

    Returns:
        set: 已索引的视频路径集合
    """
    if not os.path.exists(lance_path):
        return set()
    try:
        ds = lance.dataset(lance_path)
        col = ds.to_table(columns=["video_path"]).column("video_path").to_pylist()
        return set(col)
    except Exception:
        return set()


def read_lance_index_raw(lance_path):
    """
    从 Lance 索引读取原始场景数据（替代 _load_single_pkl + NPZ 缓存）

    Args:
        lance_path: .lance 目录路径

    Returns:
        tuple: (features_np, scene_map, feature_counts)
            - features_np: np.ndarray [M, dim] 所有特征向量（float16）
            - scene_map: list[dict] 每个场景的元信息
            - feature_counts: list[int] 每个场景的向量数量
    """
    ds = lance.dataset(lance_path)
    table = ds.to_table()

    video_paths = table.column("video_path").to_pylist()
    start_frames = table.column("start_frame").to_pylist()
    end_frames = table.column("end_frame").to_pylist()
    fps_vals = table.column("fps").to_pylist()
    feat_counts = table.column("feature_count").to_pylist()
    feat_blobs = table.column("features_blob").to_pylist()

    all_features = []
    scene_map = []
    feature_counts = []

    for i in range(len(video_paths)):
        blob = feat_blobs[i]
        count = feat_counts[i]
        if count <= 0 or not blob:
            continue

        feats = np.frombuffer(blob, dtype=np.float16).copy()
        dim = feats.shape[0] // count
        feats = feats.reshape(count, dim)
        all_features.append(feats)

        scene_map.append(
            {
                "video_path": video_paths[i],
                "start_frame": int(start_frames[i]),
                "end_frame": int(end_frames[i]),
                "fps": float(fps_vals[i]),
                "source_lance": lance_path,
            }
        )
        feature_counts.append(count)

    if not all_features:
        return None, [], []

    features_np = np.vstack(all_features).astype(np.float16)
    return features_np, scene_map, feature_counts


def read_lance_index_for_reranker(lance_path, needed_scene_keys):
    """
    从 Lance 索引按需读取场景特征（用于 Reranker 等按需场景）

    Args:
        lance_path: .lance 目录路径
        needed_scene_keys: 需要的场景键集合，格式 "{video_path}_{start_frame}_{end_frame}"

    Returns:
        dict: {scene_key: np.ndarray} 场景特征向量字典
    """
    ds = lance.dataset(lance_path)
    table = ds.to_table()

    video_paths = table.column("video_path").to_pylist()
    start_frames = table.column("start_frame").to_pylist()
    end_frames = table.column("end_frame").to_pylist()
    feat_counts = table.column("feature_count").to_pylist()
    feat_blobs = table.column("features_blob").to_pylist()

    scene_features = {}
    for i in range(len(video_paths)):
        scene_key = f"{video_paths[i]}_{start_frames[i]}_{end_frames[i]}"
        if scene_key not in needed_scene_keys:
            continue

        blob = feat_blobs[i]
        count = feat_counts[i]
        if count <= 0 or not blob:
            continue

        feats = np.frombuffer(blob, dtype=np.float16).copy()
        dim = feats.shape[0] // count
        feats = feats.reshape(count, dim)
        scene_features[scene_key] = feats

    return scene_features


class AsyncLanceWriter:
    """异步 Lance 索引写入器 - 后台线程写入，GPU 不阻塞等 I/O"""

    def __init__(self, lance_path: str, max_workers: int = 2):
        self.lance_path = lance_path
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_futures = []
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending_count = 0
        self._error_count = 0
        self._write_count = 0

    def submit(self, scene_rows: list):
        """提交一批场景行到后台写入队列"""
        if not scene_rows:
            return None
        with self._lock:
            self._pending_count += 1
        future = self._executor.submit(self._write_task, scene_rows)
        future.add_done_callback(self._on_complete)
        with self._lock:
            self._pending_futures = [f for f in self._pending_futures if not f.done()]
            self._pending_futures.append(future)
        return future

    def _on_complete(self, future):
        with self._lock:
            self._pending_count -= 1
            if future.exception():
                self._error_count += 1

    def _write_task(self, scene_rows: list):
        """实际写入 Lance 索引（线程安全）"""
        with self._write_lock:
            append_lance_index(self.lance_path, scene_rows)
            self._write_count += 1

    def wait_all(self, timeout: float = None):
        """等待所有挂起的写入完成"""
        with self._lock:
            futures = list(self._pending_futures)
        for future in futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                print(f"[Error] 等待 Lance 写入时出错 - {e}")

    def shutdown(self):
        """等待所有写入完成并关闭线程池"""
        self.wait_all()
        self._executor.shutdown(wait=True)

    @property
    def write_count(self):
        return self._write_count
