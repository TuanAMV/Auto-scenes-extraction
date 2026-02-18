# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
# batch_text_search.py
# 批量文搜图引擎 - 预加载 + 批量计算优化版
# v1.0: 初始版本
# v1.1: 添加 LMDB 支持，批量事务写入磁盘，解决内存问题
# v1.2: 添加 prompt向量缓存支持，避免重复编码
#
# 优化点：
# 1. Lance特征预加载到GPU，常驻显存
# 2. 批量编码多个prompt
# 3. GPU批量矩阵计算
# 4. 支持跨Lance/跨视频的场景管理
# 5. 使用 LMDB 批量事务写入磁盘，避免内存溢出
# 6. 支持预计算的prompt向量缓存（无需重复编码和归一化）
import os
import sys
import gc
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Generator, Union, Any, Callable, Iterator, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
# 导入 LMDB（必需）
try:
    import lmdb

    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False
    lmdb = None
    raise ImportError("LMDB 未安装，请运行: pip install lmdb")


class LMDBCache:
    """
    LMDB 缓存封装类。
    优势：
    - 高性能读写
    - 零拷贝读取（内存映射）
    - 支持并发读取（无锁）
    - 支持断点续传（checkpoint 机制）
    - 支持按前缀查询/删除
    """
    # 特殊 key 前缀
    CHECKPOINT_KEY = '__checkpoint__'
    META_KEY = '__meta__'
    RESULT_PREFIX = 'result:'
    CANDIDATE_PREFIX = 'candidate:'

    def __init__(self, path: str, map_size: int = 10 * 1024 * 1024 * 1024):
        """
        初始化 LMDB 缓存。
        Args:
            path: 缓存目录路径
            map_size: 最大数据库大小（默认 10GB）
        """
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.env = lmdb.open(
            path,
            map_size=map_size,
            max_dbs=1,
            writemap=True,  # 使用写映射提高写入性能
            map_async=True,  # 异步刷新
        )
        self._closed = False

    def __setitem__(self, key: str, value):
        """写入数据。"""
        key_bytes = key.encode('utf-8')
        value_bytes = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        with self.env.begin(write=True) as txn:
            txn.put(key_bytes, value_bytes)

    def __getitem__(self, key: str):
        """读取数据。"""
        key_bytes = key.encode('utf-8')
        with self.env.begin(buffers=True) as txn:
            value_bytes = txn.get(key_bytes)
            if value_bytes is None:
                raise KeyError(key)

            return pickle.loads(value_bytes)

    def __contains__(self, key: str) -> bool:
        """检查键是否存在。"""
        key_bytes = key.encode('utf-8')
        with self.env.begin() as txn:
            return txn.get(key_bytes) is not None

    def get(self, key: str, default=None):
        """获取数据，不存在时返回默认值。"""
        try:
            return self[key]

        except KeyError:
            return default

    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """批量读取多个 key。"""
        results = {}
        if not keys:
            return results

        with self.env.begin(buffers=True) as txn:
            for key in keys:
                value_bytes = txn.get(key.encode('utf-8'))
                if value_bytes is not None:
                    results[key] = pickle.loads(value_bytes)
        return results

    def delete(self, key: str) -> bool:
        """删除指定 key，返回是否删除成功。"""
        key_bytes = key.encode('utf-8')
        with self.env.begin(write=True) as txn:
            return txn.delete(key_bytes)

    def delete_many(self, keys: List[str]) -> int:
        """批量删除多个 key，返回删除数量。"""
        if not keys:
            return 0

        deleted = 0
        with self.env.begin(write=True) as txn:
            for key in keys:
                if txn.delete(key.encode('utf-8')):
                    deleted += 1
        return deleted

    def put_many(self, items: Dict[str, Any]):
        """批量写入数据（单事务）。"""
        if not items:
            return
        with self.env.begin(write=True) as txn:
            for key, value in items.items():
                key_bytes = key.encode('utf-8')
                value_bytes = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                txn.put(key_bytes, value_bytes)

    def iterkeys(self):
        """迭代所有 key。"""
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                yield key.decode('utf-8')

    def keys(self):
        """返回所有 key 列表。"""
        return list(self.iterkeys())

    def keys_with_prefix(self, prefix: str):
        """返回指定前缀的所有 key。"""
        prefix_bytes = prefix.encode('utf-8')
        result = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix_bytes):
                for key, _ in cursor:
                    key_str = key.decode('utf-8')
                    if key_str.startswith(prefix):
                        result.append(key_str)
                    else:
                        break

        return result

    def items_with_prefix(self, prefix: str):
        """返回指定前缀的所有键值对。"""
        prefix_bytes = prefix.encode('utf-8')
        result = []
        with self.env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix_bytes):
                for key, value in cursor:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else bytes(key).decode('utf-8')
                    if key_str.startswith(prefix):
                        result.append((key_str, pickle.loads(value)))
                    else:
                        break

        return result

    def __len__(self):
        """返回记录数。"""
        with self.env.begin() as txn:
            return txn.stat()['entries']

    def clear_all(self):
        """清空所有数据。"""
        with self.env.begin(write=True) as txn:
            txn.drop(db=txn.db, delete=False)

    def save_checkpoint(self, data: dict):
        """保存断点信息。"""
        self[self.CHECKPOINT_KEY] = data

    def load_checkpoint(self) -> Optional[dict]:
        """加载断点信息，不存在返回 None。"""
        return self.get(self.CHECKPOINT_KEY)

    def save_meta(self, data: dict):
        """保存搜索元数据。"""
        self[self.META_KEY] = data

    def load_meta(self) -> Optional[dict]:
        """加载搜索元数据，不存在返回 None。"""
        return self.get(self.META_KEY)

    def put_result(self, scene_key: str, data: dict):
        """写入最终搜索结果。"""
        self[self.RESULT_PREFIX + scene_key] = data

    def put_results_many(self, results: Dict[str, dict]):
        """批量写入最终搜索结果。"""
        if not results:
            return
        data = {
            self.RESULT_PREFIX + scene_key: result_data
            for scene_key, result_data in results.items()
        }
        self.put_many(data)

    def get_result(self, scene_key: str) -> Optional[dict]:
        """获取最终搜索结果。"""
        return self.get(self.RESULT_PREFIX + scene_key)

    def get_all_results(self) -> Dict[str, dict]:
        """获取全部最终搜索结果。"""
        results = {}
        for key, value in self.items_with_prefix(self.RESULT_PREFIX):
            scene_key = key[len(self.RESULT_PREFIX):]
            results[scene_key] = value
        return results

    def put_candidates_many(self, candidates: Dict[str, dict]):
        """批量写入候选数据（自动添加 `candidate:` 前缀）。"""
        if not candidates:
            return
        data = {
            self.CANDIDATE_PREFIX + scene_key: candidate_data
            for scene_key, candidate_data in candidates.items()
        }
        self.put_many(data)

    def get_candidate(self, scene_key: str) -> Optional[dict]:
        return self.get(self.CANDIDATE_PREFIX + scene_key)

    def get_candidates_many(self, scene_keys: List[str]) -> Dict[str, dict]:
        if not scene_keys:
            return {}

        prefixed = [self.CANDIDATE_PREFIX + key for key in scene_keys]
        data = self.get_many(prefixed)
        out: Dict[str, dict] = {}
        for key, value in data.items():
            out[key[len(self.CANDIDATE_PREFIX):]] = value
        return out

    def candidate_keys(self) -> List[str]:
        keys = self.keys_with_prefix(self.CANDIDATE_PREFIX)
        return [key[len(self.CANDIDATE_PREFIX):] for key in keys]

    def clear_candidates(self):
        stale_keys = self.keys_with_prefix(self.CANDIDATE_PREFIX)
        if stale_keys:
            self.delete_many(stale_keys)

    def clear_results(self):
        stale_keys = self.keys_with_prefix(self.RESULT_PREFIX)
        if stale_keys:
            self.delete_many(stale_keys)

    def count_prefix(self, prefix: str) -> int:
        prefix_bytes = prefix.encode('utf-8')
        count = 0
        with self.env.begin() as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix_bytes):
                for key, _ in cursor:
                    key_str = key.decode('utf-8')
                    if key_str.startswith(prefix):
                        count += 1
                    else:
                        break

        return count

    def count_results(self) -> int:
        return self.count_prefix(self.RESULT_PREFIX)

    def iter_results_batches(self, batch_size: int = 1000) -> Iterator[List[Tuple[str, dict]]]:
        batch_size = max(1, int(batch_size))
        prefix_bytes = self.RESULT_PREFIX.encode('utf-8')
        with self.env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            if not cursor.set_range(prefix_bytes):
                return
            batch: List[Tuple[str, dict]] = []
            for key, value in cursor:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else bytes(key).decode('utf-8')
                if not key_str.startswith(self.RESULT_PREFIX):
                    break

                scene_key = key_str[len(self.RESULT_PREFIX):]
                batch.append((scene_key, pickle.loads(value)))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    def iter_items_with_prefix(self, prefix: str, batch_size: int = 1000) -> Iterator[List[Tuple[str, Any]]]:
        batch_size = max(1, int(batch_size))
        prefix_bytes = prefix.encode('utf-8')
        with self.env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            if not cursor.set_range(prefix_bytes):
                return
            batch: List[Tuple[str, Any]] = []
            for key, value in cursor:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else bytes(key).decode('utf-8')
                if not key_str.startswith(prefix):
                    break

                batch.append((key_str, pickle.loads(value)))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    def close(self):
        """关闭数据库。"""
        if not self._closed:
            self.env.close()
            self._closed = True

    def destroy(self):
        """关闭并删除整个 LMDB 目录。"""
        self.close()
        import shutil

        if os.path.exists(self.path):
            shutil.rmtree(self.path, ignore_errors=True)

    def __del__(self):
        """析构时关闭资源。"""
        self.close()


class LMDBResultView:
    """
    LMDB 结果流式视图，只暴露 result:*。
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._cache = LMDBCache(cache_dir)
        self._closed = False

    def iter_batches(self, batch_size: int = 1000) -> Iterator[List[Tuple[str, dict]]]:
        yield from self._cache.iter_results_batches(batch_size=batch_size)

    def __iter__(self) -> Iterator[Tuple[str, dict]]:
        for batch in self.iter_batches(batch_size=1000):
            for item in batch:
                yield item

    def count(self) -> int:
        return self._cache.count_results()

    def get(self, scene_key: str) -> Optional[dict]:
        return self._cache.get_result(scene_key)

    def to_dict(self) -> Dict[str, dict]:
        return self._cache.get_all_results()

    def close(self):
        if not self._closed:
            self._cache.close()
            self._closed = True

    def __len__(self):
        return self.count()

    def __del__(self):
        self.close()


# ============================================================
#  路径设置 - 确保能找到项目根目录模块


# ============================================================
_current_file = os.path.abspath(__file__)
_search_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_search_dir)
_cut_detect_scene_dir = os.path.dirname(_a_core_utils_dir)
if _cut_detect_scene_dir not in sys.path:
    sys.path.insert(0, _cut_detect_scene_dir)
# 导入路径解析器
from path_resolver import PathResolver
# 导入视频工具
from A_coreUtils.video_processing.video_utils import VideoMetaHelper, SceneFeatureExtractor
from A_coreUtils.lance_index_io import read_lance_index_raw
# 全局设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_BATCH_LOAD_WORKERS = 4
DEFAULT_BATCH_LMDB_WRITE_BATCH_SIZE = 1000
DEFAULT_BATCH_RERANK_TOP_K = 10
DEFAULT_BATCH_RERANK_BATCH_SIZE = 7
DEFAULT_BATCH_RESULT_FLUSH_THRESHOLD = 2000


def _normalize_optional_positive_int(
    value: Optional[Any],
    *,
    field_name: str = "value",
    default: Optional[int] = None
) -> Optional[int]:
    if default is not None and int(default) <= 0:
        raise ValueError(f"default for {field_name} must be positive or None, got {default!r}")

    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('', 'none', 'null', 'unlimited', 'default', 'auto', '默认', '不限'):
            return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be positive int or None, got {value!r}") from exc
    if parsed <= 0:
        return default
    return parsed


def _normalize_top_k(value: Optional[Any]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('', 'none', 'null', 'unlimited', 'default', 'auto', '默认', '不限'):
            return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"top_k must be int or None, got {value!r}") from exc
    return parsed if parsed > 0 else None


class BatchTextSearchEngine:
    """
    批量文搜图引擎：预加载 + 批量计算。
    优化点：
    1. Lance 特征预加载到 GPU，常驻显存。
    2. 批量加载 prompt 向量（可来自缓存）。
    3. GPU 批量矩阵计算。
    4. 支持跨 Lance / 跨视频的场景管理。
    使用方式：
    ```python
    engine = BatchTextSearchEngine(
        processor,
        index_paths,
        video_name_format="{主体}_{动作}_{起始帧}_{视频解析名}",
    )
    results = engine.search_with_batched_cache(cache_iterator, threshold=20.0)
    # 释放资源
    engine.cleanup()
    ```
    """

    def __init__(self,
                 processor=None,
                 index_paths: Optional[List[str]] = None,
                 cache_dir: str = None,
                 load_workers: Optional[int] = DEFAULT_BATCH_LOAD_WORKERS,
                 use_fp16: bool = True,
                 lance_batch_size: Optional[int] = None,
                 video_name_format: str = None,
                 search_mode: int = 0,
                 top_k: Optional[int] = None,
                 lmdb_write_batch_size: Optional[int] = DEFAULT_BATCH_LMDB_WRITE_BATCH_SIZE,
                 truncate_dim: Optional[int] = None,
                 logit_scale: float = 100.0):
        """
        初始化批量搜索引擎。
        Args:
            processor: `EmbeddingModelProcessor` 实例
            index_paths: Lance 索引文件路径列表
            cache_dir: 缓存目录
            load_workers: 索引加载线程数
            use_fp16: 是否使用 FP16 存储特征（降低显存）
            lance_batch_size: 每批加载的 Lance 数量
              - `None` 或 `>= Lance 总数`：一次性全部加载到 GPU
              - `< Lance 总数`：分批加载，用完释放
            video_name_format: 视频命名格式模板（必填）
            search_mode: 搜索模式
              - `-1`：按视频模式，每个视频独立返回 `top_k`
              - `0`：按 Lance 模式，每个 Lance 独立返回 `top_k`
              - `1`：全局模式，返回全局 `top_k`
            top_k: 每组返回的最大结果数；`None` 表示不限制
            lmdb_write_batch_size: LMDB 单事务写入批大小
        """
        if video_name_format is None:
            raise ValueError("video_name_format is required")

        self.processor = processor
        self.index_paths = index_paths or []
        self.load_workers = _normalize_optional_positive_int(
            load_workers,
            field_name='load_workers',
        )
        self.lmdb_write_batch_size = _normalize_optional_positive_int(
            lmdb_write_batch_size,
            field_name='lmdb_write_batch_size',
        )
        self.use_fp16 = use_fp16
        self.video_name_format = video_name_format
        self.search_mode = search_mode
        self.top_k = _normalize_top_k(top_k)
        self.truncate_dim = truncate_dim if truncate_dim is not None else getattr(processor, 'truncate_dim', None)
        self.logit_scale_value = float(logit_scale)
        lance_batch_size = _normalize_optional_positive_int(
            lance_batch_size,
            field_name='lance_batch_size',
        )
        # 处理 lance_batch_size：None 或 >= 总数表示全量加载
        total_lances = len(self.index_paths)
        if lance_batch_size is None or lance_batch_size >= total_lances:
            self.lance_batch_size = total_lances  # 全量加载
            self._preload_all = True
        else:
            self.lance_batch_size = lance_batch_size
            self._preload_all = False
        # 缓存目录
        if cache_dir is None:
            resolver = PathResolver()
            cache_dir = str(resolver.project_root / 'temp' / 'cache')
        self.cache_dir = cache_dir
        # 预加载特征（合并模式）
        self.all_features_gpu: Optional[torch.Tensor] = None  # [M, dim] GPU tensor
        self.scene_map: List[Dict] = []
        self.feature_counts: List[int] = []
        self.lance_boundaries: List[int] = []
        # GPU 索引缓存（用于 scatter_reduce）
        self.scene_indices_gpu: Optional[torch.Tensor] = None
        self.frame_indices_gpu: Optional[torch.Tensor] = None
        self.scene_frame_indices_gpu: Optional[torch.Tensor] = None
        self.num_scenes: int = 0
        # 按视频分组的数据结构（search_mode = -1 时使用）
        self.video_features: Dict[str, torch.Tensor] = {}      # {video_path: [N, dim] GPU tensor}
        self.video_scene_maps: Dict[str, List[Dict]] = {}      # {video_path: [scene_info, ...]}
        self.video_feature_counts: Dict[str, List[int]] = {}   # {video_path: [count, ...]}
        # logit_scale 缓存
        self._logit_scale: Optional[torch.Tensor] = torch.tensor(self.logit_scale_value, device=DEVICE, dtype=torch.float32)
        # 根据配置决定是否立即全量预加载
        if self._preload_all:
            self._preload_all_features()

    def _ensure_logit_scale(self):
        if self._logit_scale is None:
            self._logit_scale = torch.tensor(self.logit_scale_value, device=DEVICE, dtype=torch.float32)

    def _load_single_lance(self, lance_path: str) -> Tuple[Optional[np.ndarray], List[Dict], List[int]]:
        """
        从 Lance 索引直接读取特征（无需 NPZ 缓存层）
        Returns:
            tuple: (features_np, scene_map, feature_counts)
        """
        try:
            features_np, scene_map, feature_counts = read_lance_index_raw(lance_path)
            return features_np, scene_map, feature_counts

        except Exception as e:
            print(f"    [Error] 无法加载 Lance 索引 {os.path.basename(lance_path)}: {e}")
            return None, [], []

    def _preload_all_features(self):
        """
        按 Lance 分组预加载特征到 GPU。
        每个 Lance 独立存储，支持按 Lance 独立搜索（不跨 Lance 合并）。
        """
        # 静默：预加载开始时间
        start_time = time.time()
        # 按 Lance 分组存储
        self.lance_features: Dict[str, torch.Tensor] = {}  # {lance_path: features_gpu}
        self.lance_scene_maps: Dict[str, List[Dict]] = {}  # {lance_path: scene_maps}
        self.lance_feature_counts: Dict[str, List[int]] = {}  # {lance_path: feature_counts}
        self.lance_scene_indices: Dict[str, torch.Tensor] = {}  # {lance_path: scene_indices_gpu}
        total_scenes = 0
        total_vectors = 0
        total_memory_mb = 0.0
        # CPU 并行加载 Lance
        with ThreadPoolExecutor(max_workers=self.load_workers) as executor:
            futures = {executor.submit(self._load_single_lance, p): p for p in self.index_paths}
            for future in as_completed(futures):
                lance_path = futures[future]
                try:
                    features_np, scene_map, feature_counts = future.result()
                    if features_np is not None and len(scene_map) > 0:
                        # 检查维度截断
                        truncate_dim = self.truncate_dim
                        if truncate_dim is not None and features_np.shape[1] > truncate_dim:
                            features_np = features_np[:, :truncate_dim]
                            # 截断后重新归一化
                            norms = np.linalg.norm(features_np, axis=1, keepdims=True)
                            features_np = features_np / (norms + 1e-8)
                        # 送入 GPU
                        dtype = torch.float16 if self.use_fp16 else torch.float32
                        features_gpu = torch.tensor(features_np, device=DEVICE, dtype=dtype)
                        # 构建场景索引张量
                        scene_indices = []
                        for scene_idx, count in enumerate(feature_counts):
                            scene_indices.extend([scene_idx] * count)
                        scene_indices_gpu = torch.tensor(scene_indices, device=DEVICE, dtype=torch.long)
                        # 存储
                        self.lance_features[lance_path] = features_gpu
                        self.lance_scene_maps[lance_path] = scene_map
                        self.lance_feature_counts[lance_path] = feature_counts
                        self.lance_scene_indices[lance_path] = scene_indices_gpu
                        # 统计
                        total_scenes += len(scene_map)
                        total_vectors += features_np.shape[0]
                        total_memory_mb += features_gpu.element_size() * features_gpu.nelement() / (1024 * 1024)
                        # 静默：单个 Lance 加载信息
                except Exception as e:
                    print(f"[错误] {os.path.basename(lance_path)}: 加载失败 - {e}")
        if not self.lance_features:
            print("[警告] 没有加载到任何特征数据")
            return
        # 获取 logit_scale
        self._ensure_logit_scale()
        # 设置总场景数（兼容旧流程）
        self.num_scenes = total_scenes
        # 合并所有 Lance 特征到 all_features_gpu（用于 search_with_batched_cache）
        all_features_list = []
        all_scene_maps = []
        all_feature_counts = []
        all_scene_indices = []
        all_frame_indices = []  # 记录每个特征是场景中的第几帧（0=首帧, 1=中帧, 2=尾帧）
        scene_offset = 0
        for lance_path in self.lance_features:
            features_gpu = self.lance_features[lance_path]
            scene_map = self.lance_scene_maps[lance_path]
            feature_counts = self.lance_feature_counts[lance_path]
            all_features_list.append(features_gpu)
            all_scene_maps.extend(scene_map)
            all_feature_counts.extend(feature_counts)
            # 构建场景索引和帧索引（带偏移）
            for scene_idx, count in enumerate(feature_counts):
                all_scene_indices.extend([scene_offset + scene_idx] * count)
                all_frame_indices.extend(list(range(count)))
            scene_offset += len(scene_map)
        # 合并到单个大张量
        self.all_features_gpu = torch.cat(all_features_list, dim=0)
        self.scene_map = all_scene_maps
        self.feature_counts = all_feature_counts
        self.scene_indices_gpu = torch.tensor(all_scene_indices, device=DEVICE, dtype=torch.long)
        self.frame_indices_gpu = torch.tensor(all_frame_indices, device=DEVICE, dtype=torch.long)
        # 构建复合索引：scene_idx * 3 + frame_idx
        self.scene_frame_indices_gpu = self.scene_indices_gpu * 3 + self.frame_indices_gpu
        # 记录 Lance 数量（释放前）
        loaded_lance_count = len(self.lance_features) if hasattr(self, 'lance_features') else 0
        # 释放按 Lance 保留的 GPU 张量，避免与 all_features_gpu 双份占用显存
        if hasattr(self, 'lance_features'):
            for lance_path in list(self.lance_features.keys()):
                if self.lance_features[lance_path] is not None:
                    del self.lance_features[lance_path]
            self.lance_features.clear()
        if hasattr(self, 'lance_scene_indices'):
            for lance_path in list(self.lance_scene_indices.keys()):
                if self.lance_scene_indices[lance_path] is not None:
                    del self.lance_scene_indices[lance_path]
            self.lance_scene_indices.clear()
        del all_features_list
        load_time = time.time() - start_time
        print(f"[预加载] 已加载 {loaded_lance_count} 个Lance, {total_scenes} 个场景, {total_vectors} 个向量, 耗时 {load_time:.2f}s")

    def _preload_features_per_video(self):
        """
        按视频分组预加载特征到 GPU。
        与 `_preload_all_features()` 的区别：
        - 合并模式：所有视频场景合并到一个 GPU 张量
        - 按视频模式：每个视频独立存储并独立搜索
        """
        # 静默：预加载开始时间
        start_time = time.time()
        # 临时存储：{video_path: {'features': [], 'scenes': [], 'counts': []}}
        video_data = defaultdict(lambda: {'features': [], 'scenes': [], 'counts': []})
        # CPU 并行加载 Lance
        with ThreadPoolExecutor(max_workers=self.load_workers) as executor:
            futures = {executor.submit(self._load_single_lance, p): p for p in self.index_paths}
            for future in as_completed(futures):
                lance_path = futures[future]
                try:
                    features_np, scene_map, feature_counts = future.result()
                    if features_np is not None and len(scene_map) > 0:
                        # 按视频分组
                        feat_idx = 0
                        for scene_idx, scene_info in enumerate(scene_map):
                            video_path = scene_info['video_path']
                            count = feature_counts[scene_idx]
                            # 提取该场景特征
                            scene_features = features_np[feat_idx:feat_idx + count]
                            feat_idx += count
                            video_data[video_path]['features'].append(scene_features)
                            video_data[video_path]['scenes'].append(scene_info)
                            video_data[video_path]['counts'].append(count)
                        # 静默：单个 Lance 加载信息
                        pass

                except Exception as e:
                    print(f"[错误] {os.path.basename(lance_path)}: 加载失败 - {e}")
        if not video_data:
            print("[警告] 没有加载到任何特征数据")
            return
        # 按视频构建 GPU 张量
        dtype = torch.float16 if self.use_fp16 else torch.float32
        total_scenes = 0
        total_vectors = 0
        for video_path, vdata in video_data.items():
            if not vdata['features']:
                continue

            # 合并该视频的所有特征
            features_np = np.vstack(vdata['features']).astype(np.float32)
            # 维度截断
            truncate_dim = self.truncate_dim
            if truncate_dim is not None and features_np.shape[1] > truncate_dim:
                features_np = features_np[:, :truncate_dim]
                norms = np.linalg.norm(features_np, axis=1, keepdims=True)
                features_np = features_np / (norms + 1e-8)
            self.video_features[video_path] = torch.tensor(features_np, device=DEVICE, dtype=dtype)
            self.video_scene_maps[video_path] = vdata['scenes']
            self.video_feature_counts[video_path] = vdata['counts']
            total_scenes += len(vdata['scenes'])
            total_vectors += features_np.shape[0]
        # 获取 logit_scale
        self._ensure_logit_scale()
        load_time = time.time() - start_time
        print(f"[预加载] 已加载 {len(self.video_features)} 个视频, {total_scenes} 个场景, {total_vectors} 个向量, 耗时 {load_time:.2f}s")

    @staticmethod

    def _sanitize_filename(name: str) -> str:
        """清理文件名中的非法字符。"""
        if not name:
            return ""

        # 替换非法字符
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            name = name.replace(char, '_')
        return name.strip('_')

    def _format_video_name(self, meta: Dict, start_frame: int, video_parsed_name: str, video_path: str = None) -> str:
        import re

        if not isinstance(meta, dict):
            meta = {}
        placeholders = re.findall(r"\{(\w+)\}", self.video_name_format)
        format_dict = {ph: "" for ph in placeholders}
        if "起始帧" in format_dict:
            format_dict["起始帧"] = str(start_frame)
        if "视频解析名" in format_dict:
            format_dict["视频解析名"] = self._sanitize_filename(video_parsed_name)
        if "扩展名" in format_dict:
            extension = ""
            resolved_video_path = video_path or meta.get("video_path", "")
            if isinstance(resolved_video_path, str) and resolved_video_path:
                extension = os.path.splitext(resolved_video_path)[1]
            if extension.startswith("."):
                extension = extension[1:]
            format_dict["扩展名"] = self._sanitize_filename(extension)
        if "prompt_cn" in format_dict:
            format_dict["prompt_cn"] = self._sanitize_filename(str(meta.get("prompt_cn", "")))
        if "prompt_en" in format_dict:
            format_dict["prompt_en"] = self._sanitize_filename(str(meta.get("prompt_en", "")))
        pl_category_mapping = {
            "镜头": "lens_cn",
            "情绪": "mood_cn",
            "场景": "scene_cn",
            "主体": "subject_cn",
            "动作": "action_cn",
        }
        for category_name, meta_key in pl_category_mapping.items():
            if category_name in format_dict:
                format_dict[category_name] = self._sanitize_filename(str(meta.get(meta_key, "")))
        for key, value in meta.items():
            if key.endswith("_cn"):
                category_name = key[:-3]
                if category_name in format_dict:
                    format_dict[category_name] = self._sanitize_filename(str(value))
        labels = meta.get("labels", {})
        if isinstance(labels, dict):
            for subcat_name, label_tuple in labels.items():
                if subcat_name in format_dict and isinstance(label_tuple, (tuple, list)) and len(label_tuple) >= 1:
                    format_dict[subcat_name] = self._sanitize_filename(str(label_tuple[0]))
        result_name = self.video_name_format.format(**format_dict)
        separators = re.findall(r"\}([^\{]+)\{", self.video_name_format)
        for sep in set(separators):
            if not sep:
                continue

            double_sep = sep + sep
            while double_sep in result_name:
                result_name = result_name.replace(double_sep, sep)
            result_name = result_name.strip(sep)
        return result_name

    def search_with_batched_cache(self,
                                   cache_iterator,
                                   threshold: float = 20.0,
                                   initial_threshold: float = None,
                                   video_name_parser=None,
                                   use_diskcache: bool = True,
                                   cache_dir: str = None,
                                   rerank_top_k: Optional[int] = DEFAULT_BATCH_RERANK_TOP_K,
                                   use_reranker: bool = False,
                                   reranker=None,
                                   reranker_loader: Optional[Callable[[], Any]] = None,
                                   reranker_weight: float = 0.6,
                                   reranker_output_resolution: str = '384',
                                   rerank_batch_size: Optional[int] = DEFAULT_BATCH_RERANK_BATCH_SIZE,
                                   search_mode: int = 0,
                                   result_top_k: Optional[int] = None,
                                   config_hash: str = None,
                                   candidate_batch_size: Optional[int] = None,
                                   skip_clip_search: bool = False,
                                   append_to_lmdb_cache: bool = False,
                                   prompt_meta_lookup: Any = None,
                                   result_flush_threshold: Optional[int] = DEFAULT_BATCH_RESULT_FLUSH_THRESHOLD) -> LMDBResultView:
        """
        统一入口：执行批量搜索并返回 LMDBResultView（result:* 流式视图）。
        """
        rerank_top_k = _normalize_optional_positive_int(
            rerank_top_k,
            field_name='rerank_top_k',
        )
        rerank_batch_size = _normalize_optional_positive_int(
            rerank_batch_size,
            field_name='rerank_batch_size',
        )
        result_top_k = _normalize_top_k(result_top_k)
        candidate_batch_size = _normalize_optional_positive_int(
            candidate_batch_size,
            field_name='candidate_batch_size',
        )
        result_flush_threshold = _normalize_optional_positive_int(
            result_flush_threshold,
            field_name='result_flush_threshold',
        )
        return self._search_with_batched_cache_stream(
            cache_iterator=cache_iterator,
            threshold=threshold,
            initial_threshold=initial_threshold,
            video_name_parser=video_name_parser,
            use_diskcache=use_diskcache,
            cache_dir=cache_dir,
            rerank_top_k=rerank_top_k,
            use_reranker=use_reranker,
            reranker=reranker,
            reranker_loader=reranker_loader,
            reranker_weight=reranker_weight,
            reranker_output_resolution=reranker_output_resolution,
            rerank_batch_size=rerank_batch_size,
            search_mode=search_mode,
            result_top_k=result_top_k,
            config_hash=config_hash,
            candidate_batch_size=candidate_batch_size,
            skip_clip_search=skip_clip_search,
            append_to_lmdb_cache=append_to_lmdb_cache,
            prompt_meta_lookup=prompt_meta_lookup,
            result_flush_threshold=result_flush_threshold,
        )

    def _search_with_batched_cache_stream(self,
                                          cache_iterator,
                                          threshold: float = 20.0,
                                          initial_threshold: float = None,
                                          video_name_parser=None,
                                          use_diskcache: bool = True,
                                          cache_dir: str = None,
                                          rerank_top_k: Optional[int] = DEFAULT_BATCH_RERANK_TOP_K,
                                          use_reranker: bool = False,
                                          reranker=None,
                                          reranker_loader: Optional[Callable[[], Any]] = None,
                                          reranker_weight: float = 0.6,
                                          reranker_output_resolution: str = '384',
                                          rerank_batch_size: Optional[int] = DEFAULT_BATCH_RERANK_BATCH_SIZE,
                                          search_mode: int = 0,
                                          result_top_k: Optional[int] = None,
                                          config_hash: str = None,
                                          candidate_batch_size: Optional[int] = None,
                                          skip_clip_search: bool = False,
                                          append_to_lmdb_cache: bool = False,
                                          prompt_meta_lookup: Any = None,
                                          result_flush_threshold: Optional[int] = DEFAULT_BATCH_RESULT_FLUSH_THRESHOLD) -> LMDBResultView:
        import heapq
        rerank_top_k = _normalize_optional_positive_int(
            rerank_top_k,
            field_name='rerank_top_k',
        )
        rerank_batch_size = _normalize_optional_positive_int(
            rerank_batch_size,
            field_name='rerank_batch_size',
        )
        result_top_k = _normalize_top_k(result_top_k)
        candidate_batch_size = _normalize_optional_positive_int(
            candidate_batch_size,
            field_name='candidate_batch_size',
        )
        result_flush_threshold = _normalize_optional_positive_int(
            result_flush_threshold,
            field_name='result_flush_threshold',
        )

        reranker_loaded_by_current_call = False
        if not skip_clip_search and self.all_features_gpu is None:
            raise RuntimeError("特征未预加载，请先调用 _preload_all_features()")

        if append_to_lmdb_cache and cache_iterator is None:
            raise RuntimeError("append_to_lmdb_cache=True 需要提供 cache_iterator")

        if cache_dir is None:
            resolver = PathResolver()
            cache_dir = str(resolver.project_root / 'temp' / 'cache' / 'search_results')
        if not use_diskcache:
            cache_dir = os.path.join(cache_dir, f"memory_mode_{int(time.time() * 1000)}")
        os.makedirs(cache_dir, exist_ok=True)
        lmdb_cache = LMDBCache(cache_dir, map_size=10 * 1024 * 1024 * 1024)
        if skip_clip_search and cache_iterator is None:
            class _EmptyCacheIterator:
                total_prompts = 0
                num_batches = 0

                def __iter__(self):
                    return iter(())

            cache_iterator = _EmptyCacheIterator()
        write_batch_size = max(1, int(self.lmdb_write_batch_size or 0), 256)
        result_flush_threshold = max(write_batch_size, int(result_flush_threshold or 0), 256)
        clip_threshold = initial_threshold if initial_threshold is not None else threshold

        def _put_candidates_many_chunked(items: Dict[str, Any]):
            if not items:
                return
            item_list = list(items.items())
            for idx in range(0, len(item_list), write_batch_size):
                lmdb_cache.put_candidates_many(dict(item_list[idx:idx + write_batch_size]))

        def _put_results_many_chunked(items: Dict[str, Any]):
            if not items:
                return
            item_list = list(items.items())
            for idx in range(0, len(item_list), write_batch_size):
                lmdb_cache.put_results_many(dict(item_list[idx:idx + write_batch_size]))
        resume_from_batch = 0
        if not skip_clip_search and not append_to_lmdb_cache:
            checkpoint = lmdb_cache.load_checkpoint() if config_hash is not None else None
            if checkpoint is not None and checkpoint.get('config_hash') == config_hash:
                checkpoint_phase = checkpoint.get('phase', '')
                if checkpoint_phase == 'completed' and lmdb_cache.count_results() > 0:
                    print("[分批搜索-断点续传] 命中已完成缓存，直接返回结果视图")
                    lmdb_cache.close()
                    return LMDBResultView(cache_dir)

                resume_from_batch = int(checkpoint.get('last_completed_batch', -1) or -1) + 1
            else:
                lmdb_cache.clear_candidates()
                lmdb_cache.clear_results()
        if (not skip_clip_search) and (not append_to_lmdb_cache) and config_hash is not None:
            lmdb_cache.save_meta({
                'config_hash': config_hash,
                'threshold': threshold,
                'clip_threshold': clip_threshold,
                'search_mode': search_mode,
                'result_top_k': result_top_k,
                'total_prompts': getattr(cache_iterator, 'total_prompts', 0),
                'total_batches': getattr(cache_iterator, 'num_batches', 0),
            })
        start_time = time.time()
        if not skip_clip_search:
            compute_dtype = self.all_features_gpu.dtype
            device = self.all_features_gpu.device
            total_prompts = getattr(cache_iterator, 'total_prompts', 0)
            total_batches = getattr(cache_iterator, 'num_batches', 0)
            print(f"[分批搜索] 开始 CLIP 候选聚合: {total_prompts} prompts / {total_batches} batches")
            for batch_vectors, _, _, batch_info in cache_iterator:
                batch_idx = int(batch_info['batch_idx'])
                if batch_idx < resume_from_batch:
                    continue

                batch_start = int(batch_info['start_idx'])
                batch_end = int(batch_info['end_idx'])
                actual_batch_size = batch_end - batch_start
                if batch_vectors.device != device:
                    batch_vectors = batch_vectors.to(device)
                if batch_vectors.dtype != compute_dtype:
                    batch_vectors = batch_vectors.to(compute_dtype)
                with torch.no_grad():
                    similarities = self._logit_scale * batch_vectors @ self.all_features_gpu.T
                    num_scene_frames = self.num_scenes * 3
                    scene_frame_max_sims = torch.full(
                        (actual_batch_size, num_scene_frames),
                        float('-inf'),
                        device=device,
                        dtype=compute_dtype
                    )
                    expanded_sf_indices = self.scene_frame_indices_gpu.unsqueeze(0).expand(actual_batch_size, -1)
                    scene_frame_max_sims.scatter_reduce_(1, expanded_sf_indices, similarities, reduce='amax')
                    scene_frame_max_sims = scene_frame_max_sims.view(actual_batch_size, self.num_scenes, 3)
                    scene_max_sims, best_frame_indices = scene_frame_max_sims.max(dim=2)
                    scene_prompt_sims = scene_max_sims.T
                    best_frame_per_scene = best_frame_indices.T
                    scene_prompt_sims_cpu = scene_prompt_sims.float().cpu().numpy()
                    best_frame_per_scene_cpu = best_frame_per_scene.cpu().numpy()
                batch_candidate_updates: Dict[str, dict] = {}
                for scene_idx in range(self.num_scenes):
                    scene_sims = scene_prompt_sims_cpu[scene_idx]
                    scene_best_frames = best_frame_per_scene_cpu[scene_idx]
                    above_threshold_mask = scene_sims >= clip_threshold
                    if not above_threshold_mask.any():
                        continue

                    above_indices = np.where(above_threshold_mask)[0]
                    scene_info = self.scene_map[scene_idx]
                    video_path = scene_info['video_path']
                    start_frame = scene_info['start_frame']
                    end_frame = scene_info['end_frame']
                    video_name = os.path.basename(video_path) if video_path else ''
                    scene_key = f"{start_frame}_{video_name}"
                    existing = lmdb_cache.get_candidate(scene_key)
                    if existing is None:
                        candidates: List[Tuple[float, int, int]] = []
                        best_frame_idx = 1
                        scene_data = {
                            'video_path': video_path,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'fps': scene_info.get('fps', 25.0),
                            'source_lance': scene_info.get('source_lance', 'unknown'),
                        }
                    else:
                        candidates = existing.get('candidates', [])
                        best_frame_idx = int(existing.get('best_frame_idx', 1) or 1)
                        scene_data = {
                            'video_path': existing.get('video_path', video_path),
                            'start_frame': existing.get('start_frame', start_frame),
                            'end_frame': existing.get('end_frame', end_frame),
                            'fps': existing.get('fps', 25.0),
                            'source_lance': existing.get('source_lance', scene_info.get('source_lance', 'unknown')),
                        }
                    for batch_prompt_idx in above_indices:
                        similarity = float(scene_sims[batch_prompt_idx])
                        best_frame_for_prompt = int(scene_best_frames[batch_prompt_idx])
                        global_prompt_idx = batch_start + int(batch_prompt_idx)
                        # light candidate tuple: (sim, prompt_idx, frame_idx)
                        if rerank_top_k is None:
                            candidates.append((similarity, global_prompt_idx, best_frame_for_prompt))
                        elif len(candidates) < rerank_top_k:
                            heapq.heappush(candidates, (similarity, global_prompt_idx, best_frame_for_prompt))
                        elif similarity > candidates[0][0]:
                            heapq.heapreplace(candidates, (similarity, global_prompt_idx, best_frame_for_prompt))
                    if candidates:
                        best_candidate = max(candidates, key=lambda x: x[0])
                        best_frame_idx = int(best_candidate[2])
                    batch_candidate_updates[scene_key] = {
                        'candidates': candidates,
                        'best_frame_idx': best_frame_idx,
                        'video_path': scene_data['video_path'],
                        'start_frame': scene_data['start_frame'],
                        'end_frame': scene_data['end_frame'],
                        'fps': scene_data.get('fps', 25.0),
                        'source_lance': scene_data.get('source_lance', 'unknown'),
                    }
                if batch_candidate_updates:
                    _put_candidates_many_chunked(batch_candidate_updates)
                del batch_vectors, similarities, scene_frame_max_sims, scene_max_sims, scene_prompt_sims
                del best_frame_indices, best_frame_per_scene
                if batch_idx % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if (not append_to_lmdb_cache) and config_hash is not None:
                    lmdb_cache.save_checkpoint({
                        'config_hash': config_hash,
                        'last_completed_batch': batch_idx,
                        'total_batches': total_batches,
                        'total_prompts': total_prompts,
                        'phase': 'clip_search',
                        'timestamp': time.time(),
                    })
                if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                    elapsed = time.time() - start_time
                    progress = (batch_idx + 1) / max(1, total_batches) * 100
                    print(f"  [CLIP批次 {batch_idx + 1}/{total_batches}] {progress:.1f}%, {elapsed:.1f}s")
        clip_time = time.time() - start_time
        if append_to_lmdb_cache:
            lmdb_cache.close()
            print(f"[分批搜索] append 模式完成，耗时 {clip_time:.2f}s")
            return LMDBResultView(cache_dir)

        def _sort_candidate_keys_by_video_path(keys: List[str]) -> List[str]:
            if not keys:
                return keys

            ordered = []
            chunk_size = max(256, write_batch_size)
            for offset in range(0, len(keys), chunk_size):
                chunk_keys = keys[offset:offset + chunk_size]
                chunk_data = lmdb_cache.get_candidates_many(chunk_keys)
                for scene_key in chunk_keys:
                    data = chunk_data.get(scene_key, {})
                    video_path = data.get('video_path', '')
                    start_frame = int(data.get('start_frame', 0) or 0)
                    ordered.append((video_path, start_frame, scene_key))
            ordered.sort(key=lambda item: (item[0], item[1], item[2]))
            return [item[2] for item in ordered]

        candidate_keys = lmdb_cache.candidate_keys()
        candidate_keys = _sort_candidate_keys_by_video_path(candidate_keys)
        if not candidate_keys:
            lmdb_cache.clear_results()
            if config_hash is not None:
                lmdb_cache.save_checkpoint({
                    'config_hash': config_hash,
                    'last_completed_batch': getattr(cache_iterator, 'num_batches', 0) - 1,
                    'total_batches': getattr(cache_iterator, 'num_batches', 0),
                    'total_prompts': getattr(cache_iterator, 'total_prompts', 0),
                    'phase': 'completed',
                    'result_count': 0,
                    'timestamp': time.time(),
                })
            lmdb_cache.close()
            print(f"[分批搜索] 完成! 总耗时 {time.time() - start_time:.2f}s, 0 个场景")
            return LMDBResultView(cache_dir)

        # Reranker 延迟到帧提取完成后再加载，避免与 decord 同时占用大量内存
        # CLIP 搜索已完成，释放 GPU 特征张量，为 Reranker 腾出显存/内存
        if self.all_features_gpu is not None:
            del self.all_features_gpu
            self.all_features_gpu = None
        if hasattr(self, 'scene_indices_gpu') and self.scene_indices_gpu is not None:
            del self.scene_indices_gpu
            self.scene_indices_gpu = None
        if hasattr(self, 'frame_indices_gpu') and self.frame_indices_gpu is not None:
            del self.frame_indices_gpu
            self.frame_indices_gpu = None
        if hasattr(self, 'scene_frame_indices_gpu') and self.scene_frame_indices_gpu is not None:
            del self.scene_frame_indices_gpu
            self.scene_frame_indices_gpu = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        lmdb_cache.clear_results()
        grouped_result_heaps = defaultdict(list) if (result_top_k is not None and int(result_top_k) > 0) else None
        grouped_selected_results: Dict[str, dict] = {}
        result_buffer: Dict[str, dict] = {}
        video_parsed_cache: Dict[str, str] = {}
        prompt_meta_cache: Dict[int, dict] = {}

        def _load_prompt_meta(indices: Iterable[int]) -> Dict[int, dict]:
            unique_indices = sorted({int(i) for i in indices if i is not None and int(i) >= 0})
            missing = [idx for idx in unique_indices if idx not in prompt_meta_cache]
            fetched: Dict[int, dict] = {}
            if missing:
                if prompt_meta_lookup is not None:
                    if hasattr(prompt_meta_lookup, 'get_many'):
                        fetched = prompt_meta_lookup.get_many(missing) or {}
                    elif hasattr(prompt_meta_lookup, 'get_metadata_by_indices'):
                        fetched = prompt_meta_lookup.get_metadata_by_indices(missing) or {}
                if (not fetched) and cache_iterator is not None and hasattr(cache_iterator, 'get_metadata_by_indices'):
                    fetched = cache_iterator.get_metadata_by_indices(missing) or {}
                for idx in missing:
                    data = fetched.get(idx, {})
                    if not isinstance(data, dict):
                        data = {}
                    prompt_meta_cache[idx] = data
            return {idx: prompt_meta_cache.get(idx, {}) for idx in unique_indices}

        def _resolve_group_key(scene_key: str, result_data: dict) -> str:
            if search_mode == -1:
                return result_data.get('video_path', '') or 'unknown_video'

            if search_mode == 0:
                return result_data.get('source_lance', 'unknown')

            return '__global__'

        def _flush_result_buffer(force: bool = False):
            if not result_buffer:
                return
            if force or len(result_buffer) >= result_flush_threshold:
                _put_results_many_chunked(result_buffer)
                result_buffer.clear()

        def _add_result(scene_key: str, result_data: dict):
            if grouped_result_heaps is None:
                result_buffer[scene_key] = result_data
                _flush_result_buffer(force=False)
                return
            group_key = _resolve_group_key(scene_key, result_data)
            group_heap = grouped_result_heaps[group_key]
            sim = float(result_data.get('similarity', float('-inf')))
            heap_item = (sim, scene_key)
            if len(group_heap) < int(result_top_k):
                heapq.heappush(group_heap, heap_item)
                grouped_selected_results[scene_key] = result_data
            elif sim > group_heap[0][0]:
                _, old_scene_key = heapq.heapreplace(group_heap, heap_item)
                if old_scene_key != scene_key:
                    grouped_selected_results.pop(old_scene_key, None)
                grouped_selected_results[scene_key] = result_data

        def _get_video_parsed_name(video_path: str) -> str:
            if video_path in video_parsed_cache:
                return video_parsed_cache[video_path]

            video_name = os.path.basename(video_path) if video_path else ''
            video_parsed_name = ''
            if video_path and video_name_parser:
                try:
                    parsed = video_name_parser.parse_filename(video_path)
                    if parsed:
                        video_parsed_name = parsed.rstrip('_')
                except Exception:
                    video_parsed_name = os.path.splitext(video_name)[0]
            elif video_path:
                video_parsed_name = os.path.splitext(video_name)[0]
            video_parsed_cache[video_path] = video_parsed_name
            return video_parsed_name

        def _build_result_data(scene_info: dict,
                               best_sim: float,
                               best_prompt_idx: int,
                               prompt_meta_map: Dict[int, dict]) -> dict:
            best_meta = prompt_meta_map.get(int(best_prompt_idx), {})
            if not isinstance(best_meta, dict):
                best_meta = {}
            video_parsed_name = _get_video_parsed_name(scene_info['video_path'])
            result_name = self._format_video_name(
                best_meta,
                int(scene_info['start_frame']),
                video_parsed_name,
                scene_info.get('video_path')
            )
            result_data = {
                'result_name': result_name,
                'similarity': float(best_sim),
                'video_path': scene_info['video_path'],
                'start_frame': int(scene_info['start_frame']),
                'end_frame': int(scene_info['end_frame']),
                'fps': float(scene_info.get('fps', 25.0) or 25.0),
                'source_lance': scene_info.get('source_lance', 'unknown'),
                'prompt_idx': int(best_prompt_idx),
            }
            if 'prompt' in best_meta:
                result_data['prompt'] = best_meta.get('prompt', '')
            for key, value in best_meta.items():
                if key in ('prompt_cn', 'prompt_en', 'labels') or key.endswith('_cn') or key.endswith('_en'):
                    result_data[key] = value
            return result_data

        frame_extractor = None
        if use_reranker:
            from A_coreUtils.search.reranker_frame_extractor import RerankerFrameExtractor

            resolver = PathResolver()
            rerank_frames_dir = str(resolver.project_root / 'temp' / 'cache' / 'rerank_frames')
            frame_extractor = RerankerFrameExtractor(
                output_resolution=reranker_output_resolution,
                cache_dir=rerank_frames_dir
            )
        try:
            if candidate_batch_size is not None and int(candidate_batch_size) > 0:
                batch_size = int(candidate_batch_size)
            else:
                batch_size = len(candidate_keys)
            total_keys = len(candidate_keys)
            num_batches = (total_keys + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                batch_start_idx = batch_idx * batch_size
                batch_end_idx = min(batch_start_idx + batch_size, total_keys)
                batch_keys = candidate_keys[batch_start_idx:batch_end_idx]
                batch_candidate_data = lmdb_cache.get_candidates_many(batch_keys)
                if not batch_candidate_data:
                    continue

                prompt_indices = []
                for data in batch_candidate_data.values():
                    for candidate in data.get('candidates', []):
                        if len(candidate) >= 2:
                            prompt_indices.append(int(candidate[1]))
                prompt_meta_map = _load_prompt_meta(prompt_indices)
                frame_paths = {}
                if frame_extractor is not None:
                    scenes_to_extract = []
                    for data in batch_candidate_data.values():
                        scene_info = {
                            'video_path': data.get('video_path', ''),
                            'start_frame': int(data.get('start_frame', 0) or 0),
                            'end_frame': int(data.get('end_frame', 0) or 0),
                            'fps': float(data.get('fps', 25.0) or 25.0),
                        }
                        best_frame_idx = int(data.get('best_frame_idx', 1) or 1)
                        scenes_to_extract.append({
                            'video_path': scene_info['video_path'],
                            'start_frame': scene_info['start_frame'],
                            'end_frame': scene_info['end_frame'],
                            'fps': scene_info['fps'],
                            'target_frame_idx': best_frame_idx,
                        })
                    frame_paths = frame_extractor.extract_batch(scenes_to_extract)
                # 帧提取完成后再懒加载 Reranker，避免与 decord 同时占用大量内存
                if use_reranker and reranker is None and callable(reranker_loader):
                    reranker = reranker_loader()
                    reranker_loaded_by_current_call = reranker is not None
                    if reranker is None:
                        raise RuntimeError("Reranker 加载失败: reranker_loader 返回 None")
                for scene_key, data in batch_candidate_data.items():
                    candidates = data.get('candidates', [])
                    if not candidates:
                        continue

                    scene_info = {
                        'video_path': data.get('video_path', ''),
                        'start_frame': int(data.get('start_frame', 0) or 0),
                        'end_frame': int(data.get('end_frame', 0) or 0),
                        'fps': float(data.get('fps', 25.0) or 25.0),
                        'source_lance': data.get('source_lance', 'unknown'),
                    }
                    best_sim = float('-inf')
                    best_prompt_idx = None
                    if frame_extractor is None:
                        best_candidate = max(candidates, key=lambda x: x[0])
                        best_sim = float(best_candidate[0])
                        best_prompt_idx = int(best_candidate[1])
                    else:
                        frame_key = f"{scene_info['start_frame']}_{os.path.basename(scene_info['video_path'])}"
                        frame_path = frame_paths.get(frame_key)
                        before_best_candidate = max(candidates, key=lambda x: x[0])
                        before_best_sim = float(before_best_candidate[0])
                        before_best_prompt_idx = int(before_best_candidate[1])
                        if not frame_path or not os.path.exists(frame_path):
                            best_sim = before_best_sim
                            best_prompt_idx = before_best_prompt_idx
                        else:
                            sorted_candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
                            documents = []
                            for candidate in sorted_candidates:
                                prompt_idx = int(candidate[1])
                                meta = prompt_meta_map.get(prompt_idx, {})
                                documents.append({'text': meta.get('prompt', '')})
                            try:
                                rerank_scores = reranker.process(
                                    {'query': {'image': frame_path}, 'documents': documents},
                                    batch_size=(rerank_batch_size if rerank_batch_size is not None else max(1, len(documents)))
                                )
                                for candidate, rerank_score in zip(sorted_candidates, rerank_scores):
                                    clip_sim = float(candidate[0])
                                    prompt_idx = int(candidate[1])
                                    final_score = clip_sim * (1 - reranker_weight) + float(rerank_score) * 100.0 * reranker_weight
                                    if final_score > best_sim:
                                        best_sim = final_score
                                        best_prompt_idx = prompt_idx
                                # 打印 Reranker 前后最佳 prompt 对比
                                before_meta = prompt_meta_map.get(before_best_prompt_idx, {})
                                after_meta = prompt_meta_map.get(best_prompt_idx, {})
                                before_prompt_text = before_meta.get('prompt', f'idx={before_best_prompt_idx}')
                                after_prompt_text = after_meta.get('prompt', f'idx={best_prompt_idx}')
                                if best_prompt_idx != before_best_prompt_idx:
                                    print(f"    [Reranker] {scene_key}: "
                                          f"前=[{before_best_prompt_idx}] {before_prompt_text[:60]} (sim={before_best_sim:.1f}) → "
                                          f"后=[{best_prompt_idx}] {after_prompt_text[:60]} (sim={best_sim:.1f})")
                                else:
                                    print(f"    [Reranker] {scene_key}: "
                                          f"最佳prompt不变 [{best_prompt_idx}] {after_prompt_text[:60]} "
                                          f"(sim: {before_best_sim:.1f} → {best_sim:.1f})")
                            except Exception as e:
                                print(f"  [Reranker错误] 场景 {scene_key} 推理失败: {e}")
                                best_sim = before_best_sim
                                best_prompt_idx = before_best_prompt_idx
                    if best_prompt_idx is None or best_sim < threshold:
                        continue

                    result_data = _build_result_data(
                        scene_info=scene_info,
                        best_sim=best_sim,
                        best_prompt_idx=best_prompt_idx,
                        prompt_meta_map=prompt_meta_map
                    )
                    _add_result(scene_key, result_data)
                if frame_extractor is not None:
                    frame_extractor.cleanup(remove_files=False)
                print(f"  [候选处理] 批次 {batch_idx + 1}/{num_batches} 完成, 已处理 {batch_end_idx}/{total_keys}")
        finally:
            if frame_extractor is not None:
                frame_extractor.cleanup(remove_files=True)
        if grouped_result_heaps is None:
            _flush_result_buffer(force=True)
        else:
            _put_results_many_chunked(grouped_selected_results)
            if search_mode == -1:
                print(f"[分批搜索] 按视频模式过滤: {len(grouped_result_heaps)} 个视频, 每视频最多 {result_top_k} 个场景")
            elif search_mode == 0:
                print(f"[分批搜索] 按Lance模式过滤: {len(grouped_result_heaps)} 个Lance索引, 每Lance最多 {result_top_k} 个场景")
            else:
                print(f"[分批搜索] 全局模式过滤: 最多 {result_top_k} 个场景")
        if reranker_loaded_by_current_call and reranker is not None:
            try:
                reranker.cleanup()
            except Exception as e:
                print(f"[警告] 释放懒加载 Reranker 失败: {e}")
        result_count = lmdb_cache.count_results()
        if config_hash is not None:
            lmdb_cache.save_checkpoint({
                'config_hash': config_hash,
                'last_completed_batch': getattr(cache_iterator, 'num_batches', 0) - 1,
                'total_batches': getattr(cache_iterator, 'num_batches', 0),
                'total_prompts': getattr(cache_iterator, 'total_prompts', 0),
                'phase': 'completed',
                'result_count': result_count,
                'timestamp': time.time(),
            })
        total_time = time.time() - start_time
        print(f"[分批搜索] 完成! 总耗时 {total_time:.2f}s, {result_count} 个场景")
        lmdb_cache.close()
        return LMDBResultView(cache_dir)

    def _load_lance_batch_to_merged(self, lance_paths: List[str]):
        """
        将一批 Lance 加载并合并到 `all_features_gpu`（用于分批 Lance 搜索）。
        逻辑与 `_preload_all_features()` 类似，但仅加载指定 Lance 子集。
        调用后可直接执行 `search_with_batched_cache()`，搜索完成后调用
        `_unload_merged_features()` 释放显存。
        Args:
            lance_paths: 要加载的 Lance 文件路径列表
        """
        start_time = time.time()
        # 临时存储
        batch_lance_features = {}
        batch_lance_scene_maps = {}
        batch_lance_feature_counts = {}
        total_scenes = 0
        total_vectors = 0
        # CPU 并行加载 Lance
        with ThreadPoolExecutor(max_workers=self.load_workers) as executor:
            futures = {executor.submit(self._load_single_lance, p): p for p in lance_paths}
            for future in as_completed(futures):
                lance_path = futures[future]
                try:
                    features_np, scene_map, feature_counts = future.result()
                    if features_np is not None and len(scene_map) > 0:
                        # 检查维度截断
                        truncate_dim = self.truncate_dim
                        if truncate_dim is not None and features_np.shape[1] > truncate_dim:
                            features_np = features_np[:, :truncate_dim]
                            norms = np.linalg.norm(features_np, axis=1, keepdims=True)
                            features_np = features_np / (norms + 1e-8)
                        # 送入 GPU
                        dtype = torch.float16 if self.use_fp16 else torch.float32
                        features_gpu = torch.tensor(features_np, device=DEVICE, dtype=dtype)
                        batch_lance_features[lance_path] = features_gpu
                        batch_lance_scene_maps[lance_path] = scene_map
                        batch_lance_feature_counts[lance_path] = feature_counts
                        total_scenes += len(scene_map)
                        total_vectors += features_np.shape[0]
                except Exception as e:
                    print(f"[错误] {os.path.basename(lance_path)}: 加载失败 - {e}")
        if not batch_lance_features:
            print("[警告] 本批次没有加载到任何特征数据")
            return
        # 获取 logit_scale（如果还没有）
        if self._logit_scale is None:
            self._ensure_logit_scale()
        # 设置总场景数
        self.num_scenes = total_scenes
        # 合并所有 Lance 特征到 all_features_gpu
        all_features_list = []
        all_scene_maps = []
        all_feature_counts = []
        all_scene_indices = []
        all_frame_indices = []
        scene_offset = 0
        # 同时更新 lance_features 等字典（兼容旧逻辑）
        self.lance_features = batch_lance_features
        self.lance_scene_maps = batch_lance_scene_maps
        self.lance_feature_counts = batch_lance_feature_counts
        for lance_path in batch_lance_features:
            features_gpu = batch_lance_features[lance_path]
            scene_map = batch_lance_scene_maps[lance_path]
            feature_counts = batch_lance_feature_counts[lance_path]
            all_features_list.append(features_gpu)
            all_scene_maps.extend(scene_map)
            all_feature_counts.extend(feature_counts)
            for scene_idx, count in enumerate(feature_counts):
                all_scene_indices.extend([scene_offset + scene_idx] * count)
                all_frame_indices.extend(list(range(count)))
            scene_offset += len(scene_map)
        # 合并到单个大张量
        self.all_features_gpu = torch.cat(all_features_list, dim=0)
        self.scene_map = all_scene_maps
        self.feature_counts = all_feature_counts
        self.scene_indices_gpu = torch.tensor(all_scene_indices, device=DEVICE, dtype=torch.long)
        self.frame_indices_gpu = torch.tensor(all_frame_indices, device=DEVICE, dtype=torch.long)
        self.scene_frame_indices_gpu = self.scene_indices_gpu * 3 + self.frame_indices_gpu
        load_time = time.time() - start_time
        print(f"[分批Lance加载] {len(batch_lance_features)} 个Lance, {total_scenes} 个场景, {total_vectors} 个向量, 耗时 {load_time:.2f}s")

    def _unload_merged_features(self):
        """
        释放合并特征数据（用于分批 Lance 搜索时的显存回收）。
        """
        if self.all_features_gpu is not None:
            del self.all_features_gpu
            self.all_features_gpu = None
        if self.scene_indices_gpu is not None:
            del self.scene_indices_gpu
            self.scene_indices_gpu = None
        if hasattr(self, 'frame_indices_gpu') and self.frame_indices_gpu is not None:
            del self.frame_indices_gpu
            self.frame_indices_gpu = None
        if hasattr(self, 'scene_frame_indices_gpu') and self.scene_frame_indices_gpu is not None:
            del self.scene_frame_indices_gpu
            self.scene_frame_indices_gpu = None
        # 释放 lance_features 中的 GPU 张量
        if hasattr(self, 'lance_features'):
            for lance_path in list(self.lance_features.keys()):
                if self.lance_features[lance_path] is not None:
                    del self.lance_features[lance_path]
            self.lance_features = {}
        if hasattr(self, 'lance_scene_indices'):
            for lance_path in list(self.lance_scene_indices.keys()):
                if self.lance_scene_indices[lance_path] is not None:
                    del self.lance_scene_indices[lance_path]
            self.lance_scene_indices = {}
        self.scene_map = []
        self.feature_counts = []
        self.num_scenes = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cleanup(self):
        """释放 GPU/CPU 资源。"""
        if hasattr(self, "lance_features") and self.lance_features:
            self.lance_features.clear()
        if hasattr(self, "lance_scene_indices"):
            self.lance_scene_indices.clear()
        if hasattr(self, "lance_scene_maps"):
            self.lance_scene_maps.clear()
        if hasattr(self, "lance_feature_counts"):
            self.lance_feature_counts.clear()
        if self.all_features_gpu is not None:
            del self.all_features_gpu
            self.all_features_gpu = None
        if self.scene_indices_gpu is not None:
            del self.scene_indices_gpu
            self.scene_indices_gpu = None
        if self.frame_indices_gpu is not None:
            del self.frame_indices_gpu
            self.frame_indices_gpu = None
        if self.scene_frame_indices_gpu is not None:
            del self.scene_frame_indices_gpu
            self.scene_frame_indices_gpu = None
        if self.video_features:
            self.video_features.clear()
        self.video_scene_maps.clear()
        self.video_feature_counts.clear()
        if self._logit_scale is not None:
            del self._logit_scale
            self._logit_scale = None
        self.scene_map = []
        self.feature_counts = []
        self.lance_boundaries = []
        self.num_scenes = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
#  LMDB 流式分组 Top-K
# ============================================================


def apply_search_mode_grouping_lmdb(
    result_view: 'LMDBResultView',
    search_mode: int,
    top_k: Optional[int],
    scene_key_to_lance_map: Dict[str, str] = None,
    cache_dir: str = None,
    batch_size: int = 1000,
) -> 'LMDBResultView':
    """
    对 LMDBResultView 中的搜索结果按 search_mode 分组取 Top-K（流式版）。
    替代 AutoSceneSearcher._apply_search_mode_grouping（dict 版）。

    Args:
        result_view: 输入结果视图
        search_mode: -1=按视频, 0=按Lance, 1=跨Lance全局
        top_k: 每组最大结果数，None 或 <=0 不限制
        scene_key_to_lance_map: {scene_key: source_lance} 映射（search_mode=0 时需要）
        cache_dir: 输出 LMDB 目录，None 则自动生成
        batch_size: 迭代批大小

    Returns:
        过滤后的 LMDBResultView
    """
    import heapq as _heapq

    # 归一化 top_k
    if top_k is not None:
        top_k = int(top_k)
        if top_k <= 0:
            top_k = None
    if top_k is None:
        return result_view

    # Pass 1: 分组堆排序，收集每组 top_k 个 scene_key
    grouped_heaps: Dict[str, list] = defaultdict(list)
    for batch in result_view.iter_batches(batch_size=batch_size):
        for scene_key, data in batch:
            similarity = float(data.get('similarity', 0))
            if search_mode == -1:
                group_key = str(data.get('video_path', ''))
            elif search_mode == 0:
                group_key = (scene_key_to_lance_map or {}).get(scene_key, 'unknown')
            else:
                group_key = '__global__'
            heap = grouped_heaps[group_key]
            if len(heap) < top_k:
                _heapq.heappush(heap, (similarity, scene_key))
            elif similarity > heap[0][0]:
                _heapq.heapreplace(heap, (similarity, scene_key))

    keep_keys = set()
    for heap in grouped_heaps.values():
        for _, scene_key in heap:
            keep_keys.add(scene_key)
    del grouped_heaps

    total_in = result_view.count()
    print(f"[Grouping LMDB] mode={search_mode}, top_k={top_k}, {total_in} -> {len(keep_keys)}")

    if len(keep_keys) == total_in:
        return result_view

    # Pass 2: 过滤写入新 LMDB
    if cache_dir is None:
        from path_resolver import PathResolver
        _resolver = PathResolver()
        cache_dir = os.path.join(str(_resolver.project_root), 'temp', 'cache', 'search_results', 'grouping')
    os.makedirs(cache_dir, exist_ok=True)
    out_cache = LMDBCache(cache_dir, map_size=10 * 1024 * 1024 * 1024)
    out_cache.clear_results()
    out_buffer = {}
    for batch in result_view.iter_batches(batch_size=batch_size):
        for scene_key, data in batch:
            if scene_key in keep_keys:
                out_buffer[f"result:{scene_key}"] = data
                if len(out_buffer) >= batch_size:
                    out_cache.put_many(out_buffer)
                    out_buffer.clear()
    if out_buffer:
        out_cache.put_many(out_buffer)
        out_buffer.clear()
    out_cache.close()

    result_view.close()
    return LMDBResultView(cache_dir)


# ============================================================
#  向量去重工具函数


# ============================================================


def extract_label_key_from_result_name(
    result_name: str,
    video_name_format: str,
    exclude_fields: list = None
) -> str:
    """
    从 `result_name` 中提取标签组合键（用于分组）。
    Args:
        result_name: 例如 `P模式_男人_跑步_城市_开心_1234_视频名`
        video_name_format: 例如 `P模式_{主体}_{动作}_{场景}_{情绪}_{起始帧}_{视频解析名}`
        exclude_fields: 排除字段，默认 `['起始帧', '视频解析名']`
    Returns:
        标签组合键，例如 `P模式_男人_跑步_城市_开心`
    """
    import re

    if exclude_fields is None:
        exclude_fields = ["起始帧", "视频解析名"]
    # 解析格式字符串中的全部占位符
    placeholders = re.findall(r'\{([^}]+)\}', video_name_format)
    # 将格式模板转换为正则表达式
    # 例如：'P模式_{主体}_{动作}' -> 'P模式_(.+?)_(.+?)'
    pattern = video_name_format
    for ph in placeholders:
        pattern = pattern.replace(f'{{{ph}}}', '(.+?)')
    pattern = pattern + '$'
    # 匹配 result_name
    match = re.match(pattern, result_name)
    if not match:
        return result_name  # 无法解析时返回原值

    # 提取标签值（排除指定字段）
    label_values = []
    for i, ph in enumerate(placeholders):
        if ph not in exclude_fields:
            label_values.append(match.group(i + 1))
    return '_'.join(label_values)


def extract_video_parsed_name(
    result_name: str,
    video_name_format: str
) -> str:
    """
    从 `result_name` 中提取视频解析名。
    Args:
        result_name: 例如 `P模式_男人_跑步_城市_开心_1234_视频名_OP`
        video_name_format: 例如 `P模式_{主体}_{动作}_{场景}_{情绪}_{起始帧}_{视频解析名}`
    Returns:
        视频解析名，例如 `视频名_OP`
    """
    import re

    placeholders = re.findall(r'\{([^}]+)\}', video_name_format)
    pattern = video_name_format
    for ph in placeholders:
        pattern = pattern.replace(f'{{{ph}}}', '(.+?)')
    pattern = pattern + '$'
    match = re.match(pattern, result_name)
    if not match:
        return ''

    # 找到“视频解析名”占位符的位置
    for i, ph in enumerate(placeholders):
        if ph == "视频解析名":
            return match.group(i + 1)

    return ''


def is_op_ed_video(video_parsed_name: str) -> bool:
    """
    检查视频解析名是否包含 OP/ED 标记（不区分大小写）。
    支持格式：
    - 标准格式：`op`, `ed`, `_op_`, `_ed_`, `op_`, `ed_`, `_op`, `_ed`
    - 带编号：`op01`, `ed01`, `op1`, `ed1`, `op_01`, `ed_01`
    - NC 版本：`ncop`, `nced`, `ncop01`, `nced01`
    - 混合格式：`video_op01_xxx`, `video_ncop_xxx`
    Args:
        video_parsed_name: 视频解析名
    Returns:
        是否为 OP/ED 视频
    """
    import re

    if not video_parsed_name:
        return False

    name_lower = video_parsed_name.lower()
    # 匹配模式列表（按优先级排序）
    patterns = [
        # NC 版本（无字幕）：ncop, nced, ncop01, nced01, ncop_01
        r'(?:^|_)nc(?:op|ed)(?:\d+|_\d+)?(?:_|$)',
        # 带编号 OP/ED：op01, ed01, op1, ed1, op_01, ed_01
        r'(?:^|_)(?:op|ed)(?:\d+|_\d+)(?:_|$)',
        # 标准 OP/ED：_op_, _ed_, op_, ed_, _op, _ed
        r'(?:^|_)(?:op|ed)(?:_|$)',
    ]
    for pattern in patterns:
        if re.search(pattern, name_lower):
            return True

    return False


def _deduplicate_vectors_matrix(
    scene_vectors_list: list,
    threshold: float
) -> list:
    """
    用矩阵运算执行余弦相似度去重（支持多帧向量）。
    比较两个场景时，计算两者所有帧向量间的最大相似度。
    例如：场景 A 有 [a1,a2,a3]，场景 B 有 [b1,b2,b3]。
    计算 3x3=9 个相似度，取最大值作为两个场景的相似度。
    Args:
        scene_vectors_list: 场景向量列表，每个元素是 [k, D]（k 帧向量）
        threshold: 余弦相似度阈值
    Returns:
        保留的索引列表
    """
    n = len(scene_vectors_list)
    if n <= 1:
        return list(range(n))

    # 获取每个场景的帧数
    frame_counts = [v.shape[0] for v in scene_vectors_list]
    total_frames = sum(frame_counts)
    # 展平所有向量为 [total_frames, D]
    all_vectors = np.vstack(scene_vectors_list).astype(np.float32)
    # 向量已归一化，可直接计算相似度矩阵 [total_frames, total_frames]
    similarity_matrix = all_vectors @ all_vectors.T
    # 构建场景边界索引
    scene_boundaries = [0]
    for count in frame_counts:
        scene_boundaries.append(scene_boundaries[-1] + count)
    # 计算场景间最大相似度矩阵 [N, N]
    scene_max_sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        i_start, i_end = scene_boundaries[i], scene_boundaries[i+1]
        for j in range(i, n):
            j_start, j_end = scene_boundaries[j], scene_boundaries[j+1]
            # 取场景 i 和场景 j 之间所有帧向量的最大相似度
            block = similarity_matrix[i_start:i_end, j_start:j_end]
            max_sim = block.max()
            scene_max_sim[i, j] = max_sim
            scene_max_sim[j, i] = max_sim
    # 贪心去重：遍历上三角
    keep_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep_mask[i]:
            continue

        # 标记与场景 i 相似度超过阈值的后续场景为重复
        for j in range(i + 1, n):
            if keep_mask[j] and scene_max_sim[i, j] > threshold:
                keep_mask[j] = False
    return np.where(keep_mask)[0].tolist()


def _deduplicate_vectors_matrix_cross_video(
    scene_vectors_list: list,
    scene_video_ids: list,
    threshold: float
) -> list:
    """
    跨视频去重版本：仅比较不同视频之间的场景相似度；
    同一视频内场景不去重。
    """
    n = len(scene_vectors_list)
    if n <= 1:
        return list(range(n))

    frame_counts = [vectors.shape[0] for vectors in scene_vectors_list]
    all_vectors = np.vstack(scene_vectors_list).astype(np.float32)
    similarity_matrix = all_vectors @ all_vectors.T
    scene_boundaries = [0]
    for count in frame_counts:
        scene_boundaries.append(scene_boundaries[-1] + count)
    scene_max_sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        i_start, i_end = scene_boundaries[i], scene_boundaries[i + 1]
        for j in range(i, n):
            j_start, j_end = scene_boundaries[j], scene_boundaries[j + 1]
            block = similarity_matrix[i_start:i_end, j_start:j_end]
            max_sim = block.max()
            scene_max_sim[i, j] = max_sim
            scene_max_sim[j, i] = max_sim
    keep_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep_mask[i]:
            continue

        for j in range(i + 1, n):
            if not keep_mask[j]:
                continue

            if scene_video_ids[i] == scene_video_ids[j]:
                continue

            if scene_max_sim[i, j] > threshold:
                keep_mask[j] = False
    return np.where(keep_mask)[0].tolist()


def deduplicate_by_vector_similarity(
    best_matches: dict,
    scene_features: dict,
    video_name_format: str,
    similarity_threshold: float = 0.95,
    exclude_fields: list = None,
    scene_pkl_map: dict = None
) -> dict:
    """
    在视频导出前进行同标签向量去重（同 Lance 内、跨视频）。
    优先规则：
    - 若同标签组中存在 OP/ED 视频，仅保留 OP/ED 视频，并跳过向量去重
    - 否则执行常规向量相似度去重
    去重范围：
    - 仅在同一 Lance 内去重，不跨 Lance
    - 同一 Lance + 同一标签组下，仅做跨视频去重（同视频内不去重）
    Args:
        best_matches: 搜索结果字典 {scene_key: {...}}
        scene_features: 场景特征向量 {scene_key: np.ndarray}
        video_name_format: 视频命名格式
        similarity_threshold: 余弦相似度阈值（0-1），超过则去重
        exclude_fields: 去重时排除的字段，默认 ['起始帧', '视频解析名']
        scene_pkl_map: 场景到 Lance 的映射 {scene_key: source_lance}，用于同 Lance 内去重
    Returns:
        去重后的结果字典
    """
    if exclude_fields is None:
        exclude_fields = ["起始帧", "视频解析名"]
    if not best_matches or similarity_threshold is None:
        return best_matches

    # Step 1: 按 (Lance, 标签组合) 分组，实现同 Lance 内跨视频去重
    # 格式: {(lance_path, label_key): [scene_key, ...]}
    groups = defaultdict(list)
    for scene_key, data in best_matches.items():
        result_name = data.get('result_name', '')
        label_key = extract_label_key_from_result_name(
            result_name, video_name_format, exclude_fields
        )
        # 获取该场景的 Lance 来源
        source_path = scene_pkl_map.get(scene_key, 'unknown') if scene_pkl_map else 'unknown'
        # 使用 (source_path, label_key) 作为分组键，实现同 Lance 内去重
        group_key = (source_path, label_key)
        groups[group_key].append(scene_key)
    # Step 2: 对每个分组执行去重
    keep_scene_keys = set()
    op_ed_priority_count = 0
    vector_dedup_count = 0
    for group_key, scene_keys in groups.items():
        if len(scene_keys) == 1:
            # 只有一个候选，直接保留
            keep_scene_keys.add(scene_keys[0])
            continue

        # ========== OP/ED 优先规则 ==========
        # 检查组内是否存在 OP/ED 视频
        op_ed_keys = []
        non_op_ed_keys = []
        for sk in scene_keys:
            data = best_matches[sk]
            result_name = data.get('result_name', '')
            video_parsed_name = extract_video_parsed_name(result_name, video_name_format)
            if is_op_ed_video(video_parsed_name):
                op_ed_keys.append(sk)
            else:
                non_op_ed_keys.append(sk)
        # 如果有 OP/ED 视频，只保留 OP/ED 视频（并跳过向量去重）
        if op_ed_keys:
            keep_scene_keys.update(op_ed_keys)
            op_ed_priority_count += len(non_op_ed_keys)
            continue

        # ========== 向量相似度去重（场景级 + 跨视频） ==========
        scene_vectors_list = []
        valid_keys = []
        valid_video_ids = []
        for sk in scene_keys:
            vectors = scene_features.get(sk)
            if vectors is None:
                # 无向量无法判重，直接保留
                keep_scene_keys.add(sk)
                continue

            scene_vectors_list.append(vectors)
            valid_keys.append(sk)
            valid_video_ids.append(best_matches[sk].get('video_path') or '__unknown_video__')
        if len(scene_vectors_list) <= 1:
            keep_scene_keys.update(valid_keys)
            continue

        before_count = len(valid_keys)
        keep_indices = _deduplicate_vectors_matrix_cross_video(
            scene_vectors_list,
            valid_video_ids,
            similarity_threshold
        )
        after_count = len(keep_indices)
        vector_dedup_count += (before_count - after_count)
        for idx in keep_indices:
            keep_scene_keys.add(valid_keys[idx])
    # 打印统计信息
    total_removed = len(best_matches) - len(keep_scene_keys)
    print(f"[向量去重] OP/ED优先过滤: {op_ed_priority_count}, 向量相似度去重: {vector_dedup_count}")
    print(f"[向量去重] 总计去除: {total_removed}, 保留: {len(keep_scene_keys)}")
    # Step 3: 返回去重后的结果
    return {k: v for k, v in best_matches.items() if k in keep_scene_keys}


def deduplicate_by_vector_lmdb(
    result_view: 'LMDBResultView',
    scene_features: dict,
    video_name_format: str,
    similarity_threshold: float = 0.95,
    scene_lance_map: Dict[str, str] = None,
    cache_dir: str = None,
    batch_size: int = 1000,
) -> 'LMDBResultView':
    """
    LMDB 流式版向量去重，替代 deduplicate_by_vector_similarity 的 dict 版。
    逻辑与 dict 版完全一致：按 (source_lance, label_key) 分组，组内 OP/ED 优先 + 向量去重。

    Args:
        result_view: 输入结果视图
        scene_features: {scene_key: np.ndarray} 场景特征向量
        video_name_format: 视频命名格式
        similarity_threshold: 余弦相似度阈值
        scene_lance_map: {scene_key: source_lance} 映射
        cache_dir: 输出 LMDB 目录
        batch_size: 迭代批大小

    Returns:
        去重后的 LMDBResultView
    """
    import hashlib as _hashlib

    if not scene_features or similarity_threshold is None:
        return result_view

    total_in = result_view.count()
    if total_in == 0:
        return result_view

    # 准备输出目录
    if cache_dir is None:
        from path_resolver import PathResolver
        _resolver = PathResolver()
        cache_dir = os.path.join(str(_resolver.project_root), 'temp', 'cache', 'search_results', 'vector_dedup')

    group_cache_dir = os.path.join(cache_dir, 'groups')
    output_cache_dir = os.path.join(cache_dir, 'output')
    os.makedirs(group_cache_dir, exist_ok=True)
    os.makedirs(output_cache_dir, exist_ok=True)

    group_cache = LMDBCache(group_cache_dir, map_size=10 * 1024 * 1024 * 1024)
    group_cache.clear_all()

    # Pass 1: 按 (source_lance, label_key) 分组写入 group LMDB
    exclude_fields = ["起始帧", "视频解析名"]
    group_buffer = {}
    for batch in result_view.iter_batches(batch_size=batch_size):
        for scene_key, data in batch:
            result_name = data.get('result_name', '')
            label_key = extract_label_key_from_result_name(result_name, video_name_format, exclude_fields)
            source_lance = data.get('source_lance', 'unknown')
            if scene_lance_map:
                source_lance = scene_lance_map.get(scene_key, source_lance)
            group_id = _hashlib.md5(f"{source_lance}|{label_key}".encode('utf-8')).hexdigest()
            group_buffer[f"group:{group_id}:{scene_key}"] = data
            if len(group_buffer) >= batch_size:
                group_cache.put_many(group_buffer)
                group_buffer.clear()
    if group_buffer:
        group_cache.put_many(group_buffer)
        group_buffer.clear()

    # Pass 2: 遍历 group LMDB（有序），同组去重
    out_cache = LMDBCache(output_cache_dir, map_size=10 * 1024 * 1024 * 1024)
    out_cache.clear_all()
    out_buffer = {}
    op_ed_priority_count = 0
    vector_dedup_count = 0

    current_group_id = None
    current_group_data = {}  # {scene_key: data}

    def _flush_current_group():
        nonlocal current_group_data, out_buffer, op_ed_priority_count, vector_dedup_count
        if not current_group_data:
            return
        scene_keys = list(current_group_data.keys())
        if len(scene_keys) == 1:
            sk = scene_keys[0]
            out_buffer[f"result:{sk}"] = current_group_data[sk]
        else:
            # OP/ED 优先
            op_ed_keys = []
            non_op_ed_keys = []
            for sk in scene_keys:
                rn = current_group_data[sk].get('result_name', '')
                vpn = extract_video_parsed_name(rn, video_name_format)
                if is_op_ed_video(vpn):
                    op_ed_keys.append(sk)
                else:
                    non_op_ed_keys.append(sk)
            if op_ed_keys:
                for sk in op_ed_keys:
                    out_buffer[f"result:{sk}"] = current_group_data[sk]
                op_ed_priority_count += len(non_op_ed_keys)
            else:
                # 向量去重
                scene_vectors_list = []
                valid_keys = []
                valid_video_ids = []
                for sk in scene_keys:
                    vectors = scene_features.get(sk)
                    if vectors is None:
                        out_buffer[f"result:{sk}"] = current_group_data[sk]
                        continue
                    scene_vectors_list.append(vectors)
                    valid_keys.append(sk)
                    valid_video_ids.append(current_group_data[sk].get('video_path') or '__unknown__')
                if len(scene_vectors_list) <= 1:
                    for sk in valid_keys:
                        out_buffer[f"result:{sk}"] = current_group_data[sk]
                else:
                    before_count = len(valid_keys)
                    keep_indices = _deduplicate_vectors_matrix_cross_video(
                        scene_vectors_list, valid_video_ids, similarity_threshold
                    )
                    vector_dedup_count += (before_count - len(keep_indices))
                    for idx in keep_indices:
                        sk = valid_keys[idx]
                        out_buffer[f"result:{sk}"] = current_group_data[sk]
        current_group_data = {}
        if len(out_buffer) >= batch_size:
            out_cache.put_many(out_buffer)
            out_buffer.clear()

    for group_batch in group_cache.iter_items_with_prefix("group:", batch_size=batch_size):
        for full_key, data in group_batch:
            # full_key = "group:{group_id}:{scene_key}"
            parts = full_key.split(":", 2)
            group_id = parts[1] if len(parts) > 1 else ''
            scene_key = parts[2] if len(parts) > 2 else full_key
            if group_id != current_group_id:
                _flush_current_group()
                current_group_id = group_id
            current_group_data[scene_key] = data
    _flush_current_group()

    if out_buffer:
        out_cache.put_many(out_buffer)
        out_buffer.clear()

    total_out = out_cache.count_results()
    total_removed = total_in - total_out
    print(f"[向量去重 LMDB] OP/ED优先: {op_ed_priority_count}, 向量去重: {vector_dedup_count}")
    print(f"[向量去重 LMDB] 总计去除: {total_removed}, 保留: {total_out}")

    group_cache.close()
    out_cache.close()
    result_view.close()
    return LMDBResultView(output_cache_dir)


# ============================================================
#  测试入口
# ============================================================
if __name__ == "__main__":
    print("batch_text_search module loaded")
