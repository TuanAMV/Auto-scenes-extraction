# batch_text_search.py
# 批量文搜图引擎 - 预加载 + 批量计算优化版
# v1.0: 初始版本
# v1.1: 添加 LMDB 支持，批量事务写入磁盘，解决内存问题
# v1.2: 添加 prompt向量缓存支持，避免重复编码
#
# 优化点：
# 1. PKL特征预加载到GPU，常驻显存
# 2. 批量编码多个prompt
# 3. GPU批量矩阵计算
# 4. 支持跨PKL/跨视频的场景管理
# 5. 使用 LMDB 批量事务写入磁盘，避免内存溢出
# 6. 支持预计算的prompt向量缓存（无需重复编码和归一化）

import os
import sys
import gc
import time
import pickle
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Generator, Union, Any, Callable
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
    LMDB 缓存包装类
    
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
        初始化 LMDB 缓存
        
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
        """写入数据"""
        key_bytes = key.encode('utf-8')
        value_bytes = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        with self.env.begin(write=True) as txn:
            txn.put(key_bytes, value_bytes)
    
    def __getitem__(self, key: str):
        """读取数据"""
        key_bytes = key.encode('utf-8')
        with self.env.begin(buffers=True) as txn:
            value_bytes = txn.get(key_bytes)
            if value_bytes is None:
                raise KeyError(key)
            return pickle.loads(value_bytes)
    
    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        key_bytes = key.encode('utf-8')
        with self.env.begin() as txn:
            return txn.get(key_bytes) is not None
    
    def get(self, key: str, default=None):
        """获取数据，不存在返回默认值"""
        try:
            return self[key]
        except KeyError:
            return default
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """鎵归噺鑾峰彇鏁版嵁锛屼粎杩斿洖瀛樺湪鐨勯敭"""
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
        """删除指定键，返回是否成功"""
        key_bytes = key.encode('utf-8')
        with self.env.begin(write=True) as txn:
            return txn.delete(key_bytes)
    
    def delete_many(self, keys: List[str]) -> int:
        """鎵归噺鍒犻櫎鎸囧畾 key锛岃繑鍥炲垹闄ゆ暟閲?"""
        if not keys:
            return 0
        deleted = 0
        with self.env.begin(write=True) as txn:
            for key in keys:
                if txn.delete(key.encode('utf-8')):
                    deleted += 1
        return deleted

    def put_many(self, items: Dict[str, Any]):
        """鎵归噺鍐欏叆鏁版嵁锛堝崟浜嬪姟锛?"""
        if not items:
            return
        with self.env.begin(write=True) as txn:
            for key, value in items.items():
                key_bytes = key.encode('utf-8')
                value_bytes = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                txn.put(key_bytes, value_bytes)

    def iterkeys(self):
        """迭代所有键"""
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                yield key.decode('utf-8')
    
    def keys(self):
        """返回所有键的列表"""
        return list(self.iterkeys())
    
    def keys_with_prefix(self, prefix: str):
        """返回指定前缀的所有键"""
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
        """返回指定前缀的所有键值对"""
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
        """返回记录数"""
        with self.env.begin() as txn:
            return txn.stat()['entries']
    
    def clear_all(self):
        """清空所有数据"""
        with self.env.begin(write=True) as txn:
            txn.drop(db=txn.db, delete=False)
    
    def save_checkpoint(self, data: dict):
        """保存断点信息"""
        self[self.CHECKPOINT_KEY] = data
    
    def load_checkpoint(self) -> Optional[dict]:
        """加载断点信息，不存在返回 None"""
        return self.get(self.CHECKPOINT_KEY)
    
    def save_meta(self, data: dict):
        """保存搜索元数据"""
        self[self.META_KEY] = data
    
    def load_meta(self) -> Optional[dict]:
        """加载搜索元数据，不存在返回 None"""
        return self.get(self.META_KEY)
    
    def put_result(self, scene_key: str, data: dict):
        """写入最终搜索结果"""
        self[self.RESULT_PREFIX + scene_key] = data
    
    def put_results_many(self, results: Dict[str, dict]):
        """鎵归噺鍐欏叆鏈€缁堟悳绱㈢粨鏋?"""
        if not results:
            return
        data = {
            self.RESULT_PREFIX + scene_key: result_data
            for scene_key, result_data in results.items()
        }
        self.put_many(data)

    def get_result(self, scene_key: str) -> Optional[dict]:
        """获取最终搜索结果"""
        return self.get(self.RESULT_PREFIX + scene_key)
    
    def get_all_results(self) -> Dict[str, dict]:
        """获取所有最终搜索结果"""
        results = {}
        for key, value in self.items_with_prefix(self.RESULT_PREFIX):
            scene_key = key[len(self.RESULT_PREFIX):]
            results[scene_key] = value
        return results
    
    def close(self):
        """关闭数据库"""
        if not self._closed:
            self.env.close()
            self._closed = True
    
    def destroy(self):
        """关闭并删除整个 LMDB 目录"""
        self.close()
        import shutil
        if os.path.exists(self.path):
            shutil.rmtree(self.path, ignore_errors=True)
    
    def __del__(self):
        """析构时关闭"""
        self.close()

# ============================================================
#  路径设置 - 确保能找到项目根目录的模块
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

# 全局设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BatchTextSearchEngine:
    """
    批量文搜图引擎 - 预加载 + 批量计算
    
    优化点：
    1. PKL特征预加载到GPU，常驻显存
    2. 批量加载prompt向量（从缓存）
    3. GPU批量矩阵计算
    4. 支持跨PKL/跨视频的场景管理
    
    使用方式：
    ```python
    # 初始化（预加载所有PKL）
    engine = BatchTextSearchEngine(processor, index_paths, video_name_format="{主体}_{动作}_{起始帧}_{视频解析名}")
    
    # 使用缓存迭代器进行搜索
    results = engine.search_with_batched_cache(cache_iterator, threshold=20.0)
    
    # 释放资源
    engine.cleanup()
    ```
    """
    
    def __init__(self,
                 processor=None,
                 index_paths: Optional[List[str]] = None,
                 cache_dir: str = None,
                 load_workers: int = 4,
                 use_fp16: bool = True,
                 pkl_batch_size: int = None,
                 video_name_format: str = None,
                 search_mode: int = 0,
                 top_k: int = 50,
                 lmdb_write_batch_size: int = 1000,
                 truncate_dim: Optional[int] = None,
                 logit_scale: float = 100.0):
        """
        初始化批量搜索引擎
        
        Args:
            processor: EmbeddingModelProcessor 实例
            index_paths: PKL索引文件路径列表
            cache_dir: 特征缓存目录
            load_workers: PKL加载线程数（全量预加载时使用）
            use_fp16: 是否使用FP16存储特征（显存减半）
            pkl_batch_size: 每批加载的PKL数量
                - None 或 >= PKL总数: 一次性全部加载到GPU（显存占用高但搜索快）
                - < PKL总数: 分批加载，用完释放（显存占用低但稍慢）
            video_name_format: 视频名称格式模板（必需参数）
            search_mode: 搜索模式选择
                - -1（按视频模式）: 每个视频独立搜索，每个视频返回 top_k 个结果
                - 0（按PKL模式）: 每个PKL文件独立搜索，每个PKL返回 top_k 个结果
                - 1（跨PKL模式）: 全局搜索，返回全局 top_k 个结果
            top_k: 每组返回的最大结果数，None则不限制
            lmdb_write_batch_size: LMDB单事务写入批大小（分批加载时使用）
        """
        if video_name_format is None:
            raise ValueError("video_name_format 参数是必需的，请传入视频名称格式模板")
        
        self.processor = processor
        self.index_paths = index_paths or []
        self.load_workers = load_workers
        self.lmdb_write_batch_size = lmdb_write_batch_size
        self.use_fp16 = use_fp16
        self.video_name_format = video_name_format
        self.search_mode = search_mode
        self.top_k = top_k
        self.truncate_dim = truncate_dim if truncate_dim is not None else getattr(processor, 'truncate_dim', None)
        self.logit_scale_value = float(logit_scale)
        
        # 处理 pkl_batch_size：None 或 >= PKL总数 表示全量加载
        total_pkls = len(self.index_paths)
        if pkl_batch_size is None or pkl_batch_size >= total_pkls:
            self.pkl_batch_size = total_pkls  # 全量加载
            self._preload_all = True
        else:
            self.pkl_batch_size = pkl_batch_size
            self._preload_all = False
        
        # 缓存目录
        if cache_dir is None:
            resolver = PathResolver()
            cache_dir = str(resolver.project_root / 'temp' / 'cache')
        self.cache_dir = cache_dir
        self.feature_cache_dir = os.path.join(cache_dir, 'features')
        os.makedirs(self.feature_cache_dir, exist_ok=True)
        
        # 预加载的特征数据（合并模式）
        self.all_features_gpu: Optional[torch.Tensor] = None  # [M, dim] GPU tensor
        self.scene_map: List[Dict] = []           # 场景信息列表
        self.feature_counts: List[int] = []       # 每个场景的向量数量
        self.pkl_boundaries: List[int] = []       # PKL边界索引
        
        # GPU优化：场景索引张量（用于 scatter_reduce）
        self.scene_indices_gpu: Optional[torch.Tensor] = None  # [M] 每个向量对应的场景索引
        self.num_scenes: int = 0  # 场景总数
        
        # 按视频分组的数据结构（search_mode=-1时使用）
        self.video_features: Dict[str, torch.Tensor] = {}      # {video_path: [N, dim] GPU tensor}
        self.video_scene_maps: Dict[str, List[Dict]] = {}      # {video_path: [scene_info, ...]}
        self.video_feature_counts: Dict[str, List[int]] = {}   # {video_path: [count, ...]}
        
        # logit_scale 缓存
        self._logit_scale: Optional[torch.Tensor] = torch.tensor(self.logit_scale_value, device=DEVICE, dtype=torch.float32)
        
        # 根据配置决定预加载方式
        if self._preload_all:
            self._preload_all_features()
    
    def _ensure_logit_scale(self):
        if self._logit_scale is None:
            self._logit_scale = torch.tensor(self.logit_scale_value, device=DEVICE, dtype=torch.float32)

    def _get_cache_path(self, pkl_path: str) -> str:
        """根据 pkl 路径生成磁盘缓存文件路径"""
        mtime = os.path.getmtime(pkl_path)
        cache_key = f"{pkl_path}_{mtime}".encode('utf-8')
        file_hash = hashlib.md5(cache_key).hexdigest()
        return os.path.join(self.feature_cache_dir, f"{file_hash}.npz")
    
    def _load_single_pkl(self, pkl_path: str) -> Tuple[Optional[np.ndarray], List[Dict], List[int]]:
        """
        加载单个PKL文件的特征（带磁盘缓存）
        
        Returns:
            tuple: (features_np, scene_map, feature_counts)
        """
        cache_path = self._get_cache_path(pkl_path)
        
        # 检查磁盘缓存
        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path, allow_pickle=True)
                if 'feature_counts' in cached:
                    features_np = cached['features']
                    scene_map = cached['scene_map'].tolist()
                    feature_counts = cached['feature_counts'].tolist()
                    return features_np, scene_map, feature_counts
            except Exception as e:
                print(f"    [缓存损坏] {os.path.basename(pkl_path)}: {e}")
                try:
                    os.remove(cache_path)
                except:
                    pass
        
        # 从PKL加载
        try:
            with open(pkl_path, 'rb') as f:
                data_dict = pickle.load(f)
        except Exception as e:
            print(f"    [Error] 无法加载 pkl: {e}")
            return None, [], []
        
        # 收集特征和场景信息
        local_features = []
        scene_map = []
        feature_counts = []
        
        for video_path, data in data_dict.items():
            scenes = data.get('scenes', []) if isinstance(data, dict) else data
            fps = data.get('fps', 25.0) if isinstance(data, dict) else 25.0
            
            for scene in scenes:
                # 提取场景的所有特征向量
                all_feats = SceneFeatureExtractor.extract_all_from_scene(scene)
                if not all_feats:
                    continue
                
                # 添加所有特征向量
                for feat in all_feats:
                    local_features.append(feat)
                
                # 记录场景信息
                scene_map.append({
                    "video_path": video_path,
                    "start_frame": scene.get("start_frame", 0),
                    "end_frame": scene.get("end_frame", 0),
                    "fps": fps,
                    "source_pkl": pkl_path
                })
                feature_counts.append(len(all_feats))
        
        if not local_features:
            return None, [], []
        
        # 构建特征矩阵
        features_np = np.vstack(local_features).astype(np.float32)
        
        # 保存到磁盘缓存
        try:
            np.savez(cache_path, 
                     features=features_np, 
                     scene_map=np.array(scene_map, dtype=object),
                     feature_counts=np.array(feature_counts, dtype=np.int32))
        except Exception as e:
            print(f"    [缓存写入失败] {e}")
        
        return features_np, scene_map, feature_counts
    
    def _preload_all_features(self):
        """
        按PKL分组预加载特征到GPU
        
        每个PKL独立存储，支持按PKL独立搜索（不跨PKL合并）
        """
        # 静默：预加载开始信息
        start_time = time.time()
        
        # 按PKL分组存储
        self.pkl_features: Dict[str, torch.Tensor] = {}  # {pkl_path: features_gpu}
        self.pkl_scene_maps: Dict[str, List[Dict]] = {}  # {pkl_path: scene_maps}
        self.pkl_feature_counts: Dict[str, List[int]] = {}  # {pkl_path: feature_counts}
        self.pkl_scene_indices: Dict[str, torch.Tensor] = {}  # {pkl_path: scene_indices_gpu}
        
        total_scenes = 0
        total_vectors = 0
        total_memory_mb = 0.0
        
        # CPU并行加载PKL
        with ThreadPoolExecutor(max_workers=self.load_workers) as executor:
            futures = {executor.submit(self._load_single_pkl, p): p for p in self.index_paths}
            
            for future in as_completed(futures):
                pkl_path = futures[future]
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
                        
                        # 送入GPU
                        dtype = torch.float16 if self.use_fp16 else torch.float32
                        features_gpu = torch.tensor(features_np, device=DEVICE, dtype=dtype)
                        
                        # 构建场景索引张量
                        scene_indices = []
                        for scene_idx, count in enumerate(feature_counts):
                            scene_indices.extend([scene_idx] * count)
                        scene_indices_gpu = torch.tensor(scene_indices, device=DEVICE, dtype=torch.long)
                        
                        # 存储
                        self.pkl_features[pkl_path] = features_gpu
                        self.pkl_scene_maps[pkl_path] = scene_map
                        self.pkl_feature_counts[pkl_path] = feature_counts
                        self.pkl_scene_indices[pkl_path] = scene_indices_gpu
                        
                        # 统计
                        total_scenes += len(scene_map)
                        total_vectors += features_np.shape[0]
                        total_memory_mb += features_gpu.element_size() * features_gpu.nelement() / (1024 * 1024)
                        # 静默：单个PKL加载信息
                        
                except Exception as e:
                    print(f"[错误] {os.path.basename(pkl_path)}: 加载失败 - {e}")
        
        if not self.pkl_features:
            print("[警告] 没有加载到任何特征数据")
            return
        
        # 获取 logit_scale
        self._ensure_logit_scale()
        
        # 设置总场景数（用于兼容性）
        self.num_scenes = total_scenes
        
        # 合并所有PKL特征到 all_features_gpu（用于 search_with_batched_cache）
        all_features_list = []
        all_scene_maps = []
        all_feature_counts = []
        all_scene_indices = []
        all_frame_indices = []  # 新增：记录每个特征是场景的第几帧（0=首帧, 1=中帧, 2=尾帧）
        scene_offset = 0
        
        for pkl_path in self.pkl_features:
            features_gpu = self.pkl_features[pkl_path]
            scene_map = self.pkl_scene_maps[pkl_path]
            feature_counts = self.pkl_feature_counts[pkl_path]
            
            all_features_list.append(features_gpu)
            all_scene_maps.extend(scene_map)
            all_feature_counts.extend(feature_counts)
            
            # 构建场景索引和帧索引（偏移后）
            for scene_idx, count in enumerate(feature_counts):
                all_scene_indices.extend([scene_offset + scene_idx] * count)
                # 帧索引：0=首帧, 1=中帧, 2=尾帧（如果只有1帧则为0）
                all_frame_indices.extend(list(range(count)))
            scene_offset += len(scene_map)
        
        # 合并到单个张量
        self.all_features_gpu = torch.cat(all_features_list, dim=0)
        self.scene_map = all_scene_maps
        self.feature_counts = all_feature_counts
        self.scene_indices_gpu = torch.tensor(all_scene_indices, device=DEVICE, dtype=torch.long)
        self.frame_indices_gpu = torch.tensor(all_frame_indices, device=DEVICE, dtype=torch.long)  # 新增
        # 构建复合索引：scene_idx * 3 + frame_idx（用于 GPU 矩阵运算）
        self.scene_frame_indices_gpu = self.scene_indices_gpu * 3 + self.frame_indices_gpu

        # 记录PKL数量（释放前）
        loaded_pkl_count = len(self.pkl_features) if hasattr(self, 'pkl_features') else 0

        # 释放按PKL分组保留的GPU张量，避免与 all_features_gpu 双份占用显存
        if hasattr(self, 'pkl_features'):
            for pkl_path in list(self.pkl_features.keys()):
                if self.pkl_features[pkl_path] is not None:
                    del self.pkl_features[pkl_path]
            self.pkl_features.clear()
        if hasattr(self, 'pkl_scene_indices'):
            for pkl_path in list(self.pkl_scene_indices.keys()):
                if self.pkl_scene_indices[pkl_path] is not None:
                    del self.pkl_scene_indices[pkl_path]
            self.pkl_scene_indices.clear()
        del all_features_list
        
        load_time = time.time() - start_time
        print(f"[预加载] {loaded_pkl_count} 个PKL, {total_scenes} 场景, {total_vectors} 向量, 耗时 {load_time:.2f}s")
    
    def _preload_features_per_video(self):
        """
        按视频分组预加载特征到GPU
        
        与 _preload_all_features() 的区别：
        - 合并模式：所有视频场景合并到一个GPU张量
        - 按视频模式：每个视频独立存储，支持按视频独立搜索
        """
        # 静默：预加载开始信息
        start_time = time.time()
        
        # 临时存储：{video_path: {'features': [], 'scenes': [], 'counts': []}}
        video_data = defaultdict(lambda: {'features': [], 'scenes': [], 'counts': []})
        
        # CPU并行加载PKL
        with ThreadPoolExecutor(max_workers=self.load_workers) as executor:
            futures = {executor.submit(self._load_single_pkl, p): p for p in self.index_paths}
            
            for future in as_completed(futures):
                pkl_path = futures[future]
                try:
                    features_np, scene_map, feature_counts = future.result()
                    
                    if features_np is not None and len(scene_map) > 0:
                        # 按视频分组
                        feat_idx = 0
                        for scene_idx, scene_info in enumerate(scene_map):
                            video_path = scene_info['video_path']
                            count = feature_counts[scene_idx]
                            
                            # 提取该场景的特征
                            scene_features = features_np[feat_idx:feat_idx + count]
                            feat_idx += count
                            
                            video_data[video_path]['features'].append(scene_features)
                            video_data[video_path]['scenes'].append(scene_info)
                            video_data[video_path]['counts'].append(count)
                        
                        # 静默：单个PKL加载信息
                        pass
                        
                except Exception as e:
                    print(f"[错误] {os.path.basename(pkl_path)}: 加载失败 - {e}")
        
        if not video_data:
            print("[警告] 没有加载到任何特征数据")
            return
        
        # 按视频构建GPU张量
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
        print(f"[预加载] {len(self.video_features)} 视频, {total_scenes} 场景, {total_vectors} 向量, 耗时 {load_time:.2f}s")
    
    @staticmethod
    def _seconds_to_timecode(seconds: float) -> str:
        """秒数转时间码"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """清理文件名中的非法字符"""
        if not name:
            return ""
        # 替换非法字符
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            name = name.replace(char, '_')
        return name.strip('_')
    
    def _format_video_name(self, meta: Dict, start_frame: int, video_parsed_name: str, video_path: str = None) -> str:
        """
        根据格式模板生成视频名称（动态占位符处理）
        
        Args:
            meta: 元数据字典，包含各大类的中文值
            start_frame: 起始帧号
            video_parsed_name: 解析后的视频名称
            video_path: 视频路径（用于提取扩展名）
        
        Returns:
            格式化后的视频名称（不含扩展名）
        """
        import re
        
        # 兼容异常元数据类型（例如历史缓存中的纯字符串）
        if not isinstance(meta, dict):
            meta = {}

        # 1. 从 video_name_format 中动态解析所有占位符
        placeholders = re.findall(r'\{(\w+)\}', self.video_name_format)
        
        # 2. 为所有占位符预设空字符串默认值
        format_dict = {ph: "" for ph in placeholders}
        
        # 3. 填充系统占位符
        format_dict['起始帧'] = str(start_frame)
        format_dict['视频解析名'] = self._sanitize_filename(video_parsed_name)
        if '扩展名' in format_dict:
            extension = ''
            resolved_video_path = video_path or meta.get('video_path', '')
            if isinstance(resolved_video_path, str) and resolved_video_path:
                extension = os.path.splitext(resolved_video_path)[1]
            if extension.startswith('.'):
                extension = extension[1:]
            format_dict['扩展名'] = self._sanitize_filename(extension)
        if 'prompt_cn' in format_dict:
            format_dict['prompt_cn'] = self._sanitize_filename(str(meta.get('prompt_cn', '')))
        if 'prompt_en' in format_dict:
            format_dict['prompt_en'] = self._sanitize_filename(str(meta.get('prompt_en', '')))
        
        # 4. 填充 P/L 模式的标准大类（从 meta 的 xxx_cn 字段）
        pl_category_mapping = {
            '镜头': 'lens_cn',
            '情绪': 'mood_cn',
            '场景': 'scene_cn',
            '主体': 'subject_cn',
            '动作': 'action_cn',
        }
        for category_name, meta_key in pl_category_mapping.items():
            if category_name in format_dict:
                format_dict[category_name] = self._sanitize_filename(meta.get(meta_key, ""))
        
        # 5. 填充扩展大类的中文值（从 meta 的 xxx_cn 字段）
        for key, value in meta.items():
            if key.endswith('_cn'):
                category_name = key[:-3]
                if category_name in format_dict:
                    format_dict[category_name] = self._sanitize_filename(value)
        
        # 6. 填充 C 模式（选词填空模式）的 labels 字典
        # 格式: {子类名: (中文标签, 英文标签)}
        labels = meta.get("labels", {})
        if labels and isinstance(labels, dict):
            for subcat_name, label_tuple in labels.items():
                if subcat_name in format_dict:
                    if isinstance(label_tuple, (tuple, list)) and len(label_tuple) >= 1:
                        # 使用中文标签（第一个元素）
                        format_dict[subcat_name] = self._sanitize_filename(label_tuple[0])
        
        # 7. 使用格式模板生成名称（所有占位符都已预设默认值，不会 KeyError）
        result_name = self.video_name_format.format(**format_dict)
        
        # 8. 动态清理连续分隔符（从格式模板中提取分隔符）
        # 提取占位符之间的分隔符
        separators = re.findall(r'\}([^\{]+)\{', self.video_name_format)
        for sep in set(separators):
            if sep:
                # 清理连续的分隔符（2个及以上）
                double_sep = sep + sep
                while double_sep in result_name:
                    result_name = result_name.replace(double_sep, sep)
        # 清理首尾分隔符
        for sep in set(separators):
            if sep:
                result_name = result_name.strip(sep)
        
        return result_name
    
    def search_with_batched_cache(self,
                                   cache_iterator,
                                   threshold: float = 20.0,
                                   initial_threshold: float = None,
                                   video_name_parser=None,
                                   use_diskcache: bool = True,
                                   cache_dir: str = None,
                                   rerank_top_k: int = 10,
                                   use_reranker: bool = False,
                                   reranker=None,
                                   reranker_loader: Optional[Callable[[], Any]] = None,
                                   reranker_weight: float = 0.6,
                                   reranker_output_resolution: str = '384',
                                   rerank_batch_size: int = 7,
                                   search_mode: int = 0,
                                   result_top_k: int = None,
                                   config_hash: str = None,
                                   candidate_batch_size: int = None,
                                   skip_clip_search: bool = False,
                                   append_to_lmdb_cache: bool = False) -> Dict[str, Dict]:
        """
        使用分批加载的缓存进行搜索（内存友好版本，支持 Reranker + 断点续传）
        
        适用于大规模 prompt 缓存（70万+），分批加载向量，使用 LMDB 存储中间结果。
        支持断点续传：程序中断后重启可从上次完成的 prompt 批次继续搜索。
        
        流程：
        1. 检查 LMDB 中的 checkpoint，决定从哪个批次开始
        2. 分批加载 prompt 向量（每批 batch_size 个）
        3. 每批进行 GPU 矩阵计算
        4. 更新 LMDB 中每个场景的 Top-K 候选（实时去重）
        5. 每批完成后保存 checkpoint 到 LMDB
        6. 释放当前批次内存，加载下一批
        7. 搜索完成后，分批从 LMDB 读取候选进行 Reranker 验证（内存友好）
        8. 最终结果实时写入 LMDB
        9. 按搜索模式分组，每组返回 Top-K 结果
        
        Args:
            cache_iterator: PromptVectorBatchIterator 迭代器
            threshold: 最终相似度阈值（用于 Reranker 混合分数筛选）
            initial_threshold: CLIP 初始阈值（用于候选筛选），None则使用 threshold
            video_name_parser: 视频名称解析器
            use_diskcache: 是否使用 LMDB 存储（强烈建议 True）
            cache_dir: LMDB 缓存目录
            rerank_top_k: 每个场景保留的 Top-K 候选数量（CLIP 阶段堆大小）
            use_reranker: 是否使用 Reranker
            reranker: Reranker 实例
            reranker_weight: Reranker 分数权重
            reranker_output_resolution: Reranker 帧分辨率
            rerank_batch_size: Reranker 模型推理分批大小（所有 rerank_top_k 个候选都会送入 Reranker，按此大小分批推理）
            search_mode: 搜索模式选择
                - -1（按视频模式）: 每个视频独立搜索，每个视频返回 result_top_k 个结果
                - 0（按PKL模式）: 每个PKL独立搜索，每个PKL返回 result_top_k 个结果
                - 1（跨PKL模式）: 全局搜索，返回全局 result_top_k 个结果
            result_top_k: 每组返回的最大结果数，None则不限制
            config_hash: 搜索配置哈希值（用于断点续传验证），None则不启用断点续传
            candidate_batch_size: 候选处理阶段的分批大小
                - None（默认）: 全量读取候选到内存（兼容旧行为）
                - 正整数: 每批从 LMDB 读取指定数量的场景候选进行处理，处理完释放
                - 推荐值: 1000-5000（根据内存大小调整）
                - 仅在 use_diskcache=True 时生效
            append_to_lmdb_cache: 仅追加候选到 LMDB，不执行候选后处理/重排
                - 主要用于分批 PKL 场景：多次调用后统一进行一次 Reranker
        
        Returns:
            去重后的最佳匹配字典 {scene_key: {result_name, similarity, video_path, start_frame, end_frame, candidates}}
        """
        import heapq
        reranker_loaded_by_current_call = False
        
        if not skip_clip_search and self.all_features_gpu is None:
            raise RuntimeError("特征未预加载，请先调用 _preload_all_features()")
        if skip_clip_search and not use_diskcache:
            raise RuntimeError("skip_clip_search=True 需要 use_diskcache=True 并从 LMDB 读取候选")
        if append_to_lmdb_cache and not use_diskcache:
            raise RuntimeError("append_to_lmdb_cache=True 需要 use_diskcache=True")
        if append_to_lmdb_cache and cache_iterator is None:
            raise RuntimeError("append_to_lmdb_cache=True 需要提供 cache_iterator")

        if skip_clip_search and cache_iterator is None:
            class _EmptyCacheIterator:
                total_prompts = 0
                num_batches = 0

                def __iter__(self):
                    return iter(())

            cache_iterator = _EmptyCacheIterator()

        write_batch_size = max(1, int(self.lmdb_write_batch_size))

        def _put_many_chunked(items: Dict[str, Any]):
            if not items:
                return
            item_list = list(items.items())
            for idx in range(0, len(item_list), write_batch_size):
                lmdb_cache.put_many(dict(item_list[idx:idx + write_batch_size]))

        def _put_results_many_chunked(items: Dict[str, Any]):
            if not items:
                return
            item_list = list(items.items())
            for idx in range(0, len(item_list), write_batch_size):
                lmdb_cache.put_results_many(dict(item_list[idx:idx + write_batch_size]))
        
        # 设置 CLIP 初始阈值（用于候选筛选）
        # 如果使用 Reranker，应该使用较低的 CLIP 阈值来召回更多候选
        clip_threshold = initial_threshold if initial_threshold is not None else threshold
        
        # 初始化 LMDB 缓存（支持断点续传）
        resume_from_batch = 0  # 默认从第0批开始
        if use_diskcache:
            if cache_dir is None:
                resolver = PathResolver()
                cache_dir = str(resolver.project_root / 'temp' / 'cache' / 'search_results')
            
            if not LMDB_AVAILABLE:
                raise ImportError("LMDB 未安装，请运行: pip install lmdb")
            
            # 断点续传检查
            if skip_clip_search or append_to_lmdb_cache:
                os.makedirs(cache_dir, exist_ok=True)
                lmdb_cache = LMDBCache(cache_dir, map_size=10 * 1024 * 1024 * 1024)
            elif config_hash is not None and os.path.exists(cache_dir):
                try:
                    lmdb_cache = LMDBCache(cache_dir, map_size=10 * 1024 * 1024 * 1024)
                    checkpoint = lmdb_cache.load_checkpoint()
                    if checkpoint is not None and checkpoint.get('config_hash') == config_hash:
                        last_batch = checkpoint.get('last_completed_batch', -1)
                        checkpoint_phase = checkpoint.get('phase', 'search')
                        if checkpoint_phase == 'completed':
                            # 搜索已完成，直接从 LMDB 读取最终结果
                            print(f"[分批搜索-断点续传] 搜索已完成，从 LMDB 读取最终结果...")
                            best_matches = lmdb_cache.get_all_results()
                            print(f"[分批搜索-断点续传] 恢复 {len(best_matches)} 个最终结果")
                            return best_matches
                        else:
                            resume_from_batch = last_batch + 1
                            total_batches = cache_iterator.num_batches
                            if resume_from_batch < total_batches:
                                print(f"[分批搜索-断点续传] 检测到有效断点: 已完成 {resume_from_batch}/{total_batches} 批")
                                print(f"[分批搜索-断点续传] 从第 {resume_from_batch + 1} 批继续搜索")
                            else:
                                print(f"[分批搜索-断点续传] 所有 prompt 批次已完成，进入候选处理阶段")
                    else:
                        # config_hash 不匹配，清空当前缓存目录重来
                        print(f"[分批搜索-断点续传] 配置已变更，清空当前缓存目录重新搜索: {cache_dir}")
                        lmdb_cache.close()
                        from A_coreUtils.video_processing.video_utils import cleanup_temp_folder as _cleanup_temp
                        _cleanup_temp(cache_dir)
                        os.makedirs(cache_dir, exist_ok=True)
                        lmdb_cache = LMDBCache(cache_dir, map_size=10 * 1024 * 1024 * 1024)
                except Exception as e:
                    print(f"[分批搜索-断点续传] 读取断点失败: {e}，清空当前缓存目录重新搜索: {cache_dir}")
                    from A_coreUtils.video_processing.video_utils import cleanup_temp_folder as _cleanup_temp
                    _cleanup_temp(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                    lmdb_cache = LMDBCache(cache_dir, map_size=10 * 1024 * 1024 * 1024)
            else:
                # 无断点续传或目录不存在，清空当前缓存目录重建
                from A_coreUtils.video_processing.video_utils import cleanup_temp_folder as _cleanup_temp
                _cleanup_temp(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                lmdb_cache = LMDBCache(cache_dir, map_size=10 * 1024 * 1024 * 1024)
            
            # 保存搜索元数据
            if (not skip_clip_search) and (not append_to_lmdb_cache) and config_hash is not None:
                lmdb_cache.save_meta({
                    'config_hash': config_hash,
                    'threshold': threshold,
                    'clip_threshold': clip_threshold,
                    'search_mode': search_mode,
                    'result_top_k': result_top_k,
                    'total_prompts': cache_iterator.total_prompts,
                    'total_batches': cache_iterator.num_batches
                })
        else:
            lmdb_cache = None
        
        # 内存中的场景候选（如果不用 LMDB）
        # 格式: {scene_key: {'candidates': [(sim, prompt_idx, meta, frame_idx), ...], 'scene_info': {...}, 'best_frame_idx': int}}
        scene_candidates = {} if not use_diskcache else None
        
        # 获取计算精度
        compute_dtype = self.all_features_gpu.dtype if self.all_features_gpu is not None else torch.float32
        device = self.all_features_gpu.device if self.all_features_gpu is not None else torch.device(DEVICE)
        
        # 预缓存视频名称解析结果
        video_parsed_cache = {}
        
        start_time = time.time()
        total_prompts = cache_iterator.total_prompts
        
        print(f"[分批搜索] 开始搜索 {total_prompts} 个 prompt，分 {cache_iterator.num_batches} 批")
        print(f"[分批搜索] 每个场景保留 Top-{rerank_top_k} 候选")
        if resume_from_batch > 0:
            print(f"[分批搜索] 断点续传: 跳过前 {resume_from_batch} 批")
        
        # 遍历每批 prompt 向量
        for batch_vectors, batch_prompts, batch_metadata, batch_info in cache_iterator:
            batch_idx = batch_info['batch_idx']
            
            # 断点续传：跳过已完成的批次
            if batch_idx < resume_from_batch:
                continue
            batch_start = batch_info['start_idx']
            batch_end = batch_info['end_idx']
            actual_batch_size = batch_end - batch_start
            
            # 确保向量在正确的设备和精度上
            if batch_vectors.device != device:
                batch_vectors = batch_vectors.to(device)
            if batch_vectors.dtype != compute_dtype:
                batch_vectors = batch_vectors.to(compute_dtype)
            
            # GPU 批量计算相似度 [batch, M]
            with torch.no_grad():
                similarities = self._logit_scale * batch_vectors @ self.all_features_gpu.T
                
                # ========== GPU 矩阵运算优化：按场景+帧分组 ==========
                # 使用 scatter_reduce 按场景+帧分组取最大值 [batch, num_scenes * 3]
                num_scene_frames = self.num_scenes * 3
                scene_frame_max_sims = torch.full(
                    (actual_batch_size, num_scene_frames),
                    float('-inf'),
                    device=device,
                    dtype=compute_dtype
                )
                expanded_sf_indices = self.scene_frame_indices_gpu.unsqueeze(0).expand(actual_batch_size, -1)
                scene_frame_max_sims.scatter_reduce_(1, expanded_sf_indices, similarities, reduce='amax')
                
                # reshape 为 [batch, num_scenes, 3]
                scene_frame_max_sims = scene_frame_max_sims.view(actual_batch_size, self.num_scenes, 3)
                
                # 找每个场景的最佳帧和最大相似度 [batch, num_scenes]
                scene_max_sims, best_frame_indices = scene_frame_max_sims.max(dim=2)
                
                # 转置为 [num_scenes, batch]，方便按场景处理
                scene_prompt_sims = scene_max_sims.T  # [num_scenes, batch]
                best_frame_per_scene = best_frame_indices.T  # [num_scenes, batch]
                
                # 传输到 CPU
                scene_prompt_sims_cpu = scene_prompt_sims.float().cpu().numpy()
                best_frame_per_scene_cpu = best_frame_per_scene.cpu().numpy()
            
            # 更新每个场景的 Top-K 候选
            batch_candidate_updates = {} if use_diskcache else None
            for scene_idx in range(self.num_scenes):
                scene_sims = scene_prompt_sims_cpu[scene_idx]  # [batch]
                scene_best_frames = best_frame_per_scene_cpu[scene_idx]  # [batch] 每个 prompt 的最佳帧
                
                # 找到超过 CLIP 阈值的 prompt
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
                
                # 获取或创建该场景的候选列表
                if use_diskcache:
                    existing = lmdb_cache.get(scene_key)
                    if existing is None:
                        candidates = []
                        best_frame_idx = 1  # 默认中间帧
                        scene_data = {
                            'video_path': video_path,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'fps': scene_info.get('fps', 25.0)
                        }
                    else:
                        candidates = existing.get('candidates', [])
                        best_frame_idx = existing.get('best_frame_idx', 1)
                        scene_data = {
                            'video_path': existing['video_path'],
                            'start_frame': existing['start_frame'],
                            'end_frame': existing['end_frame'],
                            'fps': existing.get('fps', 25.0)
                        }
                else:
                    if scene_key not in scene_candidates:
                        scene_candidates[scene_key] = {
                            'candidates': [],
                            'best_frame_idx': 1,  # 默认中间帧
                            'scene_info': {
                                'video_path': video_path,
                                'start_frame': start_frame,
                                'end_frame': end_frame,
                                'fps': scene_info.get('fps', 25.0)
                            }
                        }
                    candidates = scene_candidates[scene_key]['candidates']
                    best_frame_idx = scene_candidates[scene_key].get('best_frame_idx', 1)
                    scene_data = scene_candidates[scene_key]['scene_info']
                
                # 使用堆维护 Top-K（带帧索引）- 使用 GPU 预计算的最佳帧
                for batch_prompt_idx in above_indices:
                    similarity = float(scene_sims[batch_prompt_idx])
                    best_frame_for_prompt = int(scene_best_frames[batch_prompt_idx])
                    
                    global_prompt_idx = batch_start + batch_prompt_idx
                    meta = batch_metadata[batch_prompt_idx] if batch_metadata else {}
                    
                    # 堆元素: (similarity, global_prompt_idx, meta, frame_idx)
                    # 使用最小堆，保留最大的 K 个
                    if len(candidates) < rerank_top_k:
                        heapq.heappush(candidates, (similarity, global_prompt_idx, meta, best_frame_for_prompt))
                    elif similarity > candidates[0][0]:
                        heapq.heapreplace(candidates, (similarity, global_prompt_idx, meta, best_frame_for_prompt))
                
                # 更新最佳帧索引（取候选中相似度最高的那个的帧索引）
                if candidates:
                    best_candidate = max(candidates, key=lambda x: x[0])
                    best_frame_idx = best_candidate[3] if len(best_candidate) > 3 else 1
                
                # 保存回 LMDB 或内存
                if use_diskcache:
                    batch_candidate_updates[scene_key] = {
                        'candidates': candidates,
                        'best_frame_idx': best_frame_idx,
                        'video_path': scene_data['video_path'],
                        'start_frame': scene_data['start_frame'],
                        'end_frame': scene_data['end_frame'],
                        'fps': scene_data.get('fps', 25.0)
                    }
                else:
                    scene_candidates[scene_key]['best_frame_idx'] = best_frame_idx
            
            # 释放当前批次的 GPU 内存
            if use_diskcache and batch_candidate_updates:
                _put_many_chunked(batch_candidate_updates)
            del batch_vectors, similarities, scene_frame_max_sims, scene_max_sims, scene_prompt_sims, best_frame_indices, best_frame_per_scene
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 保存 checkpoint（每批完成后）
            if use_diskcache and lmdb_cache is not None and (not append_to_lmdb_cache) and config_hash is not None:
                lmdb_cache.save_checkpoint({
                    'config_hash': config_hash,
                    'last_completed_batch': batch_idx,
                    'total_batches': cache_iterator.num_batches,
                    'total_prompts': total_prompts,
                    'phase': 'clip_search',  # 标记当前阶段
                    'timestamp': time.time()
                })
            
            # 进度输出
            if batch_idx % 5 == 0 or batch_idx == cache_iterator.num_batches - 1:
                elapsed = time.time() - start_time
                progress = (batch_idx + 1) / cache_iterator.num_batches * 100
                print(f"  [批次 {batch_idx + 1}/{cache_iterator.num_batches}] {progress:.1f}%, 耗时 {elapsed:.1f}s")
        
        search_time = time.time() - start_time

        if append_to_lmdb_cache:
            if lmdb_cache is not None:
                lmdb_cache.close()
            print(f"[分批搜索] CLIP 候选聚合完成! 耗时 {search_time:.2f}s（append 模式）")
            return {}

        def _sort_candidate_keys_by_video_path(keys: List[str]) -> List[str]:
            if not keys or lmdb_cache is None:
                return keys
            ordered = []
            chunk_size = max(256, write_batch_size)
            for offset in range(0, len(keys), chunk_size):
                chunk_keys = keys[offset:offset + chunk_size]
                chunk_data = lmdb_cache.get_many(chunk_keys)
                for key in chunk_keys:
                    data = chunk_data.get(key, {})
                    video_path = data.get('video_path', '')
                    start_frame = int(data.get('start_frame', 0) or 0)
                    ordered.append((video_path, start_frame, key))
            ordered.sort(key=lambda item: (item[0], item[1], item[2]))
            return [item[2] for item in ordered]
        
        # 从 LMDB 读取候选（是否分批仅由 candidate_batch_size 决定）
        # candidate_batch_size=None 表示不启用候选分批
        use_batched_candidate = (candidate_batch_size is not None
                                 and use_diskcache
                                 and candidate_batch_size > 0)
        
        if use_diskcache:
            if use_batched_candidate:
                # 分批模式：只获取候选 key 列表，不全量读回数据
                print(f"[分批搜索] 从 LMDB 获取候选 key 列表（分批模式, batch_size={candidate_batch_size}）...")
                candidate_keys = []
                for key in lmdb_cache.keys():
                    if key.startswith('__') or key.startswith(LMDBCache.RESULT_PREFIX):
                        continue
                    candidate_keys.append(key)
                candidate_keys = _sort_candidate_keys_by_video_path(candidate_keys)
                scene_candidates = None  # 分批模式下不使用全量字典
                print(f"[分批搜索] CLIP 搜索完成! 耗时 {search_time:.2f}s, {len(candidate_keys)} 个场景有候选")
            else:
                # 全量模式：一次性读取所有候选到内存（兼容旧行为）
                print(f"[分批搜索] 从 LMDB 读取候选...")
                scene_candidates = {}
                candidate_keys = []
                for key in lmdb_cache.keys():
                    # 跳过特殊 key（checkpoint、meta、result 前缀）
                    if key.startswith('__') or key.startswith(LMDBCache.RESULT_PREFIX):
                        continue
                    candidate_keys.append(key)
                candidate_data_map = lmdb_cache.get_many(candidate_keys)
                for key, data in candidate_data_map.items():
                    scene_candidates[key] = {
                        'candidates': data.get('candidates', []),
                        'best_frame_idx': data.get('best_frame_idx', 1),
                        'scene_info': {
                            'video_path': data['video_path'],
                            'start_frame': data['start_frame'],
                            'end_frame': data['end_frame'],
                            'fps': data.get('fps', 25.0)
                        }
                    }
                candidate_keys = None  # 全量模式下不使用 key 列表
                print(f"[分批搜索] CLIP 搜索完成! 耗时 {search_time:.2f}s, {len(scene_candidates)} 个场景有候选")
        
        # ========== Reranker 阶段 ==========
        # 判断是否有候选需要处理
        has_candidates = (use_batched_candidate and candidate_keys and len(candidate_keys) > 0) or \
                         (not use_batched_candidate and scene_candidates and len(scene_candidates) > 0)
        
        best_matches = {}
        grouped_result_heaps = defaultdict(list) if result_top_k is not None else None
        scene_key_to_pkl = {}
        if result_top_k is not None and search_mode == 0:
            for scene_info in self.scene_map:
                video_name = os.path.basename(scene_info['video_path']) if scene_info.get('video_path') else ''
                scene_key = f"{scene_info['start_frame']}_{video_name}"
                scene_key_to_pkl[scene_key] = scene_info.get('source_pkl', 'unknown')

        def _resolve_group_key(scene_key: str, result_data: dict) -> str:
            if search_mode == -1:
                return result_data.get('video_path', '') or 'unknown_video'
            if search_mode == 0:
                return scene_key_to_pkl.get(scene_key, 'unknown')
            return '__global__'

        def _add_result(scene_key: str, result_data: dict):
            if result_top_k is None:
                best_matches[scene_key] = result_data
                return
            if result_top_k <= 0:
                return
            group_key = _resolve_group_key(scene_key, result_data)
            group_heap = grouped_result_heaps[group_key]
            sim = float(result_data.get('similarity', float('-inf')))
            heap_item = (sim, scene_key, result_data)
            if len(group_heap) < result_top_k:
                heapq.heappush(group_heap, heap_item)
            elif sim > group_heap[0][0]:
                heapq.heapreplace(group_heap, heap_item)

        if use_reranker and has_candidates and reranker is None:
            if callable(reranker_loader):
                try:
                    reranker = reranker_loader()
                    reranker_loaded_by_current_call = reranker is not None
                except Exception as e:
                    print(f"[警告] 懒加载 Reranker 模型失败: {e}")
                    reranker = None
            else:
                print("[警告] 已启用 Reranker，但未提供模型实例或加载器，回退为纯 CLIP")
                reranker = None

            if reranker is None:
                use_reranker = False
                threshold = clip_threshold
                print(f"[警告] 阈值已调整为纯 CLIP 阈值: {threshold}")

        if use_reranker and reranker is not None and has_candidates:
            print(f"\n[分批搜索] 开始 Reranker 验证...")
            rerank_start = time.time()
            
            # 导入帧提取器（单线程版本）
            from A_coreUtils.search.reranker_frame_extractor import RerankerFrameExtractor
            resolver = PathResolver()
            rerank_frames_dir = str(resolver.project_root / 'temp' / 'cache' / 'rerank_frames')
            
            # RerankerFrameExtractor 初始化时会自动清空缓存目录
            frame_extractor = RerankerFrameExtractor(
                output_resolution=reranker_output_resolution,
                cache_dir=rerank_frames_dir
            )
            
            if use_batched_candidate:
                # ========== 分批模式：分批从 LMDB 读取候选进行 Reranker 处理 ==========
                total_keys = len(candidate_keys)
                batch_size = candidate_batch_size
                num_batches = (total_keys + batch_size - 1) // batch_size
                
                print(f"  [Reranker分批] 共 {total_keys} 个候选, 分 {num_batches} 批处理 (batch_size={batch_size})")
                
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, total_keys)
                    batch_keys = candidate_keys[batch_start:batch_end]
                    
                    # 从 LMDB 读取当前批次的候选数据
                    batch_scene_candidates = {}
                    batch_candidate_data = lmdb_cache.get_many(batch_keys)
                    for key, data in batch_candidate_data.items():
                        batch_scene_candidates[key] = {
                            'candidates': data.get('candidates', []),
                            'best_frame_idx': data.get('best_frame_idx', 1),
                            'scene_info': {
                                'video_path': data['video_path'],
                                'start_frame': data['start_frame'],
                                'end_frame': data['end_frame'],
                                'fps': data.get('fps', 25.0)
                            }
                        }
                    
                    # 收集需要提取帧的场景（带最佳帧索引）
                    scenes_to_extract = []
                    for scene_key, data in batch_scene_candidates.items():
                        scene_info = data['scene_info']
                        best_frame_idx = data.get('best_frame_idx', 1)
                        scenes_to_extract.append({
                            'video_path': scene_info['video_path'],
                            'start_frame': scene_info['start_frame'],
                            'end_frame': scene_info['end_frame'],
                            'fps': scene_info.get('fps', 25.0),
                            'target_frame_idx': best_frame_idx
                        })
                    
                    # 批量提取帧
                    frame_paths = frame_extractor.extract_batch(scenes_to_extract)
                    
                    # 对当前批次的每个场景进行 Reranker 验证
                    for scene_key, data in batch_scene_candidates.items():
                        candidates = data['candidates']
                        scene_info = data['scene_info']
                        
                        if not candidates:
                            continue
                        
                        # 获取帧路径
                        frame_key = f"{scene_info['start_frame']}_{os.path.basename(scene_info['video_path'])}"
                        frame_path = frame_paths.get(frame_key)
                        
                        if not frame_path or not os.path.exists(frame_path):
                            best_candidate = max(candidates, key=lambda x: x[0])
                            best_sim, best_prompt_idx, best_meta, _ = best_candidate
                        else:
                            before_best_candidate = max(candidates, key=lambda x: x[0])
                            before_best_sim, before_best_prompt_idx, before_best_meta, _ = before_best_candidate
                            before_best_query = before_best_meta.get('prompt', '')
                            
                            try:
                                # 按相似度降序排列所有候选（堆内部无序）
                                sorted_candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
                                documents = []
                                for candidate in sorted_candidates:
                                    sim, prompt_idx, meta = candidate[0], candidate[1], candidate[2]
                                    query_text = meta.get('prompt', '')
                                    documents.append({'text': query_text})
                                
                                inputs = {
                                    'query': {'image': frame_path},
                                    'documents': documents
                                }
                                
                                rerank_scores = reranker.process(inputs, batch_size=rerank_batch_size)
                                
                                best_sim = float('-inf')
                                best_meta = None
                                best_prompt_idx = None
                                
                                for candidate, rerank_score in zip(sorted_candidates, rerank_scores):
                                    sim, prompt_idx, meta = candidate[0], candidate[1], candidate[2]
                                    final_score = sim * (1 - reranker_weight) + rerank_score * 100.0 * reranker_weight
                                    
                                    if final_score > best_sim:
                                        best_sim = final_score
                                        best_meta = meta
                                        best_prompt_idx = prompt_idx
                                
                                if best_meta is not None:
                                    after_best_query = best_meta.get('prompt', '')
                                    print(f"    [Reranker] 场景 {scene_key}:")
                                    print(f"      重排前: 相似度={before_best_sim:.2f}, prompt='{before_best_query}'")
                                    print(f"      重排后: 相似度={best_sim:.2f}, prompt='{after_best_query}'")
                            
                            except Exception as e:
                                print(f"      [Reranker错误] 场景 {scene_key} 批量推理失败: {e}")
                                best_sim = before_best_sim
                                best_meta = before_best_meta
                                best_prompt_idx = before_best_prompt_idx
                        
                        if best_meta is None:
                            best_candidate = max(candidates, key=lambda x: x[0])
                            best_sim, _, best_meta = best_candidate
                        
                        # 解析视频名称
                        video_path = scene_info['video_path']
                        video_name = os.path.basename(video_path) if video_path else ''
                        if video_path not in video_parsed_cache:
                            video_parsed_name = ""
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
                        else:
                            video_parsed_name = video_parsed_cache[video_path]
                        
                        # 最终阈值过滤
                        if best_sim < threshold:
                            continue
                        
                        # 生成结果名称
                        result_name = self._format_video_name(
                            best_meta, scene_info['start_frame'], video_parsed_name, scene_info.get('video_path')
                        )
                        
                        result_data = {
                            'result_name': result_name,
                            'similarity': best_sim,
                            'video_path': scene_info['video_path'],
                            'start_frame': scene_info['start_frame'],
                            'end_frame': scene_info['end_frame']
                        }
                        
                        _add_result(scene_key, result_data)
                    
                    # 中间批次仅清理 batch_* 临时目录，保留已提取帧缓存
                    frame_extractor.cleanup(remove_files=False)
                    del batch_scene_candidates
                    del scenes_to_extract
                    del frame_paths
                    
                    print(f"  [Reranker分批] 批次 {batch_idx + 1}/{num_batches} 完成, 已处理 {batch_end}/{total_keys} 个候选")
                
                rerank_time = time.time() - rerank_start
                print(f"  [Reranker] 完成! 耗时 {rerank_time:.2f}s")
                frame_extractor.cleanup(remove_files=True)
                
            else:
                # ========== 全量模式：原有逻辑（兼容旧行为）==========
                # 收集需要提取帧的场景（带最佳帧索引）
                scenes_to_extract = []
                for scene_key, data in scene_candidates.items():
                    scene_info = data['scene_info']
                    best_frame_idx = data.get('best_frame_idx', 1)  # 默认中间帧
                    scenes_to_extract.append({
                        'video_path': scene_info['video_path'],
                        'start_frame': scene_info['start_frame'],
                        'end_frame': scene_info['end_frame'],
                        'fps': scene_info.get('fps', 25.0),
                        'target_frame_idx': best_frame_idx  # 新增：指定提取哪一帧（0=首帧, 1=中帧, 2=尾帧）
                    })
                
                # 批量提取帧
                print(f"  [Reranker] 提取 {len(scenes_to_extract)} 个场景的帧...")
                frame_paths = frame_extractor.extract_batch(scenes_to_extract)
                
                for scene_key, data in scene_candidates.items():
                    candidates = data['candidates']
                    scene_info = data['scene_info']
                    
                    if not candidates:
                        continue
                    
                    # 获取帧路径
                    frame_key = f"{scene_info['start_frame']}_{os.path.basename(scene_info['video_path'])}"
                    frame_path = frame_paths.get(frame_key)
                    
                    if not frame_path or not os.path.exists(frame_path):
                        # 没有帧，使用 CLIP 最高分
                        best_candidate = max(candidates, key=lambda x: x[0])
                        best_sim, best_prompt_idx, best_meta, _ = best_candidate
                    else:
                        # 有帧，进行 Reranker 验证
                        # ========== 记录重排前的最高相似度 ==========
                        before_best_candidate = max(candidates, key=lambda x: x[0])
                        before_best_sim, before_best_prompt_idx, before_best_meta, _ = before_best_candidate
                        
                        # 构建重排前的查询文本（直接使用 CLIP 的 prompt，与 CLIP Query 保持一致）
                        before_best_query = before_best_meta.get('prompt', '')
                        
                        # ========== 批量推理：单张图像 + 多个文本候选 ==========
                        try:
                            # 按相似度降序排列所有候选（堆内部无序）
                            sorted_candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
                            
                            # 构建批量输入（直接使用 CLIP 的 prompt，与 CLIP Query 保持一致）
                            documents = []
                            for candidate in sorted_candidates:
                                sim, prompt_idx, meta = candidate[0], candidate[1], candidate[2]
                                query_text = meta.get('prompt', '')
                                documents.append({'text': query_text})
                            
                            inputs = {
                                'query': {'image': frame_path},  # 单张图像作为 query
                                'documents': documents           # 多个文本作为 documents
                            }
                            
                            # 一次推理返回所有分数
                            rerank_scores = reranker.process(inputs, batch_size=rerank_batch_size)
                            
                            # 找出最佳匹配
                            best_sim = float('-inf')
                            best_meta = None
                            best_prompt_idx = None
                            
                            for candidate, rerank_score in zip(sorted_candidates, rerank_scores):
                                sim, prompt_idx, meta = candidate[0], candidate[1], candidate[2]
                                # 融合分数: final = clip * (1 - weight) + rerank * 100 * weight
                                final_score = sim * (1 - reranker_weight) + rerank_score * 100.0 * reranker_weight
                                
                                if final_score > best_sim:
                                    best_sim = final_score
                                    best_meta = meta
                                    best_prompt_idx = prompt_idx
                            
                            # ========== 打印重排前后对比 ==========
                            if best_meta is not None:
                                # 构建重排后的查询文本（直接使用 CLIP 的 prompt，与 CLIP Query 保持一致）
                                after_best_query = best_meta.get('prompt', '')
                                
                                # 打印 Reranker 前后对比
                                print(f"    [Reranker] 场景 {scene_key}:")
                                print(f"      重排前: 相似度={before_best_sim:.2f}, prompt='{before_best_query}'")
                                print(f"      重排后: 相似度={best_sim:.2f}, prompt='{after_best_query}'")
                        
                        except Exception as e:
                            # 批量推理失败，回退到 CLIP 最高分
                            print(f"      [Reranker错误] 场景 {scene_key} 批量推理失败: {e}")
                            best_sim = before_best_sim
                            best_meta = before_best_meta
                            best_prompt_idx = before_best_prompt_idx
                    
                    if best_meta is None:
                        best_candidate = max(candidates, key=lambda x: x[0])
                        best_sim, _, best_meta = best_candidate
                    
                    # 解析视频名称
                    video_path = scene_info['video_path']
                    video_name = os.path.basename(video_path) if video_path else ''
                    if video_path not in video_parsed_cache:
                        video_parsed_name = ""
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
                    else:
                        video_parsed_name = video_parsed_cache[video_path]
                    
                    # 最终阈值过滤
                    if best_sim < threshold:
                        continue
                    
                    # 生成结果名称
                    result_name = self._format_video_name(
                        best_meta, scene_info['start_frame'], video_parsed_name, scene_info.get('video_path')
                    )
                    
                    result_data = {
                        'result_name': result_name,
                        'similarity': best_sim,
                        'video_path': scene_info['video_path'],
                        'start_frame': scene_info['start_frame'],
                        'end_frame': scene_info['end_frame']
                    }
                    _add_result(scene_key, result_data)
                
                rerank_time = time.time() - rerank_start
                print(f"  [Reranker] 完成! 耗时 {rerank_time:.2f}s")
                
                # 清理帧文件
                frame_extractor.cleanup(remove_files=True)
            
            # 释放候选数据内存（Reranker 阶段已处理完毕，不再需要）
            if scene_candidates is not None:
                scene_candidates.clear()
                del scene_candidates
                scene_candidates = None
            if candidate_keys is not None:
                del candidate_keys
                candidate_keys = None
            gc.collect()
        else:
            # 不使用 Reranker，直接取每个场景的最高分候选
            if use_batched_candidate:
                # ========== 分批模式：分批从 LMDB 读取候选进行处理 ==========
                total_keys = len(candidate_keys)
                batch_size = candidate_batch_size
                num_batches = (total_keys + batch_size - 1) // batch_size
                
                print(f"  [非Reranker分批] 共 {total_keys} 个候选, 分 {num_batches} 批处理 (batch_size={batch_size})")
                
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, total_keys)
                    batch_keys = candidate_keys[batch_start:batch_end]
                    
                    # 从 LMDB 读取当前批次的候选数据
                    batch_candidate_data = lmdb_cache.get_many(batch_keys)
                    for key, data in batch_candidate_data.items():
                        candidates = data.get('candidates', [])
                        scene_info = {
                            'video_path': data['video_path'],
                            'start_frame': data['start_frame'],
                            'end_frame': data['end_frame'],
                            'fps': data.get('fps', 25.0)
                        }
                        
                        if not candidates:
                            continue
                        
                        # 取最高分候选
                        best_candidate = max(candidates, key=lambda x: x[0])
                        best_sim, best_prompt_idx, best_meta, _ = best_candidate
                        
                        # 解析视频名称
                        video_path = scene_info['video_path']
                        video_name = os.path.basename(video_path) if video_path else ''
                        if video_path not in video_parsed_cache:
                            video_parsed_name = ""
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
                        else:
                            video_parsed_name = video_parsed_cache[video_path]
                        
                        # 最终阈值过滤
                        if best_sim < threshold:
                            continue
                        
                        # 生成结果名称
                        result_name = self._format_video_name(
                            best_meta, scene_info['start_frame'], video_parsed_name, scene_info.get('video_path')
                        )
                        
                        result_data = {
                            'result_name': result_name,
                            'similarity': best_sim,
                            'video_path': scene_info['video_path'],
                            'start_frame': scene_info['start_frame'],
                            'end_frame': scene_info['end_frame']
                        }
                        
                        _add_result(key, result_data)
                    
                    print(f"  [非Reranker分批] 批次 {batch_idx + 1}/{num_batches} 完成, 已处理 {batch_end}/{total_keys} 个候选")
            else:
                # ========== 全量模式：原有逻辑（兼容旧行为）==========
                for scene_key, data in scene_candidates.items():
                    candidates = data['candidates']
                    scene_info = data['scene_info']
                    
                    if not candidates:
                        continue
                    
                    # 取最高分候选
                    best_candidate = max(candidates, key=lambda x: x[0])
                    best_sim, best_prompt_idx, best_meta, _ = best_candidate
                    
                    # 解析视频名称
                    video_path = scene_info['video_path']
                    video_name = os.path.basename(video_path) if video_path else ''
                    if video_path not in video_parsed_cache:
                        video_parsed_name = ""
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
                    else:
                        video_parsed_name = video_parsed_cache[video_path]
                    
                    # 最终阈值过滤
                    if best_sim < threshold:
                        continue
                    
                    # 生成结果名称
                    result_name = self._format_video_name(
                        best_meta, scene_info['start_frame'], video_parsed_name, scene_info.get('video_path')
                    )
                    
                    result_data = {
                        'result_name': result_name,
                        'similarity': best_sim,
                        'video_path': scene_info['video_path'],
                        'start_frame': scene_info['start_frame'],
                        'end_frame': scene_info['end_frame']
                    }
                    _add_result(scene_key, result_data)
            
            # 释放候选数据内存（非Reranker 阶段已处理完毕，不再需要）
            if scene_candidates is not None:
                scene_candidates.clear()
                del scene_candidates
                scene_candidates = None
            if candidate_keys is not None:
                del candidate_keys
                candidate_keys = None
            gc.collect()
        
        if reranker_loaded_by_current_call and reranker is not None:
            try:
                reranker.cleanup()
            except Exception as e:
                print(f"[警告] 释放懒加载 Reranker 模型时出错: {e}")
            del reranker
            reranker = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 按搜索模式分组，每组返回 Top-K 结果
        if result_top_k is not None:
            filtered_matches = {}
            for group_heap in grouped_result_heaps.values():
                for _, scene_key, data in sorted(group_heap, key=lambda x: x[0], reverse=True):
                    filtered_matches[scene_key] = data
            best_matches = filtered_matches
            if search_mode == -1:
                print(f"[分批搜索] 按视频模式过滤: {len(grouped_result_heaps)} 个视频, 每视频最多 {result_top_k} 个场景")
            elif search_mode == 0:
                print(f"[分批搜索] 按PKL模式过滤: {len(grouped_result_heaps)} 个PKL, 每PKL最多 {result_top_k} 个场景")
            else:
                print(f"[分批搜索] 跨PKL模式过滤: 全局最多 {result_top_k} 个场景")
        # 将最终结果写入 LMDB（用于断点续传和外部读取）
        if use_diskcache and lmdb_cache is not None and not append_to_lmdb_cache:
            print(f"[分批搜索] 将 {len(best_matches)} 个最终结果批量写入 LMDB...")
            # 更新 checkpoint 标记搜索完成
            stale_result_keys = lmdb_cache.keys_with_prefix(LMDBCache.RESULT_PREFIX)
            if stale_result_keys:
                lmdb_cache.delete_many(stale_result_keys)
            _put_results_many_chunked(best_matches)
            if config_hash is not None:
                lmdb_cache.save_checkpoint({
                    'config_hash': config_hash,
                    'last_completed_batch': cache_iterator.num_batches - 1,
                    'total_batches': cache_iterator.num_batches,
                    'total_prompts': total_prompts,
                    'phase': 'completed',
                    'result_count': len(best_matches),
                    'timestamp': time.time()
                })
            lmdb_cache.close()
        
        total_time = time.time() - start_time
        print(f"[分批搜索] 完成! 总耗时 {total_time:.2f}s, {len(best_matches)} 个场景")
        
        return best_matches
    
    def _load_pkl_batch_to_merged(self, pkl_paths: List[str]):
        """
        加载一批PKL到合并的 all_features_gpu 张量（用于分批PKL搜索）
        
        与 _preload_all_features() 类似，但只加载指定的PKL子集。
        调用后可使用 search_with_batched_cache() 进行搜索。
        搜索完成后调用 _unload_merged_features() 释放显存。
        
        Args:
            pkl_paths: 要加载的PKL文件路径列表
        """
        start_time = time.time()
        
        # 临时存储
        batch_pkl_features = {}
        batch_pkl_scene_maps = {}
        batch_pkl_feature_counts = {}
        
        total_scenes = 0
        total_vectors = 0
        
        # CPU并行加载PKL
        with ThreadPoolExecutor(max_workers=self.load_workers) as executor:
            futures = {executor.submit(self._load_single_pkl, p): p for p in pkl_paths}
            
            for future in as_completed(futures):
                pkl_path = futures[future]
                try:
                    features_np, scene_map, feature_counts = future.result()
                    
                    if features_np is not None and len(scene_map) > 0:
                        # 检查维度截断
                        truncate_dim = self.truncate_dim
                        if truncate_dim is not None and features_np.shape[1] > truncate_dim:
                            features_np = features_np[:, :truncate_dim]
                            norms = np.linalg.norm(features_np, axis=1, keepdims=True)
                            features_np = features_np / (norms + 1e-8)
                        
                        # 送入GPU
                        dtype = torch.float16 if self.use_fp16 else torch.float32
                        features_gpu = torch.tensor(features_np, device=DEVICE, dtype=dtype)
                        
                        batch_pkl_features[pkl_path] = features_gpu
                        batch_pkl_scene_maps[pkl_path] = scene_map
                        batch_pkl_feature_counts[pkl_path] = feature_counts
                        
                        total_scenes += len(scene_map)
                        total_vectors += features_np.shape[0]
                        
                except Exception as e:
                    print(f"[错误] {os.path.basename(pkl_path)}: 加载失败 - {e}")
        
        if not batch_pkl_features:
            print("[警告] 该批次没有加载到任何特征数据")
            return
        
        # 获取 logit_scale（如果还没有）
        if self._logit_scale is None:
            self._ensure_logit_scale()
        
        # 设置总场景数
        self.num_scenes = total_scenes
        
        # 合并所有PKL特征到 all_features_gpu
        all_features_list = []
        all_scene_maps = []
        all_feature_counts = []
        all_scene_indices = []
        all_frame_indices = []
        scene_offset = 0
        
        # 同时更新 pkl_features 等字典（用于兼容性）
        self.pkl_features = batch_pkl_features
        self.pkl_scene_maps = batch_pkl_scene_maps
        self.pkl_feature_counts = batch_pkl_feature_counts
        
        for pkl_path in batch_pkl_features:
            features_gpu = batch_pkl_features[pkl_path]
            scene_map = batch_pkl_scene_maps[pkl_path]
            feature_counts = batch_pkl_feature_counts[pkl_path]
            
            all_features_list.append(features_gpu)
            all_scene_maps.extend(scene_map)
            all_feature_counts.extend(feature_counts)
            
            for scene_idx, count in enumerate(feature_counts):
                all_scene_indices.extend([scene_offset + scene_idx] * count)
                all_frame_indices.extend(list(range(count)))
            scene_offset += len(scene_map)
        
        # 合并到单个张量
        self.all_features_gpu = torch.cat(all_features_list, dim=0)
        self.scene_map = all_scene_maps
        self.feature_counts = all_feature_counts
        self.scene_indices_gpu = torch.tensor(all_scene_indices, device=DEVICE, dtype=torch.long)
        self.frame_indices_gpu = torch.tensor(all_frame_indices, device=DEVICE, dtype=torch.long)
        self.scene_frame_indices_gpu = self.scene_indices_gpu * 3 + self.frame_indices_gpu
        
        load_time = time.time() - start_time
        print(f"[分批PKL加载] {len(batch_pkl_features)} 个PKL, {total_scenes} 场景, {total_vectors} 向量, 耗时 {load_time:.2f}s")
    
    def _unload_merged_features(self):
        """
        释放合并的特征数据（用于分批PKL搜索，释放显存后加载下一批）
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
        
        # 释放 pkl_features 中的 GPU 张量
        if hasattr(self, 'pkl_features'):
            for pkl_path in list(self.pkl_features.keys()):
                if self.pkl_features[pkl_path] is not None:
                    del self.pkl_features[pkl_path]
            self.pkl_features = {}
        if hasattr(self, 'pkl_scene_indices'):
            for pkl_path in list(self.pkl_scene_indices.keys()):
                if self.pkl_scene_indices[pkl_path] is not None:
                    del self.pkl_scene_indices[pkl_path]
            self.pkl_scene_indices = {}
        
        self.scene_map = []
        self.feature_counts = []
        self.num_scenes = 0
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def cleanup(self):
        """释放GPU资源"""
        # 清理按PKL模式的数据
        if hasattr(self, 'pkl_features') and self.pkl_features:
            # 静默：清理信息
            for pkl_path in list(self.pkl_features.keys()):
                del self.pkl_features[pkl_path]
            self.pkl_features.clear()
        
        if hasattr(self, 'pkl_scene_indices'):
            for pkl_path in list(self.pkl_scene_indices.keys()):
                del self.pkl_scene_indices[pkl_path]
            self.pkl_scene_indices.clear()
        
        if hasattr(self, 'pkl_scene_maps'):
            self.pkl_scene_maps.clear()
        
        if hasattr(self, 'pkl_feature_counts'):
            self.pkl_feature_counts.clear()
        
        # 清理旧的合并模式数据（兼容性）
        if self.all_features_gpu is not None:
            # 静默：清理信息
            del self.all_features_gpu
            self.all_features_gpu = None
        
        if self.scene_indices_gpu is not None:
            del self.scene_indices_gpu
            self.scene_indices_gpu = None
        
        # 清理帧索引 GPU 张量（frame_indices_gpu, scene_frame_indices_gpu）
        if hasattr(self, 'frame_indices_gpu') and self.frame_indices_gpu is not None:
            del self.frame_indices_gpu
            self.frame_indices_gpu = None
        if hasattr(self, 'scene_frame_indices_gpu') and self.scene_frame_indices_gpu is not None:
            del self.scene_frame_indices_gpu
            self.scene_frame_indices_gpu = None
        
        # 清理按视频模式的数据
        if self.video_features:
            # 静默：清理信息
            for video_path in list(self.video_features.keys()):
                del self.video_features[video_path]
            self.video_features.clear()
        
        self.video_scene_maps.clear()
        self.video_feature_counts.clear()
        
        if self._logit_scale is not None:
            del self._logit_scale
            self._logit_scale = None
        
        self.scene_map = []
        self.feature_counts = []
        self.pkl_boundaries = []
        self.num_scenes = 0
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 静默：清理完成信息


# ============================================================
#  向量去重工具函数
# ============================================================

def extract_label_key_from_result_name(
    result_name: str,
    video_name_format: str,
    exclude_fields: list = None
) -> str:
    """
    从 result_name 中提取标签组合键（用于分组）
    
    Args:
        result_name: 如 'P模式_男人_跑步_城市_开心_1234_视频名'
        video_name_format: 如 'P模式_{主体}_{动作}_{场景}_{情绪}_{起始帧}_{视频解析名}'
        exclude_fields: 排除的字段，默认 ['起始帧', '视频解析名']
    
    Returns:
        标签组合键，如 'P模式_男人_跑步_城市_开心'
    """
    import re
    
    if exclude_fields is None:
        exclude_fields = ['起始帧', '视频解析名']
    
    # 解析格式字符串中的所有占位符
    placeholders = re.findall(r'\{([^}]+)\}', video_name_format)
    
    # 将格式字符串转换为正则表达式
    # 例如: 'P模式_{主体}_{动作}' -> 'P模式_(.+?)_(.+?)'
    pattern = video_name_format
    for ph in placeholders:
        pattern = pattern.replace(f'{{{ph}}}', '(.+?)')
    pattern = pattern + '$'
    
    # 匹配 result_name
    match = re.match(pattern, result_name)
    if not match:
        return result_name  # 无法解析，返回原名
    
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
    从 result_name 中提取视频解析名
    
    Args:
        result_name: 如 'P模式_男人_跑步_城市_开心_1234_视频名_OP'
        video_name_format: 如 'P模式_{主体}_{动作}_{场景}_{情绪}_{起始帧}_{视频解析名}'
    
    Returns:
        视频解析名，如 '视频名_OP'
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
    
    # 找到 '视频解析名' 占位符的位置
    for i, ph in enumerate(placeholders):
        if ph == '视频解析名':
            return match.group(i + 1)
    
    return ''


def is_op_ed_video(video_parsed_name: str) -> bool:
    """
    检查视频解析名是否包含 OP 或 ED 字段（不分大小写）
    
    支持的格式：
    - 标准格式: op, ed, _op_, _ed_, op_, ed_, _op, _ed
    - 带编号: op01, ed01, op1, ed1, op_01, ed_01
    - NC版本: ncop, nced, ncop01, nced01
    - 混合格式: video_op01_xxx, video_ncop_xxx
    
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
        # NC版本（无字幕版）: ncop, nced, ncop01, nced01, ncop_01
        r'(?:^|_)nc(?:op|ed)(?:\d+|_\d+)?(?:_|$)',
        
        # 带编号的 OP/ED: op01, ed01, op1, ed1, op_01, ed_01
        r'(?:^|_)(?:op|ed)(?:\d+|_\d+)(?:_|$)',
        
        # 标准 OP/ED: _op_, _ed_, op_, ed_, _op, _ed
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
    使用矩阵运算进行余弦相似度去重（支持多帧向量）
    
    比较两个场景时，计算所有帧向量之间的最大相似度。
    例如：场景A有3帧 [a1,a2,a3]，场景B有3帧 [b1,b2,b3]
    计算 3×3=9 个相似度，取最大值作为两个场景的相似度。
    
    Args:
        scene_vectors_list: 场景向量列表，每个元素是 [k, D] 的矩阵（k帧向量）
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
    
    # 向量已经归一化，直接计算相似度矩阵 [total_frames, total_frames]
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
        # 标记与 i 相似度超过阈值的后续场景为重复
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
    跨视频去重版本：仅比较不同视频之间的场景相似度，同视频场景互不去重。
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
    在视频导出前进行同标签向量去重（同PKL内、跨视频去重）
    
    优先规则：
    - 如果同标签组内有 OP/ED 视频，只保留 OP/ED 视频，跳过向量去重
    - 否则进行正常的向量相似度去重
    
    去重范围：
    - 只在同一PKL内进行去重，不跨PKL去重
    - 同一PKL+同标签组下，仅做跨视频去重（同视频内不去重）
    
    Args:
        best_matches: 搜索结果字典 {scene_key: {...}}
        scene_features: 场景特征向量 {scene_key: np.ndarray}
        video_name_format: 视频命名格式
        similarity_threshold: 余弦相似度阈值（0-1），超过则去重
        exclude_fields: 去重时排除的字段，默认 ['起始帧', '视频解析名']
        scene_pkl_map: 场景到PKL的映射 {scene_key: source_pkl}，用于同PKL内去重
    
    Returns:
        去重后的结果字典
    """
    if exclude_fields is None:
        exclude_fields = ['起始帧', '视频解析名']
    
    if not best_matches or similarity_threshold is None:
        return best_matches
    
    # Step 1: 按 (PKL, 标签组合) 分组，实现同PKL内跨视频去重
    # 格式: {(pkl_path, label_key): [scene_key, ...]}
    groups = defaultdict(list)
    
    for scene_key, data in best_matches.items():
        result_name = data.get('result_name', '')
        label_key = extract_label_key_from_result_name(
            result_name, video_name_format, exclude_fields
        )
        # 获取该场景的 PKL 来源
        pkl_path = scene_pkl_map.get(scene_key, 'unknown') if scene_pkl_map else 'unknown'
        # 使用 (pkl_path, label_key) 作为分组键，实现同PKL内去重
        group_key = (pkl_path, label_key)
        groups[group_key].append(scene_key)
    
    # Step 2: 对每个组进行去重
    keep_scene_keys = set()
    op_ed_priority_count = 0
    vector_dedup_count = 0
    
    for group_key, scene_keys in groups.items():
        if len(scene_keys) == 1:
            # 只有一个，直接保留
            keep_scene_keys.add(scene_keys[0])
            continue
        
        # ========== OP/ED 优先规则 ==========
        # 检查组内是否有 OP/ED 视频
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
    print(f"[向量去重] OP/ED优先过滤: {op_ed_priority_count} 个, 向量相似度去重: {vector_dedup_count} 个")
    print(f"[向量去重] 总计去除: {total_removed} 个, 保留: {len(keep_scene_keys)} 个")
    
    # Step 3: 返回去重后的结果
    return {k: v for k, v in best_matches.items() if k in keep_scene_keys}


# ============================================================
#  测试入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("批量文搜图引擎 - 测试")
    print("=" * 60)
    
    # 延迟导入
    from A_coreUtils.embedding.embedding_model import EmbeddingModelProcessor
    from A_coreUtils.prompt.prompt_vector_cache import PromptVectorCache
    
    # 查找索引文件
    resolver = PathResolver()
    index_dir = str(resolver.project_root / 'indexes')
    
    if os.path.isdir(index_dir):
        pkl_files = [os.path.join(index_dir, f) for f in os.listdir(index_dir) if f.endswith('.pkl')]
        
        if pkl_files:
            print(f"找到 {len(pkl_files)} 个PKL文件")
            
            # 从第一个PKL提取模型名
            import re
            filename = os.path.basename(pkl_files[0])
            basename = filename.replace('.pkl', '')
            first_underscore = basename.find('_')
            if first_underscore != -1:
                model_name = basename[first_underscore + 1:]
                # 去掉维度后缀
                dim_match = re.search(r'_d\d+$', model_name)
                if dim_match:
                    model_name = model_name[:dim_match.start()]
                
                print(f"检测到模型: {model_name}")
                
                # 创建处理器
                processor = EmbeddingModelProcessor(model_name=model_name, use_fp16=True)
                
                # 创建批量搜索引擎
                engine = BatchTextSearchEngine(
                    processor, pkl_files,
                    video_name_format="{主体}_{动作}_{起始帧}_{视频解析名}"
                )
                
                # 预加载特征
                print("\n预加载PKL特征到GPU...")
                engine._preload_all_features()
                
                # 创建 prompt 向量缓存
                print("\n创建 prompt 向量缓存...")
                prompt_template = "A {情绪} photo of a {主体} {动作} in {场景}."
                prompt_cache = PromptVectorCache(
                    processor=processor,
                    prompt_template=prompt_template,
                    batch_size=512
                )
                
                # 自动生成或使用现有缓存
                prompt_cache.generate_cache()
                
                # 获取缓存迭代器
                cache_iterator = prompt_cache.load_cache_batched(
                    model_name=model_name,
                    batch_size=1024
                )
                
                print(f"\n测试 search_with_batched_cache:")
                print(f"  - 总 prompt 数: {cache_iterator.total_prompts}")
                print(f"  - 批次数: {cache_iterator.num_batches}")
                
                # 使用 search_with_batched_cache 进行搜索
                results = engine.search_with_batched_cache(
                    cache_iterator=cache_iterator,
                    threshold=15.0,
                    use_diskcache=True,
                    search_mode=0,  # 按PKL模式
                    result_top_k=10
                )
                
                print(f"\n搜索结果: {len(results)} 个场景")
                for scene_key, data in list(results.items())[:5]:
                    print(f"  - {scene_key}: sim={data['similarity']:.2f}, {data['result_name']}")
                
                # 清理
                engine.cleanup()
        else:
            print("未找到PKL文件")
    else:
        print(f"索引目录不存在: {index_dir}")
    
    print("\n" + "=" * 60)
