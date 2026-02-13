# label_traverse_search.py
# 遍历模式标签匹配搜索
# v1.0: 初始版本
# v2.0: 添加 LMDB 磁盘缓存支持
# v3.0: 移除 Reranker，改为跨帧选择最高相似度标签
#       - 每个场景有N帧特征向量（通常为3帧：起始帧、中间帧、结束帧）
#       - 对每个大类的每个标签，分别计算与N帧的相似度
#       - 选择N帧中相似度最高的那个作为该标签的最终相似度
#       - 每个大类选择相似度最高的标签
# v4.0: GPU矩阵运算优化
#       - 将跨帧选择从Python循环改为GPU矩阵运算
#       - 使用 tensor.view() + tensor.max(dim=2) 实现快速跨帧取最大值
#       - 最小化GPU-CPU数据传输，只传输有效结果
#       - 支持快速路径（所有场景帧数相同）和慢速路径（帧数不同）
# v5.0: 移除分配规则
#       - 标签模式不使用分配规则，必须跳过 '分配规则' 和 '选词填空规则'
#       - 只提取纯标签数据，每个大类独立选择相似度最高的标签
#
# 功能说明：
# 1. 遍历每个大类的所有子标签，用单个英文词计算与场景的相似度
# 2. 每个场景的多帧特征向量中，选择相似度最高的（跨帧选择）
# 3. 每个场景记录每个大类中相似度最高且大于阈值的标签
# 4. 如果某个大类没有大于阈值的标签，则丢弃该场景
# 注意：标签模式不使用分配规则（与Prompt模式不同）

import os
import sys
import json
import pickle
import hashlib
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import gc
from typing import Dict, List, Tuple, Optional, Any, Union, Generator
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入 LMDB（必需）
try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False
    lmdb = None

# ============================================================
#  路径设置
# ============================================================
_current_file = os.path.abspath(__file__)
_search_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_search_dir)
_cut_detect_scene_dir = os.path.dirname(_a_core_utils_dir)
if _cut_detect_scene_dir not in sys.path:
    sys.path.insert(0, _cut_detect_scene_dir)

from path_resolver import PathResolver

# GPU设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class LabelMatch:
    """单个标签的匹配结果"""
    category: str           # 大类名称（如"主体"）
    subcategory: str        # 子类名称（如"生物"）
    label_cn: str           # 中文标签
    label_en: str           # 英文标签
    similarity: float       # 相似度
    
    def to_dict(self) -> Dict:
        """转换为字典（用于序列化）"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LabelMatch':
        """从字典创建（用于反序列化）"""
        return cls(**data)


@dataclass
class SceneLabelResult:
    """场景的标签匹配结果"""
    scene_key: str                          # 场景唯一标识
    video_path: str                         # 视频路径
    start_frame: int                        # 起始帧
    end_frame: int                          # 结束帧
    fps: float                              # 帧率
    labels: Dict[str, LabelMatch] = field(default_factory=dict)  # 每个大类的最佳匹配
    valid: bool = True                      # 是否有效（所有大类都有匹配）
    
    def to_dict(self) -> Dict:
        """转换为字典（用于序列化）"""
        return {
            'scene_key': self.scene_key,
            'video_path': self.video_path,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'fps': self.fps,
            'labels': {k: v.to_dict() for k, v in self.labels.items()},
            'valid': self.valid
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SceneLabelResult':
        """从字典创建（用于反序列化）"""
        labels = {k: LabelMatch.from_dict(v) for k, v in data.get('labels', {}).items()}
        return cls(
            scene_key=data['scene_key'],
            video_path=data['video_path'],
            start_frame=data['start_frame'],
            end_frame=data['end_frame'],
            fps=data['fps'],
            labels=labels,
            valid=data.get('valid', True)
        )


def validate_label_video_name_format(video_name_format: str, categories: List[str]) -> Tuple[bool, str]:
    """
    验证标签模式的 video_name_format 是否合法
    
    检查格式字符串中的占位符是否都是有效的大类名称或系统占位符。
    
    Args:
        video_name_format: 视频名称格式字符串
        categories: 有效的大类名称列表
    
    Returns:
        (是否合法, 错误信息)
    """
    import re
    
    # 系统占位符
    system_placeholders = {'起始帧', '视频解析名', '扩展名'}
    
    # 提取所有占位符
    placeholders = re.findall(r'\{(\w+)\}', video_name_format)
    
    # 检查每个占位符
    invalid_placeholders = []
    for ph in placeholders:
        if ph not in system_placeholders and ph not in categories:
            invalid_placeholders.append(ph)
    
    if invalid_placeholders:
        valid_list = list(categories) + list(system_placeholders)
        return False, f"video_name_format 中存在无效占位符: {invalid_placeholders}，有效占位符: {valid_list}"
    
    return True, ""


def convert_label_results_to_dict(
    results: List['SceneLabelResult'],
    video_name_format: str
) -> Dict[str, Dict]:
    """
    将 SceneLabelResult 列表转换为 best_matches 字典格式
    
    用于复用 P模式的向量去重和相邻片段合并逻辑
    
    Args:
        results: SceneLabelResult 列表
        video_name_format: 视频名称格式模板
    
    Returns:
        best_matches 格式的字典 {scene_key: {...}}
    """
    import re
    from A_coreUtils.video_processing.video_name_parser import VideoNameParser
    
    parser = VideoNameParser()
    best_matches = {}
    
    for result in results:
        scene_key = result.scene_key
        
        # 解析视频名称
        video_basename = os.path.basename(result.video_path)
        video_name_no_ext = os.path.splitext(video_basename)[0]
        parsed_name = parser.parse(video_name_no_ext)
        
        # 构建替换字典
        placeholders = re.findall(r'\{(\w+)\}', video_name_format)
        replacements = {ph: "" for ph in placeholders}
        
        # 填充系统占位符
        replacements['起始帧'] = str(result.start_frame)
        replacements['视频解析名'] = parsed_name
        
        # 填充标签
        avg_similarity = 0
        for category, match in result.labels.items():
            if category in replacements:
                replacements[category] = match.label_cn
            # 添加 xxx_cn 字段用于去重分组
            replacements[f'{category}_cn'] = match.label_cn
            avg_similarity += match.similarity
        
        if result.labels:
            avg_similarity /= len(result.labels)
        
        # 生成 result_name
        result_name = video_name_format.format(**replacements)
        
        # 清理连续分隔符
        separators = re.findall(r'\}([^\{]+)\{', video_name_format)
        for sep in set(separators):
            if sep:
                double_sep = sep + sep
                while double_sep in result_name:
                    result_name = result_name.replace(double_sep, sep)
        for sep in set(separators):
            if sep:
                result_name = result_name.strip(sep)
        
        # 构建数据字典
        data = {
            'result_name': result_name,
            'similarity': avg_similarity,
            'video_path': result.video_path,
            'start_frame': result.start_frame,
            'end_frame': result.end_frame,
            'fps': result.fps,
            'source_pkl': getattr(result, 'source_pkl', 'unknown'),
        }
        
        # 添加标签字段
        for category, match in result.labels.items():
            data[f'{category}_cn'] = match.label_cn
            data[f'{category}_en'] = match.label_en
        
        best_matches[scene_key] = data
    
    return best_matches


def convert_dict_to_label_results(
    best_matches: Dict[str, Dict],
    categories: List[str]
) -> List['SceneLabelResult']:
    """
    将 best_matches 字典转换回 SceneLabelResult 列表
    
    用于去重和合并后恢复原始数据结构
    
    Args:
        best_matches: best_matches 格式的字典
        categories: 大类名称列表
    
    Returns:
        SceneLabelResult 列表
    """
    results = []
    
    for scene_key, data in best_matches.items():
        # 重建标签字典
        labels = {}
        for category in categories:
            cn_key = f'{category}_cn'
            en_key = f'{category}_en'
            if cn_key in data and data[cn_key]:
                labels[category] = LabelMatch(
                    category=category,
                    subcategory='',  # 合并后可能丢失子类信息
                    label_cn=data[cn_key],
                    label_en=data.get(en_key, ''),
                    similarity=data.get('similarity', 0)
                )
        
        result = SceneLabelResult(
            scene_key=scene_key,
            video_path=data.get('video_path', ''),
            start_frame=data.get('start_frame', 0),
            end_frame=data.get('end_frame', 0),
            fps=data.get('fps', 25.0),
            labels=labels,
            valid=True
        )
        results.append(result)
    
    return results


def generate_default_label_video_name_format(categories: List[str], prefix: str = "标签模式") -> str:
    """
    动态生成标签模式的默认 video_name_format
    
    Args:
        categories: 大类名称列表
        prefix: 前缀字符串
    
    Returns:
        默认的 video_name_format 字符串
    """
    format_parts = [prefix] if prefix else []
    format_parts.extend([f"{{{cat}}}" for cat in categories])
    format_parts.extend(["{起始帧}", "{视频解析名}"])
    return "_".join(format_parts)


class LabelResultLMDBCache:
    """
    标签搜索结果的 LMDB 缓存
    
    用于存储大量场景的标签匹配结果，避免内存溢出。
    支持断点续传：通过 checkpoint 机制记录已完成的 PKL，程序中断后可从断点继续。
    """
    
    # 特殊 key
    CHECKPOINT_KEY = b'__checkpoint__'
    
    def __init__(self, cache_dir: str, map_size: int = 10 * 1024 * 1024 * 1024, config_hash: str = None):
        """
        初始化 LMDB 缓存（支持断点续传）
        
        Args:
            cache_dir: 缓存目录
            map_size: LMDB 最大大小（默认 10GB）
            config_hash: 搜索配置哈希，用于断点续传验证。
                         如果与已有 checkpoint 的 hash 不匹配，则清空重建。
        """
        if not LMDB_AVAILABLE:
            raise ImportError("LMDB 未安装，请运行: pip install lmdb")
        
        self.cache_dir = cache_dir
        self._config_hash = config_hash
        
        # 检查是否可以断点续传
        should_clear = True
        if config_hash is not None and os.path.exists(cache_dir):
            try:
                # 尝试打开已有 LMDB 检查 checkpoint
                test_env = lmdb.open(cache_dir, map_size=map_size, max_dbs=0, readonly=True)
                with test_env.begin() as txn:
                    cp_data = txn.get(self.CHECKPOINT_KEY)
                    if cp_data is not None:
                        checkpoint = pickle.loads(cp_data)
                        if checkpoint.get('config_hash') == config_hash:
                            should_clear = False
                            completed = checkpoint.get('completed_pkls', [])
                            print(f"[标签搜索LMDB] 发现有效断点: 已完成 {len(completed)} 个PKL")
                test_env.close()
            except Exception as e:
                print(f"[标签搜索LMDB] 检查断点失败: {e}，将清空重建")
                should_clear = True
        
        if should_clear:
            # 仅清理当前 LMDB 目录，避免误删其它 temp 内容
            if os.path.exists(cache_dir):
                print(f"[标签搜索LMDB] 清理当前缓存目录: {cache_dir}")
                from A_coreUtils.video_processing.video_utils import cleanup_temp_folder
                cleanup_temp_folder(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            if config_hash is not None:
                print(f"[标签搜索LMDB] 全新搜索，config_hash={config_hash[:8]}...")
        
        self.env = lmdb.open(
            cache_dir,
            map_size=map_size,
            max_dbs=0,
            sync=False,
            writemap=True
        )
        self._count = 0
        
        # 统计已有记录数
        if not should_clear:
            with self.env.begin() as txn:
                self._count = txn.stat()['entries']
                # 减去 checkpoint key
                if txn.get(self.CHECKPOINT_KEY) is not None:
                    self._count -= 1
    
    def put(self, scene_key: str, result: SceneLabelResult):
        """存储结果"""
        with self.env.begin(write=True) as txn:
            key_bytes = scene_key.encode('utf-8')
            existed = txn.get(key_bytes) is not None
            data = pickle.dumps(result.to_dict())
            txn.put(key_bytes, data)
            if not existed:
                self._count += 1
    
    def put_many(self, items: List[Tuple[str, SceneLabelResult]]):
        """鎵归噺瀛樺偍缁撴灉锛堝崟浜嬪姟锛夈€?"""
        if not items:
            return

        with self.env.begin(write=True) as txn:
            for scene_key, result in items:
                key_bytes = scene_key.encode('utf-8')
                existed = txn.get(key_bytes) is not None
                data = pickle.dumps(result.to_dict())
                txn.put(key_bytes, data)
                if not existed:
                    self._count += 1

    def get(self, scene_key: str) -> Optional[SceneLabelResult]:
        """获取结果"""
        with self.env.begin() as txn:
            data = txn.get(scene_key.encode('utf-8'))
            if data is None:
                return None
            return SceneLabelResult.from_dict(pickle.loads(data))
    
    def get_all(self) -> List[SceneLabelResult]:
        """获取所有结果（跳过特殊 key）"""
        return list(self.iter_results())

    def iter_results(self) -> Generator[SceneLabelResult, None, None]:
        """stream all results except checkpoint key"""
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                # 跳过 checkpoint key
                if key == self.CHECKPOINT_KEY:
                    continue
                yield SceneLabelResult.from_dict(pickle.loads(value))
    
    def save_checkpoint(self, completed_pkls: list):
        """保存断点信息"""
        import time as _time
        checkpoint = {
            'config_hash': self._config_hash,
            'completed_pkls': completed_pkls,
            'timestamp': _time.time()
        }
        with self.env.begin(write=True) as txn:
            txn.put(self.CHECKPOINT_KEY, pickle.dumps(checkpoint))
    
    def load_checkpoint(self) -> Optional[dict]:
        """加载断点信息"""
        with self.env.begin() as txn:
            data = txn.get(self.CHECKPOINT_KEY)
            if data is None:
                return None
            return pickle.loads(data)
    
    def get_completed_pkls(self) -> set:
        """获取已完成的 PKL 路径集合"""
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return set()
        return set(checkpoint.get('completed_pkls', []))
    
    def __len__(self) -> int:
        return self._count
    
    def close(self):
        """关闭缓存"""
        if self.env:
            self.env.close()
            self.env = None
    
    def destroy(self):
        """关闭并删除整个 LMDB 目录"""
        self.close()
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir, ignore_errors=True)


class KeywordManager:
    """
    关键词管理器 - 从 logic_keywords.json 的 "PL标签" 下加载并管理关键词
    
    注意：标签模式不使用分配规则，只提取纯标签数据
    """
    
    # 系统保留字段（不作为大类处理，标签模式必须跳过）
    RESERVED_KEYS = {'分配规则', '选词填空规则'}
    
    def __init__(self, keywords_path: str = None):
        """
        初始化关键词管理器
        
        Args:
            keywords_path: logic_keywords.json 路径，None则自动查找
        """
        if keywords_path is None:
            resolver = PathResolver()
            keywords_path = str(resolver.project_root / 'logic_keywords.json')
        
        self.keywords_path = keywords_path
        self._data: Dict = {}
        self._categories: List[str] = []
        
        self._load_keywords()
    
    def _load_keywords(self):
        """
        加载关键词配置
        
        从 "PL标签" 下读取标签数据。
        标签模式不使用分配规则，必须跳过 RESERVED_KEYS 中的字段。
        """
        with open(self.keywords_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        
        # 从 "PL标签" 下提取数据（必须存在）
        if "PL标签" not in full_data:
            raise KeyError("logic_keywords.json 中必须包含 'PL标签' 键")
        self._data = full_data["PL标签"]
        
        # 从 _data 中移除保留字段（分配规则、选词填空规则），标签模式不使用
        for key in self.RESERVED_KEYS:
            if key in self._data:
                del self._data[key]
        
        # 获取所有大类（排除说明字段）
        self._categories = [k for k in self._data.keys() if not k.startswith('_')]
        
        # 静默：大类加载信息
    
    @property
    def categories(self) -> List[str]:
        """获取所有大类名称"""
        return self._categories
    
    def get_category_data(self, category: str) -> Dict[str, Dict[str, str]]:
        """
        获取某个大类的所有子类和标签
        
        Returns:
            {子类名: {中文标签: 英文标签}}
        """
        return self._data.get(category, {})
    
    def get_all_labels_flat(self, category: str) -> List[Tuple[str, str, str]]:
        """
        获取某个大类的所有标签（扁平化）
        
        Returns:
            [(子类名, 中文标签, 英文标签), ...]
        """
        result = []
        category_data = self.get_category_data(category)
        for subcategory, labels in category_data.items():
            if subcategory.startswith('_'):
                continue
            for cn, en in labels.items():
                result.append((subcategory, cn, en))
        return result
    
    def count_total_labels(self) -> int:
        """
        计算总标签数（数学计算，不遍历）
        
        Returns:
            所有大类的标签数总和
        """
        total = 0
        for category in self._categories:
            category_data = self.get_category_data(category)
            for subcategory, labels in category_data.items():
                if subcategory.startswith('_'):
                    continue
                total += len(labels)
        return total
    
    def iterate_labels_flat(self) -> Generator[Tuple[str, str, str, str], None, None]:
        """
        轻量版遍历：生成所有标签的元组
        
        用于保存缓存时避免内存爆炸。
        
        Yields:
            (category, subcategory, cn, en) 元组
        """
        for category in self._categories:
            category_data = self.get_category_data(category)
            for subcategory, labels in category_data.items():
                if subcategory.startswith('_'):
                    continue
                for cn, en in labels.items():
                    yield (category, subcategory, cn, en)


class LabelVectorCache:
    """
    标签向量缓存器 - 预计算所有标签的归一化向量
    
    v4.0: memmap 流式写入优化
    - 向量: .dat 文件（memmap 流式写入，极低内存占用）
    - 向量元信息: .meta.json 文件（存储 shape 和 dtype）
    - 标签信息: .pkl 文件（一次性加载）
    
    缓存文件命名格式:
    - label_cache_{model_name}_{lang}_vectors.dat (向量，memmap)
    - label_cache_{model_name}_{lang}_vectors.meta.json (向量元信息)
    - label_cache_{model_name}_{lang}_labels.pkl (标签信息 + 元数据)
    """
    
    def __init__(self,
                 processor=None,
                 keywords_path: str = None,
                 cache_dir: str = None,
                 model_name: str = None,
                 use_chinese_labels: bool = False):
        """
        初始化标签向量缓存器
        
        Args:
            processor: EmbeddingModelProcessor 实例（生成缓存时需要）
            keywords_path: logic_keywords.json 路径
            cache_dir: 缓存目录，None则使用 templates/prompt_cache
            model_name: 模型名称（用于缓存文件命名）
            use_chinese_labels: 是否使用中文标签模式
                - False（默认）: 使用英文标签值编码向量，哈希基于英文标签
                - True: 使用中文标签键名编码向量，哈希基于中文标签
        """
        self.processor = processor
        self.use_chinese_labels = use_chinese_labels
        
        # 路径设置
        resolver = PathResolver()
        if keywords_path is None:
            keywords_path = str(resolver.project_root / 'logic_keywords.json')
        self.keywords_path = keywords_path
        
        if cache_dir is None:
            cache_dir = str(resolver.project_root / 'templates' / 'prompt_cache')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 模型名称
        self.model_name = model_name or "unknown"
        
        # 加载关键词
        self.keyword_manager = KeywordManager(keywords_path)
        
        # 计算关键词哈希（用于缓存验证）
        self._keywords_hash = self._compute_keywords_hash()
    
    def _compute_keywords_hash(self) -> str:
        """
        计算关键词配置的哈希值
        
        根据 use_chinese_labels 参数选择哈希内容：
        - False: 基于英文标签值计算哈希
        - True: 基于中文标签键名计算哈希
        """
        # 收集所有标签
        all_labels = []
        for category in self.keyword_manager.categories:
            labels = self.keyword_manager.get_all_labels_flat(category)
            if self.use_chinese_labels:
                # 中文模式：使用中文标签键名
                for _, cn, _ in labels:
                    all_labels.append(cn)
            else:
                # 英文模式：使用英文标签值
                for _, _, en in labels:
                    all_labels.append(en)
        
        # 排序后计算哈希
        all_labels.sort()
        content = json.dumps(all_labels, ensure_ascii=False)
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    
    def _get_safe_model_name(self) -> str:
        """获取安全的模型名称（用于文件命名）"""
        return self.model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    def get_cache_path(self) -> str:
        """
        获取缓存基础路径（不带扩展名）
        
        v2.0: 返回基础路径，各文件通过辅助方法获取：
        - get_vectors_path(): .npy 向量文件
        - get_labels_path(): .pkl 标签信息文件
        
        v3.0: 根据 use_chinese_labels 区分缓存文件：
        - 英文模式: label_cache_{model_name}_en
        - 中文模式: label_cache_{model_name}_cn
        """
        safe_model_name = self._get_safe_model_name()
        lang_suffix = "cn" if self.use_chinese_labels else "en"
        base_name = f"label_cache_{safe_model_name}_{lang_suffix}"
        return os.path.join(self.cache_dir, base_name)
    
    def get_vectors_path(self) -> str:
        """获取向量文件路径 (.dat，memmap格式)"""
        return self.get_cache_path() + "_vectors.dat"
    
    def get_vectors_meta_path(self) -> str:
        """获取向量元信息文件路径 (.meta.json)"""
        return self.get_cache_path() + "_vectors.meta.json"
    
    def get_labels_path(self) -> str:
        """获取标签信息文件路径 (.pkl)"""
        return self.get_cache_path() + "_labels.pkl"
    
    def cache_exists(self) -> bool:
        """
        检查缓存是否存在且有效（v4.0 memmap 格式）
        
        检查三个文件是否都存在：
        - vectors.dat: memmap 向量文件
        - vectors.meta.json: 向量元信息文件
        - labels.pkl: 标签信息 + 元数据
        
        同时验证哈希值：如果 logic_keywords.json 变化，缓存将被视为过期。
        """
        vectors_path = self.get_vectors_path()
        vectors_meta_path = self.get_vectors_meta_path()
        labels_path = self.get_labels_path()
        
        # 检查文件是否存在
        if not os.path.exists(vectors_path):
            return False
        if not os.path.exists(vectors_meta_path):
            return False
        if not os.path.exists(labels_path):
            return False
        
        # 验证 labels 文件有效性
        try:
            with open(labels_path, 'rb') as f:
                data = pickle.load(f)
            
            # 检查必要字段
            required_fields = ['labels', 'metadata']
            for field in required_fields:
                if field not in data:
                    print(f"[标签缓存无效] labels文件缺少字段: {field}")
                    return False
            
            # 验证哈希值（检测 keywords 是否变化）
            cached_hash = data['metadata'].get('keywords_hash', '')
            current_hash = self._keywords_hash
            if cached_hash != current_hash:
                print(f"[标签缓存过期] 配置已变化 (缓存哈希: {cached_hash}, 当前哈希: {current_hash})")
                print(f"  -> logic_keywords.json 已修改，需要重新生成缓存")
                return False
            
            # 验证向量元信息文件
            with open(vectors_meta_path, 'r', encoding='utf-8') as f:
                vectors_meta = json.load(f)
            
            # 验证向量文件可读且数量匹配
            vectors_mmap = np.memmap(
                vectors_path,
                dtype=vectors_meta['dtype'],
                mode='r',
                shape=tuple(vectors_meta['shape'])
            )
            if len(vectors_mmap) != len(data['labels']):
                print(f"[标签缓存无效] 向量数量({len(vectors_mmap)})与标签数量({len(data['labels'])})不匹配")
                del vectors_mmap
                return False
            del vectors_mmap
            
            return True
            
        except Exception as e:
            print(f"[标签缓存损坏] {e}")
            return False
    
    def generate_cache(self, force_regenerate: bool = False, batch_size: int = 512) -> str:
        """
        生成标签向量缓存（v5.0 memmap + 生产者-消费者并行模式）
        
        v5.0: memmap 流式写入 + 生产者-消费者并行
        - 生产者线程：GPU 编码向量
        - 消费者线程：memmap 写入
        - 向量: .dat 文件（memmap 流式写入）
        - 向量元信息: .meta.json 文件（存储 shape 和 dtype）
        - 标签信息: .pkl 文件（一次性加载）
        
        v3.0: 支持中英文标签模式
        - use_chinese_labels=False: 编码英文标签值
        - use_chinese_labels=True: 编码中文标签键名
        
        Args:
            force_regenerate: 是否强制重新生成
            batch_size: 批量编码大小（用于大规模标签时分批编码）
        
        Returns:
            缓存基础路径
        """
        import time
        from queue import Queue
        from threading import Thread, Event
        
        cache_base_path = self.get_cache_path()
        vectors_path = self.get_vectors_path()
        vectors_meta_path = self.get_vectors_meta_path()
        labels_path = self.get_labels_path()
        
        if self.cache_exists() and not force_regenerate:
            print(f"[标签缓存] 缓存已存在: {cache_base_path}")
            return cache_base_path
        
        if self.processor is None:
            raise RuntimeError("生成缓存需要 processor 实例")
        
        # 删除旧文件
        for path in [vectors_path, vectors_meta_path, labels_path]:
            if os.path.exists(path):
                os.remove(path)
        
        lang_mode = "中文" if self.use_chinese_labels else "英文"
        print("=" * 70)
        print(f"📝 标签向量缓存生成器 (v5.0 memmap + 生产者-消费者并行)")
        print("=" * 70)
        print(f"  模型: {self.model_name}")
        print(f"  缓存目录: {self.cache_dir}")
        print(f"  标签语言: {lang_mode}")
        print(f"  批量大小: {batch_size}")
        
        # 统计标签总数（数学计算，不遍历）
        total_labels = self.keyword_manager.count_total_labels()
        print(f"  标签数量: {total_labels}")
        
        # 根据模式提取标签文本（使用生成器，不累积）
        if self.use_chinese_labels:
            # 中文模式：使用中文标签键名
            texts = [cn for _, _, cn, _ in self.keyword_manager.iterate_labels_flat()]
        else:
            # 英文模式：使用英文标签值
            texts = [en for _, _, _, en in self.keyword_manager.iterate_labels_flat()]
        
        # 获取向量维度（通过编码一个样本）
        sample_vector = self.processor.encode_text([texts[0]])
        vector_dim = sample_vector.shape[1]
        print(f"  向量维度: {vector_dim}")
        
        # 创建 memmap 文件
        print(f"\n[步骤1] 创建 memmap 文件...")
        vectors_mmap = np.memmap(
            vectors_path,
            dtype='float32',
            mode='w+',
            shape=(total_labels, vector_dim)
        )
        
        # 保存向量元信息
        vectors_meta = {
            'shape': [total_labels, vector_dim],
            'dtype': 'float32',
            'total_labels': total_labels,
            'vector_dim': vector_dim
        }
        with open(vectors_meta_path, 'w', encoding='utf-8') as f:
            json.dump(vectors_meta, f, indent=2)
        
        # 生产者-消费者并行模式
        print(f"\n[步骤2] 生产者-消费者并行编码 (batch_size={batch_size})...")
        start_time = time.time()
        
        # 队列和事件
        write_queue = Queue(maxsize=4)
        stop_event = Event()
        
        # 消费者线程：负责 memmap 写入
        def writer_thread():
            while not stop_event.is_set() or not write_queue.empty():
                try:
                    item = write_queue.get(timeout=0.1)
                    if item is None:
                        break
                    idx, batch_vectors = item
                    vectors_mmap[idx:idx + len(batch_vectors)] = batch_vectors
                    vectors_mmap.flush()
                    write_queue.task_done()
                except:
                    continue
        
        # 启动写入线程
        writer = Thread(target=writer_thread, daemon=True)
        writer.start()
        
        # 生产者：编码并放入队列
        current_idx = 0
        for i in range(0, total_labels, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_vectors = self.processor.encode_text(batch_texts)
            
            # 放入写入队列
            write_queue.put((current_idx, batch_vectors))
            
            current_idx += len(batch_texts)
            
            if total_labels > batch_size:
                print(f"    [{current_idx}/{total_labels}]")
        
        # 等待写入完成
        write_queue.put(None)
        stop_event.set()
        writer.join()
        
        # 关闭 memmap
        del vectors_mmap
        
        elapsed_time = time.time() - start_time
        print(f"\n  编码完成! 耗时: {elapsed_time:.2f}s")
        
        vectors_size = os.path.getsize(vectors_path) / 1024
        print(f"    -> {os.path.basename(vectors_path)}: {vectors_size:.2f} KB")
        
        # 保存标签信息到 .pkl 文件（使用生成器重新生成，避免内存累积）
        print(f"\n[步骤3] 保存标签信息文件...")
        regenerated_labels = list(self.keyword_manager.iterate_labels_flat())
        labels_data = {
            'labels': regenerated_labels,
            'metadata': {
                'model_name': self.model_name,
                'keywords_hash': self._keywords_hash,
                'total_labels': total_labels,
                'vector_dim': vector_dim,
                'created_at': datetime.now().isoformat(),
                'keywords_path': self.keywords_path,
                'normalized': True,
                'format_version': '5.0',
                'use_chinese_labels': self.use_chinese_labels,
            }
        }
        with open(labels_path, 'wb') as f:
            pickle.dump(labels_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        del regenerated_labels, labels_data
        gc.collect()
        labels_size = os.path.getsize(labels_path) / 1024
        print(f"    -> {os.path.basename(labels_path)}: {labels_size:.2f} KB")
        
        print("\n" + "=" * 70)
        print("✅ 标签缓存生成完成!")
        print("=" * 70)
        
        gc.collect()
        return cache_base_path
    
    def load_cache(self, use_mmap: bool = True) -> Tuple[np.ndarray, List[Tuple[str, str, str, str]], Dict]:
        """
        加载缓存（v4.0 memmap 格式）
        
        Args:
            use_mmap: 是否使用内存映射加载向量（默认True）
        
        Returns:
            (vectors, labels, metadata)
            - vectors: [N, dim] 归一化向量（mmap或普通数组）
            - labels: [(category, subcategory, cn, en), ...]
            - metadata: 元数据字典
        """
        vectors_path = self.get_vectors_path()
        vectors_meta_path = self.get_vectors_meta_path()
        labels_path = self.get_labels_path()
        
        if not self.cache_exists():
            raise FileNotFoundError(f"缓存文件不存在: {self.get_cache_path()}")
        
        # 加载向量元信息
        with open(vectors_meta_path, 'r', encoding='utf-8') as f:
            vectors_meta = json.load(f)
        
        # 加载向量（使用memmap）
        if use_mmap:
            vectors = np.memmap(
                vectors_path,
                dtype=vectors_meta['dtype'],
                mode='r',
                shape=tuple(vectors_meta['shape'])
            )
        else:
            # 完整加载到内存
            vectors_mmap = np.memmap(
                vectors_path,
                dtype=vectors_meta['dtype'],
                mode='r',
                shape=tuple(vectors_meta['shape'])
            )
            vectors = np.array(vectors_mmap)
            del vectors_mmap
        
        # 加载标签信息
        with open(labels_path, 'rb') as f:
            labels_data = pickle.load(f)
        
        labels = labels_data['labels']
        metadata = labels_data['metadata']
        
        print(f"[标签缓存] 已加载 {len(labels)} 个标签向量 (mmap={use_mmap})")
        return vectors, labels, metadata


class LabelTraverseSearcher:
    """
    遍历模式标签匹配搜索器
    
    核心特色：跨帧选择最高相似度标签
    - 每个场景有N帧特征向量（通常为3帧：起始帧、中间帧、结束帧）
    - 对每个大类的每个标签，分别计算与N帧的相似度
    - 选择N帧中相似度最高的那个作为该标签的最终相似度
    - 每个大类选择相似度最高的标签
    
    核心逻辑：
    1. 预加载所有场景特征到GPU
    2. 对每个大类的每个子标签（英文词）编码并计算与所有场景的相似度
    3. 每个场景的多帧中选择相似度最高的（跨帧选择）
    4. 每个场景记录每个大类中相似度最高且大于阈值的标签
    5. 过滤掉任何大类没有匹配的场景
    
    注意：标签模式不使用分配规则（与Prompt模式不同）
    """
    
    def __init__(self,
                 processor,
                 index_paths: List[str],
                 keywords_path: str = None,
                 similarity_threshold: float = 20.0,
                 prompt_search_batch_size: int = 1024,
                 use_fp16: bool = True,
                 feature_fp16: Optional[bool] = None,
                 use_label_cache: bool = True,
                 cache_dir: str = None,
                 model_name: str = None,
                 # PKL加载参数
                 pkl_batch_size: int = None,
                 pkl_load_workers: int = 4,
                 # 搜索模式参数
                 search_mode: int = 0,
                 top_k: int = 50,
                 scene_chunk_size: Optional[int] = 1000,
                 # LMDB 缓存参数
                 use_diskcache: bool = True,
                 diskcache_dir: str = None,
                 lmdb_write_batch_size: int = 1000,
                 # 中文标签模式
                 use_chinese_labels: bool = False,
                 # 标签缓存批处理大小
                 label_cache_batch_size: int = 512):
        """
        初始化搜索器
        
        核心特色：跨帧选择最高相似度标签
        - 每个场景有3帧特征向量（起始帧、中间帧、结束帧）
        - 对每个大类的每个标签，分别计算与3帧的相似度
        - 选择3帧中相似度最高的那个作为该标签的最终相似度
        - 每个大类选择相似度最高的标签
        
        Args:
            processor: EmbeddingModelProcessor 实例
            index_paths: PKL索引文件路径列表
            keywords_path: logic_keywords.json 路径
            similarity_threshold: 相似度阈值
            prompt_search_batch_size: 搜索时每批加载的标签向量数量
            use_fp16: 是否使用FP16
            use_label_cache: 是否使用标签向量缓存（自动检测：有就用，没有就生成）
            cache_dir: 缓存目录
            model_name: 模型名称（用于缓存文件命名）
            pkl_batch_size: 每批加载的PKL数量
                - None 或 >= PKL总数: 一次性全部加载到GPU（显存占用高但搜索快）
                - < PKL总数: 分批加载，用完释放（显存占用低但稍慢）
            pkl_load_workers: PKL加载线程数（全量预加载时使用）
            search_mode: 搜索模式选择
                - -1（按视频模式）: 每个视频独立搜索，每个视频返回 top_k 个结果
                - 0（按PKL模式）: 每个PKL文件独立搜索，每个PKL返回 top_k 个结果
                - 1（跨PKL模式）: 全局搜索，返回全局 top_k 个结果
            top_k: 每组返回的最大结果数，None则不限制
            use_diskcache: 是否使用 LMDB 磁盘缓存存储结果（解决内存问题）
            diskcache_dir: LMDB 缓存目录，None则使用默认目录
            lmdb_write_batch_size: LMDB单事务写入批大小（分批加载时使用）
            use_chinese_labels: 是否使用中文标签模式
                - False（默认）: 使用英文标签值编码向量
                - True: 使用中文标签键名编码向量
            label_cache_batch_size: 标签缓存生成时的批处理大小
                - 用于分批编码标签向量，避免大规模标签时内存溢出
                - 推荐值: 256-1024
        """
        self.processor = processor
        self.index_paths = index_paths
        self.similarity_threshold = similarity_threshold
        self.prompt_search_batch_size = prompt_search_batch_size
        self.use_fp16 = use_fp16
        if feature_fp16 is None:
            feature_fp16 = use_fp16
        self.feature_fp16 = feature_fp16
        self.feature_dtype = torch.float16 if (self.feature_fp16 and DEVICE.type == 'cuda') else torch.float32
        self.use_label_cache = use_label_cache
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.use_chinese_labels = use_chinese_labels
        self.pkl_load_workers = pkl_load_workers
        self.lmdb_write_batch_size = lmdb_write_batch_size
        self.label_cache_batch_size = label_cache_batch_size
        
        # 处理 pkl_batch_size：None 或 >= PKL总数 表示全量加载
        total_pkls = len(index_paths)
        if pkl_batch_size is None or pkl_batch_size >= total_pkls:
            self.pkl_batch_size = total_pkls  # 全量加载
            self._preload_all = True
        else:
            self.pkl_batch_size = pkl_batch_size
            self._preload_all = False
        
        # 搜索模式参数
        if search_mode not in (-1, 0, 1):
            raise ValueError(f"search_mode only supports -1/0/1, got {search_mode}")
        self.search_mode = search_mode
        self.top_k = top_k
        self.scene_chunk_size = scene_chunk_size if (scene_chunk_size is not None and scene_chunk_size > 0) else None
        
        # LMDB 缓存参数
        self.use_diskcache = use_diskcache
        if diskcache_dir is None:
            resolver = PathResolver()
            diskcache_root = str(resolver.project_root / 'temp' / 'cache' / 'label_search_results')
        else:
            diskcache_root = diskcache_dir
        self.diskcache_dir = os.path.join(diskcache_root, 'pkl_merge')
        if self.use_diskcache:
            os.makedirs(self.diskcache_dir, exist_ok=True)
        self._lmdb_cache: Optional[LabelResultLMDBCache] = None
        self._label_cache_config_hash: Optional[str] = None
        
        # 关键词管理器
        self.keyword_manager = KeywordManager(keywords_path)
        
        # 标签向量缓存
        self._label_vectors: Optional[torch.Tensor] = None  # [N, dim] GPU tensor
        self._label_info: List[Tuple[str, str, str, str]] = []  # [(category, subcategory, cn, en), ...]
        
        # 按视频分组的数据结构（search_mode=-1时使用）
        self.video_features: Dict[str, torch.Tensor] = {}      # {video_path: [N, dim] GPU tensor}
        self.video_scene_maps: Dict[str, List[Dict]] = {}      # {video_path: [scene_info, ...]}
        self.video_feature_counts: Dict[str, List[int]] = {}   # {video_path: [count, ...]}
        
        # logit_scale
        self._logit_scale: Optional[torch.Tensor] = None
        
        # 预加载特征（根据模式和配置选择）
        if self._preload_all:
            if self.search_mode == -1:
                self._preload_features_per_video()
            else:
                self._preload_features()
        # 如果 _preload_all=False，则不预加载，搜索时分批加载
        
        # 预加载标签向量缓存
        if self.use_label_cache:
            self._preload_label_vectors()
    
    def _preload_label_vectors(self):
        """预加载标签向量缓存（自动检测：有就用，没有就生成）"""
        lang_mode = "中文" if self.use_chinese_labels else "英文"
        cache = LabelVectorCache(
            processor=self.processor,
            keywords_path=None,  # 使用默认路径
            cache_dir=self.cache_dir,
            model_name=self.model_name,
            use_chinese_labels=self.use_chinese_labels
        )
        
        # 自动检测缓存：有就用，没有就生成（cache_exists 内部会验证哈希）
        if not cache.cache_exists():
            print(f"[遍历搜索] 标签向量缓存不存在或已过期，正在生成（{lang_mode}模式）...")
            cache.generate_cache(batch_size=self.label_cache_batch_size)
        
        # 加载缓存
        vectors, labels, metadata = cache.load_cache()
        self._label_cache_config_hash = getattr(cache, '_config_hash', None)
        
        # 转换为GPU张量
        self._label_vectors = torch.tensor(vectors, device=DEVICE, dtype=self.feature_dtype)
        self._label_info = labels
        
        print(f"[遍历搜索] 已加载 {len(labels)} 个标签向量到GPU（{lang_mode}模式）")
    
    def _load_single_pkl(self, pkl_path: str) -> Tuple[Optional[np.ndarray], List[Dict], List[int]]:
        """
        加载单个PKL文件的特征
        
        Returns:
            tuple: (features_np, scene_map, feature_counts)
        """
        try:
            with open(pkl_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            pkl_features = []
            pkl_scene_map = []
            pkl_feature_counts = []
            
            for video_path, data in data_dict.items():
                scenes = data.get('scenes', []) if isinstance(data, dict) else data
                fps = data.get('fps', 25.0) if isinstance(data, dict) else 25.0
                
                for scene in scenes:
                    # 提取场景的所有特征向量
                    feats = self._extract_scene_features(scene)
                    if not feats:
                        continue
                    
                    for feat in feats:
                        pkl_features.append(feat)
                    
                    pkl_scene_map.append({
                        "video_path": video_path,
                        "start_frame": scene.get("start_frame", 0),
                        "end_frame": scene.get("end_frame", 0),
                        "fps": fps,
                        "source_pkl": pkl_path
                    })
                    pkl_feature_counts.append(len(feats))
            
            if not pkl_features:
                return None, [], []
            
            features_np = np.vstack(pkl_features).astype(np.float32)
            return features_np, pkl_scene_map, pkl_feature_counts
            
        except Exception as e:
            print(f"  [Error] 加载 {pkl_path} 失败: {e}")
            return None, [], []
    
    def _preload_features(self):
        """
        按PKL分组预加载特征到GPU（不跨PKL合并）
        
        每个PKL独立存储，支持按PKL独立搜索
        使用多线程并行加载PKL文件
        """
        print(f"[遍历搜索-按PKL模式] 预加载 {len(self.index_paths)} 个PKL文件...")
        
        # 按PKL分组存储
        self.pkl_features: Dict[str, torch.Tensor] = {}  # {pkl_path: features_gpu}
        self.pkl_scene_maps: Dict[str, List[Dict]] = {}  # {pkl_path: scene_maps}
        self.pkl_feature_counts: Dict[str, List[int]] = {}  # {pkl_path: feature_counts}
        
        total_scenes = 0
        total_vectors = 0
        
        # CPU并行加载PKL
        with ThreadPoolExecutor(max_workers=self.pkl_load_workers) as executor:
            futures = {executor.submit(self._load_single_pkl, p): p for p in self.index_paths}
            
            for future in as_completed(futures):
                pkl_path = futures[future]
                try:
                    features_np, pkl_scene_map, pkl_feature_counts = future.result()
                    
                    if features_np is not None and len(pkl_scene_map) > 0:
                        # 构建GPU张量（PKL中的向量已在索引构建时归一化，无需再次归一化）
                        features_gpu = torch.tensor(features_np, device=DEVICE, dtype=self.feature_dtype)
                        
                        # 存储
                        self.pkl_features[pkl_path] = features_gpu
                        self.pkl_scene_maps[pkl_path] = pkl_scene_map
                        self.pkl_feature_counts[pkl_path] = pkl_feature_counts
                        
                        total_scenes += len(pkl_scene_map)
                        total_vectors += features_np.shape[0]
                        
                        print(f"  ✔ {os.path.basename(pkl_path)}: {len(pkl_scene_map)} 场景, {features_np.shape[0]} 向量")
                    else:
                        print(f"  ✘ {os.path.basename(pkl_path)}: 无有效数据")
                        
                except Exception as e:
                    print(f"  [Error] 加载 {pkl_path} 失败: {e}")
        
        if not self.pkl_features:
            raise RuntimeError("没有加载到任何特征数据")
        
        # 获取 logit_scale
        self._logit_scale = self._get_logit_scale()
        
        print(f"[遍历搜索-按PKL模式] 加载完成: {len(self.pkl_features)} 个PKL, {total_scenes} 个场景, {total_vectors} 个向量")
        print(f"    模式: 按PKL独立搜索（不跨PKL合并）")
    
    def _preload_features_per_video(self):
        """按视频分组预加载特征到GPU，使用多线程并行加载PKL文件"""
        print(f"[遍历搜索-按视频模式] 预加载 {len(self.index_paths)} 个PKL文件...")
        
        # 临时存储：{video_path: {'features': [], 'scenes': [], 'counts': []}}
        from collections import defaultdict
        video_data = defaultdict(lambda: {'features': [], 'scenes': [], 'counts': []})
        
        # CPU并行加载PKL
        with ThreadPoolExecutor(max_workers=self.pkl_load_workers) as executor:
            futures = {executor.submit(self._load_single_pkl, p): p for p in self.index_paths}
            
            for future in as_completed(futures):
                pkl_path = futures[future]
                try:
                    features_np, pkl_scene_map, pkl_feature_counts = future.result()
                    
                    if features_np is not None and len(pkl_scene_map) > 0:
                        # 按视频分组
                        feat_idx = 0
                        for scene_idx, scene_info in enumerate(pkl_scene_map):
                            video_path = scene_info['video_path']
                            count = pkl_feature_counts[scene_idx]
                            
                            # 提取该场景的特征
                            scene_features = features_np[feat_idx:feat_idx + count]
                            feat_idx += count
                            
                            video_data[video_path]['features'].append(scene_features)
                            video_data[video_path]['scenes'].append(scene_info)
                            video_data[video_path]['counts'].append(count)
                            
                except Exception as e:
                    print(f"  [Error] 加载 {pkl_path} 失败: {e}")
        
        if not video_data:
            raise RuntimeError("没有加载到任何特征数据")
        
        # 按视频构建GPU张量
        total_scenes = 0
        total_vectors = 0
        
        for video_path, vdata in video_data.items():
            if not vdata['features']:
                continue
            
            # 合并该视频的所有特征（PKL中的向量已在索引构建时归一化，无需再次归一化）
            features_np = np.vstack(vdata['features']).astype(np.float32)
            features_tensor = torch.tensor(features_np, device=DEVICE, dtype=self.feature_dtype)
            
            self.video_features[video_path] = features_tensor
            self.video_scene_maps[video_path] = vdata['scenes']
            self.video_feature_counts[video_path] = vdata['counts']
            
            total_scenes += len(vdata['scenes'])
            total_vectors += features_np.shape[0]
        
        # 获取 logit_scale
        self._logit_scale = self._get_logit_scale()
        
        print(f"[遍历搜索-按视频模式] 加载完成:")
        print(f"    - 视频数: {len(self.video_features)} 个")
        print(f"    - 场景数: {total_scenes} 个")
        print(f"    - 特征向量: {total_vectors} 个")
        print(f"    - top_k: {self.top_k}")
    
    def _extract_scene_features(self, scene: Dict) -> List[np.ndarray]:
        """从场景数据中提取所有特征向量"""
        feats = []
        
        # 尝试多种特征格式
        if 'feature' in scene:
            feat = scene['feature']
            if isinstance(feat, np.ndarray):
                feats.append(feat.reshape(-1))
            elif isinstance(feat, torch.Tensor):
                feats.append(feat.cpu().numpy().reshape(-1))
        
        if 'features' in scene:
            for feat in scene['features']:
                if isinstance(feat, np.ndarray):
                    feats.append(feat.reshape(-1))
                elif isinstance(feat, torch.Tensor):
                    feats.append(feat.cpu().numpy().reshape(-1))
        
        # 三元组格式
        for key in ['start_feature', 'mid_feature', 'end_feature']:
            if key in scene:
                feat = scene[key]
                if isinstance(feat, np.ndarray):
                    feats.append(feat.reshape(-1))
                elif isinstance(feat, torch.Tensor):
                    feats.append(feat.cpu().numpy().reshape(-1))
        
        return feats
    
    def _get_logit_scale(self) -> torch.Tensor:
        """获取模型的 logit_scale"""
        if hasattr(self.processor, '_model') and hasattr(self.processor._model, 'logit_scale'):
            scale = self.processor._model.logit_scale
            if isinstance(scale, torch.nn.Parameter):
                return scale.exp().to(DEVICE)
            return torch.tensor(scale, device=DEVICE)
        
        if hasattr(self.processor, '_model_raw'):
            if hasattr(self.processor._model_raw, 'logit_scale'):
                scale = self.processor._model_raw.logit_scale
                if isinstance(scale, torch.nn.Parameter):
                    return scale.exp().to(DEVICE)
                return torch.tensor(scale, device=DEVICE)
        
        # 默认值
        return torch.tensor(100.0, device=DEVICE)
    
    def search(self) -> List[SceneLabelResult]:
        """
        执行遍历搜索（标签模式不使用分配规则）
        
        核心特色：跨帧选择最高相似度标签
        - 每个场景有N帧特征向量（通常为3帧：起始帧、中间帧、结束帧）
        - 对每个大类的每个标签，分别计算与N帧的相似度
        - 选择N帧中相似度最高的那个作为该标签的最终相似度
        - 每个大类选择相似度最高的标签
        
        逻辑流程：
        1. 先搜索所有大类，保存所有超过阈值的候选（跨帧取最大相似度）
        2. 每个大类选择相似度最高的标签
        3. 过滤掉任何大类没有匹配的场景
        4. 按搜索模式：每组最多返回 top_k 个有效场景
        
        Returns:
            有效场景的标签匹配结果列表
        """
        # 按视频模式（预加载模式）：调用专用方法
        if self.use_diskcache and self._lmdb_cache is None:
            import hashlib as _hashlib
            config_payload = {
                'similarity_threshold': self.similarity_threshold,
                'search_mode': self.search_mode,
                'top_k': self.top_k,
                'scene_chunk_size': self.scene_chunk_size,
                'prompt_search_batch_size': self.prompt_search_batch_size,
                'use_fp16': self.use_fp16,
                'feature_fp16': self.feature_fp16,
                'pkl_batch_size': self.pkl_batch_size,
                'use_diskcache': self.use_diskcache,
                'use_chinese_labels': self.use_chinese_labels,
                'label_cache_batch_size': self.label_cache_batch_size,
                'label_cache_config_hash': self._label_cache_config_hash,
                'index_files': sorted([os.path.basename(p) for p in self.index_paths]),
            }
            config_str = json.dumps(config_payload, ensure_ascii=False, sort_keys=True)
            config_hash = _hashlib.md5(config_str.encode('utf-8')).hexdigest()
            self._lmdb_cache = LabelResultLMDBCache(
                self.diskcache_dir,
                config_hash=config_hash
            )

        if self.search_mode == -1 and self._preload_all:
            return self._search_per_video()
        
        # 分批加载模式（包括 search_mode=-1 时的分批加载）：调用分批搜索方法
        if not self._preload_all:
            return self._search_batch_by_pkl()
        
        # 按PKL模式：每个PKL独立搜索（预加载模式）
        if not self.pkl_features:
            raise RuntimeError("特征未预加载，请先调用 _preload_features()")
        
        total_scenes = sum(len(scenes) for scenes in self.pkl_scene_maps.values())
        print(f"\n[遍历搜索-按PKL模式] 开始搜索 {len(self.pkl_features)} 个PKL, 共 {total_scenes} 个场景...")
        print(f"[遍历搜索-按PKL模式] 相似度阈值: {self.similarity_threshold}")
        print(f"[遍历搜索-按PKL模式] 每PKL Top-K: {self.top_k}")
        print(f"[遍历搜索-按PKL模式] 标签选择: 跨帧选择最高相似度")
        
        # 获取所有大类
        categories = self.keyword_manager.categories
        ordered_categories = list(categories)
        
        all_valid_results = []
        use_grouped_heap = (self.top_k is not None and self.search_mode in (-1, 1))
        grouped_result_heaps = None
        if use_grouped_heap:
            from collections import defaultdict
            import heapq as _heapq

            grouped_result_heaps = defaultdict(list)

            def _add_grouped_result(result_obj):
                avg_sim = sum(m.similarity for m in result_obj.labels.values()) / len(result_obj.labels) if result_obj.labels else 0
                group_key = result_obj.video_path if self.search_mode == -1 else '__global__'
                group_heap = grouped_result_heaps[group_key]
                heap_item = (avg_sim, result_obj.scene_key, result_obj)
                if len(group_heap) < self.top_k:
                    _heapq.heappush(group_heap, heap_item)
                elif avg_sim > group_heap[0][0]:
                    _heapq.heapreplace(group_heap, heap_item)
        
        # 遍历每个PKL独立搜索
        completed_units = set()
        if self._lmdb_cache is not None:
            checkpoint = self._lmdb_cache.load_checkpoint()
            if checkpoint is not None:
                completed_units = set(checkpoint.get('completed_pkls', []))
                if completed_units:
                    existing_count = 0
                    for result_obj in self._lmdb_cache.iter_results():
                        existing_count += 1
                        if use_grouped_heap:
                            _add_grouped_result(result_obj)
                        else:
                            all_valid_results.append(result_obj)
                    print(f"[遍历搜索-断点续传] 已完成 {len(completed_units)} 个PKL, 已有 {existing_count} 个结果, 继续搜索...")

        for pkl_idx, (pkl_path, features_gpu) in enumerate(self.pkl_features.items()):
            pkl_name = os.path.basename(pkl_path)
            scene_maps = self.pkl_scene_maps[pkl_path]
            feature_counts = self.pkl_feature_counts[pkl_path]
            num_scenes = len(scene_maps)

            if pkl_path in completed_units:
                print(f"\n[PKL {pkl_idx + 1}/{len(self.pkl_features)}] {pkl_name}: 已完成，跳过")
                continue
            
            print(f"\n[PKL {pkl_idx + 1}/{len(self.pkl_features)}] {pkl_name}: {num_scenes} 个场景")
            
            # 初始化该PKL的结果
            results: Dict[int, SceneLabelResult] = {}
            for scene_idx, scene_info in enumerate(scene_maps):
                scene_key = f"{scene_info['start_frame']}_{os.path.basename(scene_info['video_path'])}"
                results[scene_idx] = SceneLabelResult(
                    scene_key=scene_key,
                    video_path=scene_info['video_path'],
                    start_frame=scene_info['start_frame'],
                    end_frame=scene_info['end_frame'],
                    fps=scene_info['fps']
                )
            
            # 搜索所有大类的候选（针对该PKL）
            # 核心：跨帧选择最高相似度的标签
            all_candidates: Dict[str, Dict[int, List[LabelMatch]]] = {}
            
            for category in categories:
                candidates = self._search_category_for_pkl(
                    category, features_gpu, scene_maps, feature_counts
                )
                all_candidates[category] = candidates
                # 调试输出：显示每个大类有多少场景有候选
                print(f"  [CLIP召回] 大类'{category}': {len(candidates)} 个场景有候选 (阈值={self.similarity_threshold})")
            
            # 对每个大类选择最佳标签（直接选择CLIP相似度最高的）
            
            for category in ordered_categories:
                candidates = all_candidates.get(category, {})
                
                # 直接选择每个场景中相似度最高的标签（跨帧已在 _search_category_for_pkl 中处理）
                category_matches = self._select_best_label_for_category(category, candidates)
                
                
                # 更新结果
                for scene_idx, match in category_matches.items():
                    if scene_idx in results:
                        results[scene_idx].labels[category] = match
            
            # 过滤有效场景
            pkl_valid_results = []
            for scene_idx, result in results.items():
                if len(result.labels) == len(categories):
                    # 计算平均相似度用于排序
                    avg_sim = sum(m.similarity for m in result.labels.values()) / len(result.labels)
                    pkl_valid_results.append((avg_sim, result, scene_idx))
            
            # 按PKL模式(search_mode=0)：每个PKL内部取TopK
            # 跨PKL模式(search_mode=1)：先收集全部有效场景，最后做全局TopK
            pkl_valid_results.sort(key=lambda x: x[0], reverse=True)
            top_results = pkl_valid_results[:self.top_k] if (self.search_mode == 0 and self.top_k is not None) else pkl_valid_results
            if self.search_mode == 0:
                print(f"  有效场景: {len(pkl_valid_results)}, 取 Top-{self.top_k}: {len(top_results)}")
            else:
                print(f"  有效场景: {len(pkl_valid_results)}")
            new_results = [r for _, r, _ in top_results]
            if use_grouped_heap:
                for result_obj in new_results:
                    _add_grouped_result(result_obj)
            else:
                all_valid_results.extend(new_results)

            if self._lmdb_cache is not None:
                if new_results:
                    write_batch_size = max(1, self.lmdb_write_batch_size)
                    for idx in range(0, len(new_results), write_batch_size):
                        batch_items = [(result.scene_key, result) for result in new_results[idx:idx + write_batch_size]]
                        self._lmdb_cache.put_many(batch_items)
                completed_units.add(pkl_path)
                self._lmdb_cache.save_checkpoint(list(completed_units))
        
        total_result_count = sum(len(heap) for heap in grouped_result_heaps.values()) if use_grouped_heap else len(all_valid_results)
        print(f"\n[遍历搜索-按PKL模式] 完成: 共 {total_result_count} 个有效场景")
        
        # 跨PKL模式：全局排序后取TopK
        if self.search_mode == 1 and self.top_k is not None:
            if use_grouped_heap:
                global_heap = grouped_result_heaps.get('__global__', [])
                all_valid_results = [r for _, _, r in sorted(global_heap, key=lambda x: x[0], reverse=True)]
            else:
                all_valid_results_with_sim = []
                for result in all_valid_results:
                    avg_sim = sum(m.similarity for m in result.labels.values()) / len(result.labels) if result.labels else 0
                    all_valid_results_with_sim.append((avg_sim, result))
                all_valid_results_with_sim.sort(key=lambda x: x[0], reverse=True)
                all_valid_results = [r for _, r in all_valid_results_with_sim[:self.top_k]]
            print(f"[遍历搜索-跨PKL模式] 全局过滤: 取 Top-{self.top_k}, 最终 {len(all_valid_results)} 个场景")
        
        return all_valid_results
    
    def _search_batch_by_pkl(self) -> List[SceneLabelResult]:
        """
        分批加载PKL并搜索（显存优化模式，支持断点续传）
        
        每批加载 pkl_batch_size 个PKL，搜索后释放GPU显存。
        支持断点续传：通过 LMDB checkpoint 记录已完成的 PKL，程序中断后可从断点继续。
        
        支持三种 search_mode：
        - -1（按视频模式）: 收集所有有效场景后按视频分组取 Top-K
        - 0（按PKL模式）: 每个PKL独立取 Top-K
        - 1（跨PKL模式）: 收集所有有效场景后全局取 Top-K
        """
        total_pkls = len(self.index_paths)
        total_batches = (total_pkls + self.pkl_batch_size - 1) // self.pkl_batch_size
        
        mode_names = {-1: '按视频', 0: '按PKL', 1: '跨PKL'}
        mode_name = mode_names.get(self.search_mode, '按PKL')
        
        print(f"\n[遍历搜索-分批加载模式] 开始搜索 {total_pkls} 个PKL...")
        print(f"[遍历搜索-分批加载模式] 每批加载: {self.pkl_batch_size} 个PKL, 共 {total_batches} 批")
        print(f"[遍历搜索-分批加载模式] 相似度阈值: {self.similarity_threshold}")
        print(f"[遍历搜索-分批加载模式] 搜索模式: {mode_name} (search_mode={self.search_mode})")
        print(f"[遍历搜索-分批加载模式] Top-K: {self.top_k}")
        
        # 获取所有大类
        categories = self.keyword_manager.categories
        
        # 获取 logit_scale
        if self._logit_scale is None:
            self._logit_scale = self._get_logit_scale()
        
        all_valid_results = []
        use_grouped_heap = (self.top_k is not None and self.search_mode in (-1, 1))
        grouped_result_heaps = None
        if use_grouped_heap:
            from collections import defaultdict
            import heapq as _heapq

            grouped_result_heaps = defaultdict(list)

            def _add_grouped_result(result_obj):
                avg_sim = sum(m.similarity for m in result_obj.labels.values()) / len(result_obj.labels) if result_obj.labels else 0
                group_key = result_obj.video_path if self.search_mode == -1 else '__global__'
                group_heap = grouped_result_heaps[group_key]
                heap_item = (avg_sim, result_obj.scene_key, result_obj)
                if len(group_heap) < self.top_k:
                    _heapq.heappush(group_heap, heap_item)
                elif avg_sim > group_heap[0][0]:
                    _heapq.heapreplace(group_heap, heap_item)
        
        # 断点续传：检查已完成的 PKL
        completed_pkls = set()
        if self._lmdb_cache is not None:
            checkpoint = self._lmdb_cache.load_checkpoint()
            if checkpoint is not None:
                completed_pkls = set(checkpoint.get('completed_pkls', []))
                if completed_pkls:
                    # 从 LMDB 读取已有结果
                    existing_count = 0
                    for result_obj in self._lmdb_cache.iter_results():
                        existing_count += 1
                        if use_grouped_heap:
                            _add_grouped_result(result_obj)
                        else:
                            all_valid_results.append(result_obj)
                    print(f"[遍历搜索-断点续传] 已完成 {len(completed_pkls)} 个PKL, 已有 {existing_count} 个结果, 继续搜索...")
        
        # 分批处理PKL
        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.pkl_batch_size
            batch_end = min(batch_start + self.pkl_batch_size, total_pkls)
            batch_paths = self.index_paths[batch_start:batch_end]
            
            # 过滤掉已完成的 PKL
            remaining_paths = [p for p in batch_paths if p not in completed_pkls]
            if not remaining_paths:
                print(f"\n[批次 {batch_idx + 1}/{total_batches}] 全部已完成，跳过")
                continue
            
            print(f"\n[批次 {batch_idx + 1}/{total_batches}] 加载 {len(remaining_paths)} 个PKL（跳过 {len(batch_paths) - len(remaining_paths)} 个已完成）...")
            
            # 加载这一批PKL（只处理未完成的）
            preloaded_batch = {}
            with ThreadPoolExecutor(max_workers=self.pkl_load_workers) as executor:
                futures = {executor.submit(self._load_single_pkl, p): p for p in remaining_paths}
                for future in as_completed(futures):
                    pkl_path = futures[future]
                    try:
                        preloaded_batch[pkl_path] = future.result()
                    except Exception as e:
                        print(f"  [Error] 预加载 {os.path.basename(pkl_path)} 失败: {e}")
                        preloaded_batch[pkl_path] = (None, [], [])

            for pkl_path in remaining_paths:
                pkl_name = os.path.basename(pkl_path)
                
                # 加载单个PKL
                features_np, scene_maps, feature_counts = preloaded_batch.get(pkl_path, (None, [], []))
                features_gpu = None
                if features_np is not None and len(scene_maps) > 0:
                    features_gpu = torch.tensor(features_np, device=DEVICE, dtype=self.feature_dtype)
                
                if features_gpu is None or len(scene_maps) == 0:
                    print(f"  ✘ {pkl_name}: 无有效数据")
                    # 标记为已完成（即使无数据）
                    completed_pkls.add(pkl_path)
                    if self._lmdb_cache is not None:
                        self._lmdb_cache.save_checkpoint(list(completed_pkls))
                    continue
                
                num_scenes = len(scene_maps)
                print(f"  ✔ {pkl_name}: {num_scenes} 个场景, {features_gpu.shape[0]} 向量")
                
                # 初始化该PKL的结果
                results: Dict[int, SceneLabelResult] = {}
                for scene_idx, scene_info in enumerate(scene_maps):
                    scene_key = f"{scene_info['start_frame']}_{os.path.basename(scene_info['video_path'])}"
                    results[scene_idx] = SceneLabelResult(
                        scene_key=scene_key,
                        video_path=scene_info['video_path'],
                        start_frame=scene_info['start_frame'],
                        end_frame=scene_info['end_frame'],
                        fps=scene_info['fps']
                    )
                
                # 搜索并就地归并每个大类的最佳标签（避免保存 all_candidates 中间大字典）
                for category in categories:
                    candidates = self._search_category_for_pkl(
                        category, features_gpu, scene_maps, feature_counts
                    )
                    category_matches = self._select_best_label_for_category(category, candidates)
                    for scene_idx, match in category_matches.items():
                        if scene_idx in results:
                            results[scene_idx].labels[category] = match
                
                # 过滤有效场景
                pkl_valid_results = []
                for scene_idx, result in results.items():
                    if len(result.labels) == len(categories):
                        avg_sim = sum(m.similarity for m in result.labels.values()) / len(result.labels)
                        pkl_valid_results.append((avg_sim, result, scene_idx))
                
                # 按PKL模式(search_mode=0)：每个PKL内部取TopK
                # 按视频模式(search_mode=-1)和跨PKL模式(search_mode=1)：先收集所有，后续统一处理
                if self.search_mode == 0:
                    pkl_valid_results.sort(key=lambda x: x[0], reverse=True)
                    top_results = pkl_valid_results[:self.top_k] if self.top_k is not None else pkl_valid_results
                    print(f"    有效场景: {len(pkl_valid_results)}, 取 Top-{self.top_k}: {len(top_results)}")
                    new_results = [r for _, r, _ in top_results]
                else:
                    # search_mode=-1 或 1：收集所有有效场景，后续统一处理
                    new_results = [r for _, r, _ in pkl_valid_results]
                    print(f"    有效场景: {len(pkl_valid_results)}")
                
                if use_grouped_heap:
                    for result_obj in new_results:
                        _add_grouped_result(result_obj)
                else:
                    all_valid_results.extend(new_results)
                
                # 将结果写入 LMDB 并更新 checkpoint
                if self._lmdb_cache is not None:
                    if new_results:
                        write_batch_size = max(1, self.lmdb_write_batch_size)
                        for idx in range(0, len(new_results), write_batch_size):
                            batch_items = [(result.scene_key, result) for result in new_results[idx:idx + write_batch_size]]
                            self._lmdb_cache.put_many(batch_items)
                    completed_pkls.add(pkl_path)
                    self._lmdb_cache.save_checkpoint(list(completed_pkls))
                
                # 释放GPU显存
                del features_gpu
                if features_np is not None:
                    del features_np
            
            # 每批处理完后清理GPU缓存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_result_count = sum(len(heap) for heap in grouped_result_heaps.values()) if use_grouped_heap else len(all_valid_results)
        print(f"\n[遍历搜索-分批加载模式] 完成: 共 {total_result_count} 个有效场景")
        
        # 按视频模式(search_mode=-1)：按视频分组后每组取TopK
        if self.search_mode == -1 and self.top_k is not None:
            if use_grouped_heap:
                grouped_results = []
                for video_path, video_heap in grouped_result_heaps.items():
                    video_name = os.path.basename(video_path)
                    top_video_results = [r for _, _, r in sorted(video_heap, key=lambda x: x[0], reverse=True)]
                    print(f"[遍历搜索-按视频模式] {video_name}: Top-{self.top_k}: {len(top_video_results)}")
                    grouped_results.extend(top_video_results)
                all_valid_results = grouped_results
            else:
                from collections import defaultdict
                video_groups = defaultdict(list)
                for result in all_valid_results:
                    video_groups[result.video_path].append(result)
                
                grouped_results = []
                for video_path, video_results in video_groups.items():
                    video_name = os.path.basename(video_path)
                    video_results_with_sim = []
                    for r in video_results:
                        avg_sim = sum(m.similarity for m in r.labels.values()) / len(r.labels) if r.labels else 0
                        video_results_with_sim.append((avg_sim, r))
                    video_results_with_sim.sort(key=lambda x: x[0], reverse=True)
                    top_video_results = [r for _, r in video_results_with_sim[:self.top_k]]
                    print(f"[遍历搜索-按视频模式] {video_name}: {len(video_results)} → Top-{self.top_k}: {len(top_video_results)}")
                    grouped_results.extend(top_video_results)
                all_valid_results = grouped_results
            print(f"[遍历搜索-按视频模式] 最终: {len(all_valid_results)} 个场景")
        
        # 跨PKL模式(search_mode=1)：全局排序后取TopK
        elif self.search_mode == 1 and self.top_k is not None:
            if use_grouped_heap:
                global_heap = grouped_result_heaps.get('__global__', [])
                all_valid_results = [r for _, _, r in sorted(global_heap, key=lambda x: x[0], reverse=True)]
            else:
                all_valid_results_with_sim = []
                for result in all_valid_results:
                    avg_sim = sum(m.similarity for m in result.labels.values()) / len(result.labels) if result.labels else 0
                    all_valid_results_with_sim.append((avg_sim, result))
                all_valid_results_with_sim.sort(key=lambda x: x[0], reverse=True)
                all_valid_results = [r for _, r in all_valid_results_with_sim[:self.top_k]]
            print(f"[遍历搜索-跨PKL模式] 全局过滤: 取 Top-{self.top_k}, 最终 {len(all_valid_results)} 个场景")
        
        return all_valid_results
    
    def _select_best_label_for_category(self, category: str,
                                         candidates: Dict[int, List[LabelMatch]]) -> Dict[int, LabelMatch]:
        """
        为每个场景选择该大类中相似度最高的标签
        
        跨帧选择已在 _search_category_for_pkl 中完成，这里直接选择候选列表中的第一个（最高相似度）
        
        Args:
            category: 大类名称
            candidates: {scene_idx: [LabelMatch, ...]} 每个场景的候选标签列表（已按相似度降序排序）
        
        Returns:
            {scene_idx: LabelMatch} 每个场景的最佳标签
        """
        result: Dict[int, LabelMatch] = {}
        
        for scene_idx, scene_candidates in candidates.items():
            if scene_candidates:
                # 候选列表已按相似度降序排序，直接取第一个
                result[scene_idx] = scene_candidates[0]
        
        return result
    
    def _search_category_for_pkl(self, category: str, features_gpu: torch.Tensor,
                                  scene_maps: List[Dict], feature_counts: List[int]) -> Dict[int, List[LabelMatch]]:
        """
        针对单个PKL搜索某个大类的所有候选（GPU矩阵运算优化版）。

        核心优化：
        1. 所有相似度计算在 GPU 上完成；
        2. 跨帧取最大值使用 GPU 矩阵运算；
        3. 最小化 GPU-CPU 数据传输；
        4. 对候选按相似度排序后返回。
        """
        label_indices = []
        label_info = []
        for idx, (cat, subcategory, cn, en) in enumerate(self._label_info):
            if cat != category:
                continue
            label_indices.append(idx)
            label_info.append((subcategory, cn, en))

        if not label_indices:
            return {}

        num_labels = len(label_indices)
        num_scenes = len(feature_counts)
        label_batch_size = self.prompt_search_batch_size if self.prompt_search_batch_size and self.prompt_search_batch_size > 0 else num_labels
        label_batch_size = max(1, min(label_batch_size, num_labels))
        all_candidates: Dict[int, List[LabelMatch]] = {}

        frames_per_scene = feature_counts[0] if feature_counts else 3
        all_same_frames = all(c == frames_per_scene for c in feature_counts)

        scene_boundaries = None
        if (not all_same_frames) and num_scenes > 0:
            scene_boundaries = torch.zeros(num_scenes + 1, dtype=torch.long, device=DEVICE)
            scene_boundaries[1:] = torch.cumsum(torch.tensor(feature_counts, device=DEVICE), dim=0)

        total_batches = (num_labels + label_batch_size - 1) // label_batch_size

        for batch_idx, start_idx in enumerate(range(0, num_labels, label_batch_size)):
            end_idx = min(start_idx + label_batch_size, num_labels)
            batch_label_indices = label_indices[start_idx:end_idx]
            batch_label_info = label_info[start_idx:end_idx]
            batch_label_count = len(batch_label_indices)
            batch_label_vectors = self._label_vectors[batch_label_indices]

            with torch.no_grad():
                all_sims = self._logit_scale * batch_label_vectors @ features_gpu.T

                if all_same_frames and num_scenes > 0:
                    all_sims_reshaped = all_sims.view(batch_label_count, num_scenes, frames_per_scene)
                    label_scene_max_sims, _ = all_sims_reshaped.max(dim=2)
                else:
                    label_scene_max_sims = torch.empty(batch_label_count, num_scenes, device=DEVICE, dtype=all_sims.dtype)
                    for scene_idx in range(num_scenes):
                        s = scene_boundaries[scene_idx].item()
                        e = scene_boundaries[scene_idx + 1].item()
                        label_scene_max_sims[:, scene_idx] = all_sims[:, s:e].max(dim=1)[0]

                threshold_mask = label_scene_max_sims >= self.similarity_threshold
                scene_has_candidates = threshold_mask.any(dim=0)
                valid_scene_indices = torch.where(scene_has_candidates)[0]

                if len(valid_scene_indices) == 0:
                    del all_sims, label_scene_max_sims, threshold_mask, scene_has_candidates, valid_scene_indices
                    if batch_idx % 10 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    continue

                candidate_chunk_size = self.scene_chunk_size or len(valid_scene_indices)
                candidate_chunk_size = max(1, int(candidate_chunk_size))

                for chunk_start in range(0, len(valid_scene_indices), candidate_chunk_size):
                    chunk_end = min(chunk_start + candidate_chunk_size, len(valid_scene_indices))
                    chunk_scene_indices = valid_scene_indices[chunk_start:chunk_end]

                    chunk_sims = label_scene_max_sims[:, chunk_scene_indices].cpu().numpy()
                    chunk_mask = threshold_mask[:, chunk_scene_indices].cpu().numpy()
                    chunk_scene_indices_np = chunk_scene_indices.cpu().numpy()

                    for valid_idx, scene_idx in enumerate(chunk_scene_indices_np):
                        scene_idx = int(scene_idx)
                        scene_candidates = all_candidates.get(scene_idx)
                        if scene_candidates is None:
                            scene_candidates = []
                            all_candidates[scene_idx] = scene_candidates

                        for label_idx, (subcategory, cn, en) in enumerate(batch_label_info):
                            if chunk_mask[label_idx, valid_idx]:
                                scene_candidates.append(LabelMatch(
                                    category=category,
                                    subcategory=subcategory,
                                    label_cn=cn,
                                    label_en=en,
                                    similarity=float(chunk_sims[label_idx, valid_idx])
                                ))

                    del chunk_sims, chunk_mask, chunk_scene_indices_np

            del all_sims, label_scene_max_sims, threshold_mask, scene_has_candidates, valid_scene_indices

            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for scene_candidates in all_candidates.values():
            scene_candidates.sort(key=lambda x: x.similarity, reverse=True)

        return all_candidates

    def _search_per_video(self) -> List[SceneLabelResult]:
        """
        按视频模式执行遍历搜索（标签模式不使用分配规则）
        
        核心特色：跨帧选择最高相似度标签
        - 每个场景有N帧特征向量（通常为3帧：起始帧、中间帧、结束帧）
        - 对每个大类的每个标签，分别计算与N帧的相似度
        - 选择N帧中相似度最高的那个作为该标签的最终相似度
        - 每个大类选择相似度最高的标签
        
        每个视频独立处理，每个视频最多返回 top_k 个有效场景
        
        Returns:
            有效场景的标签匹配结果列表
        """
        total_scenes = sum(len(scenes) for scenes in self.video_scene_maps.values())
        print(f"\n[遍历搜索-按视频模式] 开始搜索 {len(self.video_features)} 个视频, 共 {total_scenes} 个场景...")
        print(f"[遍历搜索-按视频模式] 相似度阈值: {self.similarity_threshold}")
        print(f"[遍历搜索-按视频模式] 每视频 Top-K: {self.top_k}")
        print(f"[遍历搜索-按视频模式] 标签选择: 跨帧选择最高相似度")
        
        # 获取所有大类
        categories = self.keyword_manager.categories
        ordered_categories = list(categories)
        
        all_valid_results = []
        
        # 遍历每个视频
        completed_units = set()
        if self._lmdb_cache is not None:
            checkpoint = self._lmdb_cache.load_checkpoint()
            if checkpoint is not None:
                completed_units = set(checkpoint.get('completed_pkls', []))
                if completed_units:
                    existing_count = 0
                    for result_obj in self._lmdb_cache.iter_results():
                        existing_count += 1
                        all_valid_results.append(result_obj)
                    print(f"[遍历搜索-断点续传] 已完成 {len(completed_units)} 个视频, 已有 {existing_count} 个结果, 继续搜索...")

        for video_idx, (video_path, features) in enumerate(self.video_features.items()):
            video_name = os.path.basename(video_path)
            scene_maps = self.video_scene_maps[video_path]
            feature_counts = self.video_feature_counts[video_path]
            num_scenes = len(scene_maps)

            if video_path in completed_units:
                print(f"\n[视频 {video_idx + 1}/{len(self.video_features)}] {video_name}: 已完成，跳过")
                continue
            
            print(f"\n[视频 {video_idx + 1}/{len(self.video_features)}] {video_name}: {num_scenes} 个场景")
            
            # 初始化该视频的结果
            results: Dict[int, SceneLabelResult] = {}
            for scene_idx, scene_info in enumerate(scene_maps):
                scene_key = f"{scene_info['start_frame']}_{video_name}"
                results[scene_idx] = SceneLabelResult(
                    scene_key=scene_key,
                    video_path=scene_info['video_path'],
                    start_frame=scene_info['start_frame'],
                    end_frame=scene_info['end_frame'],
                    fps=scene_info['fps']
                )
            
            # 搜索并就地归并每个大类的最佳标签（避免保存 all_candidates 中间大字典）
            for category in ordered_categories:
                candidates = self._search_category_for_video(
                    category, features, scene_maps, feature_counts
                )
                print(f"  [CLIP召回] 大类'{category}': {len(candidates)} 个场景有候选 (阈值={self.similarity_threshold})")
                category_matches = self._select_best_label_for_category(category, candidates)
                for scene_idx, match in category_matches.items():
                    if scene_idx in results:
                        results[scene_idx].labels[category] = match
            
            # 过滤有效场景
            video_valid_results = []
            for scene_idx, result in results.items():
                if len(result.labels) == len(categories):
                    # 计算平均相似度用于排序
                    avg_sim = sum(m.similarity for m in result.labels.values()) / len(result.labels)
                    video_valid_results.append((avg_sim, result, scene_idx))
            
            # 按相似度排序
            video_valid_results.sort(key=lambda x: x[0], reverse=True)
            
            # 取 TopK
            top_results = [(sim, r, idx) for sim, r, idx in video_valid_results[:self.top_k]] if self.top_k is not None else video_valid_results
            print(f"  有效场景: {len(video_valid_results)}, 取 Top-{self.top_k}: {len(top_results)}")
            
            new_results = [r for _, r, _ in top_results]
            all_valid_results.extend(new_results)

            if self._lmdb_cache is not None:
                if new_results:
                    write_batch_size = max(1, self.lmdb_write_batch_size)
                    for idx in range(0, len(new_results), write_batch_size):
                        batch_items = [(result.scene_key, result) for result in new_results[idx:idx + write_batch_size]]
                        self._lmdb_cache.put_many(batch_items)
                completed_units.add(video_path)
                self._lmdb_cache.save_checkpoint(list(completed_units))
        
        print(f"\n[遍历搜索-按视频模式] 完成: 共 {len(all_valid_results)} 个有效场景")
        
        return all_valid_results
    
    def _search_category_for_video(self, category: str, features: torch.Tensor,
                                    scene_maps: List[Dict], feature_counts: List[int]) -> Dict[int, List[LabelMatch]]:
        """
        针对单个视频搜索某个大类的所有候选（GPU矩阵运算优化版）
        
        核心优化：
        1. 所有相似度计算在GPU上完成
        2. 跨帧取最大值使用GPU矩阵运算
        3. 最小化GPU-CPU数据传输
        """
        # 筛选该大类的标签索引
        label_indices = []
        label_info = []
        for idx, (cat, subcategory, cn, en) in enumerate(self._label_info):
            if cat != category:
                continue
            label_indices.append(idx)
            label_info.append((subcategory, cn, en))
        
        if not label_indices:
            return {}
        
        num_labels = len(label_indices)
        num_scenes = len(feature_counts)
        
        # 获取对应的向量
        label_vectors = self._label_vectors[label_indices]  # [N_labels, dim]
        
        with torch.no_grad():
            # 1. 计算相似度 [N_labels, M_features]
            all_sims = self._logit_scale * label_vectors @ features.T
            
            # 2. 跨帧取最大值 - GPU矩阵运算
            # 检查是否所有场景帧数相同（常见情况：都是3帧）
            frames_per_scene = feature_counts[0] if feature_counts else 3
            all_same_frames = all(c == frames_per_scene for c in feature_counts)
            
            if all_same_frames and num_scenes > 0:
                # 快速路径：所有场景帧数相同，可以直接reshape
                # [N_labels, num_scenes * frames] -> [N_labels, num_scenes, frames]
                all_sims_reshaped = all_sims.view(num_labels, num_scenes, frames_per_scene)
                # 跨帧取最大值 [N_labels, num_scenes]
                label_scene_max_sims, _ = all_sims_reshaped.max(dim=2)
            else:
                # 慢速路径：场景帧数不同，使用segment_reduce或手动索引
                # 构建场景边界索引
                scene_boundaries = torch.zeros(num_scenes + 1, dtype=torch.long, device=DEVICE)
                scene_boundaries[1:] = torch.cumsum(torch.tensor(feature_counts, device=DEVICE), dim=0)
                
                # 预分配结果张量
                label_scene_max_sims = torch.empty(num_labels, num_scenes, device=DEVICE, dtype=all_sims.dtype)
                
                # 使用向量化的segment max（比Python循环快很多）
                for scene_idx in range(num_scenes):
                    start_idx = scene_boundaries[scene_idx].item()
                    end_idx = scene_boundaries[scene_idx + 1].item()
                    # 对所有标签同时取该场景的最大值
                    label_scene_max_sims[:, scene_idx] = all_sims[:, start_idx:end_idx].max(dim=1)[0]
            
            # 3. 应用阈值过滤（在GPU上）
            threshold_mask = label_scene_max_sims >= self.similarity_threshold  # [N_labels, num_scenes]
            
            # 4. 只传输需要的数据到CPU
            # 找出有候选的场景
            scene_has_candidates = threshold_mask.any(dim=0)  # [num_scenes]
            valid_scene_indices = torch.where(scene_has_candidates)[0]
            
            if len(valid_scene_indices) == 0:
                return {}
        
        # 5. 在CPU上构建结果（只处理有效场景）
        all_candidates: Dict[int, List[LabelMatch]] = {}

        candidate_chunk_size = self.scene_chunk_size or len(valid_scene_indices)
        candidate_chunk_size = max(1, int(candidate_chunk_size))

        for chunk_start in range(0, len(valid_scene_indices), candidate_chunk_size):
            chunk_end = min(chunk_start + candidate_chunk_size, len(valid_scene_indices))
            chunk_scene_indices = valid_scene_indices[chunk_start:chunk_end]

            # 只传输当前分片的有效数据到 CPU
            chunk_sims = label_scene_max_sims[:, chunk_scene_indices].cpu().numpy()  # [N_labels, chunk]
            chunk_mask = threshold_mask[:, chunk_scene_indices].cpu().numpy()  # [N_labels, chunk]
            chunk_scene_indices_np = chunk_scene_indices.cpu().numpy()

            for valid_idx, scene_idx in enumerate(chunk_scene_indices_np):
                scene_idx = int(scene_idx)
                scene_candidates = []

                for label_idx, (subcategory, cn, en) in enumerate(label_info):
                    if chunk_mask[label_idx, valid_idx]:
                        scene_candidates.append(LabelMatch(
                            category=category,
                            subcategory=subcategory,
                            label_cn=cn,
                            label_en=en,
                            similarity=float(chunk_sims[label_idx, valid_idx])
                        ))

                if scene_candidates:
                    # 按相似度降序排序
                    scene_candidates.sort(key=lambda x: x.similarity, reverse=True)
                    all_candidates[scene_idx] = scene_candidates

            del chunk_sims, chunk_mask, chunk_scene_indices_np
        
        return all_candidates
    
    def cleanup(self):
        """清理资源"""
        if self._lmdb_cache is not None:
            self._lmdb_cache.close()
            self._lmdb_cache = None

        # 清理按PKL模式的数据
        if hasattr(self, 'pkl_features') and self.pkl_features:
            print(f"[清理] 释放 {len(self.pkl_features)} 个PKL的GPU特征缓存...")
            for pkl_path in list(self.pkl_features.keys()):
                del self.pkl_features[pkl_path]
            self.pkl_features.clear()
        
        if hasattr(self, 'pkl_scene_maps'):
            self.pkl_scene_maps.clear()
        
        if hasattr(self, 'pkl_feature_counts'):
            self.pkl_feature_counts.clear()
        
        # 清理按视频模式的数据
        if hasattr(self, 'video_features') and self.video_features:
            for video_path in list(self.video_features.keys()):
                del self.video_features[video_path]
            self.video_features.clear()
        
        if hasattr(self, 'video_scene_maps'):
            self.video_scene_maps.clear()
        
        if hasattr(self, 'video_feature_counts'):
            self.video_feature_counts.clear()
        
        # 清理GPU缓存
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    

def run_label_traverse_search(
    index_directory: str = None,
    output_directory: str = None,
    video_output_directory: str = None,
    video_copy_mode: bool = False,
    video_name_format: str = None,
    debug_similarity: bool = False,
    # 优化参数
    prompt_search_batch_size: int = 1024,
    pkl_batch_size: int = None,
    # 搜索模式参数
    search_mode: int = 0,
    top_k: int = 50,
    scene_chunk_size: Optional[int] = 1000,
    # 视频导出帧偏移参数
    start_frame_offset: int = None,
    end_frame_offset: int = None,
    # LMDB 缓存参数
    use_diskcache: bool = True,
    diskcache_dir: str = None,
    # 中文标签模式
    use_chinese_labels: bool = False,
    # 线程配置参数
    pkl_load_workers: int = 4,
    lmdb_write_batch_size: int = 1000,
    # 标签缓存批处理大小
    label_cache_batch_size: int = 512,
    # 向量去重参数
    vector_dedup_threshold: float = None,
    # 相邻片段合并参数
    adjacent_merge_frames: int = None,
    use_fp16: bool = True,
    feature_fp16: Optional[bool] = None
) -> List[SceneLabelResult]:
    """
    运行遍历模式标签匹配搜索
    
    核心特色：跨帧选择最高相似度标签
    - 每个场景有3帧特征向量（起始帧、中间帧、结束帧）
    - 对每个大类的每个标签，分别计算与3帧的相似度
    - 选择3帧中相似度最高的那个作为该标签的最终相似度
    - 每个大类选择相似度最高的标签
    
    Args:
        index_directory: 索引文件目录
        output_directory: 输出目录
        video_output_directory: 视频输出目录
        video_copy_mode: 视频切割模式
        video_name_format: 视频名称格式
        debug_similarity: 调试模式
        prompt_search_batch_size: 搜索时每批加载的标签向量数量
        pkl_batch_size: 每批加载的PKL数量
            - None 或 >= PKL总数: 一次性全部加载到GPU（显存占用高但搜索快）
            - < PKL总数: 分批加载，用完释放（显存占用低但稍慢）
        search_mode: 搜索模式选择
            - -1（按视频模式）: 每个视频独立搜索，每个视频返回 top_k 个结果
            - 0（按PKL模式）: 每个PKL文件独立搜索，每个PKL返回 top_k 个结果
            - 1（跨PKL模式）: 全局搜索，返回全局 top_k 个结果
        top_k: 每组返回的最大场景数，None则不限制
        start_frame_offset: 起始帧偏移量（负数向前，正数向后）
            - None: 使用默认值（copy模式=0，精确切割=-2）
            - 例如：-2 表示起始帧向前偏移2帧
        end_frame_offset: 结束帧偏移量（负数向前，正数向后）
            - None: 使用默认值（copy模式=2，精确切割=2）
            - 例如：2 表示结束帧向后偏移2帧
        use_diskcache: 是否使用 LMDB 磁盘缓存存储结果（解决内存问题）
        diskcache_dir: LMDB 缓存目录，None则使用默认目录
        use_chinese_labels: 是否使用中文标签模式
            - False（默认）: 使用英文标签值编码向量，适合英文CLIP模型
            - True: 使用中文标签键名编码向量，适合中文CLIP模型
        pkl_load_workers: PKL加载线程数（全量预加载时使用）
        lmdb_write_batch_size: LMDB单事务写入批大小（分批加载时使用）
        label_cache_batch_size: 标签缓存生成时的批处理大小
            - 用于分批编码标签向量，避免大规模标签时内存溢出
            - 推荐值: 256-1024
        vector_dedup_threshold: 向量去重余弦相似度阈值
            - None（默认）: 不进行向量去重
            - 0.90 ~ 0.98: 超过此阈值的同标签视频只保留一个
            - 优先规则: 同标签组内如有 OP/ED 视频，只保留 OP/ED 视频
        adjacent_merge_frames: 相邻片段合并帧阈值
            - None（默认）: 不进行相邻片段合并
            - N（正整数）: 当片段A的endframe与片段B的startframe差值≤N时合并
    
    自动配置（无需传参）：
        - 相似度阈值：根据模型类型自动选择（CLIP Large: 21, FG-CLIP2: 14）
        - 标签向量缓存：自动检测（有就用，没有就生成），目录固定为 templates/prompt_cache
    
    Returns:
        有效场景的标签匹配结果列表
    """
    # 路径解析
    resolver = PathResolver()
    
    if index_directory is None:
        index_directory = str(resolver.project_root / 'indexes')
    
    if output_directory is None:
        output_directory = str(resolver.project_root / 'output')
    
    os.makedirs(output_directory, exist_ok=True)
    
    # 固定缓存目录为 templates/prompt_cache
    cache_dir = str(resolver.project_root / 'templates' / 'prompt_cache')
    
    # 查找PKL文件
    pkl_files = []
    for f in os.listdir(index_directory):
        if f.endswith('.pkl'):
            pkl_files.append(os.path.join(index_directory, f))
    
    if not pkl_files:
        print(f"❌ 错误: 在 {index_directory} 中没有找到PKL文件")
        return []
    
    # 从PKL文件名提取模型名称
    model_name = None
    for pkl_file in pkl_files:
        basename = os.path.basename(pkl_file)
        # 格式: VideoName_modelname.pkl
        parts = basename.rsplit('_', 1)
        if len(parts) == 2:
            model_name = parts[1].replace('.pkl', '')
            break
    
    if model_name is None:
        model_name = "openai-clip-vit-large-patch14"
    
    # 加载模型
    from A_coreUtils.embedding.embedding_model import EmbeddingModelProcessor
    from A_coreUtils.search.auto_scene_search import (
        detect_model_type_from_name,
        SimilarityThresholdConfig,
        export_video_matches,
        cleanup_temp_after_export,
    )
    
    processor = EmbeddingModelProcessor(
        model_name=model_name,
        use_fp16=use_fp16
    )
    
    # 确定相似度阈值（优先使用 config.json 配置，否则使用默认值）
    model_type = detect_model_type_from_name(model_name)
    similarity_threshold = SimilarityThresholdConfig.get_threshold(model_type, use_reranker=False)
    
    lang_mode = "中文" if use_chinese_labels else "英文"
    print(f"[遍历搜索] 相似度阈值: {similarity_threshold}")
    print(f"[遍历搜索] 标签模式: {lang_mode}")
    
    # 预先创建 KeywordManager 获取大类名称（用于验证 video_name_format）
    keyword_manager = KeywordManager()
    categories = keyword_manager.categories
    
    # 验证或生成 video_name_format（在搜索开始前验证）
    if video_name_format is None:
        video_name_format = generate_default_label_video_name_format(categories, prefix="标签模式")
        print(f"[遍历搜索] 使用默认 video_name_format: {video_name_format}")
    else:
        is_valid, error_msg = validate_label_video_name_format(video_name_format, categories)
        if not is_valid:
            raise ValueError(f"[遍历搜索] {error_msg}")
        print(f"[遍历搜索] 使用自定义 video_name_format: {video_name_format}")
    
    # 创建搜索器（标签缓存自动检测：有就用，没有就生成）
    searcher = LabelTraverseSearcher(
        processor=processor,
        index_paths=pkl_files,
        similarity_threshold=similarity_threshold,
        prompt_search_batch_size=prompt_search_batch_size,
        use_fp16=use_fp16,
        feature_fp16=feature_fp16,
        use_label_cache=True,  # 始终启用缓存，自动检测
        cache_dir=cache_dir,   # 固定为 templates
        model_name=model_name,
        # PKL加载参数
        pkl_batch_size=pkl_batch_size,
        pkl_load_workers=pkl_load_workers,
        # 搜索模式参数
        search_mode=search_mode,
        top_k=top_k,
        scene_chunk_size=scene_chunk_size,
        # LMDB 缓存参数
        use_diskcache=use_diskcache,
        diskcache_dir=diskcache_dir,
        lmdb_write_batch_size=lmdb_write_batch_size,
        # 中文标签模式
        use_chinese_labels=use_chinese_labels,
        # 标签缓存批处理大小
        label_cache_batch_size=label_cache_batch_size
    )
    
    # 提取场景特征向量（用于向量去重）
    scene_features = {}  # {scene_key: np.ndarray}
    scene_pkl_map = {}  # {scene_key: source_pkl}
    
    try:
        # 执行搜索
        results = searcher.search()
        
        # 如果启用向量去重，提取场景特征向量
        if vector_dedup_threshold is not None and results:
            print(f"[遍历搜索] 提取场景特征向量用于去重...")
            result_keys = {r.scene_key for r in results}

            # 从 searcher 中提取特征向量（全量预加载路径）
            if hasattr(searcher, 'pkl_features') and searcher.pkl_features:
                for pkl_path, features_gpu in searcher.pkl_features.items():
                    scene_maps = searcher.pkl_scene_maps.get(pkl_path, [])
                    feature_counts = searcher.pkl_feature_counts.get(pkl_path, [])

                    features_cpu = features_gpu.float().cpu().numpy()
                    feat_idx = 0
                    for scene_idx, scene_info in enumerate(scene_maps):
                        video_name = os.path.basename(scene_info['video_path']) if scene_info.get('video_path') else ''
                        scene_key = f"{scene_info['start_frame']}_{video_name}"
                        count = feature_counts[scene_idx] if scene_idx < len(feature_counts) else 1

                        if scene_key in result_keys:
                            scene_vectors = features_cpu[feat_idx:feat_idx + count]
                            scene_features[scene_key] = scene_vectors
                            scene_pkl_map[scene_key] = pkl_path

                        feat_idx += count

            elif hasattr(searcher, 'video_features') and searcher.video_features:
                for video_path, features_gpu in searcher.video_features.items():
                    scene_maps = searcher.video_scene_maps.get(video_path, [])
                    feature_counts = searcher.video_feature_counts.get(video_path, [])

                    features_cpu = features_gpu.float().cpu().numpy()
                    feat_idx = 0
                    for scene_idx, scene_info in enumerate(scene_maps):
                        video_name = os.path.basename(scene_info['video_path']) if scene_info.get('video_path') else ''
                        scene_key = f"{scene_info['start_frame']}_{video_name}"
                        count = feature_counts[scene_idx] if scene_idx < len(feature_counts) else 1

                        if scene_key in result_keys:
                            scene_vectors = features_cpu[feat_idx:feat_idx + count]
                            scene_features[scene_key] = scene_vectors
                            scene_pkl_map[scene_key] = scene_info.get('source_pkl', 'unknown')

                        feat_idx += count

            # 分批PKL路径：按批回读PKL提取命中结果的特征向量（与P模式对齐）
            elif hasattr(searcher, 'index_paths') and hasattr(searcher, 'pkl_batch_size'):
                pkl_bs = searcher.pkl_batch_size
                total_pkls = len(searcher.index_paths)
                total_batches = (total_pkls + pkl_bs - 1) // pkl_bs

                for batch_idx in range(total_batches):
                    batch_start = batch_idx * pkl_bs
                    batch_end = min(batch_start + pkl_bs, total_pkls)
                    batch_paths = searcher.index_paths[batch_start:batch_end]

                    preloaded_batch = {}
                    worker_count = max(1, int(getattr(searcher, 'pkl_load_workers', 4)))
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        futures = {executor.submit(searcher._load_single_pkl, p): p for p in batch_paths}
                        for future in as_completed(futures):
                            pkl_path = futures[future]
                            try:
                                preloaded_batch[pkl_path] = future.result()
                            except Exception as e:
                                print(f"  [Error] 预加载 {os.path.basename(pkl_path)} 失败: {e}")
                                preloaded_batch[pkl_path] = (None, [], [])

                    for pkl_path in batch_paths:
                        features_np, scene_maps, feature_counts = preloaded_batch.get(pkl_path, (None, [], []))
                        if features_np is None or len(scene_maps) == 0:
                            continue

                        feat_idx = 0
                        for scene_idx, scene_info in enumerate(scene_maps):
                            video_name = os.path.basename(scene_info['video_path']) if scene_info.get('video_path') else ''
                            scene_key = f"{scene_info['start_frame']}_{video_name}"
                            count = feature_counts[scene_idx] if scene_idx < len(feature_counts) else 1

                            if scene_key in result_keys and scene_key not in scene_features:
                                scene_vectors = features_np[feat_idx:feat_idx + count]
                                scene_features[scene_key] = scene_vectors
                                scene_pkl_map[scene_key] = scene_info.get('source_pkl', pkl_path)

                            feat_idx += count

                        del features_np

                    gc.collect()

                    if len(scene_features) >= len(result_keys):
                        break

            print(f"[遍历搜索] 提取了 {len(scene_features)} 个场景的特征向量")
    finally:
        # 清理资源
        searcher.cleanup()
    
    # ========== 向量去重和相邻片段合并（视频导出前）==========
    if results and (vector_dedup_threshold is not None or adjacent_merge_frames is not None):
        # 转换为 best_matches 格式
        best_matches = convert_label_results_to_dict(results, video_name_format)
        
        # 向量去重（如果启用）
        if vector_dedup_threshold is not None and scene_features:
            print(f"\n[向量去重] 开始向量去重，阈值={vector_dedup_threshold}，模式=同PKL内去重")
            from .batch_text_search import deduplicate_by_vector_similarity
            best_matches = deduplicate_by_vector_similarity(
                best_matches=best_matches,
                scene_features=scene_features,
                video_name_format=video_name_format,
                similarity_threshold=vector_dedup_threshold,
                scene_pkl_map=scene_pkl_map
            )
        
        # 相邻片段合并（如果启用）
        if adjacent_merge_frames is not None and adjacent_merge_frames >= 0:
            from A_coreUtils.search.auto_scene_search import merge_adjacent_scenes
            best_matches = merge_adjacent_scenes(
                best_matches=best_matches,
                adjacent_merge_frames=adjacent_merge_frames,
                video_name_format=video_name_format
            )
        
        # 转换回 SceneLabelResult 格式
        results = convert_dict_to_label_results(best_matches, categories)
    
    # 导出视频（必须执行）
    if results:
        print("\n" + "=" * 70)
        print("📹 导出视频片段")
        print("=" * 70)
        
        best_matches_for_export = convert_label_results_to_dict(results, video_name_format)
        export_stats, video_output_directory = export_video_matches(
            best_matches=best_matches_for_export,
            resolver=resolver,
            output_directory=output_directory,
            video_output_directory=video_output_directory,
            video_copy_mode=video_copy_mode,
            start_frame_offset=start_frame_offset,
            end_frame_offset=end_frame_offset,
            debug_similarity=debug_similarity
        )
        print(f"  导出完成: 成功 {export_stats['success']}, 失败 {export_stats['failed']}, 跳过 {export_stats['skipped']}")
        if export_stats['failed_files']:
            print(f"  ⚠️ 失败文件: {len(export_stats['failed_files'])} 个")
        
        cleanup_temp_after_export(resolver)
    
    return results


def generate_label_vector_cache(
    model_name: str = "openai-clip-vit-large-patch14",
    cache_dir: str = None,
    force_regenerate: bool = False
) -> str:
    """
    生成标签向量缓存
    
    Args:
        model_name: 模型名称
        cache_dir: 缓存目录，None则使用 templates/prompt_cache
        force_regenerate: 是否强制重新生成
    
    Returns:
        缓存文件路径
    """
    print("=" * 70)
    print("📦 标签向量缓存生成器")
    print("=" * 70)
    
    print(f"\n配置:")
    print(f"  模型: {model_name}")
    print(f"  强制重新生成: {force_regenerate}")
    
    # 加载模型
    from A_coreUtils.embedding.embedding_model import EmbeddingModelProcessor
    
    print(f"\n[步骤1] 加载模型...")
    processor = EmbeddingModelProcessor(
        model_name=model_name,
        use_fp16=True
    )
    
    # 创建缓存器
    print(f"\n[步骤2] 初始化缓存器...")
    cache = LabelVectorCache(
        processor=processor,
        cache_dir=cache_dir,
        model_name=model_name
    )
    
    # 生成缓存
    print(f"\n[步骤3] 生成缓存...")
    cache_path = cache.generate_cache(force_regenerate=force_regenerate)
    
    print(f"\n✅ 缓存生成完成!")
    print(f"   文件位置: {cache_path}")
    
    return cache_path


# ============================================================
#  测试入口
# ============================================================
if __name__ == '__main__':
    # 测试运行
    results = run_label_traverse_search(
        prompt_search_batch_size=1024,
        debug_similarity=True
    )
    
    print(f"\n✅ 测试完成，共 {len(results)} 个有效场景")
