# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
# auto_scene_search.py
# 自动化场景搜索器 - 遍历所有合理的关键词组合进行文搜图
# 基于 logic_keywords.json，按自动提纯模板生成 prompt
# v1.0: 初始版本
# v1.1: 添加去重功能 - 同一场景保留相似度最高的prompt结果

import json
import hashlib
import math
import os
import pickle
import re
import sys
import time
import itertools
from functools import lru_cache
from typing import List, Dict, Optional, Callable, Tuple, Any
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
from A_coreUtils.aftertreatment.jev_rerank import score_prefix

# 导入视频名称解析器
from A_coreUtils.video_processing.video_name_parser import VideoNameParser

_MERGED_NAME_PARSER = VideoNameParser()


@lru_cache(maxsize=4096)
def _parse_merged_video_name(name: str) -> str:
    return _MERGED_NAME_PARSER.parse_filename(name)

# 延迟导入嵌入模型处理器（避免在只需要PromptGenerator时加载重型依赖）
EmbeddingModelProcessor = None
BatchTextSearchEngine = None
_BATCH_SEARCH_AVAILABLE = False

# Backward-compatible system placeholders used by sibling search modules.
_SYS_PH_START_FRAME = "\u8d77\u59cb\u5e27"
_SYS_PH_VIDEO_NAME = "\u89c6\u9891\u89e3\u6790\u540d"
SYSTEM_PLACEHOLDERS = {
    _SYS_PH_START_FRAME,
    _SYS_PH_VIDEO_NAME,
    "\u6269\u5c55\u540d",
}

# Backend defaults (used when caller omits optional params).
DEFAULT_PROMPT_SEARCH_BATCH_SIZE = 1024
DEFAULT_PROMPT_CACHE_BATCH_SIZE = 512
DEFAULT_LANCE_LOAD_WORKERS = 4
DEFAULT_LMDB_WRITE_BATCH_SIZE = 1000


def _lazy_import_embedding():
    """延迟导入嵌入模型相关模块"""
    global EmbeddingModelProcessor
    if EmbeddingModelProcessor is None:
        from A_coreUtils.embedding.embedding_model import EmbeddingModelProcessor as _EMP
        EmbeddingModelProcessor = _EMP


def _lazy_import_batch_search():
    """延迟导入批量搜索引擎"""
    global BatchTextSearchEngine, _BATCH_SEARCH_AVAILABLE
    if BatchTextSearchEngine is None:
        try:
            from .batch_text_search import BatchTextSearchEngine as _BTS
            BatchTextSearchEngine = _BTS
            _BATCH_SEARCH_AVAILABLE = True
        except ImportError as e:
            print(f"[Warning] 批量搜索引擎不可用: {e}")
            _BATCH_SEARCH_AVAILABLE = False


def _sanitize_filename(name: str) -> str:
    """Sanitize filename by replacing illegal filesystem characters."""
    if not name:
        return ""
    illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in illegal_chars:
        name = name.replace(char, '_')
    return name.strip()


def normalize_top_k(top_k: Optional[Any]) -> Optional[int]:
    """
    Normalize top_k so that None / empty / non-positive means "no limit".
    """
    if top_k is None:
        return None
    if isinstance(top_k, str):
        normalized = top_k.strip().lower()
        if normalized in ('', 'none', 'null', 'unlimited', 'default', 'auto', '默认', '不限'):
            return None
    try:
        top_k_value = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"top_k must be int or None, got {top_k!r}") from exc
    return top_k_value if top_k_value > 0 else None


def normalize_optional_positive_int(
    value: Optional[Any],
    *,
    field_name: str = "value",
    default: Optional[int] = None
) -> Optional[int]:
    """
    Normalize an optional positive integer.
    - None/empty/none-like -> default
    - <=0 -> default
    - positive integer -> that integer
    """
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


def validate_prompt_video_name_format(video_name_format: str, categories: List[str]) -> Tuple[bool, str]:
    """
        验证 Prompt 模式的 video_name_format 是否合法
        
        检查格式字符串中的占位符是否都是有效的大类名称或系统占位符。
        
        Args:
            video_name_format: 视频名称格式字符串
            categories: 有效的大类名称列表（如 ['主体', '动作', '场景', '情绪']）
        
        Returns:
            (是否合法, 错误信息)
        
    """
    import re

    # 提取所有占位符
    system_placeholders = SYSTEM_PLACEHOLDERS
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


def _cleanup_gpu_memory():
    """清理 GPU 显存"""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class PromptGenerator:
    """
        Prompt生成器 - 基于logic_Prompt.py的逻辑
        
        遵循严格的逻辑约束（从JSON动态读取）：
        - 分配规则定义了大类之间的约束关系
        - 例如：主体类型决定可用的动作子类
        
        所有关键词完全从logic_keywords.json动态读取，支持后续扩展
        
        支持自定义prompt模板，可使用的占位符：
        - JSON中定义的任何大类（如 {主体}, {动作}, {场景}, {情绪} 等）
        
        大类会根据JSON文件动态检测，无需硬编码。
        
    """

    # 系统保留字段（不作为大类处理）
    RESERVED_KEYS = {"自动提纯prompt", "其他参数设置"}

    def __init__(self, keywords_path: str = None, use_chinese: bool = False):
        """
                初始化Prompt生成器
                
                Args:
                    keywords_path: logic_keywords.json的路径，None则使用默认路径
                    use_chinese: 是否使用中文模式
                        - False（默认）: 使用英文标签值生成prompt
                        - True: 使用中文标签键名生成prompt
                
        """
        self.use_chinese = use_chinese

        if keywords_path is None:
            # 使用项目根目录（Cut_DetectScene）下的 logic_keywords.json
            resolver = PathResolver()
            keywords_path = str(resolver.project_root / 'logic_keywords.json')

        with open(keywords_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)

        # 从 "标签" 下提取数据（必须存在）
        if "\u6807\u7b7e" not in full_data:
            raise KeyError("logic_keywords.json must contain key \"标签\"")
        self.data = full_data["\u6807\u7b7e"]
        self._full_data = full_data  # 保留完整JSON引用，读取顶层键

        # 动态发现JSON中的所有大类
        self._simple_categories = self._discover_simple_categories()

        # 统一模式的分类prompt模板（从JSON读取）
        raw_templates = self._full_data.get("自动提纯prompt", {})
        self._per_cat_templates = {k: v for k, v in raw_templates.items() if not k.startswith("_")}

        # 其他参数设置
        other = self._full_data.get("其他参数设置", {})
        self.name_template = other["输出视频命名格式"]
        if not isinstance(self.name_template, str) or not self.name_template.strip():
            raise ValueError("logic_keywords.json 中的输出视频命名格式不能为空")
        self._required_categories = [c for c in other.get("必有标签", []) if isinstance(c, str)]
        self._required_subcategories = other.get("必有子类", []) or []
        self._required_labels = other.get("必有孙类", []) or []

        # 构建子类→父类+标签查找索引
        self._subcategory_index = self._build_subcategory_index()

        # 一致性校验
        self._warnings = self._validate_consistency()
        for w in self._warnings:
            print(f"\u26a0\ufe0f\u26a0\ufe0f\u26a0\ufe0f [logic_keywords] {w}")

    def _validate_consistency(self) -> list:
        """校验 必有标签 ↔ 标签.* ↔ Jev-rerank验证Prompt 的一致性，返回警告列表。"""
        warnings = []
        special = self.RESERVED_KEYS | {'_说明'}
        all_cats = {k for k in self.data if k not in special and not k.startswith('_')
                    and isinstance(self.data.get(k), dict)}
        required_set = set(self._required_categories)

        for cat in required_set - all_cats:
            warnings.append(f"必有标签 '{cat}' 在 标签.* 中未定义，Jev-rerank 验证将失败")
        for cat in all_cats - required_set:
            warnings.append(f"大类 '{cat}' 未加入 必有标签，Jev-rerank 将跳过该类别验证")

        settings = self._full_data.get('其他参数设置', {})
        verify_cfg = settings.get('Jev-rerank验证Prompt', {})
        zh_tpl = verify_cfg.get('zh', {}) if isinstance(verify_cfg, dict) else {}
        tpl_keys = {k for k in zh_tpl if not k.startswith('_')}
        for cat in tpl_keys - all_cats:
            warnings.append(f"Jev-rerank验证Prompt 中的 '{cat}' 在 标签.* 中未定义")
        for cat in required_set & all_cats - tpl_keys:
            warnings.append(f"大类 '{cat}' 缺少 Jev-rerank验证Prompt 模板，将使用通用占位符")

        return warnings

    def _discover_simple_categories(self) -> Dict[str, Dict[str, str]]:
        """动态发现JSON中的所有大类，返回 {大类名: {中文: 英文}}。"""
        simple_categories = {}
        for key, value in self.data.items():
            if key in self.RESERVED_KEYS or key.startswith("_"):
                continue
            if isinstance(value, dict) and value:
                first_value = next(iter(value.values()), None)
                if isinstance(first_value, dict):
                    merged = {}
                    for subcat_name, subcat_data in value.items():
                        if isinstance(subcat_data, dict) and not subcat_name.startswith("_"):
                            merged.update(subcat_data)
                    if merged:
                        simple_categories[key] = merged
                elif isinstance(first_value, str):
                    simple_categories[key] = value
        return simple_categories

    def _build_subcategory_index(self) -> Dict[str, tuple]:
        """构建子类→(标签字典, 父大类名) 索引，用于 prompt 占位符解析。

        Returns: {"生物": ({"男人":"man",...}, "主体"), "生物行为": ({"奔跑":"running",...}, "动作"), ...}
        """
        index = {}
        for cat_name, cat_data in self.data.items():
            if cat_name in self.RESERVED_KEYS or cat_name.startswith("_"):
                continue
            if not isinstance(cat_data, dict):
                continue
            for sub_name, sub_data in cat_data.items():
                if not sub_name.startswith("_") and isinstance(sub_data, dict):
                    index[sub_name] = (sub_data, cat_name)
        # 同时注册大类名本身（如 {主体}）→ 合并该大类下所有子类的标签
        for key in self._simple_categories:
            if key not in index:
                index[key] = (self._simple_categories[key], key)
        return index

    def _resolve_placeholder(self, placeholder: str) -> Tuple[Dict[str, str], str, str]:
        """解析占位符 {名称} → (标签dict, 标签_display_name, 所属大类)。

        Returns None 若无法解析。
        """
        if placeholder in self._subcategory_index:
            labels, parent = self._subcategory_index[placeholder]
            return labels, placeholder, parent
        # 尝试模糊匹配
        for key in self._subcategory_index:
            if placeholder in key:
                labels, parent = self._subcategory_index[key]
                return labels, key, parent
        return None

    def iterate_all_labels_flat(self):
        """按\"自动提纯prompt\"模板生成独立按类别提示词。

        模板格式: {zh_prompt: en_prompt}，与标签的 zh:en 结构一致。
        根据 self.use_chinese 选择中文或英文模板，占位符用对应语言的标签值填充。
        """
        for cat_name, templates in self._per_cat_templates.items():
            if not isinstance(templates, list):
                templates = [templates]
            for template_entry in templates:
                # 解析 zh:en 模板对
                if isinstance(template_entry, str):
                    # 兼容旧格式纯字符串
                    zh_tpl = template_entry
                    en_tpl = template_entry
                elif isinstance(template_entry, dict) and template_entry:
                    zh_tpl = list(template_entry.keys())[0]
                    en_tpl = list(template_entry.values())[0]
                else:
                    continue

                tpl = zh_tpl if self.use_chinese else en_tpl
                if not tpl or not tpl.strip():
                    continue

                # 解析占位符
                phs = re.findall(r'\{([^}]+)\}', tpl)
                if not phs:
                    continue

                # 解析每个占位符 → 标签集
                label_sets = []      # [(ph_name, ph_display, {cn: en}, parent_cat)]
                for ph in phs:
                    resolved = self._resolve_placeholder(ph)
                    if resolved is None:
                        continue
                    labels, display, parent = resolved
                    if not labels:
                        continue
                    label_sets.append((ph, display, labels, parent))

                if not label_sets:
                    continue

                # 笛卡尔积
                label_dicts = [list(ls[2].items()) for ls in label_sets]  # [[(cn,en),...], ...]
                for combo in itertools.product(*label_dicts):
                    fmt = {}
                    meta = {}    # {display_name: cn_string}
                    # 向后兼容字段：映射到旧格式的大类名
                    compat = {}
                    for i, (label_cn, label_en) in enumerate(combo):
                        ph_name = label_sets[i][0]
                        display = label_sets[i][1]
                        parent = label_sets[i][3]
                        # 占位符填充值
                        fmt[ph_name] = label_cn if self.use_chinese else label_en
                        meta[display] = label_cn
                        # 动态字段：{大类名}_cn / {大类名}_en（只写当前模板所属大类）
                        if parent == cat_name:
                            compat[f"{parent}_cn"] = label_cn
                            compat[f"{parent}_en"] = label_en

                    prompt = tpl.format(**fmt)
                    result = {
                        "prompt": prompt,
                        "prompt_cn": prompt if self.use_chinese else "",
                        "prompt_en": "" if self.use_chinese else prompt,
                        "category": cat_name,
                        "label_cn": list(meta.values())[0] if len(meta) == 1 else "",
                        "label_en": list(compat.values())[1] if len(compat) >= 2 else "",
                        "labels": meta,
                    }
                    result.update(compat)
                    yield result

def extract_model_name_from_index(index_path: str) -> str:
    """从索引路径名中解析模型名称。"""
    import re
    filename = os.path.basename(index_path)
    basename = filename.replace('.lance', '')

    remaining = basename

    # 尝试匹配模式后缀
    mode_match = re.search(r'_(Single|Triplet|TripletAvg|SceneNative3f|SceneNative|SceneDetect|Video)$', basename)
    if mode_match:
        remaining = basename[:mode_match.start()]

    # 匹配可选的维度后缀
    dim_match = re.search(r'_d\d+$', remaining)
    if dim_match:
        remaining = remaining[:dim_match.start()]

    # 找到最后一个下划线后的部分作为模型名（模型名不含下划线）
    last_underscore = remaining.rfind('_')
    if last_underscore == -1:
        return None

    return remaining[last_underscore + 1:]


def detect_truncate_dim_from_index_paths(index_paths) -> int:
    """从索引文件名中自动检测 truncate_dim（_d\\d+ 后缀）。

    扫描所有索引路径，匹配文件名中的 ``_d<数字>`` 后缀。
    如果找到则返回对应的整数维度，否则返回 None。
    """
    import re as _re
    for path in index_paths:
        filename = os.path.basename(path)
        basename = filename.replace('.lance', '').replace('.pkl', '')
        match = _re.search(r'_d(\d+)(?:_|\.|$)', basename)
        if match:
            return int(match.group(1))
    return None


def detect_model_type_from_name(model_name: str) -> str:
    """
        根据模型名称检测模型类型
        
        Returns:
            'clip' 或 'fgclip2'
        
    """
    if model_name is None:
        return 'clip'

    model_name_lower = model_name.lower()

    if 'fg-clip2' in model_name_lower or 'fg_clip2' in model_name_lower or 'fgclip2' in model_name_lower:
        return 'fgclip2'

    return 'clip'


class AutoSceneSearcher:
    """
        自动化场景搜索器
        
        遍历所有合理的关键词组合，调用embedding_model的文搜图方法
        自动从pkl文件名获取模型名并加载对应模型
        
    """

    def __init__(self,
                 io_workers: int = 8,
                 use_fp16: bool = False,
                 cache_dir: str = None,
                 use_chinese: bool = False):
        """
                初始化自动化搜索器（延迟加载模型）
                
                Args:
                    io_workers: I/O工作线程数
                    use_fp16: 是否使用FP16精度
                    cache_dir: 缓存目录
                    use_chinese: 是否使用中文模式
                        - False（默认）: 使用英文标签值生成prompt
                        - True: 使用中文标签键名生成prompt
                
        """
        # 初始化路径解析器（不传参数，使用 path_resolver.py 所在目录作为项目根目录）
        self.resolver = PathResolver()

        # 保存中文模式设置
        self.use_chinese = use_chinese

        # 初始化Prompt生成器（传递 use_chinese 参数）
        self.prompt_generator = PromptGenerator(use_chinese=use_chinese)

        # 保存配置，延迟加载模型
        self.io_workers = io_workers
        self.use_fp16 = use_fp16
        self.truncate_dim = None  # 延迟检测：在 run_batch_search_optimized 中从索引文件名自动推断

        # 缓存目录
        if cache_dir is None:
            cache_dir = str(self.resolver.project_root / 'temp' / 'cache')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)


    def find_index_files(self, index_dir: str = None) -> List[str]:
        """
                查找索引文件
                
                Args:
                    index_dir: 索引目录，None则使用默认目录
                
                Returns:
                    索引文件路径列表
                
        """
        if index_dir is None:
            index_dir = self.resolver.resolve_str('indexes')

        index_files = []
        if os.path.isdir(index_dir):
            for filename in os.listdir(index_dir):
                if os.path.isdir(os.path.join(index_dir, filename)) and filename.endswith('.lance'):
                    index_files.append(os.path.join(index_dir, filename))

        return index_files

    def run_batch_search_optimized(self,
                                    index_paths: List[str] = None,
                                    similarity_threshold: float = 20.0,
                                    candidate_batch_size: Optional[int] = None,
                                    prompt_search_batch_size: Optional[int] = DEFAULT_PROMPT_SEARCH_BATCH_SIZE,
                                    feature_fp16: bool = True,
                                    lance_batch_size: Optional[int] = None,
                                    use_diskcache: bool = True,
                                    diskcache_dir: str = None,
                                    search_mode: int = 0,
                                    top_k: Optional[int] = None,
                                    prompt_cache_batch_size: Optional[int] = DEFAULT_PROMPT_CACHE_BATCH_SIZE,
                                    lance_load_workers: Optional[int] = DEFAULT_LANCE_LOAD_WORKERS,
                                    lmdb_write_batch_size: Optional[int] = DEFAULT_LMDB_WRITE_BATCH_SIZE,
                                    vector_dedup_threshold: float = None):
        prompt_search_batch_size = normalize_optional_positive_int(
            prompt_search_batch_size,
            field_name='prompt_search_batch_size',
        )
        prompt_cache_batch_size = normalize_optional_positive_int(
            prompt_cache_batch_size,
            field_name='prompt_cache_batch_size',
        )
        lance_load_workers = normalize_optional_positive_int(
            lance_load_workers,
            field_name='lance_load_workers',
        )
        lmdb_write_batch_size = normalize_optional_positive_int(
            lmdb_write_batch_size,
            field_name='lmdb_write_batch_size',
        )
        candidate_batch_size = normalize_optional_positive_int(
            candidate_batch_size,
            field_name='candidate_batch_size',
        )
        lance_batch_size = normalize_optional_positive_int(
            lance_batch_size,
            field_name='lance_batch_size',
        )
        top_k = normalize_top_k(top_k)
        if index_paths is None:
            index_paths = self.find_index_files()
        if not index_paths:
            print("[Error] 索引目录中没有 .lance 索引")
            return None
        model_names = set()
        for index_path in index_paths:
            model_name = extract_model_name_from_index(index_path)
            model_names.add(model_name if model_name else "unknown")
        if len(model_names) > 1:
            print(f"[Error] Multiple model names detected: {len(model_names)}")
            return None
        model_name = list(model_names)[0] if model_names else None
        if not model_name or model_name == "unknown":
            print("[Error] 无法从索引文件名中提取模型名")
            return None
        _lazy_import_batch_search()
        _lazy_import_embedding()
        if not _BATCH_SEARCH_AVAILABLE:
            raise RuntimeError(
                "批量搜索引擎不可用。\n"
                "Please check whether batch_text_search.py exists and can be imported."
            )
        model_type = detect_model_type_from_name(model_name)
        # 自动检测 truncate_dim：从索引文件名的 _d<数字> 后缀推断
        if self.truncate_dim is None:
            detected_dim = detect_truncate_dim_from_index_paths(index_paths)
            if detected_dim is not None:
                self.truncate_dim = detected_dim
                print(f"[优化搜索] 从索引文件名自动检测到 truncate_dim={detected_dim}")
        from A_coreUtils.prompt.prompt_vector_cache import PromptVectorCache
        prompt_cache = PromptVectorCache(batch_size=prompt_cache_batch_size or DEFAULT_PROMPT_CACHE_BATCH_SIZE,
                                         use_chinese=self.use_chinese)
        if not prompt_cache.cache_exists(model_name=model_name):
            processor = EmbeddingModelProcessor(model_name=model_name, model_type=model_type,
                                               truncate_dim=self.truncate_dim, io_workers=self.io_workers,
                                               use_fp16=self.use_fp16)
            prompt_cache.processor = processor
            prompt_cache.generate_cache()
            prompt_cache.processor = None
            del processor
            _cleanup_gpu_memory()
        prompts = prompt_cache.load_cache_batched(model_name=model_name,
                    batch_size=prompt_search_batch_size or DEFAULT_PROMPT_SEARCH_BATCH_SIZE, use_fp16=self.use_fp16)
        root = diskcache_dir or os.path.join(self.cache_dir, 'search_results')
        parallel = min(lance_load_workers or DEFAULT_LANCE_LOAD_WORKERS, lance_batch_size or len(index_paths), len(index_paths))
        engine = BatchTextSearchEngine(candidate_batch_size or 128, feature_fp16, self.truncate_dim)
        try:
            return engine.search(index_paths, prompts, self.prompt_generator, similarity_threshold,
                                 search_mode, top_k, vector_dedup_threshold, root,
                                 use_diskcache, parallel, lmdb_write_batch_size or DEFAULT_LMDB_WRITE_BATCH_SIZE)
        finally:
            prompts.close()
            _cleanup_gpu_memory()


class SimilarityThresholdConfig:
    """按本次搜索的 Lance 模型选择召回阈值。"""

    @staticmethod
    def get_threshold(model_type: str = 'clip', thresholds: Optional[Dict] = None) -> float:
        key = 'fgclip2' if model_type.lower() in ('fgclip2', 'fg-clip2', 'fg_clip2') else 'clip_large'
        if thresholds is not None and thresholds.get(key) is not None:
            return float(thresholds[key])
        try:
            with (PathResolver().project_root / 'config.json').open(encoding='utf-8') as f:
                config = json.load(f)
            value = config.get('similarity_thresholds', {}).get(key)
            if value is not None:
                return float(value)
        except (OSError, TypeError, ValueError) as exc:
            print(f"[自动提纯] 读取配置阈值失败，使用模型默认值：{exc}")
        return 14.0 if key == 'fgclip2' else 21.0


def merge_adjacent_scenes(
    result_source,
    adjacent_merge_frames: int,
    video_name_format: str,
    output_cache_dir: str = None,
    batch_size: int = 1000,
):
    """Merge adjacent scenes through a temporary, ordered LMDB store."""
    from tempfile import mkdtemp
    from .lmdb_store import LMDBCache, LMDBResultView, _sortable_int

    if adjacent_merge_frames is None or adjacent_merge_frames < 0:
        return result_source
    root = output_cache_dir or os.path.dirname(result_source.cache_dir)
    os.makedirs(root, exist_ok=True)
    output = mkdtemp(dir=root, prefix="merged_")
    stage_path = mkdtemp(dir=root, prefix="merge_stage_")
    stage = LMDBCache(stage_path)
    cache = LMDBCache(output)
    try:
        staged = {}
        for batch in result_source.iter_batches(batch_size):
            for key, item in batch:
                video = str(item.get("video_path", "") or "")
                start = int(item.get("start_frame", 0) or 0)
                sort_key = (
                    b"stage:" + video.encode("utf-8", "surrogatepass")
                    + b"\0" + _sortable_int(start) + b":"
                    + str(key).encode("utf-8", "surrogatepass")
                )
                staged[sort_key] = item
            if len(staged) >= batch_size:
                stage.put_many(staged)
                staged.clear()
        if staged:
            stage.put_many(staged)

        def segments():
            previous_video = None
            previous_end = None
            sequence = 0
            for _, value in stage.iter_raw(b"stage:"):
                row = pickle.loads(value)
                video = str(row.get("video_path", "") or "")
                start = int(row.get("start_frame", 0) or 0)
                end = int(row.get("end_frame", start) or start)
                if previous_video != video or previous_end is None or start - previous_end > adjacent_merge_frames:
                    sequence += 1
                previous_video, previous_end = video, end
                yield sequence, row

        buffer = {}
        for _, grouped_rows in itertools.groupby(segments(), key=lambda value: value[0]):
            rows = (row for _, row in grouped_rows)
            try:
                first = next(rows)
            except StopIteration:
                continue
            try:
                second = next(rows)
            except StopIteration:
                second = None
            if second is None:
                key = hashlib.sha256(
                    json.dumps(
                        [first.get("video_path", ""), first.get("start_frame", 0), first.get("end_frame", 0)],
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()
                buffer[key] = first
            else:
                base_fields = ("video_path", "start_frame", "end_frame", "source_lance", "fps")
                skipped = {*base_fields, "similarity", "result_name", "prompt"}
                merged = {field: first.get(field) for field in base_fields}
                values = {}
                value_tokens = {}
                total = 0.0
                count = 0
                for row in itertools.chain((first, second), rows):
                    count += 1
                    total += float(row.get("similarity", 0.0) or 0.0)
                    merged["end_frame"] = row.get("end_frame", merged["end_frame"])
                    for name, value in row.items():
                        if name in skipped or value in (None, "", [], {}):
                            continue
                        token = json.dumps(value, ensure_ascii=False, sort_keys=True)
                        tokens = value_tokens.setdefault(name, set())
                        if token not in tokens:
                            tokens.add(token)
                            values.setdefault(name, []).append(value)
                for name, entries in values.items():
                    merged[name] = entries[0] if len(entries) == 1 else "_".join(str(value) for value in entries)
                merged["similarity"] = total / max(1, count)
                merged["result_name"] = _generate_merged_result_name(merged, video_name_format)
                key = hashlib.sha256(
                    json.dumps(
                        [merged.get("video_path", ""), merged.get("start_frame", 0), merged.get("end_frame", 0)],
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()
                buffer[key] = merged
            if len(buffer) >= batch_size:
                cache.put_results_many(buffer)
                buffer.clear()
        cache.put_results_many(buffer)
        cache.close()
        cache = None
        return LMDBResultView(output)
    except BaseException:
        if cache is not None:
            cache.destroy()
        raise
    finally:
        stage.destroy()

def _generate_merged_result_name(merged_data: Dict, video_name_format: str) -> str:
    """
        根据合并后的数据生成 result_name（动态占位符处理）
        
        Args:
            merged_data: 合并后的场景数据
            video_name_format: 视频名称格式模板
        
        Returns:
            生成的 result_name
        
    """
    # Parsing is cached per source video to avoid repeated work.

    # 解析视频名称
    video_path = merged_data.get('video_path', '')
    video_basename = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(video_basename)[0]
    parsed_name = _parse_merged_video_name(video_name_no_ext)

    # 1. 从 video_name_format 中动态解析所有占位符
    placeholders = re.findall(r'\{(\w+)\}', video_name_format)

    # 2. 为所有占位符预设空字符串默认值
    format_dict = {ph: "" for ph in placeholders}

    # 3. 填充系统占位符
    format_dict['起始帧'] = str(merged_data.get('start_frame', 0))
    format_dict['视频解析名'] = _sanitize_filename(parsed_name)
    format_dict['扩展名'] = os.path.splitext(video_basename)[1].lstrip('.')

    # 4. 填充大类中文值（从 merged_data 的 xxx_cn 动态字段）
    for key, value in merged_data.items():
        if key.endswith('_cn'):
            category_name = key[:-3]
            if category_name in format_dict:
                format_dict[category_name] = _sanitize_filename(str(value))

    # 6. 使用格式模板生成名称（所有占位符都已预设默认值，不会 KeyError）
    result_name = video_name_format.format(**format_dict)

    # 7. 动态清理连续分隔符（从格式模板中提取分隔符）
    # 提取占位符之间的分隔符
    separators = re.findall(r'\}([^\{]+)\{', video_name_format)
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
# ============================================================
# 视频导出模块
# ============================================================


class VideoExporter:
    """
        视频导出器 - 将搜索结果中的匹配片段导出为视频文件
        
        支持两种模式:
        - copy模式: 快速无损切割（可能有关键帧偏移）
        - 精确切割模式: 重新编码，帧精确
        
        视频命名格式: {lens}_{mood}_{scene}_{subject}_{action}_{startframe}.{ext}
        
    """

    def __init__(self, copy_mode: bool = True, verbose: bool = False,
                 start_frame_offset: int = None, end_frame_offset: int = None):
        """
                初始化视频导出器
                
                Args:
                    copy_mode: True使用copy模式（快速），False使用精确切割模式
                    verbose: 是否输出详细信息
                    start_frame_offset: 起始帧偏移量（None使用默认值）
                    end_frame_offset: 结束帧偏移量（None使用默认值）
                
        """
        self.copy_mode = copy_mode
        self.verbose = verbose
        self._cutter = None

        # 设置帧偏移（如果未指定则使用默认值）
        if start_frame_offset is None:
            self.start_frame_offset = 0
        else:
            self.start_frame_offset = start_frame_offset

        if end_frame_offset is None:
            self.end_frame_offset = 0
        else:
            self.end_frame_offset = end_frame_offset

    def _ensure_cutter(self):
        """延迟初始化 FFmpegPrecisionCutter"""
        if self._cutter is None:
            from A_coreUtils.video_processing.ffmpeg_precision_cutter import FFmpegPrecisionCutter
            self._cutter = FFmpegPrecisionCutter(
                copy_mode=self.copy_mode,
                verbose=self.verbose
            )

    def _get_video_extension(self, video_path: str) -> str:
        """Return video extension, defaulting to .mp4 when missing."""
        ext = os.path.splitext(video_path)[1].lower()
        if not ext:
            ext = '.mp4'  # 默认扩展名
        return ext

    def export_single(
        self,
        video_path: str,
        output_path: str,
        start_frame: int,
        end_frame: int,
        fps: float = None
    ) -> bool:
        """
                导出单个视频片段
                
                Args:
                    video_path: 源视频路径
                    output_path: 输出路径
                    start_frame: 起始帧
                    end_frame: 结束帧
                    fps: 帧率（可选，自动检测）
                
                Returns:
                    是否成功
                
        """
        self._ensure_cutter()

        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # 使用实例变量的帧偏移参数
            # 执行切割
            success = self._cutter.cut_by_frames(
                input_video=video_path,
                output_file=output_path,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                end_inclusive=True,
                start_frame_offset=self.start_frame_offset,
                end_frame_offset=self.end_frame_offset
            )

            return success
        except Exception as e:
            if self.verbose:
                print(f"  [Error] 导出失败: {e}")
            return False

    def export_deduplicated_results(
        self,
        result_source: Any,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
        debug_similarity: bool = False,
        post_export_hook: Optional[Callable] = None,
        use_aesthetic_score: bool = True,
    ) -> Dict:
        """
                导出去重后的搜索结果
                
                Args:
                    deduplicated_data: 去重后的结果字典，支持两种格式：
                    deduplicated_data: 去重后的结果字典
                        格式: {scene_key: {result_name, similarity, video_path, start_frame, end_frame}}
                    output_dir: 输出目录
                    progress_callback: 进度回调函数 (current, total, filename)
                    debug_similarity: 调试模式，在文件名前添加相似度
                
                Returns:
                    导出统计信息
                
        """
        self._ensure_cutter()
        self._debug_similarity = debug_similarity
        os.makedirs(output_dir, exist_ok=True)
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'failed_files': []
        }
        total_items = result_source.count() if hasattr(result_source, "count") else 0
        stats['total'] = int(total_items)
        if total_items == 0:
            print("  没有需要导出的视频")
            return stats
        print(f"\n开始导出 {stats['total']} 个去重后的视频片段...")
        current = 0
        for batch in result_source.iter_batches(batch_size=256):
            for scene_key, item in batch:
                current += 1
                result_name = item.get('result_name', scene_key)
                similarity = float(item.get('similarity', 0) or 0)
                video_path = item.get('video_path')
                if not video_path or not os.path.exists(video_path):
                    stats['skipped'] += 1
                    continue
                ext = self._get_video_extension(video_path)
                shot = item.get('shot_type', '')
                move = item.get('camera_movement', '')
                prefix = ''
                if shot and shot != '未知':
                    prefix += shot
                if move and move != '未知':
                    prefix += f"_{move}" if prefix else move
                if prefix:
                    prefix += '_'
                scores = score_prefix(similarity, item if use_aesthetic_score else {},
                                      include_similarity=self._debug_similarity)
                output_filename = f"{scores}{prefix}{_sanitize_filename(result_name)}{ext}"
                output_path = os.path.join(output_dir, output_filename)
                if progress_callback:
                    progress_callback(current, total_items, output_filename)
                elif self.verbose:
                    print(f"  [{current}/{total_items}] {output_filename}")
                if os.path.exists(output_path):
                    stats['skipped'] += 1
                    continue
                success = self.export_single(
                    video_path=video_path,
                    output_path=output_path,
                    start_frame=int(item.get('start_frame', 0) or 0),
                    end_frame=int(item.get('end_frame', 0) or 0),
                    fps=item.get('fps'),
                )
                if success:
                    stats['success'] += 1
                    if post_export_hook:
                        post_export_hook(output_path, item)
                else:
                    stats['failed'] += 1
                    if len(stats['failed_files']) < 256:
                        stats['failed_files'].append(output_filename)
                    if 'failure_log' not in stats:
                        from tempfile import NamedTemporaryFile
                        with NamedTemporaryFile(dir=output_dir, prefix='export_errors_', suffix='.txt', delete=False) as log:
                            stats['failure_log'] = log.name
                    with open(stats['failure_log'], 'a', encoding='utf-8') as log:
                        print(output_filename, file=log)
        return stats
# ============================================================
# ============================================================
# ============================================================


def resolve_video_output_directory(
    resolver: PathResolver,
    output_directory: str,
    video_output_directory: str = None
) -> str:
    if video_output_directory is None:
        return os.path.join(output_directory, 'videos')
    if not os.path.isabs(video_output_directory):
        return resolver.resolve_str(video_output_directory)
    return video_output_directory


def export_video_matches(
    result_source,
    resolver: PathResolver,
    output_directory: str,
    video_output_directory: str = None,
    video_copy_mode: bool = False,
    start_frame_offset: int = None,
    end_frame_offset: int = None,
    debug_similarity: bool = False,
    post_export_hook: Optional[Callable] = None,
    use_aesthetic_score: bool = False,
):
    resolved_output_dir = resolve_video_output_directory(
        resolver=resolver,
        output_directory=output_directory,
        video_output_directory=video_output_directory
    )
    exporter = VideoExporter(
        copy_mode=video_copy_mode,
        verbose=False,
        start_frame_offset=start_frame_offset,
        end_frame_offset=end_frame_offset
    )
    export_stats = exporter.export_deduplicated_results(
        result_source=result_source,
        output_dir=resolved_output_dir,
        progress_callback=None,
        debug_similarity=debug_similarity,
        post_export_hook=post_export_hook,
        use_aesthetic_score=use_aesthetic_score,
    )
    return export_stats, resolved_output_dir


def attach_shot_analysis(result_view, temp_dir: str = None):
    """对去重后的结果做景别和镜头运动分析，结果写回原 LMDB。

    景别固定使用场景中间帧，光流固定使用中间附近最多30张连续帧。
    同一视频的请求帧只解码一次，帧以 BGR numpy 数组在内存中按帧号分发，
    不再经过 JPG 临时文件。
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from collections import defaultdict
    from A_coreUtils.aftertreatment.shot_type_classifier import DinoV2ShotClassifier
    from A_coreUtils.aftertreatment.optical_flow_analyzer import OpticalFlowAnalyzer
    from A_coreUtils.search.reranker_frame_extractor import RerankerFrameExtractor

    if result_view is None or result_view.count() == 0:
        return result_view

    resolver = PathResolver()
    if temp_dir is None:
        temp_dir = str(resolver.project_root / 'temp' / 'cache' / 'shot_analysis')
    os.makedirs(temp_dir, exist_ok=True)

    shot_clf = DinoV2ShotClassifier()
    shot_clf.load()
    of_analyzer = OpticalFlowAnalyzer(
        sample_interval=1,
        static_threshold=0.35,
        radial_threshold=0.22,
        tracking_threshold=0.50,
        pan_tilt_ratio=0.35,
        push_symmetry_threshold=0.15,
        push_vote_ratio=0.15,
        tracking_vote_ratio=0.20,
    )
    memory_extractor = RerankerFrameExtractor(
        output_resolution='384', backend='auto', decord_decode_batch_size=30
    )

    # 直接用导出前用的 result_view 的 LMDB，不新建
    cache = getattr(result_view, '_cache', None)
    if cache is None:
        return result_view

    total = result_view.count()
    print(f"\n[帧提取-共享] 开始为 {total} 个场景准备分析帧...")
    print(f"[景别分析-GPU] 使用场景中间帧进行景别判断...")
    print(f"[光流分析-CPU] 使用中间附近连续帧进行镜头运动判断...")
    # Optical-flow work is independent per scene. Keep the CPU pool bounded so
    # Farneback calculations can overlap without taking all system cores.
    flow_workers = min(4, max(1, (os.cpu_count() or 4) // 8))
    flow_pool = ThreadPoolExecutor(
        max_workers=flow_workers, thread_name_prefix="shot-flow"
    )
    shot_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shot-type-gpu")
    # 光流帧已经在解码后压缩为 320x240 灰度图，仅中间帧保留 BGR；
    # 允许 8 个在途场景可填满 4 个 CPU worker，同时仍有界控制内存。
    max_in_flight_scenes = 8
    shot_batch_size = 4
    pending_jobs = {}
    shot_batch_queue = []
    buffer = {}

    def _store_item(scene_key, item):
        """写回分析结果，并保持原有批量 LMDB 写入行为。"""
        buffer[scene_key] = item
        if len(buffer) >= 16:
            cache.put_results_many(buffer)
            buffer.clear()

    def _finish_incomplete_plan(plan):
        """解码不完整时写回未知结果，并释放该场景的帧引用。"""
        item = plan['item']
        item['shot_type'] = '未知'
        item['camera_movement'] = '未知'
        _store_item(plan['scene_key'], item)
        plan['frames'].clear()
        plan['mid_frame_data'] = None
        plan['remaining'].clear()

    def _finish_job(job_key, job):
        """收集 CPU/GPU 两条分析结果，并在此之后释放场景帧。"""
        plan = job['plan']
        flow_result = None
        shot_result = None

        flow_future = job.get('flow_future')
        if flow_future is not None:
            try:
                flow_result = flow_future.result()
            except Exception:
                flow_result = None

        shot_future = job.get('shot_future')
        if shot_future is not None:
            try:
                batch_results = shot_future.result()
                shot_index = int(job.get('shot_index', 0))
                shot_result = batch_results[shot_index]
            except Exception:
                shot_result = None

        plan['item']['shot_type'] = (
            shot_result.get('景别', '未知')
            if isinstance(shot_result, dict) else '未知'
        )
        plan['item']['camera_movement'] = (
            flow_result.get('镜头运动', '未知')
            if isinstance(flow_result, dict) else '未知'
        )
        video_name = os.path.basename(plan['item'].get('video_path', ''))
        scene_label = (
            f"{video_name} [{plan['item'].get('start_frame', 0)}-"
            f"{plan['item'].get('end_frame', 0)}]"
        )
        flow_frame_count = int(job.get('flow_frame_count', 0) or 0)
        print(
            f"[景别分析-GPU] {scene_label}: "
            f"中间帧={plan['mid_frame']} -> {plan['item']['shot_type']}"
        )
        print(
            f"[光流分析-CPU] {scene_label}: "
            f"输入帧={flow_frame_count}、帧对={max(0, flow_frame_count - 1)} "
            f"-> {plan['item']['camera_movement']}"
        )
        _store_item(plan['scene_key'], plan['item'])

        # 两条分析任务都完成后，才解除对这些 ndarray 的引用。
        plan['frames'].clear()
        plan['mid_frame_data'] = None
        plan['remaining'].clear()
        pending_jobs.pop(job_key, None)

    def _job_complete(job):
        flow_future = job.get('flow_future')
        shot_future = job.get('shot_future')
        return (
            (flow_future is None or flow_future.done())
            and not job.get('shot_pending', False)
            and (shot_future is None or shot_future.done())
        )

    def _flush_shot_batch(force=False):
        """将多个场景的中间帧合并为一次 GPU 景别推理。"""
        if not shot_batch_queue:
            return False
        if not force and len(shot_batch_queue) < shot_batch_size:
            return False

        batch_items = shot_batch_queue[:shot_batch_size]
        del shot_batch_queue[:len(batch_items)]
        frames = [item[1] for item in batch_items]
        try:
            shot_future = shot_pool.submit(shot_clf.analyze_frames_batch, frames)
        except Exception as exc:
            print(f"[景别分析-GPU] 批量任务提交失败: {exc}")
            for job_key, _ in batch_items:
                job = pending_jobs.get(job_key)
                if job is not None:
                    job['shot_pending'] = False
                    job['shot_future'] = None
            return True

        for shot_index, (job_key, _) in enumerate(batch_items):
            job = pending_jobs.get(job_key)
            if job is not None:
                job['shot_future'] = shot_future
                job['shot_index'] = shot_index
                job['shot_pending'] = False
        return True

    def _drain_jobs(block=False, min_completed=0):
        """回收已完成场景；必要时等待至少一个完整场景完成。"""
        drained = 0
        while pending_jobs:
            if block:
                _flush_shot_batch(force=True)
            completed_keys = [
                key for key, job in pending_jobs.items()
                if _job_complete(job)
            ]
            if completed_keys:
                for job_key in completed_keys:
                    job = pending_jobs.get(job_key)
                    if job is not None:
                        _finish_job(job_key, job)
                        drained += 1
                        if min_completed and drained >= min_completed:
                            return drained
                if not block:
                    return drained
                continue

            if not block:
                return drained

            futures = []
            for job in pending_jobs.values():
                if job.get('flow_future') is not None:
                    futures.append(job['flow_future'])
                if job.get('shot_future') is not None:
                    futures.append(job['shot_future'])
            if futures:
                wait(futures, return_when=FIRST_COMPLETED)
            else:
                return drained
        return drained

    def _submit_plan(plan):
        """把一个已收齐帧的场景同时送入 CPU 光流和 GPU 景别队列。"""
        frames = plan['frames']
        mid_frame = plan.get('mid_frame_data')
        flow_frames = [
            frames[frame_number]
            for frame_number in plan['flow_frames']
            if frame_number in frames
        ]

        flow_future = None
        try:
            if len(flow_frames) >= 2:
                flow_future = flow_pool.submit(
                    of_analyzer.analyze_frames,
                    flow_frames,
                )
            if mid_frame is not None:
                shot_batch_queue.append((id(plan), mid_frame))
        except Exception as exc:
            if flow_future is not None:
                flow_future.cancel()
            print(f"[景别/光流任务] 场景任务提交失败: {exc}")
            _finish_incomplete_plan(plan)
            # 已经写回未知结果，调用方将其视为已处理，避免重复写入。
            return True

        # 先登记场景任务，再触发 GPU 批量提交。否则当本场景恰好填满
        # 一个批次时，批量 future 无法回挂到当前场景。
        pending_jobs[id(plan)] = {
            'plan': plan,
            'flow_future': flow_future,
            'shot_future': None,
            'shot_pending': mid_frame is not None,
            'flow_frame_count': len(flow_frames),
        }
        _flush_shot_batch(force=False)
        return True

    def _make_plan(scene_key, item):
        """建立一个场景的帧使用表，不改变原有采样公式。"""
        start_frame = int(item.get('start_frame', 0) or 0)
        end_frame = int(item.get('end_frame', 0) or 0)
        mid_frame = (start_frame + end_frame) // 2
        flow_frame_numbers = []

        if start_frame < end_frame:
            scene_duration = end_frame - start_frame
            flow_sample_count = min(30, scene_duration)
            # 保持原逻辑：中点附近连续帧，避免均匀采样导致帧间位移过大。
            first = max(start_frame, mid_frame - flow_sample_count // 2)
            flow_frame_numbers = [min(end_frame, first + i)
                                  for i in range(flow_sample_count)]

        required = set(flow_frame_numbers)
        required.add(mid_frame)
        return {
            'scene_key': scene_key,
            'item': item,
            'mid_frame': mid_frame,
            'flow_frames': flow_frame_numbers,
            'frames': {},
            'mid_frame_data': None,
            'remaining': required,
        }

    try:
        for batch in result_view.iter_batches(batch_size=128):
            video_groups = defaultdict(list)
            for scene_key, item in batch:
                video_path = item.get('video_path', '')
                if not video_path or not os.path.exists(video_path):
                    item['shot_type'] = '未知'
                    item['camera_movement'] = '未知'
                    _store_item(scene_key, item)
                    continue
                video_groups[video_path].append(_make_plan(scene_key, item))

            for video_path, plans in video_groups.items():
                plans.sort(key=lambda plan: min(plan['remaining']))
                submitted_indices = set()
                decoded_frame_count = 0
                video_decode_start = time.perf_counter()
                try:
                    video_name = os.path.basename(video_path)
                    middle_frame_numbers = {plan['mid_frame'] for plan in plans}
                    flow_frame_requests = sum(
                        len(plan['flow_frames']) for plan in plans
                    )
                    requested_frame_count = sum(
                        len(plan['remaining']) for plan in plans
                    )
                    print(
                        f"[帧提取-共享-{memory_extractor.backend}] {video_name}: "
                        f"场景={len(plans)}，"
                        f"景别中间帧={len(middle_frame_numbers)}个，"
                        f"光流={flow_frame_requests}个场景帧请求，"
                        f"按场景连续批次读取={requested_frame_count}帧"
                    )
                    for plan_index, plan in enumerate(plans):
                        # 每个场景的光流帧在中点附近连续，单独读取可避免
                        # 把两个视频中远距离场景混成稀疏随机 get_batch。
                        scene_frame_numbers = sorted(plan['remaining'])
                        for frame_number, frame_bgr in memory_extractor.iter_memory_frames(
                                video_path, scene_frame_numbers,
                                release_reader=False):
                            decoded_frame_count += 1
                            if frame_number == plan['mid_frame']:
                                plan['mid_frame_data'] = frame_bgr
                            if frame_number in plan['flow_frames']:
                                plan['frames'][frame_number] = of_analyzer.prepare_frame(
                                    frame_bgr
                                )
                            plan['remaining'].discard(frame_number)

                        if not plan['remaining']:
                            if len(pending_jobs) >= max_in_flight_scenes:
                                _drain_jobs(block=True, min_completed=1)
                            if _submit_plan(plan):
                                submitted_indices.add(plan_index)
                        _drain_jobs(block=False)
                    memory_extractor.release_video_reader(video_path)
                except Exception as exc:
                    print(f"[帧提取-共享-{memory_extractor.backend}] {video_name}: 抽帧失败: {exc}")
                    memory_extractor.release_video_reader(video_path)

                video_decode_elapsed = time.perf_counter() - video_decode_start
                print(
                    f"[帧提取-共享-{memory_extractor.backend}] {video_name}: "
                    f"实际交付={decoded_frame_count}/{requested_frame_count}帧，"
                    f"耗时={video_decode_elapsed:.2f}s"
                )

                # 解码异常或帧越界时，仍为场景写回未知结果，避免丢失结果记录。
                for plan_index, plan in enumerate(plans):
                    if plan_index not in submitted_indices:
                        _finish_incomplete_plan(plan)

            # 当前结果批次结束前回收全部在途场景，避免帧引用跨批次累积。
            _flush_shot_batch(force=True)
            while pending_jobs:
                _drain_jobs(block=True, min_completed=1)

        if buffer:
            cache.put_results_many(buffer)
    finally:
        # 即使中途发生异常，也要先等待分析任务结束，再卸载 GPU 模型。
        _flush_shot_batch(force=True)
        while pending_jobs:
            _drain_jobs(block=True, min_completed=1)
        if buffer:
            cache.put_results_many(buffer)
            buffer.clear()
        flow_pool.shutdown(wait=True)
        shot_pool.shutdown(wait=True)
        shot_clf.unload()

    print(f"[景别分析-GPU] 完成 {total} 个场景")
    print(f"[光流分析-CPU] 完成 {total} 个场景")
    print(f"[帧提取-共享] 已完成全部分析帧交付\n")
    return result_view


def _strip_false_labels(item: dict, false_cats: set) -> dict:
    """从 item 中删除 Jev-rerank 判定为否的标签，清理所有相关字段。

    Args:
        item: LMDB 中的一条结果
        false_cats: Jev-rerank 判定为 False 的类别名集合, 如 {'动作', '情绪'}

    Returns:
        原地修改后的 item
    """
    if not false_cats:
        return item

    # 1. 删除 _cn / _en
    for cat in false_cats:
        item.pop('%s_cn' % cat, None)
        item.pop('%s_en' % cat, None)

    # 2. 清理 labels dict — value 命中 false_cats 即删
    if isinstance(item.get('labels'), dict):
        item['labels'] = {
            k: v for k, v in item['labels'].items()
            if v not in false_cats
        }

    # 3. 清理 all_labels dict — key 直接对应类别名
    if isinstance(item.get('all_labels'), dict):
        for cat in false_cats:
            item['all_labels'].pop(cat, None)

    # 4. 保险: label_cn / label_en 若命中也清空
    for key in ('label_cn', 'label_en'):
        if item.get(key) and item[key] in false_cats:
            item[key] = ''

    return item


def apply_jev_rerank(result_view, *, verify_labels, aesthetics, min_matches=4, use_chinese=False, settings=None):
    """按窗口处理镜头中间帧，使用同一份视觉特征验证标签并给保留的镜头评分。"""
    from A_coreUtils.aftertreatment.jev_rerank import JevReranker, extracted_frames

    root = PathResolver().project_root
    cache = result_view._cache
    if settings is None:
        with open(root / 'logic_keywords.json', encoding='utf-8') as f:
            settings = json.load(f)['其他参数设置']
    name_template = settings['输出视频命名格式']
    categories = settings.get('必有标签', [])
    templates = settings.get('Jev-rerank验证Prompt', {})
    templates = templates.get('zh' if use_chinese else 'en', {})
    instruction = templates.get('_instruction', '下列描述是否成立？' if use_chinese else 'Does this image match the description?')
    suffix = '_cn' if use_chinese else '_en'
    total_groups = 0
    passed_groups = 0
    removed_groups = 0
    missing_frame_groups = 0
    total_removed = 0
    threshold = 0.5
    initial_count = result_view.count()
    if verify_labels:
        print(f"[Jev-rerank] 开始复核 {initial_count} 条结果，必有标签最低匹配 {min_matches} 项，判断阈值 {threshold:.2f}")
    else:
        print(f"[Jev-rerank] 开始为 {initial_count} 条结果计算美学评分")
    with JevReranker() as jev:
        for batch in result_view.iter_batches(batch_size=1024):
            groups = {}
            for key, item in batch:
                scene = (item.get('video_path'), int(item['start_frame']), int(item['end_frame']))
                groups.setdefault(scene, []).append((key, item))
            targets = [(vp, (sf + ef) // 2) for vp, sf, ef in groups]
            updates = {}
            removed = []
            with extracted_frames(targets, root / 'temp/cache/jev_rerank') as frames:
                for ((vp, _, _), items), target in zip(groups.items(), targets):
                    total_groups += 1
                    scene = items[0][1]
                    scene_label = (
                        f"{os.path.basename(vp or '')} "
                        f"[{scene.get('start_frame', 0)}-{scene.get('end_frame', 0)}]"
                    )
                    labels = {}
                    if verify_labels:
                        for _, item in items:
                            for cat in categories:
                                value = item.get(cat + suffix)
                                if cat not in labels and isinstance(value, str) and value.strip():
                                    labels[cat] = value.strip()
                        if not labels:
                            removed.extend(key for key, _ in items)
                            removed_groups += 1
                            print(f"[Jev-rerank] {scene_label}: 没有可验证的关键词，淘汰")
                            continue
                        if not vp or not os.path.isfile(vp):
                            removed.extend(key for key, _ in items)
                            removed_groups += 1
                            missing_frame_groups += 1
                            print(f"[Jev-rerank] {scene_label}: 源视频不存在，无法复核，淘汰")
                            continue
                    path = frames.get(target)
                    if not path:
                        missing_frame_groups += 1
                        if verify_labels:
                            removed.extend(key for key, _ in items)
                            removed_groups += 1
                            print(f"[Jev-rerank] {scene_label}: 中间帧 {target[1]} 未能抽取，无法复核，淘汰")
                            continue
                        if aesthetics:
                            for key, item in items:
                                item['aesthetic_score'] = None
                                updates[key] = item
                            print(f"[Jev-rerank] {scene_label}: 中间帧 {target[1]} 未能抽取，美学评分记为未知")
                        continue

                    questions = {
                        cat: instruction + ' ' + templates.get(cat, '{cat}: {val}').format(cat=cat, val=val)
                        for cat, val in labels.items()
                    }
                    result = jev.evaluate(
                        path, questions, use_chinese=use_chinese, aesthetics=aesthetics,
                        min_matches=min_matches if verify_labels else 0,
                    )
                    false_cats = set()
                    if verify_labels:
                        decisions = []
                        for cat, val in labels.items():
                            score = result['scores'].get(cat)
                            if score is None:
                                decision = '无有效判定'
                                false_cats.add(cat)
                            else:
                                if score < threshold:
                                    false_cats.add(cat)
                                decision = f"{'否' if score < threshold else '是'} ({score:.3f})"
                            decisions.append(f"{cat}={val} {decision}")
                        label_log = ' | '.join(decisions)
                        matched_count = len(labels) - len(false_cats)
                        if matched_count < min_matches:
                            removed.extend(key for key, _ in items)
                            removed_groups += 1
                            print(
                                f"[Jev-rerank] {scene_label}: {label_log} "
                                f"| 匹配 {matched_count}/{min_matches}，淘汰"
                            )
                            continue
                        passed_groups += 1
                        print(
                            f"[Jev-rerank] {scene_label}: {label_log} "
                            f"| 匹配 {matched_count}/{min_matches}，保留"
                        )
                    for key, item in items:
                        if false_cats:
                            _strip_false_labels(item, false_cats)
                            item['result_name'] = _generate_merged_result_name(item, name_template)
                        if aesthetics:
                            item['aesthetic_score'] = result.get('aesthetic_score')
                        if false_cats or aesthetics:
                            updates[key] = item
            cache.put_results_many(updates)
            if removed:
                deleted = cache.delete_results_many(removed)
                if deleted != len(removed):
                    raise RuntimeError(f"[Jev-rerank] 应删除 {len(removed)} 条结果，LMDB 实际删除 {deleted} 条")
                total_removed += deleted
    if verify_labels:
        print(
            f"[Jev-rerank] 完成：复核场景 {total_groups}，通过 {passed_groups}，"
            f"淘汰 {removed_groups} 组 / {total_removed} 条结果，"
            f"缺帧或源视频异常 {missing_frame_groups} 组；"
            f"结果 {initial_count} -> {result_view.count()}"
        )
    else:
        print(f"[Jev-rerank] 美学评分完成：处理 {total_groups} 个场景，缺帧 {missing_frame_groups} 个")
    return result_view


def cleanup_temp_after_export(resolver: PathResolver):
    temp_dir = str(resolver.project_root / 'temp')
    if os.path.exists(temp_dir):
        print(f"  [清理] 清空 temp 文件夹: {temp_dir}")
        from A_coreUtils.video_processing.video_utils import cleanup_temp_folder
        cleanup_temp_folder(temp_dir)


def run_interactive_search(
    index_directory: str = None,
    output_directory: str = None,
    use_fp16: bool = True,
    candidate_batch_size: Optional[int] = None,
    video_output_directory: str = None,
    video_copy_mode: bool = False,
    start_frame_offset: int = None,
    end_frame_offset: int = None,
    post_export_hook: Optional[Callable] = None,
    prompt_search_batch_size: Optional[int] = DEFAULT_PROMPT_SEARCH_BATCH_SIZE,
    feature_fp16: Optional[bool] = None,
    lance_batch_size: Optional[int] = None,
    use_diskcache: bool = True,
    diskcache_dir: str = None,
    debug_similarity: bool = False,
    search_mode: int = 0,
    top_k: Optional[int] = None,
    prompt_cache_batch_size: Optional[int] = DEFAULT_PROMPT_CACHE_BATCH_SIZE,
    use_chinese: bool = False,
    lance_load_workers: Optional[int] = DEFAULT_LANCE_LOAD_WORKERS,
    lmdb_write_batch_size: Optional[int] = DEFAULT_LMDB_WRITE_BATCH_SIZE,
    vector_dedup_threshold: float = None,
    adjacent_merge_frames: int = None,
    use_shot_analysis: bool = False,
    use_jev_rerank: bool = False,
    jev_rerank_min_matches: int = 4,
    search_threshold: float = None,
    lance_indexes: Optional[list] = None,
    use_aesthetic_score: bool = False,
    *,
    similarity_thresholds: Optional[Dict] = None,
) -> Dict:
    resolver = PathResolver()
    if index_directory is None:
        index_directory = str(resolver.project_root / 'indexes')
    elif not os.path.isabs(index_directory):
        index_directory = str(resolver.resolve(index_directory))
    if output_directory is None:
        output_directory = str(resolver.project_root / 'output')
    elif not os.path.isabs(output_directory):
        output_directory = str(resolver.resolve(output_directory))
    if feature_fp16 is None:
        feature_fp16 = use_fp16
    prompt_search_batch_size = normalize_optional_positive_int(
        prompt_search_batch_size,
        field_name='prompt_search_batch_size',
    )
    prompt_cache_batch_size = normalize_optional_positive_int(
        prompt_cache_batch_size,
        field_name='prompt_cache_batch_size',
    )
    lance_load_workers = normalize_optional_positive_int(
        lance_load_workers,
        field_name='lance_load_workers',
    )
    lmdb_write_batch_size = normalize_optional_positive_int(
        lmdb_write_batch_size,
        field_name='lmdb_write_batch_size',
    )
    candidate_batch_size = normalize_optional_positive_int(
        candidate_batch_size,
        field_name='candidate_batch_size',
    )
    lance_batch_size = normalize_optional_positive_int(
        lance_batch_size,
        field_name='lance_batch_size',
    )
    top_k = normalize_top_k(top_k)
    if not os.path.exists(index_directory):
        return {
            'success': False,
            'error': f'索引目录不存在: {index_directory}',
            'result_count': 0,
        }
    os.makedirs(output_directory, exist_ok=True)
    index_paths = []
    for filename in os.listdir(index_directory):
        path = os.path.join(index_directory, filename)
        if os.path.isdir(path) and filename.endswith('.lance'):
            index_paths.append(path)
    if not index_paths:
        return {
            'success': False,
            'error': '索引目录中没有 .lance 索引',
            'result_count': 0,
        }
    # 如果指定了 lance_indexes，只保留选中的索引
    if lance_indexes is not None:
        index_set = set(lance_indexes)
        index_paths = [p for p in index_paths if os.path.basename(p) in index_set]
        if not index_paths:
            return {
                'success': False,
                'error': '没有匹配的选中索引',
                'result_count': 0,
            }
    model_names = set()
    for index_path in index_paths:
        model_name = extract_model_name_from_index(index_path)
        model_names.add(model_name if model_name else 'unknown')
    if len(model_names) > 1:
        return {
            'success': False,
            'error': f'Multiple model names detected across Lance indexes: {len(model_names)}',
            'result_count': 0,
        }
    detected_model_name = list(model_names)[0]
    if detected_model_name == 'unknown':
        return {
            'success': False,
            'error': '无法从索引文件名中提取模型名',
            'result_count': 0,
        }
    detected_model_type = detect_model_type_from_name(detected_model_name)
    searcher = AutoSceneSearcher(use_fp16=use_fp16, use_chinese=use_chinese)
    generator = searcher.prompt_generator
    name_template = generator.name_template
    all_categories = list(generator._simple_categories)
    is_valid, err = validate_prompt_video_name_format(name_template, all_categories)
    if not is_valid:
        raise ValueError(f"[自动提纯] 输出视频命名格式无效: {err}")
    final_threshold = (SimilarityThresholdConfig.get_threshold(detected_model_type, similarity_thresholds)
                       if search_threshold is None else float(search_threshold))
    if not math.isfinite(final_threshold):
        raise ValueError("搜索阈值必须是有限数值")
    print(f"[自动提纯] 搜索模型 {detected_model_name}，召回阈值 {final_threshold:g}")
    start_time = time.time()
    result_view = None
    merged_count = 0
    export_succeeded = False
    export_stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'failed_files': [],
    }
    resolved_video_output_dir = resolve_video_output_directory(
        resolver=resolver,
        output_directory=output_directory,
        video_output_directory=video_output_directory,
    )
    try:
        result_view = searcher.run_batch_search_optimized(
            index_paths=index_paths,
            similarity_threshold=final_threshold,
            prompt_search_batch_size=prompt_search_batch_size,
            feature_fp16=feature_fp16,
            lance_batch_size=lance_batch_size,
            use_diskcache=use_diskcache,
            diskcache_dir=diskcache_dir,
            search_mode=search_mode,
            top_k=top_k,
            prompt_cache_batch_size=prompt_cache_batch_size,
            lance_load_workers=lance_load_workers,
            lmdb_write_batch_size=lmdb_write_batch_size,
            vector_dedup_threshold=vector_dedup_threshold,
            candidate_batch_size=candidate_batch_size,
        )
        if result_view is None:
            return {
                'success': False,
                'error': 'Search did not return a result view',
                'result_count': 0,
            }
        result_count = int(result_view.count())
        print(f"\n📊 搜索完成: {result_count:,} 个场景, 耗时 {time.time() - start_time:.2f}s")
        if adjacent_merge_frames is not None and adjacent_merge_frames >= 0 and result_count > 0:
            merged_view = merge_adjacent_scenes(
                result_source=result_view,
                adjacent_merge_frames=adjacent_merge_frames,
                video_name_format=name_template,
            )
            if merged_view is not result_view and hasattr(result_view, '_cache') and hasattr(result_view._cache, 'destroy'):
                result_view._cache.destroy()
            elif merged_view is not result_view and hasattr(result_view, 'close'):
                result_view.close()
            result_view = merged_view
        merged_count = int(result_view.count())
        if merged_count > 0:
            if use_jev_rerank or use_aesthetic_score:
                result_view = apply_jev_rerank(result_view, verify_labels=use_jev_rerank,
                                               aesthetics=use_aesthetic_score,
                                               min_matches=jev_rerank_min_matches, use_chinese=use_chinese,
                                               settings=generator._full_data['其他参数设置'])
                merged_count = int(result_view.count())
            if use_shot_analysis:
                result_view = attach_shot_analysis(result_view)
                merged_count = int(result_view.count())
            print(f"\n[Export] exporting {merged_count} scenes...")
            export_stats, resolved_video_output_dir = export_video_matches(
                result_source=result_view,
                resolver=resolver,
                output_directory=output_directory,
                video_output_directory=video_output_directory,
                video_copy_mode=video_copy_mode,
                start_frame_offset=start_frame_offset,
                end_frame_offset=end_frame_offset,
                debug_similarity=debug_similarity,
                post_export_hook=post_export_hook,
                use_aesthetic_score=use_aesthetic_score,
            )
            export_succeeded = True
        return {
            'success': True,
            'export_succeeded': export_succeeded,
            'result_count': result_count,
            'merged_result_count': int(result_view.count()) if result_view is not None else 0,
            'export_stats': export_stats,
            'video_output_directory': resolved_video_output_dir,
            'elapsed_seconds': time.time() - start_time,
            'validation_warnings': generator._warnings if hasattr(generator, '_warnings') else [],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'result_count': 0,
            'elapsed_seconds': time.time() - start_time,
        }
    finally:
        if result_view is not None:
            if hasattr(result_view, '_cache') and hasattr(result_view._cache, 'destroy'):
                result_view._cache.destroy()
            elif hasattr(result_view, 'close'):
                result_view.close()
        if export_succeeded:
            pass  # cleanup 由调用方 pipeline_app.py 在 shot_cleanup 后统一执行
# ============================================================
#  测试入口
# ============================================================
if __name__ == "__main__":
    generator = PromptGenerator()
    from collections import Counter
    counts = Counter(p['category'] for p in generator.iterate_all_labels_flat())
    print(f"统一模式: {sum(counts.values())} prompts, {len(counts)} 模板类别")
    for cat,count in counts.items():
        print(f"  {cat}: {count} prompts")
    print(f"命名格式: {generator.name_template}")
    print(f"必有标签: {generator._required_categories}")
