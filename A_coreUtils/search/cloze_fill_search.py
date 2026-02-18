# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
# cloze_fill_search.py
# 选词填空模式搜索
# v1.0: 初始版本
#       - 从 logic_keywords.json 的 "选词填空规则" 动态读取标签和模板
#       - 支持中英文标签切换
#       - 复用 Prompt 模式的核心搜索功能
#
# 功能说明：
# 1. 从 "选词填空规则" 读取子类标签（如 生物、物体、情绪、动作）
# 2. 从 "选词填空规则.分配规则" 读取模板（中英文对应）
# 3. 解析模板中的占位符（如 {生物}、{情绪}）
# 4. 根据占位符从对应子类获取标签，生成所有组合
# 5. 支持中英文模式切换：
#    - 英文模式：使用英文模板 + 英文标签值
#    - 中文模式：使用中文模板 + 中文标签键名

import os
import sys
import json
import re
import itertools
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Generator, Any

# ============================================================
#  路径设置
# ============================================================
_current_file = os.path.abspath(__file__)
_search_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_search_dir)
_project_root_dir = os.path.dirname(_a_core_utils_dir)
if _project_root_dir not in sys.path:
    sys.path.insert(0, _project_root_dir)

from path_resolver import PathResolver

DEFAULT_CLOZE_PROMPT_SEARCH_BATCH_SIZE = 1024
DEFAULT_CLOZE_RERANK_TOP_K = 7
DEFAULT_CLOZE_RERANK_BATCH_SIZE = 7
DEFAULT_CLOZE_PROMPT_CACHE_BATCH_SIZE = 512
DEFAULT_CLOZE_LANCE_LOAD_WORKERS = 4
DEFAULT_CLOZE_LMDB_WRITE_BATCH_SIZE = 1000


class ClozeKeywordManager:
    """
    选词填空关键词管理器
    
    从 logic_keywords.json 的 "选词填空规则" 动态读取：
    - 子类标签（如 生物、物体、情绪、动作）
    - 分配规则模板（中英文对应）
    
    支持动态增减子类名和标签，无需修改代码。
    """
    
    # 选词填空规则中的保留字段
    RESERVED_KEYS = {"分配规则", "_说明"}
    
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
        self._cloze_data: Dict = {}  # 选词填空规则数据
        self._subcategories: List[str] = []  # 子类名列表
        self._templates: Dict[str, str] = {}  # 模板 {中文: 英文}
        
        self._load_keywords()
    
    def _load_keywords(self):
        """
        加载选词填空规则
        
        从 logic_keywords.json 读取 "选词填空规则" 字段
        """
        with open(self.keywords_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        
        # 获取选词填空规则
        self._cloze_data = full_data.get("选词填空规则", {})
        
        if not self._cloze_data:
            print("[警告] logic_keywords.json 中没有找到 '选词填空规则' 字段")
            return
        
        # 提取子类名（排除保留字段）
        self._subcategories = [
            k for k in self._cloze_data.keys() 
            if k not in self.RESERVED_KEYS and not k.startswith('_')
        ]
        
        # 提取模板（从分配规则）
        allocation_rules = self._cloze_data.get("分配规则", {})
        self._templates = {
            k: v for k, v in allocation_rules.items() 
            if not k.startswith('_')
        }
        
        print(f"[选词填空] 加载 {len(self._subcategories)} 个子类: {self._subcategories}")
        print(f"[选词填空] 加载 {len(self._templates)} 个模板")
    
    @property
    def subcategories(self) -> List[str]:
        """获取所有子类名称"""
        return self._subcategories
    
    @property
    def templates(self) -> Dict[str, str]:
        """获取所有模板 {中文模板: 英文模板}"""
        return self._templates
    
    def get_subcategory_labels(self, subcategory: str) -> Dict[str, str]:
        """
        获取指定子类的所有标签
        
        Args:
            subcategory: 子类名称（如 "生物"、"物体"）
        
        Returns:
            {中文标签: 英文标签} 字典
        """
        return self._cloze_data.get(subcategory, {})
    
    def get_all_labels_flat(self) -> List[Tuple[str, str, str]]:
        """
        获取所有标签（扁平化）
        
        Returns:
            [(子类名, 中文标签, 英文标签), ...]
        """
        result = []
        for subcategory in self._subcategories:
            labels = self.get_subcategory_labels(subcategory)
            for cn, en in labels.items():
                result.append((subcategory, cn, en))
        return result
    
    def parse_template_placeholders(self, template: str) -> List[str]:
        """
        解析模板中的占位符
        
        Args:
            template: 模板字符串（如 "{生物}孤独的呆在房间里"）
        
        Returns:
            占位符列表（如 ["生物", "情绪", "物体"]）
        """
        return re.findall(r'\{(\w+)\}', template)
    
    def reload_keywords(self):
        """重新加载关键词文件（支持热更新）"""
        self._load_keywords()


def validate_cloze_video_name_format(video_name_format: str, subcategories: List[str]) -> Tuple[bool, str]:
    """
    验证选词填空模式的 video_name_format 是否合法
    
    检查格式字符串中的占位符是否都是有效的子类名称或系统占位符。
    
    Args:
        video_name_format: 视频名称格式字符串
        subcategories: 有效的子类名称列表
    
    Returns:
        (是否合法, 错误信息)
    """
    import re
    
    # 系统占位符
    system_placeholders = {'起始帧', '视频解析名', '扩展名', 'prompt_cn', 'prompt_en'}
    
    # 提取所有占位符
    placeholders = re.findall(r'\{(\w+)\}', video_name_format)
    
    # 检查每个占位符
    invalid_placeholders = []
    for ph in placeholders:
        if ph not in system_placeholders and ph not in subcategories:
            invalid_placeholders.append(ph)
    
    if invalid_placeholders:
        valid_list = list(subcategories) + list(system_placeholders)
        return False, f"video_name_format 中存在无效占位符: {invalid_placeholders}，有效占位符: {valid_list}"
    
    return True, ""


def generate_default_cloze_video_name_format(subcategories: List[str], prefix: str = "选词填空") -> str:
    """
    动态生成选词填空模式的默认 video_name_format
    
    Args:
        subcategories: 子类名称列表
        prefix: 前缀字符串
    
    Returns:
        默认的 video_name_format 字符串
    """
    format_parts = [prefix] if prefix else []
    format_parts.extend([f"{{{subcat}}}" for subcat in subcategories])
    format_parts.extend(["{起始帧}", "{视频解析名}"])
    return "_".join(format_parts)


class ClozePromptGenerator:
    """
    选词填空 Prompt 生成器
    
    根据模板和子类标签生成所有可能的 prompt 组合。
    支持中英文模式切换。
    """
    
    def __init__(self, keywords_path: str = None, use_chinese: bool = False):
        """
        初始化 Prompt 生成器
        
        Args:
            keywords_path: logic_keywords.json 路径
            use_chinese: 是否使用中文模式
                - False（默认）: 使用英文模板 + 英文标签值
                - True: 使用中文模板 + 中文标签键名
        """
        self.keyword_manager = ClozeKeywordManager(keywords_path)
        self.use_chinese = use_chinese
    
    def iterate_template_combinations(self, 
                                       template_cn: str, 
                                       template_en: str,
                                       ) -> Generator[Dict, None, None]:
        """
        遍历单个模板的所有标签组合
        
        Args:
            template_cn: 中文模板
            template_en: 英文模板
        
        Yields:
            包含组合信息的字典:
            - prompt: 生成的 prompt（根据 use_chinese 选择中/英文）
            - prompt_cn: 中文 prompt
            - prompt_en: 英文 prompt
            - template_cn: 原始中文模板
            - template_en: 原始英文模板
            - labels: {子类名: (中文标签, 英文标签)}
        """
        # 解析模板中的占位符
        placeholders = self.keyword_manager.parse_template_placeholders(template_cn)
        
        if not placeholders:
            # 没有占位符，直接返回模板本身
            yield {
                "prompt": template_cn if self.use_chinese else template_en,
                "prompt_cn": template_cn,
                "prompt_en": template_en,
                "template_cn": template_cn,
                "template_en": template_en,
                "labels": {}
            }
            return
        
        # 获取每个占位符对应的标签列表
        placeholder_labels = []
        for ph in placeholders:
            labels = self.keyword_manager.get_subcategory_labels(ph)
            if labels:
                # [(中文, 英文), ...]
                label_list = [(cn, en) for cn, en in labels.items()]
                placeholder_labels.append((ph, label_list))
            else:
                print(f"[警告] 占位符 '{ph}' 在选词填空规则中没有对应的子类")
                placeholder_labels.append((ph, [("", "")]))
        
        # 生成笛卡尔积
        label_values_list = [labels for _, labels in placeholder_labels]
        
        for combo in itertools.product(*label_values_list):
            # 构建替换字典
            cn_values = {}
            en_values = {}
            labels_dict = {}
            
            for i, (ph, _) in enumerate(placeholder_labels):
                cn_val, en_val = combo[i]
                cn_values[ph] = cn_val
                en_values[ph] = en_val
                labels_dict[ph] = (cn_val, en_val)
            
            # 生成中英文 prompt
            prompt_cn = template_cn
            prompt_en = template_en
            
            for ph in placeholders:
                prompt_cn = prompt_cn.replace(f"{{{ph}}}", cn_values.get(ph, ""))
                prompt_en = prompt_en.replace(f"{{{ph}}}", en_values.get(ph, ""))
            
            yield {
                "prompt": prompt_cn if self.use_chinese else prompt_en,
                "prompt_cn": prompt_cn,
                "prompt_en": prompt_en,
                "template_cn": template_cn,
                "template_en": template_en,
                "labels": labels_dict
            }
            
    
    def iterate_all_combinations(self) -> Generator[Dict, None, None]:
        """
        遍历所有模板的所有标签组合
        
        Args:
        
        Yields:
            包含组合信息的字典（同 iterate_template_combinations）
        """
        for template_cn, template_en in self.keyword_manager.templates.items():
            for combo in self.iterate_template_combinations(template_cn, template_en):
                yield combo
    
    def count_total_combinations(self) -> int:
        """
        计算总组合数
        
        Returns:
            所有模板的组合数总和
        """
        total = 0
        
        for template_cn in self.keyword_manager.templates.keys():
            placeholders = self.keyword_manager.parse_template_placeholders(template_cn)
            
            if not placeholders:
                total += 1
                continue
            
            combo_count = 1
            for ph in placeholders:
                labels = self.keyword_manager.get_subcategory_labels(ph)
                if labels:
                    combo_count *= len(labels)
            
            total += combo_count
        
        return total
    
    def iterate_prompts_only(self) -> Generator[str, None, None]:
        """
        轻量版遍历：只生成 prompt 字符串，不创建完整字典
        
        用于保存缓存时避免内存爆炸。
        
        Args:
        
        Yields:
            prompt 字符串
        """
        for template_cn, template_en in self.keyword_manager.templates.items():
            placeholders = self.keyword_manager.parse_template_placeholders(template_cn)
            
            if not placeholders:
                yield template_cn if self.use_chinese else template_en
                continue
            
            label_lists = []
            for ph in placeholders:
                labels = self.keyword_manager.get_subcategory_labels(ph)
                if labels:
                    label_lists.append([(cn, en) for cn, en in labels.items()])
                else:
                    label_lists.append([("", "")])
            
            for combo in itertools.product(*label_lists):
                prompt_cn = template_cn
                prompt_en = template_en
                
                for i, ph in enumerate(placeholders):
                    cn_label, en_label = combo[i]
                    prompt_cn = prompt_cn.replace(f"{{{ph}}}", cn_label)
                    prompt_en = prompt_en.replace(f"{{{ph}}}", en_label)
                
                yield prompt_cn if self.use_chinese else prompt_en
                
    
    def compute_config_hash(self) -> str:
        """
        计算当前配置的哈希值（用于缓存验证）
        
        哈希内容包括：
        - 所有模板
        - 所有子类标签
        - use_chinese 设置
        """
        hash_content = {
            "templates": self.keyword_manager.templates,
            "subcategories": {},
            "use_chinese": self.use_chinese
        }
        
        for subcat in self.keyword_manager.subcategories:
            hash_content["subcategories"][subcat] = self.keyword_manager.get_subcategory_labels(subcat)
        
        content_str = json.dumps(hash_content, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()[:8]


class ClozeVectorCache:
    """
    基于 Lance 的选词填空提示词向量缓存。

    缓存文件:
      {cache_dir}/cloze_cache_{model_name}_{lang}.lance

    说明:
    - 仅支持 Lance，不支持旧 memmap/pkl 缓存格式。
    - 对外接口保持兼容: cache_exists/generate_cache/load_cache/load_cache_batched。
    """

    def __init__(
        self,
        processor=None,
        keywords_path: str = None,
        cache_dir: str = None,
        model_name: str = None,
        use_chinese: bool = False,
    ):
        self.processor = processor
        self.use_chinese = use_chinese

        resolver = PathResolver()
        if keywords_path is None:
            keywords_path = str(resolver.project_root / "logic_keywords.json")
        self.keywords_path = keywords_path

        if cache_dir is None:
            cache_dir = str(resolver.project_root / "templates" / "prompt_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.model_name = model_name or "unknown"
        self.prompt_generator = ClozePromptGenerator(self.keywords_path, use_chinese)
        self._config_hash = self.prompt_generator.compute_config_hash()

    def _get_safe_model_name(self) -> str:
        return self.model_name.replace("/", "_").replace("\\", "_").replace(":", "_")

    def get_cache_path(self) -> str:
        safe_model_name = self._get_safe_model_name()
        lang_suffix = "cn" if self.use_chinese else "en"
        base_name = f"cloze_cache_{safe_model_name}_{lang_suffix}"
        return os.path.join(self.cache_dir, base_name)

    def get_lance_path(self) -> str:
        return self.get_cache_path() + ".lance"

    def cache_exists(self) -> bool:
        lance_path = self.get_lance_path()
        if not os.path.exists(lance_path):
            return False
        try:
            import lance

            ds = lance.dataset(lance_path)
            md = ds.metadata or {}
            if md.get("format") != "cloze_vector_cache_lance_v1":
                return False
            if md.get("config_hash") != self._config_hash:
                return False

            expected_total = int(md.get("total_prompts", "0") or 0)
            actual_total = int(ds.count_rows())
            if expected_total and expected_total != actual_total:
                return False
            return True
        except Exception:
            return False

    def generate_cache(self, batch_size: int = 512) -> bool:
        if self.processor is None:
            raise RuntimeError("生成缓存需要 processor 实例")

        import gc
        import pickle
        import shutil
        import time
        from datetime import datetime

        import torch
        import lance
        import pyarrow as pa

        lance_path = self.get_lance_path()
        if self.cache_exists():
            print(f"[选词填空缓存] 缓存已存在: {lance_path}")
            return True

        if os.path.exists(lance_path):
            shutil.rmtree(lance_path, ignore_errors=True)

        total_prompts = int(self.prompt_generator.count_total_combinations())
        if total_prompts <= 0:
            raise RuntimeError("total_prompts == 0, check logic_keywords.json / templates")

        # sample vector dim
        first_combo = next(self.prompt_generator.iterate_all_combinations())
        sample_vec = self.processor.encode_text([first_combo.get("prompt", "")])
        vector_dim = int(sample_vec.shape[1])

        schema = pa.schema(
            [
                pa.field("prompt", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), vector_dim)),
                pa.field("metadata", pa.binary()),  # pickle bytes
            ]
        )

        def _vectors_to_fixed_size_list(vectors: np.ndarray) -> pa.FixedSizeListArray:
            vec = np.asarray(vectors, dtype=np.float32, order="C")
            values = pa.array(vec.reshape(-1), type=pa.float32())
            return pa.FixedSizeListArray.from_arrays(values, vector_dim)

        def record_batches():
            current_idx = 0
            batch_prompts: List[str] = []
            batch_meta: List[Dict] = []

            start_time = time.time()
            last_log = 0

            for combo in self.prompt_generator.iterate_all_combinations():
                prompt_text = combo.get("prompt", "")
                batch_prompts.append(prompt_text)
                batch_meta.append(combo)

                if len(batch_prompts) < batch_size:
                    continue

                vectors = self.processor.encode_text(batch_prompts)
                meta_bytes = [pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL) for m in batch_meta]

                yield pa.record_batch(
                    [
                        pa.array(batch_prompts, type=pa.string()),
                        _vectors_to_fixed_size_list(vectors),
                        pa.array(meta_bytes, type=pa.binary()),
                    ],
                    names=["prompt", "vector", "metadata"],
                )

                current_idx += len(batch_prompts)
                batch_prompts.clear()
                batch_meta.clear()

                if current_idx - last_log >= max(1, batch_size * 20):
                    elapsed = time.time() - start_time
                    speed = current_idx / elapsed if elapsed > 0 else 0
                    print(f"  [{current_idx:,}/{total_prompts:,}] {speed:.1f} prompts/s")
                    last_log = current_idx

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            if batch_prompts:
                vectors = self.processor.encode_text(batch_prompts)
                meta_bytes = [pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL) for m in batch_meta]
                yield pa.record_batch(
                    [
                        pa.array(batch_prompts, type=pa.string()),
                        _vectors_to_fixed_size_list(vectors),
                        pa.array(meta_bytes, type=pa.binary()),
                    ],
                    names=["prompt", "vector", "metadata"],
                )
                current_idx += len(batch_prompts)

            if current_idx != total_prompts:
                raise RuntimeError(f"数量不一致: 预期 {total_prompts}, 实际 {current_idx}")

        print(f"[选词填空缓存] 写入 Lance 数据集: {lance_path}")
        ds = lance.write_dataset(record_batches(), lance_path, schema=schema, mode="create")
        ds.update_metadata(
            {
                "format": "cloze_vector_cache_lance_v1",
                "model_name": str(self.model_name),
                "config_hash": str(self._config_hash),
                "total_prompts": str(total_prompts),
                "vector_dim": str(vector_dim),
                "keywords_path": str(self.keywords_path),
                "use_chinese": "1" if self.use_chinese else "0",
                "created_at": datetime.now().isoformat(),
                "normalized": "1",
            }
        )

        print(f"[选词填空缓存] 写入完成: {int(ds.count_rows()):,} 行")
        return True

    def load_cache(self) -> Tuple[Any, List[Dict]]:
        import pickle
        import lance

        lance_path = self.get_lance_path()
        if not self.cache_exists():
            raise RuntimeError("缓存不存在或已过期，请先调用 generate_cache()")

        ds = lance.dataset(lance_path)
        md = ds.metadata or {}
        vector_dim = int(md.get("vector_dim", "0") or 0)
        if vector_dim <= 0:
            try:
                vector_dim = int(getattr(ds.schema.field("vector").type, "list_size", 0) or 0)
            except Exception:
                vector_dim = 0
        if vector_dim <= 0:
            raise RuntimeError("无法解析 vector_dim，请删除缓存并重新生成")

        table = ds.to_table(columns=["vector", "metadata"])
        vec_col = table.column("vector")
        vec_chunks = []
        for chunk in vec_col.chunks:
            flat = chunk.values.to_numpy(zero_copy_only=False)
            vec_chunks.append(flat.reshape(len(chunk), vector_dim))
        vectors = np.vstack(vec_chunks) if vec_chunks else np.zeros((0, vector_dim), dtype=np.float32)

        meta_col = table.column("metadata")
        all_meta: List[Dict] = []
        for chunk in meta_col.chunks:
            for b in chunk.to_pylist():
                try:
                    all_meta.append(pickle.loads(b) if b is not None else {})
                except Exception:
                    all_meta.append({})

        return vectors, all_meta

    def load_cache_batched(self, batch_size: int = 10000, use_fp16: bool = True):
        if not self.cache_exists():
            raise RuntimeError("缓存不存在或已过期，请先调用 generate_cache()")
        return ClozeVectorBatchIterator(lance_path=self.get_lance_path(), batch_size=int(batch_size), use_fp16=use_fp16)

    def load_meta_lookup(self) -> "ClozeMetaLookup":
        lance_path = self.get_lance_path()
        if not os.path.exists(lance_path):
            raise FileNotFoundError(f"Lance 数据集不存在: {lance_path}")
        return ClozeMetaLookup(lance_path=lance_path)

    def get_cache_info(self) -> Dict:
        import lance

        lance_path = self.get_lance_path()
        if not os.path.exists(lance_path):
            raise FileNotFoundError(f"Lance 数据集不存在: {lance_path}")
        ds = lance.dataset(lance_path)
        md = ds.metadata or {}
        total_prompts = int(ds.count_rows())

        size_mb = 0.0
        for root, _, files in os.walk(lance_path):
            for fn in files:
                fp = os.path.join(root, fn)
                if os.path.isfile(fp):
                    size_mb += os.path.getsize(fp) / (1024 * 1024)

        return {
            "total_prompts": total_prompts,
            "vector_dim": int(md.get("vector_dim", "0") or 0),
            "metadata": md,
            "lance_path": lance_path,
            "file_size_mb": size_mb,
        }


class ClozeMetaLookup:
    """
    按 prompt_idx 批量回查 metadata，避免候选里冗余携带 meta 副本。
    对齐 PromptMetaLookup 的接口。
    """

    def __init__(self, lance_path: str):
        import lance

        self.lance_path = lance_path
        self._ds = lance.dataset(self.lance_path)

    def get_many(self, prompt_indices: List[int]) -> Dict[int, Dict]:
        import pickle

        uniq = sorted({int(i) for i in prompt_indices if i is not None and int(i) >= 0})
        if not uniq:
            return {}

        try:
            table = self._ds.take(uniq, columns=["metadata"])
        except Exception as e:
            raise RuntimeError(f"ClozeMetaLookup.take() 失败: {e}")

        meta_col = table.column("metadata").to_pylist()
        out: Dict[int, Dict] = {}
        for idx, raw in zip(uniq, meta_col):
            if raw is None:
                out[idx] = {}
                continue
            try:
                data = pickle.loads(raw)
                out[idx] = data if isinstance(data, dict) else {}
            except Exception:
                out[idx] = {}
        return out

    def get_metadata_by_indices(self, prompt_indices: List[int]) -> Dict[int, Dict]:
        return self.get_many(prompt_indices)


class ClozeVectorBatchIterator:
    """
    ClozeVectorCache 的 Lance 分批迭代器。
    """

    def __init__(self, lance_path: str, batch_size: int, use_fp16: bool = True):
        import pickle
        import torch
        import lance

        self.lance_path = lance_path
        self.batch_size = int(batch_size)
        self._use_fp16 = use_fp16
        self._pickle = pickle
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._ds = lance.dataset(self.lance_path)
        self._ds_meta = self._ds.metadata or {}
        self.total_prompts = int(self._ds.count_rows())
        self.num_batches = (self.total_prompts + self.batch_size - 1) // self.batch_size
        self._current_batch = 0

        self._vector_dim = int(self._ds_meta.get("vector_dim", "0") or 0)
        if self._vector_dim <= 0:
            try:
                self._vector_dim = int(getattr(self._ds.schema.field("vector").type, "list_size", 0) or 0)
            except Exception:
                self._vector_dim = 0
        if self._vector_dim <= 0:
            raise RuntimeError("无法解析 vector_dim，请删除缓存并重新生成")

        self._init_scanner()

        self._cache_metadata = {}

    @property
    def metadata(self):
        return self._ds_meta

    def _init_scanner(self):
        self._scanner = self._ds.scanner(
            columns=["vector", "prompt", "metadata"],
            scan_in_order=True,
            batch_size=max(1024, self.batch_size),
        )
        self._rb_iter = iter(self._scanner.to_batches())

        self._rb = None
        self._rb_pos = 0
        self._rb_vectors = None
        self._rb_prompts = None
        self._rb_meta_bytes = None

    def _load_next_rb(self):
        rb = next(self._rb_iter)  # may raise StopIteration
        vec_col = rb.column(0)
        flat = vec_col.values.to_numpy(zero_copy_only=False)
        vecs = flat.reshape(len(vec_col), self._vector_dim)

        self._rb = rb
        self._rb_pos = 0
        self._rb_vectors = vecs
        self._rb_prompts = rb.column(1).to_pylist()
        self._rb_meta_bytes = rb.column(2).to_pylist()

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self._current_batch >= self.num_batches:
            raise StopIteration

        start_idx = self._current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_prompts)
        desired = end_idx - start_idx

        out_vec = np.empty((desired, self._vector_dim), dtype=np.float32)
        out_meta: List[Dict] = [{} for _ in range(desired)]
        out_prompts: List[str] = ["" for _ in range(desired)]

        filled = 0
        while filled < desired:
            if self._rb is None or self._rb_pos >= len(self._rb_prompts):
                self._load_next_rb()

            rb_remain = len(self._rb_prompts) - self._rb_pos
            take = min(rb_remain, desired - filled)

            out_vec[filled : filled + take] = self._rb_vectors[self._rb_pos : self._rb_pos + take]
            out_prompts[filled : filled + take] = self._rb_prompts[self._rb_pos : self._rb_pos + take]

            mb = self._rb_meta_bytes[self._rb_pos : self._rb_pos + take]
            for i, b in enumerate(mb):
                try:
                    out_meta[filled + i] = self._pickle.loads(b) if b is not None else {}
                except Exception:
                    out_meta[filled + i] = {}

            self._rb_pos += take
            filled += take

        batch_vectors = self._torch.from_numpy(out_vec).to(self._device)
        if self._use_fp16 and str(self._device).startswith('cuda'):
            batch_vectors = batch_vectors.half()
        batch_metadata = out_meta
        batch_prompts = out_prompts

        batch_info = {
            "batch_idx": self._current_batch,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "total_prompts": self.total_prompts,
            "num_batches": self.num_batches,
        }

        self._current_batch += 1
        return batch_vectors, batch_prompts, batch_metadata, batch_info

    def __len__(self) -> int:
        return self.num_batches

    def reset(self):
        self._current_batch = 0
        self._init_scanner()

    def close(self):
        self._rb = None
        self._rb_vectors = None
        self._rb_prompts = None
        self._rb_meta_bytes = None
        self._rb_iter = None
        self._scanner = None
        self._ds = None

    def __del__(self):
        self.close()


def run_cloze_fill_search(
    index_directory: str = None,
    output_directory: str = None,
    video_output_directory: str = None,
    video_copy_mode: bool = False,
    video_name_format: str = None,
    debug_similarity: bool = False,
    # 优化参数
    prompt_search_batch_size: Optional[int] = DEFAULT_CLOZE_PROMPT_SEARCH_BATCH_SIZE,
    lance_batch_size: Optional[int] = None,
    # 搜索模式参数
    search_mode: int = 0,
    top_k: Optional[int] = None,
    # 视频导出帧偏移参数
    start_frame_offset: int = None,
    end_frame_offset: int = None,
    # LMDB 缓存参数
    use_diskcache: bool = True,
    diskcache_dir: str = None,
    # 中英文模式
    use_chinese: bool = False,
    # Reranker 参数
    use_reranker: bool = False,
    rerank_top_k: Optional[int] = DEFAULT_CLOZE_RERANK_TOP_K,
    rerank_batch_size: Optional[int] = DEFAULT_CLOZE_RERANK_BATCH_SIZE,
    reranker_output_resolution: str = '384',
    candidate_batch_size: Optional[int] = None,
    # 缓存批处理大小
    prompt_cache_batch_size: Optional[int] = DEFAULT_CLOZE_PROMPT_CACHE_BATCH_SIZE,
    # 线程配置参数
    lance_load_workers: Optional[int] = DEFAULT_CLOZE_LANCE_LOAD_WORKERS,
    lmdb_write_batch_size: Optional[int] = DEFAULT_CLOZE_LMDB_WRITE_BATCH_SIZE,
    # 向量去重参数
    vector_dedup_threshold: float = None,
    # 相邻片段合并参数
    adjacent_merge_frames: int = None,
    # 计算与特征精度
    use_fp16: bool = True,
    feature_fp16: Optional[bool] = None
) -> Dict[str, Dict]:
    """
    运行选词填空模式搜索
    
    核心特色：
    - 从 "选词填空规则" 动态读取模板和标签
    - 支持中英文模式切换
    - 复用 Prompt 模式的核心搜索功能
    
    Args:
        index_directory: 索引文件目录
        output_directory: 输出目录
        video_output_directory: 视频输出目录
        video_copy_mode: 视频切割模式
        video_name_format: 视频名称格式
        debug_similarity: 调试模式
        prompt_search_batch_size: 搜索时每批加载的向量数量
        lance_batch_size: 每批加载的Lance数量
        search_mode: 搜索模式 (-1=按视频, 0=按Lance, 1=跨Lance)
        top_k: 每组返回的最大结果数；None 或 <=0 表示不限制
        start_frame_offset: 起始帧偏移量
        end_frame_offset: 结束帧偏移量
        use_diskcache: 是否使用 LMDB 磁盘缓存
        diskcache_dir: LMDB 缓存目录
        use_chinese: 是否使用中文模式
            - False（默认）: 使用英文模板 + 英文标签值
            - True: 使用中文模板 + 中文标签键名
        use_reranker: 是否使用 Reranker
        rerank_top_k: Reranker Top-K
        rerank_batch_size: Reranker 批处理大小
        reranker_output_resolution: Reranker 帧输出分辨率
        candidate_batch_size: 候选分批处理大小
        prompt_cache_batch_size: 生成缓存时的批处理大小
        lance_load_workers: Lance加载线程数（全量预加载时使用）
        lmdb_write_batch_size: LMDB单事务写入批大小（分批加载时使用）
        vector_dedup_threshold: 向量去重余弦相似度阈值
            - None（默认）: 不进行向量去重
            - 0.90 ~ 0.98: 超过此阈值的同标签视频只保留一个
            - 优先规则: 同标签组内如有 OP/ED 视频，只保留 OP/ED 视频
        adjacent_merge_frames: 相邻片段合并帧阈值
            - None（默认）: 不进行相邻片段合并
            - N（正整数）: 当片段A的endframe与片段B的startframe差值≤N时合并
    
    Returns:
        搜索结果字典
    """
    import time
    import gc
    import torch
    import numpy as np
    
    # 路径解析
    resolver = PathResolver()
    
    if index_directory is None:
        index_directory = str(resolver.project_root / 'indexes')
    
    if output_directory is None:
        output_directory = str(resolver.project_root / 'output')
    
    os.makedirs(output_directory, exist_ok=True)
    
    if feature_fp16 is None:
        feature_fp16 = use_fp16
    
    # 固定缓存目录为 templates/prompt_cache
    cache_dir = str(resolver.project_root / 'templates' / 'prompt_cache')
    
    # 查找 Lance 索引目录，并显式拒绝旧 .pkl 索引
    lance_files = []
    unsupported_pkl_files = []
    for f in os.listdir(index_directory):
        full_path = os.path.join(index_directory, f)
        if os.path.isdir(full_path) and f.endswith('.lance'):
            lance_files.append(full_path)
        elif f.endswith('.pkl'):
            unsupported_pkl_files.append(full_path)

    if unsupported_pkl_files:
        print(f"❌ 错误: 检测到不支持的 .pkl 索引，请先转换为 .lance: {unsupported_pkl_files[0]}")
        return {}

    if not lance_files:
        print(f"❌ 错误: 在 {index_directory} 中没有找到 Lance 索引")
        return {}
    
    # 从 Lance 目录名提取模型名称（复用 Prompt 模式的解析逻辑）
    from A_coreUtils.search.auto_scene_search import (
        extract_model_name_from_index,
        detect_model_type_from_name,
        detect_truncate_dim_from_index_paths,
        normalize_top_k,
        normalize_optional_positive_int,
        SimilarityThresholdConfig,
        export_video_matches,
        cleanup_temp_after_export,
    )
    model_name = None
    for lance_file in sorted(lance_files):
        parsed_model_name = extract_model_name_from_index(lance_file)
        if parsed_model_name:
            model_name = parsed_model_name
            break

    if model_name is None:
        raise RuntimeError(
            f"[选词填空搜索] 无法从 Lance 索引文件名中解析出模型名称，"
            f"请检查 {index_directory} 中的 .lance 文件命名是否包含模型名"
        )

    # 检测模型类型
    model_type = detect_model_type_from_name(model_name)

    # 从索引文件名自动检测 truncate_dim
    detected_truncate_dim = detect_truncate_dim_from_index_paths(lance_files)
    if detected_truncate_dim is not None:
        print(f"[选词填空搜索] 从索引文件名自动检测到 truncate_dim={detected_truncate_dim}")

    prompt_search_batch_size = normalize_optional_positive_int(
        prompt_search_batch_size,
        field_name='prompt_search_batch_size',
    )
    prompt_cache_batch_size = normalize_optional_positive_int(
        prompt_cache_batch_size,
        field_name='prompt_cache_batch_size',
    )
    rerank_top_k = normalize_optional_positive_int(
        rerank_top_k,
        field_name='rerank_top_k',
    )
    rerank_batch_size = normalize_optional_positive_int(
        rerank_batch_size,
        field_name='rerank_batch_size',
    )
    candidate_batch_size = normalize_optional_positive_int(
        candidate_batch_size,
        field_name='candidate_batch_size',
    )
    lance_batch_size = normalize_optional_positive_int(
        lance_batch_size,
        field_name='lance_batch_size',
    )
    lance_load_workers = normalize_optional_positive_int(
        lance_load_workers,
        field_name='lance_load_workers',
    )
    lmdb_write_batch_size = normalize_optional_positive_int(
        lmdb_write_batch_size,
        field_name='lmdb_write_batch_size',
    )
    top_k = normalize_top_k(top_k)
    
    lang_mode = "中文" if use_chinese else "英文"
    print(f"[选词填空搜索] 模型: {model_name} (类型: {model_type})")
    print(f"[选词填空搜索] 语言模式: {lang_mode}")
    
    # 加载模型
    from A_coreUtils.embedding.embedding_model import EmbeddingModelProcessor
    
    processor = None
    
    # 创建向量缓存器
    vector_cache = ClozeVectorCache(
        processor=processor,
        cache_dir=cache_dir,
        model_name=model_name,
        use_chinese=use_chinese
    )

    # 显式 None 时采用动态全量批大小（而非回退固定常量）。
    effective_prompt_cache_batch_size = prompt_cache_batch_size
    if effective_prompt_cache_batch_size is None:
        effective_prompt_cache_batch_size = max(
            1,
            int(vector_cache.prompt_generator.count_total_combinations() or 0)
        )
    
    # 生成或加载缓存
    cache_valid = vector_cache.cache_exists()
    if not cache_valid:
        print("[Cloze Search] Cache missing or invalid, loading CLIP to regenerate vectors...")
        processor = EmbeddingModelProcessor(
            model_name=model_name,
            model_type=model_type,
            truncate_dim=detected_truncate_dim,
            io_workers=8,
            use_fp16=use_fp16
        )
        vector_cache.processor = processor
        vector_cache.generate_cache(batch_size=effective_prompt_cache_batch_size)
    cache_info = vector_cache.get_cache_info()
    cached_total = int(cache_info.get('total_prompts', 0) or 0)
    effective_prompt_search_batch_size = (
        prompt_search_batch_size
        if prompt_search_batch_size is not None
        else max(1, cached_total)
    )
    cache_iterator = vector_cache.load_cache_batched(batch_size=effective_prompt_search_batch_size, use_fp16=use_fp16)
    prompt_meta_lookup = vector_cache.load_meta_lookup()
    
    if cache_iterator is None:
        raise RuntimeError("Cloze prompt 缓存向量不可用，且不允许回退到实时编码。")
    total_prompts = cache_iterator.total_prompts
    print(f"[选词填空搜索] 缓存信息: {total_prompts} 个prompt, {cache_info['file_size_mb']:.1f} MB")
    
    # 从 SimilarityThresholdConfig 获取阈值（与 Prompt 模式一致）
    similarity_threshold = SimilarityThresholdConfig.get_threshold(model_type, use_reranker)
    
    print(f"[选词填空搜索] 相似度阈值: {similarity_threshold} (Reranker: {use_reranker})")
    
    # 导入批量搜索引擎
    from A_coreUtils.search.batch_text_search import BatchTextSearchEngine
    from A_coreUtils.video_processing.video_name_parser import VideoNameParser
    
    # 获取子类名称（用于验证和默认值生成）
    subcategories = vector_cache.prompt_generator.keyword_manager.subcategories
    
    # 默认视频名称格式（动态生成，基于选词填空规则的子类名称）
    if video_name_format is None:
        video_name_format = generate_default_cloze_video_name_format(subcategories, prefix="选词填空")
        print(f"[选词填空搜索] 使用默认 video_name_format: {video_name_format}")
    else:
        # 验证用户提供的 video_name_format
        is_valid, error_msg = validate_cloze_video_name_format(video_name_format, subcategories)
        if not is_valid:
            raise ValueError(f"[选词填空搜索] {error_msg}")
        print(f"[选词填空搜索] 使用自定义 video_name_format: {video_name_format}")
    
    # 创建批量搜索引擎
    batch_engine = BatchTextSearchEngine(
        processor=None,
        index_paths=lance_files,
        cache_dir=str(resolver.project_root / 'temp' / 'cache'),
        load_workers=lance_load_workers,
        use_fp16=feature_fp16,
        lance_batch_size=lance_batch_size,
        video_name_format=video_name_format,
        search_mode=search_mode,
        top_k=top_k,
        lmdb_write_batch_size=lmdb_write_batch_size,
        truncate_dim=detected_truncate_dim,
        logit_scale=100.0
    )
    
    # 生成搜索配置哈希（用于断点续传验证）
    config_payload = {
        'similarity_threshold': similarity_threshold,
        'search_mode': search_mode,
        'top_k': top_k,
        'use_reranker': use_reranker,
        'rerank_top_k': rerank_top_k,
        'rerank_batch_size': rerank_batch_size,
        'reranker_output_resolution': reranker_output_resolution,
        'candidate_batch_size': candidate_batch_size,
        'prompt_search_batch_size': prompt_search_batch_size,
        'prompt_cache_batch_size': prompt_cache_batch_size,
        'use_fp16': use_fp16,
        'feature_fp16': feature_fp16,
        'lance_batch_size': lance_batch_size,
        'use_diskcache': use_diskcache,
        'vector_dedup_threshold': vector_dedup_threshold,
        'use_chinese': use_chinese,
        'video_name_format': video_name_format,
        'cloze_prompt_config_hash': getattr(vector_cache, '_config_hash', None),
        'index_files': sorted([os.path.basename(p) for p in lance_files]),
    }
    config_str = json.dumps(config_payload, ensure_ascii=False, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()
    
    # 预加载Lance特征（根据 lance_batch_size 决定全量或分批）
    if batch_engine._preload_all:
        print("[选词填空搜索] 全量预加载模式")
    else:
        print(f"[选词填空搜索] 分批Lance模式: 每批 {batch_engine.lance_batch_size} 个Lance")
    
    # 初始化视频名称解析器
    video_name_parser = VideoNameParser()
    
    # LMDB 缓存目录（不再无条件清空，支持断点续传）
    diskcache_root = None
    search_cache_dir = None
    lance_merge_cache_dir = None
    if use_diskcache:
        if diskcache_dir is None:
            diskcache_root = str(resolver.project_root / 'temp' / 'cache' / 'cloze_search_results')
        else:
            diskcache_root = diskcache_dir
        search_cache_dir = os.path.join(diskcache_root, 'inner_search')
        lance_merge_cache_dir = os.path.join(diskcache_root, 'lance_merge')
        os.makedirs(search_cache_dir, exist_ok=True)
        os.makedirs(lance_merge_cache_dir, exist_ok=True)
    
    # 执行搜索
    start_time = time.time()
    
    # 懒加载 Reranker（仅在候选处理阶段首次真正需要时加载）
    reranker_loader = None
    reranker_weight = SimilarityThresholdConfig.RERANKER_WEIGHT
    
    # 计算 CLIP 初始阈值（用于候选筛选）
    clip_initial_threshold = SimilarityThresholdConfig.get_threshold(model_type, use_reranker=False) if use_reranker else similarity_threshold

    if use_reranker:
        reranker_model_path = str(resolver.project_root / 'models' / 'Qwen3-VL-Reranker-2B')
        rerank_cache_dir = str(resolver.project_root / 'temp' / 'cache' / 'rerank_cache')
        print(f"[选词填空搜索] 已启用 Reranker，候选阶段将懒加载: {reranker_model_path}")

        def _lazy_load_reranker():
            from A_coreUtils.qwen_models.qwen3_vl_reranker import Qwen3VLReranker
            print(f"[选词填空搜索] 懒加载 Reranker 模型: {reranker_model_path}")
            reranker_instance = Qwen3VLReranker(
                model_name_or_path=reranker_model_path,
                cache_dir=rerank_cache_dir,
                torch_dtype="auto"
            )
            print(f"[选词填空搜索] Reranker 模型加载完成")
            return reranker_instance

        reranker_loader = _lazy_load_reranker
    
    # 搜索参数
    search_kwargs = dict(
        cache_iterator=cache_iterator,
        threshold=similarity_threshold,
        initial_threshold=clip_initial_threshold,
        video_name_parser=video_name_parser,
        use_diskcache=use_diskcache,
        cache_dir=search_cache_dir,
        rerank_top_k=rerank_top_k,
        use_reranker=use_reranker,
        reranker=None,
        reranker_loader=reranker_loader,
        reranker_weight=reranker_weight,
        reranker_output_resolution=reranker_output_resolution,
        rerank_batch_size=rerank_batch_size,
        candidate_batch_size=candidate_batch_size,
        search_mode=search_mode,
        result_top_k=top_k,
        config_hash=config_hash,
        prompt_meta_lookup=prompt_meta_lookup,
    )
    
    if batch_engine._preload_all:
        # 全量预加载模式：直接搜索（search_with_batched_cache 内部已支持断点续传）
        if batch_engine.all_features_gpu is None:
            batch_engine._preload_all_features()
        result_view = batch_engine.search_with_batched_cache(**search_kwargs)
        result_count = int(result_view.count())
        print(f"\n📊 搜索完成: {result_count:,} 个场景")
    else:
        # 分批Lance加载模式（根治版）：
        #   阶段1：每批仅做 CLIP 候选聚合，写入同一个 LMDB
        #   阶段2：全部 Lance 聚合完成后，只做一次候选后处理（含可选 Reranker）
        from A_coreUtils.search.batch_text_search import LMDBCache
        import time as _time

        total_lances = len(batch_engine.index_paths)
        lance_bs = batch_engine.lance_batch_size
        total_batches = (total_lances + lance_bs - 1) // lance_bs

        # 生成分批Lance模式的 config_hash（包含 Lance 批次信息）
        lance_batch_config_str = f"cloze_lance_batch|{config_hash}|{total_lances}|{lance_bs}"
        lance_batch_hash = hashlib.md5(lance_batch_config_str.encode()).hexdigest()

        # 候选聚合 LMDB（与阶段2统一复用）
        lance_batch_lmdb_dir = lance_merge_cache_dir
        if lance_batch_lmdb_dir is None:
            lance_batch_lmdb_dir = str(resolver.project_root / 'temp' / 'cache' / 'cloze_search_results')
        os.makedirs(lance_batch_lmdb_dir, exist_ok=True)

        lance_batch_lmdb = LMDBCache(lance_batch_lmdb_dir, map_size=10 * 1024 * 1024 * 1024)

        # 检查断点续传（仅针对阶段1：Lance 批次聚合）
        start_lance_batch = 0
        checkpoint = lance_batch_lmdb.load_checkpoint()
        if checkpoint and checkpoint.get('config_hash') == lance_batch_hash and checkpoint.get('phase') == 'lance_batch':
            start_lance_batch = checkpoint.get('last_completed_lance_batch', -1) + 1
            if start_lance_batch > 0:
                print(f"[选词填空搜索] 🔄 断点续传: 从 Lance 批次 {start_lance_batch + 1}/{total_batches} 继续")
        else:
            lance_batch_lmdb.clear_candidates()
            lance_batch_lmdb.clear_results()

        print(f"[选词填空搜索] 分批Lance模式: {total_lances} 个Lance, 每批 {lance_bs} 个, 共 {total_batches} 批")
        scene_key_to_lance_map = {}

        # 阶段1：分批聚合 CLIP 候选（不做每批后处理）
        for batch_idx in range(start_lance_batch, total_batches):
            batch_start_idx = batch_idx * lance_bs
            batch_end_idx = min(batch_start_idx + lance_bs, total_lances)
            batch_paths = batch_engine.index_paths[batch_start_idx:batch_end_idx]

            print(f"\n[Lance批次 {batch_idx + 1}/{total_batches}] 加载 {len(batch_paths)} 个Lance...")
            batch_engine._load_lance_batch_to_merged(batch_paths)

            if batch_engine.all_features_gpu is None:
                print(f"[Lance批次 {batch_idx + 1}] 无有效数据，跳过")
                lance_batch_lmdb.save_checkpoint({
                    'config_hash': lance_batch_hash,
                    'last_completed_lance_batch': batch_idx,
                    'total_batches': total_batches,
                    'phase': 'lance_batch',
                    'timestamp': _time.time()
                })
                batch_engine._unload_merged_features()
                continue

            # 记录 scene_key -> source_lance 映射（用于最终按Lance分组）
            for scene_info in batch_engine.scene_map:
                video_name = os.path.basename(scene_info['video_path']) if scene_info.get('video_path') else ''
                scene_key = f"{scene_info['start_frame']}_{video_name}"
                scene_key_to_lance_map[scene_key] = scene_info.get('source_lance', 'unknown')

            cache_iterator.reset()

            # 仅执行 CLIP 候选聚合：候选写入 LMDB
            clip_batch_kwargs = dict(search_kwargs)
            clip_batch_kwargs['cache_iterator'] = cache_iterator
            clip_batch_kwargs['threshold'] = clip_initial_threshold
            clip_batch_kwargs['use_diskcache'] = True
            clip_batch_kwargs['cache_dir'] = lance_batch_lmdb_dir
            clip_batch_kwargs['use_reranker'] = False
            clip_batch_kwargs['reranker'] = None
            clip_batch_kwargs['reranker_loader'] = None
            clip_batch_kwargs['candidate_batch_size'] = None
            clip_batch_kwargs['result_top_k'] = None
            clip_batch_kwargs['search_mode'] = 0
            clip_batch_kwargs['config_hash'] = None
            clip_batch_kwargs['append_to_lmdb_cache'] = True
            clip_batch_kwargs['prompt_meta_lookup'] = prompt_meta_lookup
            batch_engine.search_with_batched_cache(**clip_batch_kwargs)

            lance_batch_lmdb.save_checkpoint({
                'config_hash': lance_batch_hash,
                'last_completed_lance_batch': batch_idx,
                'total_batches': total_batches,
                'phase': 'lance_batch',
                'timestamp': _time.time()
            })
            print(f"[Lance批次 {batch_idx + 1}] 候选聚合完成")

            batch_engine._unload_merged_features()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        lance_batch_lmdb.close()

        # 阶段2：统一执行一次候选后处理（可选 Reranker）
        print("\n[选词填空搜索] 分批Lance候选聚合完成，开始统一后处理...")
        rerank_search_kwargs = dict(search_kwargs)
        rerank_search_kwargs['cache_iterator'] = None
        rerank_search_kwargs['use_diskcache'] = True
        rerank_search_kwargs['cache_dir'] = lance_batch_lmdb_dir
        rerank_search_kwargs['skip_clip_search'] = True
        rerank_search_kwargs['result_top_k'] = top_k
        rerank_search_kwargs['search_mode'] = search_mode
        rerank_search_kwargs['config_hash'] = None
        rerank_search_kwargs['prompt_meta_lookup'] = prompt_meta_lookup
        all_batch_view = batch_engine.search_with_batched_cache(**rerank_search_kwargs)
        print(f"[选词填空搜索] 统一后处理完成，得到 {all_batch_view.count()} 个候选结果")

        # 按 search_mode 分组取 Top-K（流式版）
        from A_coreUtils.search.batch_text_search import apply_search_mode_grouping_lmdb
        grouping_cache = os.path.join(lance_merge_cache_dir or str(resolver.project_root / 'temp' / 'cache'), 'cloze_grouping')
        result_view = apply_search_mode_grouping_lmdb(
            all_batch_view, search_mode, top_k,
            scene_key_to_lance_map, cache_dir=grouping_cache
        )
    
    search_time = time.time() - start_time
    print(f"[选词填空搜索] 搜索完成! 耗时 {search_time:.2f}s, 找到 {result_view.count()} 个结果")
    
    # ========== 向量去重（视频导出前）==========
    scene_features = {}  # {scene_key: np.ndarray}
    scene_lance_map = {}  # {scene_key: source_lance}
    if vector_dedup_threshold is not None and result_view.count() > 0 and batch_engine.all_features_gpu is not None:
        print(f"[选词填空搜索] 提取场景特征向量用于去重...")
        
        # 从 result_view 收集 scene_key 集合
        result_keys = set()
        for _batch in result_view.iter_batches(batch_size=2000):
            for _sk, _ in _batch:
                result_keys.add(_sk)
        
        all_features_cpu = batch_engine.all_features_gpu.float().cpu().numpy()
        feat_idx = 0
        for scene_idx, scene_info in enumerate(batch_engine.scene_map):
            video_name = os.path.basename(scene_info['video_path']) if scene_info.get('video_path') else ''
            scene_key = f"{scene_info['start_frame']}_{video_name}"
            count = batch_engine.feature_counts[scene_idx]
            
            if scene_key in result_keys:
                scene_vectors = all_features_cpu[feat_idx:feat_idx + count]
                scene_features[scene_key] = scene_vectors
                scene_lance_map[scene_key] = scene_info.get('source_lance', 'unknown')
            
            feat_idx += count
        print(f"[选词填空搜索] 提取了 {len(scene_features)} 个场景的特征向量（共 {len(result_keys)} 个搜索结果）")
    
    elif vector_dedup_threshold is not None and result_view.count() > 0 and not batch_engine._preload_all:
        print("[选词填空搜索] 提取场景特征向量用于去重...")
        result_keys = set()
        for _batch in result_view.iter_batches(batch_size=2000):
            for _sk, _ in _batch:
                result_keys.add(_sk)
        lance_bs = batch_engine.lance_batch_size
        total_lances = len(batch_engine.index_paths)
        total_batches = (total_lances + lance_bs - 1) // lance_bs

        for batch_idx in range(total_batches):
            batch_start_idx = batch_idx * lance_bs
            batch_end_idx = min(batch_start_idx + lance_bs, total_lances)
            batch_paths = batch_engine.index_paths[batch_start_idx:batch_end_idx]
            batch_engine._load_lance_batch_to_merged(batch_paths)

            if batch_engine.all_features_gpu is None:
                batch_engine._unload_merged_features()
                continue

            all_features_cpu = batch_engine.all_features_gpu.float().cpu().numpy()
            feat_idx = 0
            for scene_idx, scene_info in enumerate(batch_engine.scene_map):
                video_name = os.path.basename(scene_info['video_path']) if scene_info.get('video_path') else ''
                scene_key = f"{scene_info['start_frame']}_{video_name}"
                count = batch_engine.feature_counts[scene_idx]

                if scene_key in result_keys and scene_key not in scene_features:
                    scene_vectors = all_features_cpu[feat_idx:feat_idx + count]
                    scene_features[scene_key] = scene_vectors
                    scene_lance_map[scene_key] = scene_info.get('source_lance', 'unknown')

                feat_idx += count

            batch_engine._unload_merged_features()
            if len(scene_features) >= len(result_keys):
                break

        print(f"[选词填空搜索] 提取了 {len(scene_features)} 个场景的特征向量（共 {len(result_keys)} 个搜索结果）")

    batch_engine.cleanup()
    del batch_engine
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 向量去重（如果启用）- 同Lance内去重
    if vector_dedup_threshold is not None and scene_features:
        print(f"\n[向量去重] 开始向量去重，阈值={vector_dedup_threshold}，模式=同Lance内去重")
        from .batch_text_search import deduplicate_by_vector_lmdb
        dedup_cache = os.path.join(str(resolver.project_root / 'temp' / 'cache'), 'cloze_dedup')
        result_view = deduplicate_by_vector_lmdb(
            result_view=result_view,
            scene_features=scene_features,
            video_name_format=video_name_format,
            similarity_threshold=vector_dedup_threshold,
            scene_lance_map=scene_lance_map,
            cache_dir=dedup_cache
        )
    
    # 相邻片段合并（视频导出前）
    if adjacent_merge_frames is not None and adjacent_merge_frames >= 0:
        from A_coreUtils.search.auto_scene_search import merge_adjacent_scenes
        merged = merge_adjacent_scenes(
            result_source=result_view,
            adjacent_merge_frames=adjacent_merge_frames,
            video_name_format=video_name_format
        )
        if merged is not result_view and hasattr(result_view, 'close'):
            result_view.close()
        result_view = merged
    
    # 视频导出（与 Prompt 模式一致）
    if result_view.count() > 0:
        print(f"\n📹 视频导出: {result_view.count()} 个场景")
        
        export_stats, video_output_directory = export_video_matches(
            result_source=result_view,
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
    
    print("\n✅ 选词填空搜索完成!")
    
    return result_view


# 测试代码
if __name__ == '__main__':
    # 测试关键词管理器
    print("=" * 50)
    print("测试 ClozeKeywordManager")
    print("=" * 50)
    
    manager = ClozeKeywordManager()
    print(f"子类: {manager.subcategories}")
    print(f"模板数量: {len(manager.templates)}")
    
    for template_cn, template_en in manager.templates.items():
        print(f"\n模板: {template_cn}")
        print(f"英文: {template_en}")
        placeholders = manager.parse_template_placeholders(template_cn)
        print(f"占位符: {placeholders}")
    
    # 测试 Prompt 生成器
    print("\n" + "=" * 50)
    print("测试 ClozePromptGenerator")
    print("=" * 50)
    
    generator = ClozePromptGenerator(use_chinese=False)
    print(f"总组合数: {generator.count_total_combinations()}")
    
    print("\n前5个组合（英文模式）:")
    for i, combo in enumerate(generator.iterate_all_combinations()):
        if i >= 5:
            break
        print(f"  {i+1}. {combo['prompt']}")
    
    generator_cn = ClozePromptGenerator(use_chinese=True)
    print("\n前5个组合（中文模式）:")
    for i, combo in enumerate(generator_cn.iterate_all_combinations()):
        if i >= 5:
            break
        print(f"  {i+1}. {combo['prompt']}")
