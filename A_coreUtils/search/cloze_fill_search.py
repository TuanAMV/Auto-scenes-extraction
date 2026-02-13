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
_cut_detect_scene_dir = os.path.dirname(_a_core_utils_dir)
if _cut_detect_scene_dir not in sys.path:
    sys.path.insert(0, _cut_detect_scene_dir)

from path_resolver import PathResolver


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
    选词填空向量缓存器
    
    v2.1: memmap 流式写入 + 生产者-消费者并行优化
    - 向量: .dat 文件（memmap 流式写入，极低内存占用）
    - 向量元信息: .meta.json 文件（存储 shape 和 dtype）
    - 元数据: .pkl 文件（prompt 信息 + 配置哈希）
    
    缓存文件命名格式:
    - cloze_cache_{model_name}_{lang}_vectors.dat (向量，memmap)
    - cloze_cache_{model_name}_{lang}_vectors.meta.json (向量元信息)
    - cloze_cache_{model_name}_{lang}_prompts.pkl (元数据)
    """
    
    def __init__(self,
                 processor=None,
                 keywords_path: str = None,
                 cache_dir: str = None,
                 model_name: str = None,
                 use_chinese: bool = False):
        """
        初始化向量缓存器
        
        Args:
            processor: EmbeddingModelProcessor 实例（生成缓存时需要）
            keywords_path: logic_keywords.json 路径
            cache_dir: 缓存目录，None则使用 templates/prompt_cache
            model_name: 模型名称（用于缓存文件命名）
            use_chinese: 是否使用中文模式
        """
        self.processor = processor
        self.use_chinese = use_chinese
        
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
        
        # 创建 Prompt 生成器
        self.prompt_generator = ClozePromptGenerator(keywords_path, use_chinese)
        
        # 计算配置哈希
        self._config_hash = self.prompt_generator.compute_config_hash()
    
    def _get_safe_model_name(self) -> str:
        """获取安全的模型名称（用于文件命名）"""
        return self.model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    def get_cache_path(self) -> str:
        """获取缓存基础路径（不带扩展名）"""
        safe_model_name = self._get_safe_model_name()
        lang_suffix = "cn" if self.use_chinese else "en"
        base_name = f"cloze_cache_{safe_model_name}_{lang_suffix}"
        return os.path.join(self.cache_dir, base_name)
    
    def get_vectors_path(self) -> str:
        """获取向量文件路径 (.dat，memmap格式)"""
        return self.get_cache_path() + "_vectors.dat"
    
    def get_vectors_meta_path(self) -> str:
        """获取向量元信息文件路径 (.meta.json)"""
        return self.get_cache_path() + "_vectors.meta.json"
    
    def get_prompts_path(self) -> str:
        """获取 prompt 信息文件路径 (.pkl)"""
        return self.get_cache_path() + "_prompts.pkl"
    
    def cache_exists(self) -> bool:
        """检查缓存是否存在且有效（v2.1 memmap 格式）"""
        import pickle
        import numpy as np
        
        vectors_path = self.get_vectors_path()
        vectors_meta_path = self.get_vectors_meta_path()
        prompts_path = self.get_prompts_path()
        
        if not os.path.exists(vectors_path):
            return False
        if not os.path.exists(vectors_meta_path):
            return False
        if not os.path.exists(prompts_path):
            return False
        
        # 验证哈希和数据一致性
        try:
            with open(prompts_path, 'rb') as f:
                data = pickle.load(f)

            expected_format_version = '2.1'
            cached_format_version = str(data.get('format_version', ''))
            if cached_format_version != expected_format_version:
                print(f"[选词填空缓存] 缓存版本不匹配(当前={cached_format_version}, 期望={expected_format_version})，需要重建")
                return False
            
            cached_hash = data.get('config_hash', '')
            if cached_hash != self._config_hash:
                print(f"[选词填空缓存] 配置已变化，需要重新生成缓存")
                return False

            prompt_metadata = data.get('prompts')
            if not isinstance(prompt_metadata, list):
                print("[选词填空缓存无效] prompts 结构错误，期望 list")
                return False
            if prompt_metadata:
                first_item = prompt_metadata[0]
                if not isinstance(first_item, dict) or 'prompt' not in first_item:
                    print("[选词填空缓存无效] prompts 元数据缺失字典结构，需要重建")
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
            if len(vectors_mmap) != len(prompt_metadata):
                print(f"[选词填空缓存无效] 向量数量({len(vectors_mmap)})与prompts数量({len(prompt_metadata)})不匹配")
                del vectors_mmap
                return False
            del vectors_mmap
            
            return True
        except Exception as e:
            print(f"[选词填空缓存] 验证缓存失败: {e}")
            return False
    
    def generate_cache(self, batch_size: int = 512) -> bool:
        """
        生成向量缓存（v2.1 memmap 流式写入 + 生产者-消费者并行模式）
        
        Args:
            batch_size: 批处理大小
        
        Returns:
            是否成功生成
        """
        import pickle
        import numpy as np
        import torch
        import time
        import gc
        from queue import Queue
        from threading import Thread, Event
        
        if self.processor is None:
            raise RuntimeError("生成缓存需要提供 processor")
        
        # 检查缓存是否已存在
        if self.cache_exists():
            print(f"[选词填空缓存] 缓存已存在且有效，跳过生成")
            return True
        
        vectors_path = self.get_vectors_path()
        vectors_meta_path = self.get_vectors_meta_path()
        prompts_path = self.get_prompts_path()
        
        # 删除旧文件
        for path in [vectors_path, vectors_meta_path, prompts_path]:
            if os.path.exists(path):
                os.remove(path)
        
        print("=" * 70)
        print("📝 选词填空向量缓存生成器 (v2.1 memmap + 生产者-消费者并行)")
        print("=" * 70)
        print(f"  模型: {self.model_name}")
        print(f"  语言模式: {'中文' if self.use_chinese else '英文'}")
        print(f"  批量大小: {batch_size}")
        
        # 第一遍：统计总数
        print("\n[步骤1] 统计prompt总数...")
        total_prompts = self.prompt_generator.count_total_combinations()
        print(f"  总计: {total_prompts:,} 个prompt")
        
        if total_prompts == 0:
            print(f"[选词填空缓存] 没有 prompt 需要编码")
            return False
        
        # 获取向量维度（通过编码一个样本）
        first_combo = next(self.prompt_generator.iterate_all_combinations())
        sample_vector = self.processor.encode_text([first_combo["prompt"]])
        vector_dim = sample_vector.shape[1]
        print(f"  向量维度: {vector_dim}")
        
        # 创建 memmap 文件
        print(f"\n[步骤2] 创建 memmap 文件...")
        vectors_mmap = np.memmap(
            vectors_path,
            dtype='float32',
            mode='w+',
            shape=(total_prompts, vector_dim)
        )
        print(f"  memmap 文件: {total_prompts:,} x {vector_dim} (float32)")
        
        # 保存向量元信息
        vectors_meta = {
            'shape': [total_prompts, vector_dim],
            'dtype': 'float32',
            'total_prompts': total_prompts,
            'vector_dim': vector_dim
        }
        with open(vectors_meta_path, 'w', encoding='utf-8') as f:
            json.dump(vectors_meta, f, indent=2)
        
        # 生产者-消费者并行模式
        print(f"\n[步骤3] 生产者-消费者并行编码 (batch_size={batch_size})...")
        start_time = time.time()
        
        # 队列和事件
        write_queue = Queue(maxsize=4)  # 写入队列，限制大小避免内存溢出
        stop_event = Event()
        
        # 消费者线程：负责写入 memmap
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
        
        # 生产者：编码并放入队列（不累积 all_prompts 和 all_metadata）
        current_idx = 0
        batch_texts = []
        batch_meta = []
        
        for combo in self.prompt_generator.iterate_all_combinations():
            batch_texts.append(combo["prompt"])
            batch_meta.append(combo)
            
            if len(batch_texts) >= batch_size:
                # 编码当前批次
                batch_vectors = self.processor.encode_text(batch_texts)
                if isinstance(batch_vectors, torch.Tensor):
                    batch_vectors = batch_vectors.cpu().numpy()
                
                # 放入写入队列
                write_queue.put((current_idx, batch_vectors))
                
                current_idx += len(batch_texts)
                batch_texts = []
                batch_meta = []
                
                # 进度显示
                if current_idx % (batch_size * 10) == 0:
                    elapsed = time.time() - start_time
                    speed = current_idx / elapsed if elapsed > 0 else 0
                    print(f"  [{current_idx:,}/{total_prompts:,}] {speed:.1f} prompts/s")
        
        # 处理最后一个不完整的批次
        if batch_texts:
            batch_vectors = self.processor.encode_text(batch_texts)
            if isinstance(batch_vectors, torch.Tensor):
                batch_vectors = batch_vectors.cpu().numpy()
            write_queue.put((current_idx, batch_vectors))
            current_idx += len(batch_texts)
        
        # 等待写入完成
        write_queue.put(None)
        stop_event.set()
        writer.join()
        
        # 关闭 memmap
        del vectors_mmap
        
        elapsed_time = time.time() - start_time
        print(f"\n  编码完成! 耗时: {elapsed_time:.2f}s")
        
        # 保存元数据（保存完整 metadata 字典，供后续命名/导出阶段使用）
        print(f"\n[步骤4] 保存元数据文件...")
        regenerated_prompts = list(self.prompt_generator.iterate_all_combinations())
        metadata = {
            'config_hash': self._config_hash,
            'model_name': self.model_name,
            'use_chinese': self.use_chinese,
            'total_prompts': total_prompts,
            'vector_dim': vector_dim,
            'format_version': '2.1',
            'prompts': regenerated_prompts  # 保存完整 prompt metadata 列表
        }
        with open(prompts_path, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        del regenerated_prompts, metadata
        gc.collect()
        
        # 获取文件大小
        vectors_size = os.path.getsize(vectors_path) / (1024 * 1024)
        prompts_size = os.path.getsize(prompts_path) / (1024 * 1024)
        print(f"    -> {os.path.basename(vectors_path)}: {vectors_size:.2f} MB")
        print(f"    -> {os.path.basename(prompts_path)}: {prompts_size:.2f} MB")
        
        print("\n" + "=" * 70)
        print("✅ 选词填空缓存生成完成!")
        print("=" * 70)
        
        gc.collect()
        return True
    
    def load_cache(self) -> Tuple[Any, List[Dict]]:
        """
        加载缓存（v2.1 memmap 格式）
        
        Returns:
            (向量数组, prompt元数据列表)
        """
        import pickle
        import numpy as np
        
        if not self.cache_exists():
            raise RuntimeError("缓存不存在或已过期，请先调用 generate_cache()")
        
        vectors_path = self.get_vectors_path()
        vectors_meta_path = self.get_vectors_meta_path()
        prompts_path = self.get_prompts_path()
        
        # 加载向量元信息
        with open(vectors_meta_path, 'r', encoding='utf-8') as f:
            vectors_meta = json.load(f)
        
        # 加载向量（使用 memmap 节省内存）
        vectors = np.memmap(
            vectors_path,
            dtype=vectors_meta['dtype'],
            mode='r',
            shape=tuple(vectors_meta['shape'])
        )
        
        # 加载元数据
        with open(prompts_path, 'rb') as f:
            metadata = pickle.load(f)

        return vectors, metadata['prompts']

    def load_cache_batched(self, batch_size: int = 10000):
        """
        鍒嗘壒鍔犺浇缂撳瓨锛屽叧閿涓轰笌 PromptVectorCache.load_cache_batched 瀵归綈
        """
        if not self.cache_exists():
            raise RuntimeError("缂撳瓨涓嶅瓨鍦ㄦ垨宸茶繃鏈燂紝璇峰厛璋冪敤 generate_cache()")

        return ClozeVectorBatchIterator(
            vectors_path=self.get_vectors_path(),
            vectors_meta_path=self.get_vectors_meta_path(),
            prompts_path=self.get_prompts_path(),
            batch_size=batch_size
        )


class ClozeVectorBatchIterator:
    """
    C 妯″紡 prompt 鍚戦噺鍒嗘壒杩唬鍣?
    """
    def __init__(self, vectors_path: str, vectors_meta_path: str, prompts_path: str, batch_size: int):
        import pickle
        import numpy as np
        import torch

        with open(vectors_meta_path, 'r', encoding='utf-8') as f:
            vectors_meta = json.load(f)

        self._vectors = np.memmap(
            vectors_path,
            dtype=vectors_meta['dtype'],
            mode='r',
            shape=tuple(vectors_meta['shape'])
        )

        with open(prompts_path, 'rb') as f:
            prompts_data = pickle.load(f)

        self._prompts = prompts_data['prompts']
        self.batch_size = batch_size
        self.total_prompts = len(self._prompts)
        self.num_batches = (self.total_prompts + batch_size - 1) // batch_size
        self._current_batch = 0
        self._torch = torch
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._cache_metadata = {}

    @property
    def metadata(self):
        return self._cache_metadata

    def __iter__(self):
        self._current_batch = 0
        return self

    def __next__(self):
        if self._current_batch >= self.num_batches:
            raise StopIteration

        start_idx = self._current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_prompts)

        batch_vectors_np = self._vectors[start_idx:end_idx]
        batch_vectors_np = np.array(batch_vectors_np)
        batch_vectors = self._torch.from_numpy(batch_vectors_np).to(self._device)

        batch_metadata = self._prompts[start_idx:end_idx]
        batch_prompts = [meta.get('prompt', '') for meta in batch_metadata]

        batch_info = {
            'batch_idx': self._current_batch,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'total_prompts': self.total_prompts,
            'num_batches': self.num_batches
        }

        self._current_batch += 1
        return batch_vectors, batch_prompts, batch_metadata, batch_info

    def reset(self):
        self._current_batch = 0


def run_cloze_fill_search(
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
    rerank_top_k: int = 7,
    rerank_batch_size: int = 7,
    reranker_output_resolution: str = '448',
    candidate_batch_size: int = None,
    # 缓存批处理大小
    prompt_cache_batch_size: int = 512,
    # 线程配置参数
    pkl_load_workers: int = 4,
    lmdb_write_batch_size: int = 1000,
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
        pkl_batch_size: 每批加载的PKL数量
        search_mode: 搜索模式 (-1=按视频, 0=按PKL, 1=跨PKL)
        top_k: 每组返回的最大结果数
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
        pkl_load_workers: PKL加载线程数（全量预加载时使用）
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
    
    # 查找PKL文件
    pkl_files = []
    for f in os.listdir(index_directory):
        if f.endswith('.pkl'):
            pkl_files.append(os.path.join(index_directory, f))
    
    if not pkl_files:
        print(f"❌ 错误: 在 {index_directory} 中没有找到PKL文件")
        return {}
    
    # 从PKL文件名提取模型名称
    model_name = None
    for pkl_file in pkl_files:
        basename = os.path.basename(pkl_file)
        parts = basename.rsplit('_', 1)
        if len(parts) == 2:
            model_name = parts[1].replace('.pkl', '')
            break
    
    if model_name is None:
        model_name = "openai-clip-vit-large-patch14"
    
    # 检测模型类型
    from A_coreUtils.search.auto_scene_search import (
        detect_model_type_from_name,
        SimilarityThresholdConfig,
        export_video_matches,
        cleanup_temp_after_export,
    )
    model_type = detect_model_type_from_name(model_name)
    
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
    
    # 生成或加载缓存
    cache_valid = vector_cache.cache_exists()
    if not cache_valid:
        print("[Cloze Search] Cache missing or invalid, loading CLIP to regenerate vectors...")
        processor = EmbeddingModelProcessor(
            model_name=model_name,
            use_fp16=use_fp16
        )
        vector_cache.processor = processor
        vector_cache.generate_cache(batch_size=prompt_cache_batch_size)
    cache_iterator = vector_cache.load_cache_batched(batch_size=prompt_search_batch_size)
    
    if cache_iterator is None:
        raise RuntimeError("Cloze prompt 缂撳瓨鍚戦噺涓嶅彲鐢紝涓斾笉鍏佽鍥為€€鍒板疄鏃剁紪鐮併€?")
    total_prompts = cache_iterator.total_prompts
    print(f"[选词填空搜索] 共 {total_prompts} 个 prompt")
    
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
        index_paths=pkl_files,
        cache_dir=str(resolver.project_root / 'temp' / 'cache'),
        load_workers=pkl_load_workers,
        use_fp16=feature_fp16,
        pkl_batch_size=pkl_batch_size,
        video_name_format=video_name_format,
        search_mode=search_mode,
        top_k=top_k,
        lmdb_write_batch_size=lmdb_write_batch_size,
        logit_scale=100.0
    )
    
    # 生成搜索配置哈希（用于断点续传验证）
    import hashlib as _hashlib
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
        'pkl_batch_size': pkl_batch_size,
        'use_diskcache': use_diskcache,
        'vector_dedup_threshold': vector_dedup_threshold,
        'use_chinese': use_chinese,
        'video_name_format': video_name_format,
        'cloze_prompt_config_hash': getattr(vector_cache, '_config_hash', None),
        'index_files': sorted([os.path.basename(p) for p in pkl_files]),
    }
    config_str = json.dumps(config_payload, ensure_ascii=False, sort_keys=True)
    config_hash = _hashlib.md5(config_str.encode('utf-8')).hexdigest()
    
    # 预加载PKL特征（根据 pkl_batch_size 决定全量或分批）
    if batch_engine._preload_all:
        print("[选词填空搜索] 全量预加载模式")
    else:
        print(f"[选词填空搜索] 分批PKL模式: 每批 {batch_engine.pkl_batch_size} 个PKL")
    
    # 初始化视频名称解析器
    video_name_parser = VideoNameParser()
    
    # LMDB 缓存目录（不再无条件清空，支持断点续传）
    diskcache_root = None
    search_cache_dir = None
    pkl_merge_cache_dir = None
    if use_diskcache:
        if diskcache_dir is None:
            diskcache_root = str(resolver.project_root / 'temp' / 'cache' / 'cloze_search_results')
        else:
            diskcache_root = diskcache_dir
        search_cache_dir = os.path.join(diskcache_root, 'inner_search')
        pkl_merge_cache_dir = os.path.join(diskcache_root, 'pkl_merge')
        os.makedirs(search_cache_dir, exist_ok=True)
        os.makedirs(pkl_merge_cache_dir, exist_ok=True)
    
    # 执行搜索
    start_time = time.time()
    
    # 创建缓存迭代器（兼容 PromptVectorBatchIterator 的接口）
    class ClozeCacheIterator:
        def __init__(self, vectors, all_metadata, batch_size, device, feature_fp16):
            self.vectors = vectors
            self._all_metadata = all_metadata
            self.batch_size = batch_size
            self.device = device
            self.feature_fp16 = feature_fp16
            self.total_prompts = len(all_metadata)
            self.num_batches = (self.total_prompts + batch_size - 1) // batch_size
            self.current_batch = 0
            # metadata 属性（兼容 search_with_batched_cache 中的 cache_iterator.metadata）
            self._cache_metadata = {}
        
        @property
        def metadata(self):
            return self._cache_metadata
        
        def __iter__(self):
            self.current_batch = 0
            return self
        
        def __next__(self):
            if self.current_batch >= self.num_batches:
                raise StopIteration
            
            start_idx = self.current_batch * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.total_prompts)
            
            batch_vectors_np = self.vectors[start_idx:end_idx]
            batch_vectors_np = np.array(batch_vectors_np)
            batch_vectors = torch.from_numpy(batch_vectors_np).to(self.device)
            if self.feature_fp16 and str(self.device).startswith('cuda'):
                batch_vectors = batch_vectors.half()
            batch_metadata = self._all_metadata[start_idx:end_idx]
            batch_prompts = [m.get('prompt', '') for m in batch_metadata]
            
            batch_info = {
                'batch_idx': self.current_batch,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'total_prompts': self.total_prompts,
                'num_batches': self.num_batches
            }
            
            self.current_batch += 1
            
            return batch_vectors, batch_prompts, batch_metadata, batch_info
        
        def reset(self):
            """重置迭代器"""
            self.current_batch = 0
    
    # cache_iterator 宸查€氳繃 vector_cache.load_cache_batched 鍒涘缓
    
    # 懒加载 Reranker（仅在候选处理阶段首次真正需要时加载）
    reranker_loader = None
    reranker_weight = SimilarityThresholdConfig.RERANKER_WEIGHT if use_reranker else 0.0
    
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
        config_hash=config_hash
    )
    
    if batch_engine._preload_all:
        # 全量预加载模式：直接搜索（search_with_batched_cache 内部已支持断点续传）
        if batch_engine.all_features_gpu is None:
            batch_engine._preload_all_features()
        best_matches = batch_engine.search_with_batched_cache(**search_kwargs)
    else:
        # 分批PKL加载模式（根治版）：
        #   阶段1：每批仅做 CLIP 候选聚合，写入同一个 LMDB
        #   阶段2：全部 PKL 聚合完成后，只做一次候选后处理（含可选 Reranker）
        from A_coreUtils.search.batch_text_search import LMDBCache
        import time as _time

        total_pkls = len(batch_engine.index_paths)
        pkl_bs = batch_engine.pkl_batch_size
        total_batches = (total_pkls + pkl_bs - 1) // pkl_bs

        # 生成分批PKL模式的 config_hash（包含 PKL 批次信息）
        pkl_batch_config_str = f"cloze_pkl_batch|{config_hash}|{total_pkls}|{pkl_bs}"
        pkl_batch_hash = _hashlib.md5(pkl_batch_config_str.encode()).hexdigest()

        # 候选聚合 LMDB（与阶段2统一复用）
        pkl_batch_lmdb_dir = pkl_merge_cache_dir
        if pkl_batch_lmdb_dir is None:
            pkl_batch_lmdb_dir = str(resolver.project_root / 'temp' / 'cache' / 'cloze_search_results')
        os.makedirs(pkl_batch_lmdb_dir, exist_ok=True)

        pkl_batch_lmdb = LMDBCache(pkl_batch_lmdb_dir, map_size=10 * 1024 * 1024 * 1024)

        # 检查断点续传（仅针对阶段1：PKL 批次聚合）
        start_pkl_batch = 0
        checkpoint = pkl_batch_lmdb.load_checkpoint()
        if checkpoint and checkpoint.get('config_hash') == pkl_batch_hash and checkpoint.get('phase') == 'pkl_batch':
            start_pkl_batch = checkpoint.get('last_completed_pkl_batch', -1) + 1
            if start_pkl_batch > 0:
                print(f"[选词填空搜索] 🔄 断点续传: 从 PKL 批次 {start_pkl_batch + 1}/{total_batches} 继续")
        else:
            if checkpoint:
                print(f"[选词填空搜索] 配置已变化，清理当前聚合缓存目录重新搜索: {pkl_batch_lmdb_dir}")
            pkl_batch_lmdb.close()
            temp_dir = str(resolver.project_root / 'temp')
            if os.path.exists(temp_dir):
                from A_coreUtils.video_processing.video_utils import cleanup_temp_folder
                cleanup_temp_folder(temp_dir)
            os.makedirs(pkl_batch_lmdb_dir, exist_ok=True)
            pkl_batch_lmdb = LMDBCache(pkl_batch_lmdb_dir, map_size=10 * 1024 * 1024 * 1024)

        print(f"[选词填空搜索] 分批PKL模式: {total_pkls} 个PKL, 每批 {pkl_bs} 个, 共 {total_batches} 批")
        scene_key_to_pkl_map = {}

        # 阶段1：分批聚合 CLIP 候选（不做每批后处理）
        for batch_idx in range(start_pkl_batch, total_batches):
            batch_start_idx = batch_idx * pkl_bs
            batch_end_idx = min(batch_start_idx + pkl_bs, total_pkls)
            batch_paths = batch_engine.index_paths[batch_start_idx:batch_end_idx]

            print(f"\n[PKL批次 {batch_idx + 1}/{total_batches}] 加载 {len(batch_paths)} 个PKL...")
            batch_engine._load_pkl_batch_to_merged(batch_paths)

            if batch_engine.all_features_gpu is None:
                print(f"[PKL批次 {batch_idx + 1}] 无有效数据，跳过")
                pkl_batch_lmdb.save_checkpoint({
                    'config_hash': pkl_batch_hash,
                    'last_completed_pkl_batch': batch_idx,
                    'total_batches': total_batches,
                    'phase': 'pkl_batch',
                    'timestamp': _time.time()
                })
                batch_engine._unload_merged_features()
                continue

            # 记录 scene_key -> source_pkl 映射（用于最终按PKL分组）
            for scene_info in batch_engine.scene_map:
                video_name = os.path.basename(scene_info['video_path']) if scene_info.get('video_path') else ''
                scene_key = f"{scene_info['start_frame']}_{video_name}"
                scene_key_to_pkl_map[scene_key] = scene_info.get('source_pkl', 'unknown')

            cache_iterator.reset()

            # 仅执行 CLIP 候选聚合：候选写入 LMDB
            clip_batch_kwargs = dict(search_kwargs)
            clip_batch_kwargs['cache_iterator'] = cache_iterator
            clip_batch_kwargs['threshold'] = clip_initial_threshold
            clip_batch_kwargs['use_diskcache'] = True
            clip_batch_kwargs['cache_dir'] = pkl_batch_lmdb_dir
            clip_batch_kwargs['use_reranker'] = False
            clip_batch_kwargs['reranker'] = None
            clip_batch_kwargs['reranker_loader'] = None
            clip_batch_kwargs['candidate_batch_size'] = None
            clip_batch_kwargs['result_top_k'] = None
            clip_batch_kwargs['search_mode'] = 0
            clip_batch_kwargs['config_hash'] = None
            clip_batch_kwargs['append_to_lmdb_cache'] = True
            batch_engine.search_with_batched_cache(**clip_batch_kwargs)

            pkl_batch_lmdb.save_checkpoint({
                'config_hash': pkl_batch_hash,
                'last_completed_pkl_batch': batch_idx,
                'total_batches': total_batches,
                'phase': 'pkl_batch',
                'timestamp': _time.time()
            })
            print(f"[PKL批次 {batch_idx + 1}] 候选聚合完成")

            batch_engine._unload_merged_features()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pkl_batch_lmdb.close()

        # 阶段2：统一执行一次候选后处理（可选 Reranker）
        print("\n[选词填空搜索] 分批PKL候选聚合完成，开始统一后处理...")
        rerank_search_kwargs = dict(search_kwargs)
        rerank_search_kwargs['cache_iterator'] = None
        rerank_search_kwargs['use_diskcache'] = True
        rerank_search_kwargs['cache_dir'] = pkl_batch_lmdb_dir
        rerank_search_kwargs['skip_clip_search'] = True
        rerank_search_kwargs['result_top_k'] = None
        rerank_search_kwargs['search_mode'] = 0
        rerank_search_kwargs['config_hash'] = None
        all_batch_matches = batch_engine.search_with_batched_cache(**rerank_search_kwargs)
        print(f"[选词填空搜索] 统一后处理完成，得到 {len(all_batch_matches)} 个候选结果")

        # 按 search_mode 分组取 Top-K
        from A_coreUtils.search.auto_scene_search import AutoSceneSearcher
        best_matches = AutoSceneSearcher._apply_search_mode_grouping(
            all_batch_matches, search_mode, top_k, batch_engine, scene_key_to_pkl_map
        )
    
    search_time = time.time() - start_time
    print(f"[选词填空搜索] 搜索完成! 耗时 {search_time:.2f}s, 找到 {len(best_matches)} 个结果")
    
    # ========== 向量去重（视频导出前）==========
    scene_features = {}  # {scene_key: np.ndarray}
    scene_pkl_map = {}  # {scene_key: source_pkl}
    if vector_dedup_threshold is not None and best_matches and batch_engine.all_features_gpu is not None:
        print(f"[选词填空搜索] 提取场景特征向量用于去重...")
        
        # 构建 best_matches 中的 scene_key 集合
        best_match_keys = set(best_matches.keys())
        
        all_features_cpu = batch_engine.all_features_gpu.float().cpu().numpy()
        feat_idx = 0
        for scene_idx, scene_info in enumerate(batch_engine.scene_map):
            video_name = os.path.basename(scene_info['video_path']) if scene_info.get('video_path') else ''
            scene_key = f"{scene_info['start_frame']}_{video_name}"
            count = batch_engine.feature_counts[scene_idx]
            
            # 只保存 best_matches 中存在的场景的向量
            if scene_key in best_match_keys:
                # 保留该场景的所有帧向量（通常是3帧）用于去重比较
                scene_vectors = all_features_cpu[feat_idx:feat_idx + count]
                scene_features[scene_key] = scene_vectors  # 保留所有帧向量 [count, D]
                # 记录 source_pkl
                scene_pkl_map[scene_key] = scene_info.get('source_pkl', 'unknown')
            
            feat_idx += count
        print(f"[选词填空搜索] 提取了 {len(scene_features)} 个场景的特征向量（共 {len(best_match_keys)} 个搜索结果）")
    
    # 释放资源
    elif vector_dedup_threshold is not None and best_matches and not batch_engine._preload_all:
        print("[选词填空搜索] 提取场景特征向量用于去重...")
        best_match_keys = set(best_matches.keys())
        pkl_bs = batch_engine.pkl_batch_size
        total_pkls = len(batch_engine.index_paths)
        total_batches = (total_pkls + pkl_bs - 1) // pkl_bs

        for batch_idx in range(total_batches):
            batch_start_idx = batch_idx * pkl_bs
            batch_end_idx = min(batch_start_idx + pkl_bs, total_pkls)
            batch_paths = batch_engine.index_paths[batch_start_idx:batch_end_idx]
            batch_engine._load_pkl_batch_to_merged(batch_paths)

            if batch_engine.all_features_gpu is None:
                batch_engine._unload_merged_features()
                continue

            all_features_cpu = batch_engine.all_features_gpu.float().cpu().numpy()
            feat_idx = 0
            for scene_idx, scene_info in enumerate(batch_engine.scene_map):
                video_name = os.path.basename(scene_info['video_path']) if scene_info.get('video_path') else ''
                scene_key = f"{scene_info['start_frame']}_{video_name}"
                count = batch_engine.feature_counts[scene_idx]

                if scene_key in best_match_keys and scene_key not in scene_features:
                    scene_vectors = all_features_cpu[feat_idx:feat_idx + count]
                    scene_features[scene_key] = scene_vectors
                    scene_pkl_map[scene_key] = scene_info.get('source_pkl', 'unknown')

                feat_idx += count

            batch_engine._unload_merged_features()
            if len(scene_features) >= len(best_match_keys):
                break

        print(f"[选词填空搜索] 提取了 {len(scene_features)} 个场景的特征向量（共 {len(best_match_keys)} 个搜索结果）")

    batch_engine.cleanup()
    del batch_engine
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 向量去重（如果启用）- 同PKL内去重
    if vector_dedup_threshold is not None and scene_features:
        print(f"\n[向量去重] 开始向量去重，阈值={vector_dedup_threshold}，模式=同PKL内去重")
        from .batch_text_search import deduplicate_by_vector_similarity
        best_matches = deduplicate_by_vector_similarity(
            best_matches=best_matches,
            scene_features=scene_features,
            video_name_format=video_name_format,
            similarity_threshold=vector_dedup_threshold,
            scene_pkl_map=scene_pkl_map  # 传递 PKL 映射用于同PKL内去重
        )
    
    # 相邻片段合并（视频导出前）
    if adjacent_merge_frames is not None and adjacent_merge_frames >= 0:
        from A_coreUtils.search.auto_scene_search import merge_adjacent_scenes
        best_matches = merge_adjacent_scenes(
            best_matches=best_matches,
            adjacent_merge_frames=adjacent_merge_frames,
            video_name_format=video_name_format
        )
    
    # 视频导出（与 Prompt 模式一致）
    if best_matches:
        print(f"\n📹 视频导出: {len(best_matches)} 个场景")
        
        export_stats, video_output_directory = export_video_matches(
            best_matches=best_matches,
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
    
    return best_matches


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
