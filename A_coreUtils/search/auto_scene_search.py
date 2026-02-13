# auto_scene_search.py
# 自动化场景搜索器 - 遍历所有合理的关键词组合进行文搜图
# 基于 logic_Prompt.py 的逻辑，系统性遍历所有组合
# v1.0: 初始版本
# v1.1: 添加去重功能 - 同一场景保留相似度最高的prompt结果

import json
import os
import sys
import time
import itertools
from typing import List, Dict, Generator, Optional, Callable, Tuple
from datetime import datetime

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
from path_resolver import PathResolver, get_project_root, resolve_path

# 导入视频名称解析器
from A_coreUtils.video_processing.video_name_parser import VideoNameParser

# 延迟导入嵌入模型处理器（避免在只需要PromptGenerator时加载重型依赖）
EmbeddingModelProcessor = None
BatchTextSearchEngine = None
_BATCH_SEARCH_AVAILABLE = False

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
    """清理文件名中的非法字符"""
    if not name:
        return ""
    illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in illegal_chars:
        name = name.replace(char, '_')
    return name.strip()


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


def validate_prompt_template(prompt_template: str, categories: List[str]) -> Tuple[bool, str]:
    """
    验证 prompt_template 是否合法
    
    检查模板字符串中的占位符是否都是有效的大类名称。
    
    Args:
        prompt_template: prompt 模板字符串
        categories: 有效的大类名称列表（如 ['主体', '动作', '场景', '情绪']）
    
    Returns:
        (是否合法, 错误信息)
    """
    import re
    
    # 提取所有占位符
    placeholders = re.findall(r'\{(\w+)\}', prompt_template)
    
    # 检查每个占位符
    invalid_placeholders = []
    for ph in placeholders:
        if ph not in categories:
            invalid_placeholders.append(ph)
    
    if invalid_placeholders:
        return False, f"prompt_template 中存在无效占位符: {invalid_placeholders}，有效占位符: {list(categories)}"
    
    return True, ""


def generate_default_prompt_video_name_format(categories: List[str], prefix: str = "") -> str:
    """
    动态生成 Prompt 模式的默认 video_name_format
    
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
    RESERVED_KEYS = {"分配规则", "选词填空规则"}
    
    def __init__(self, keywords_path: str = None, prompt_template: str = None, use_chinese: bool = False):
        """
        初始化Prompt生成器
        
        Args:
            keywords_path: logic_keywords.json的路径，None则使用默认路径
            prompt_template: 自定义prompt模板，None则根据JSON中的大类动态生成
                支持占位符: JSON中定义的任何大类（使用中文键名）
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
        
        # 从 "PL标签" 下提取数据（必须存在）
        if "PL标签" not in full_data:
            raise KeyError("logic_keywords.json 中必须包含 'PL标签' 键")
        self.data = full_data["PL标签"]
        
        # 从JSON动态获取第一个大类的子类列表（用于分配规则）
        first_category = self._get_first_category_with_subcategories()
        self._category_list = list(self.data.get(first_category, {}).keys()) if first_category else []
        
        # 从JSON读取分配规则（必须在JSON中定义，没有默认规则）
        # 过滤掉以 _ 开头的说明字段
        raw_rules = self.data.get("分配规则", {})
        self._allocation_rules = {k: v for k, v in raw_rules.items() if not k.startswith("_")}
        
        # 预计算每个类别的动作池（基于分配规则）
        self._action_pools = self._build_action_pools()
        
        # 动态发现JSON中的所有大类
        self._simple_categories = self._discover_simple_categories()
        
        # 设置prompt模板（如果未提供，则根据JSON中的大类动态生成）
        if prompt_template is None:
            self._prompt_template = self._generate_default_template()
        else:
            self._prompt_template = prompt_template
    
    def _get_first_category_with_subcategories(self) -> Optional[str]:
        """获取第一个有子类别的大类名称（用于分配规则）"""
        for key, value in self.data.items():
            if key in self.RESERVED_KEYS or key.startswith("_"):
                continue
            if isinstance(value, dict) and value:
                first_value = next(iter(value.values()), None)
                if isinstance(first_value, dict):
                    return key
        return None
    
    @property
    def CATEGORY_LIST(self) -> List[str]:
        """动态获取主体类别列表（从JSON读取）"""
        return self._category_list
    
    def _build_action_pools(self) -> Dict[str, Dict[str, str]]:
        """
        构建每个主体类别对应的动作池
        
        基于通用分配规则，从 分配规则.主体.动作 中读取规则。
        支持：
        1. 新增主体类别自动适配
        2. 通过JSON中的"分配规则"自定义规则
        """
        action_pools = {}
        action_data = self.data.get("动作", {})
        
        # 获取 主体->动作 的分配规则
        subject_action_rules = self._allocation_rules.get("主体", {}).get("动作", {})
        
        for category in self._category_list:
            # 获取该类别的动作规则
            action_subcategories = subject_action_rules.get(category)
            
            if action_subcategories is None:
                # 未定义规则的新类别，默认使用所有动作
                action_subcategories = list(action_data.keys())
            
            # 合并所有指定的动作子类别
            pool = {}
            for subcat in action_subcategories:
                if subcat in action_data:
                    pool.update(action_data[subcat])
            
            # 如果池为空，使用所有动作作为后备
            if not pool:
                for subcat_data in action_data.values():
                    pool.update(subcat_data)
            
            action_pools[category] = pool
        
        return action_pools
    
    def _discover_simple_categories(self) -> Dict[str, Dict[str, str]]:
        """
        动态发现JSON中的所有大类
        
        大类是指除了保留字段（分配规则）之外的所有键。
        这些大类会被用于生成默认模板和遍历组合。
        例如：主体、动作、场景、情绪等。
        
        Returns:
            Dict[str, Dict[str, str]]: {大类名: {中文: 英文, ...}}
        """
        simple_categories = {}
        
        for key, value in self.data.items():
            # 跳过保留字段和以 _ 开头的说明字段
            if key in self.RESERVED_KEYS or key.startswith("_"):
                continue
            
            # 检查是否是有效的大类（包含子类别或直接是 {中文: 英文} 映射）
            if isinstance(value, dict) and value:
                # 检查是否是嵌套结构（有子类别）
                first_value = next(iter(value.values()), None)
                if isinstance(first_value, dict):
                    # 嵌套结构，合并所有子类别
                    merged = {}
                    for subcat_name, subcat_data in value.items():
                        if isinstance(subcat_data, dict) and not subcat_name.startswith("_"):
                            merged.update(subcat_data)
                    if merged:
                        simple_categories[key] = merged
                elif isinstance(first_value, str):
                    # 直接是 {中文: 英文} 映射
                    simple_categories[key] = value
        
        return simple_categories
    
    def _analyze_allocation_dependencies(self) -> Dict[str, List[str]]:
        """
        分析分配规则中的依赖关系
        
        Returns:
            Dict[目标大类, List[源大类]]: 每个目标大类依赖哪些源大类
        """
        dependencies = {}
        
        for source_cat, target_rules in self._allocation_rules.items():
            if not isinstance(target_rules, dict):
                continue
            
            for target_cat in target_rules.keys():
                if target_cat.startswith("_"):
                    continue
                if target_cat not in dependencies:
                    dependencies[target_cat] = []
                dependencies[target_cat].append(source_cat)
        
        return dependencies
    
    def _get_excluded_categories(self) -> set:
        """
        获取需要从简单大类遍历中排除的大类
        
        排除：
        1. 第一个有子类别的大类（如"主体"）- 通过 get_subjects() 单独处理
        2. 分配规则中作为目标的大类（如"动作"）- 通过 get_actions() 单独处理
        
        这些大类在 iterate_all_combinations() 中已经通过专门的方法处理，
        不应该再次包含在简单大类的笛卡尔积遍历中。
        
        Returns:
            需要排除的大类名称集合
        """
        excluded = set()
        
        # 1. 第一个有子类别的大类（如"主体"）
        first_category = self._get_first_category_with_subcategories()
        if first_category:
            excluded.add(first_category)
        
        # 2. 分配规则中作为目标的大类（如"动作"）
        for source_cat, target_rules in self._allocation_rules.items():
            if isinstance(target_rules, dict):
                for target_cat in target_rules.keys():
                    if not target_cat.startswith("_"):
                        excluded.add(target_cat)
        
        return excluded
    
    def _generate_default_template(self) -> str:
        """
        根据JSON中实际存在的大类动态生成默认模板
        
        模板格式: "A {情绪} photo of a {主体} {动作} in {场景}"
        如果某个大类不存在，则不包含在模板中。
        
        Returns:
            动态生成的模板字符串
        """
        # 基础模板部分
        parts = ["A"]
        
        # 检查情绪是否存在
        if "情绪" in self._simple_categories:
            parts.append("{情绪}")
        
        # 检查镜头是否存在
        if "镜头" in self._simple_categories:
            parts.append("{镜头}")
        
        parts.append("photo of a {主体} {动作}")
        
        # 检查场景是否存在
        if "场景" in self._simple_categories:
            parts.append("in {场景}")
        
        # 添加其他简单大类（排除已处理的）
        handled = {"情绪", "镜头", "场景"}
        for cat_name in self._simple_categories:
            if cat_name not in handled:
                parts.append(f"with {{{cat_name}}}")
        
        return " ".join(parts)
    
    def get_allocated_values_for_category(self, target_cat: str,
                                          source_values: Dict[str, str]) -> Dict[str, str]:
        """
        根据分配规则获取目标大类的允许值
        
        Args:
            target_cat: 目标大类名称
            source_values: {源大类名: 当前源子类名} 的映射
        
        Returns:
            Dict[str, str]: {中文: 英文, ...} 允许的目标值
        """
        # 检查是否有针对该目标大类的分配规则
        target_data = self.data.get(target_cat, {})
        
        # 收集所有允许的子类别
        allowed_subcats = set()
        has_rule = False
        
        for source_cat, source_subcat in source_values.items():
            if source_cat in self._allocation_rules:
                target_rules = self._allocation_rules[source_cat].get(target_cat, {})
                if target_rules and source_subcat in target_rules:
                    has_rule = True
                    allowed_subcats.update(target_rules[source_subcat])
        
        if not has_rule:
            # 没有规则，返回目标大类的所有值
            return self._simple_categories.get(target_cat, {})
        
        # 根据允许的子类别合并值
        result = {}
        for subcat_name in allowed_subcats:
            if subcat_name in target_data:
                subcat_data = target_data[subcat_name]
                if isinstance(subcat_data, dict):
                    result.update(subcat_data)
        
        # 如果结果为空，返回所有值作为后备
        if not result:
            return self._simple_categories.get(target_cat, {})
        
        return result
    
    @property
    def prompt_template(self) -> str:
        """获取当前prompt模板"""
        return self._prompt_template
    
    @prompt_template.setter
    def prompt_template(self, template: str):
        """设置prompt模板"""
        self._prompt_template = template or self._generate_default_template()
    
    def get_subjects(self, category: str) -> Dict[str, str]:
        """获取指定类别的所有主体（完全动态，从JSON读取）"""
        subjects_data = self.data.get("主体", {})
        if category in subjects_data:
            return subjects_data[category]
        # 回退到第一个可用类别
        if subjects_data:
            first_category = list(subjects_data.keys())[0]
            return subjects_data[first_category]
        return {}
    
    def get_actions(self, category: str) -> Dict[str, str]:
        """获取指定类别的所有合法动作（完全动态）"""
        if category in self._action_pools:
            return self._action_pools[category]
        # 回退到第一个可用类别
        if self._action_pools:
            first_category = list(self._action_pools.keys())[0]
            return self._action_pools[first_category]
        return {}
    
    def get_scenes(self) -> Dict[str, str]:
        """获取所有场景（动态获取第一个场景子类别）"""
        scenes_data = self.data.get("场景", {})
        # 尝试获取"所有场景"，否则获取第一个子类别
        if "所有场景" in scenes_data:
            return scenes_data["所有场景"]
        if scenes_data:
            first_key = list(scenes_data.keys())[0]
            return scenes_data[first_key]
        return {}
    
    def get_moods(self) -> Dict[str, str]:
        """获取所有情绪（动态获取第一个情绪子类别）"""
        moods_data = self.data.get("情绪", {})
        if "所有情绪" in moods_data:
            return moods_data["所有情绪"]
        if moods_data:
            first_key = list(moods_data.keys())[0]
            return moods_data[first_key]
        return {}
    
    def get_lenses(self) -> Dict[str, str]:
        """获取所有镜头（动态获取第一个镜头子类别）"""
        lenses_data = self.data.get("镜头", {})
        if "所有镜头" in lenses_data:
            return lenses_data["所有镜头"]
        if lenses_data:
            first_key = list(lenses_data.keys())[0]
            return lenses_data[first_key]
        return {}
    
    def reload_keywords(self, keywords_path: str = None, prompt_template: str = None):
        """
        重新加载关键词文件（支持热更新）
        
        Args:
            keywords_path: 新的关键词文件路径，None则使用原路径
            prompt_template: 新的prompt模板，None则保持原模板
        """
        if keywords_path is None:
            resolver = PathResolver()
            keywords_path = str(resolver.project_root / 'logic_keywords.json')
        
        with open(keywords_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        
        # 从 "PL标签" 下提取数据（必须存在）
        if "PL标签" not in full_data:
            raise KeyError("logic_keywords.json 中必须包含 'PL标签' 键")
        self.data = full_data["PL标签"]
        
        # 重新计算
        first_category = self._get_first_category_with_subcategories()
        self._category_list = list(self.data.get(first_category, {}).keys()) if first_category else []
        raw_rules = self.data.get("分配规则", {})
        self._allocation_rules = {k: v for k, v in raw_rules.items() if not k.startswith("_")}
        self._action_pools = self._build_action_pools()
        self._simple_categories = self._discover_simple_categories()
        
        # 更新模板（如果提供）
        if prompt_template is not None:
            self._prompt_template = prompt_template
    
    def iterate_all_combinations(self,
                                  categories: List[str] = None) -> Generator[Dict, None, None]:
        """
        遍历所有合理的组合
        
        动态支持JSON中定义的所有大类（主体、动作 + 简单大类如场景、情绪、镜头等）。
        大类会根据JSON文件动态检测，无需硬编码。
        支持分配规则：任意大类之间的约束关系。
        
        Args:
            categories: 要遍历的主体类别列表，None则遍历所有类别
        
        Yields:
            包含组合信息的字典，包括:
            - prompt: 生成的prompt
            - category: 主体类别
            - subject_cn/en, action_cn/en: 必需的核心大类
            - {simple_category}_cn/en: 简单大类（如场景、情绪、镜头等，根据JSON动态生成）
        """
        if categories is None:
            categories = self.CATEGORY_LIST
        
        # 分析分配规则依赖关系
        dependencies = self._analyze_allocation_dependencies()
        
        # 获取需要排除的大类（主体、动作等已单独处理的大类）
        excluded_cats = self._get_excluded_categories()
        
        # 获取简单大类名称（排除已单独处理的大类）
        simple_cat_names = [cat for cat in self._simple_categories.keys() if cat not in excluded_cats]
        
        # 分离独立大类和依赖大类
        # 独立大类：不依赖其他大类的（不在 dependencies 中作为目标）
        # 依赖大类：依赖其他大类的（在 dependencies 中作为目标）
        independent_cats = [cat for cat in simple_cat_names if cat not in dependencies]
        dependent_cats = [cat for cat in simple_cat_names if cat in dependencies]
        
        for category in categories:
            subjects = self.get_subjects(category)
            actions = self.get_actions(category)
            
            # 如果没有主体或动作，跳过
            if not subjects or not actions:
                continue
            
            # 遍历主体和动作的组合
            for (subj_cn, subj_en) in subjects.items():
                for (act_cn, act_en) in actions.items():
                    # 构建当前源值（用于分配规则查询）
                    # 注意：主体的子类是 category（如"生物"、"载具"等）
                    source_values = {"主体": category}
                    
                    # 遍历独立大类的组合
                    for independent_combo in self._iterate_categories_product(independent_cats):
                        # 更新源值（独立大类的值可能影响依赖大类）
                        current_source_values = source_values.copy()
                        for cat_name, (cat_cn, cat_en) in independent_combo.items():
                            current_source_values[cat_name] = cat_cn
                        
                        # 遍历依赖大类的组合（基于分配规则）
                        for dependent_combo in self._iterate_dependent_categories(
                            dependent_cats, current_source_values
                        ):
                            # 合并所有简单大类的值
                            simple_combo = {**independent_combo, **dependent_combo}
                            
                            # 根据 use_chinese 选择使用中文或英文标签
                            if self.use_chinese:
                                # 中文模式：使用中文标签键名
                                prompt_kwargs = {cat_name: values[0] for cat_name, values in simple_combo.items()}
                                prompt = self._generate_prompt_dynamic(
                                    subj_cn, act_cn, **prompt_kwargs
                                )
                            else:
                                # 英文模式：使用英文标签值
                                prompt_kwargs = {cat_name: values[1] for cat_name, values in simple_combo.items()}
                                prompt = self._generate_prompt_dynamic(
                                    subj_en, act_en, **prompt_kwargs
                                )
                            
                            # 构建结果字典
                            result = {
                                "prompt": prompt,
                                "category": category,
                                "subject_cn": subj_cn,
                                "subject_en": subj_en,
                                "action_cn": act_cn,
                                "action_en": act_en,
                            }
                            
                            # 添加简单大类到结果（向后兼容：同时添加旧字段名）
                            for cat_name, (cat_cn, cat_en) in simple_combo.items():
                                result[f"{cat_name}_cn"] = cat_cn
                                result[f"{cat_name}_en"] = cat_en
                                # 向后兼容：为常见大类添加英文字段名
                                if cat_name == "场景":
                                    result["scene_cn"] = cat_cn
                                    result["scene_en"] = cat_en
                                elif cat_name == "情绪":
                                    result["mood_cn"] = cat_cn
                                    result["mood_en"] = cat_en
                                elif cat_name == "镜头":
                                    result["lens_cn"] = cat_cn
                                    result["lens_en"] = cat_en
                            
                            yield result
                            
    
    def _iterate_categories_product(self,
                                    cat_names: List[str]) -> Generator[Dict[str, tuple], None, None]:
        """
        遍历指定大类的笛卡尔积
        
        Args:
            cat_names: 大类名称列表
        
        Yields:
            Dict[大类名, (中文, 英文)]
        """
        if not cat_names:
            yield {}
            return
        
        # 构建每个大类的值列表
        all_values = []
        for cat_name in cat_names:
            values_dict = self._simple_categories.get(cat_name, {})
            if values_dict:
                cat_values = [(cat_name, cn, en) for cn, en in values_dict.items()]
                all_values.append(cat_values)
            else:
                # 空大类，跳过
                all_values.append([(cat_name, "", "")])
        
        # 使用笛卡尔积遍历所有组合
        for combo in itertools.product(*all_values):
            result = {}
            for cat_name, cn, en in combo:
                if cn or en:  # 跳过空值
                    result[cat_name] = (cn, en)
            yield result
    
    def _iterate_dependent_categories(self,
                                      cat_names: List[str],
                                      source_values: Dict[str, str]) -> Generator[Dict[str, tuple], None, None]:
        """
        遍历依赖大类的组合（基于分配规则）
        
        Args:
            cat_names: 依赖大类名称列表
            source_values: {源大类名: 当前源子类名} 的映射
        
        Yields:
            Dict[大类名, (中文, 英文)]
        """
        if not cat_names:
            yield {}
            return
        
        # 构建每个依赖大类的允许值列表
        all_values = []
        for cat_name in cat_names:
            # 根据分配规则获取允许的值
            allowed_values = self.get_allocated_values_for_category(cat_name, source_values)
            if allowed_values:
                cat_values = [(cat_name, cn, en) for cn, en in allowed_values.items()]
                all_values.append(cat_values)
            else:
                # 没有允许的值，使用空值
                all_values.append([(cat_name, "", "")])
        
        # 使用笛卡尔积遍历所有组合
        for combo in itertools.product(*all_values):
            result = {}
            for cat_name, cn, en in combo:
                if cn or en:  # 跳过空值
                    result[cat_name] = (cn, en)
            yield result
    
    def _generate_prompt_dynamic(self, subject: str, action: str, **simple_values) -> str:
        """
        动态生成prompt（支持任意简单大类）
        
        Args:
            subject: 主体（英文）
            action: 动作（英文）
            **simple_values: 简单大类的值，键为大类名（中文），值为英文
        
        Returns:
            生成的prompt字符串
        """
        # 构建替换字典（使用中文键名）
        values = {
            "主体": subject,
            "动作": action,
        }
        # 添加简单大类的值
        values.update(simple_values)
        
        # 使用模板格式化
        try:
            return self._prompt_template.format(**values)
        except KeyError as e:
            # 如果模板中有未提供的占位符，尝试用空字符串替换
            print(f"[Warning] 模板占位符 {e} 未提供值，使用空字符串")
            # 找出模板中的所有占位符
            import re
            placeholders = re.findall(r'\{(\w+)\}', self._prompt_template)
            for ph in placeholders:
                if ph not in values:
                    values[ph] = ""
            return self._prompt_template.format(**values)
    
    def count_total_combinations(self, categories: List[str] = None) -> int:
        """
        计算总组合数
        
        动态计算主体、动作和所有简单大类的组合数。
        排除已通过 get_subjects() 和 get_actions() 单独处理的大类。
        """
        if categories is None:
            categories = self.CATEGORY_LIST
        
        # 获取需要排除的大类（主体、动作等已单独处理的大类）
        excluded_cats = self._get_excluded_categories()
        
        # 计算简单大类的组合数（排除已单独处理的大类）
        simple_count = 1
        for cat_name, values in self._simple_categories.items():
            if cat_name not in excluded_cats and values:
                simple_count *= len(values)
        
        total = 0
        for category in categories:
            subjects_count = len(self.get_subjects(category))
            actions_count = len(self.get_actions(category))
            total += subjects_count * actions_count * simple_count
        
        return total
    
    def iterate_prompts_only(self, categories: List[str] = None) -> Generator[str, None, None]:
        """
        轻量版遍历：只生成 prompt 字符串，不创建完整字典
        
        用于保存 prompts.pkl 时避免内存爆炸。
        与 iterate_all_combinations() 生成相同顺序的 prompt，但不创建 metadata 字典。
        
        Args:
            categories: 要遍历的主体类别列表，None则遍历所有类别
        
        Yields:
            prompt 字符串
        """
        if categories is None:
            categories = self.CATEGORY_LIST
        
        # 分析分配规则依赖关系
        dependencies = self._analyze_allocation_dependencies()
        
        # 获取需要排除的大类（主体、动作等已单独处理的大类）
        excluded_cats = self._get_excluded_categories()
        
        # 获取简单大类名称（排除已单独处理的大类）
        simple_cat_names = [cat for cat in self._simple_categories.keys() if cat not in excluded_cats]
        
        # 分离独立大类和依赖大类
        independent_cats = [cat for cat in simple_cat_names if cat not in dependencies]
        dependent_cats = [cat for cat in simple_cat_names if cat in dependencies]
        
        for category in categories:
            subjects = self.get_subjects(category)
            actions = self.get_actions(category)
            
            if not subjects or not actions:
                continue
            
            for (subj_cn, subj_en) in subjects.items():
                for (act_cn, act_en) in actions.items():
                    source_values = {"主体": category}
                    
                    for independent_combo in self._iterate_categories_product(independent_cats):
                        current_source_values = source_values.copy()
                        for cat_name, (cat_cn, cat_en) in independent_combo.items():
                            current_source_values[cat_name] = cat_cn
                        
                        for dependent_combo in self._iterate_dependent_categories(
                            dependent_cats, current_source_values
                        ):
                            simple_combo = {**independent_combo, **dependent_combo}
                            
                            # 根据 use_chinese 选择使用中文或英文标签
                            if self.use_chinese:
                                prompt_kwargs = {cat_name: values[0] for cat_name, values in simple_combo.items()}
                                prompt = self._generate_prompt_dynamic(subj_cn, act_cn, **prompt_kwargs)
                            else:
                                prompt_kwargs = {cat_name: values[1] for cat_name, values in simple_combo.items()}
                                prompt = self._generate_prompt_dynamic(subj_en, act_en, **prompt_kwargs)
                            
                            yield prompt
                            


def extract_model_name_from_pkl(pkl_path: str) -> str:
    """
    从PKL文件名中提取模型名
    
    支持格式:
    1. {source}_{model_name}{_dXXX}_{mode}.pkl
    2. {source}_{model_name}.pkl
    
    示例:
    - "Judas Anne Shirley_openai-clip-vit-large-patch14.pkl" -> "openai-clip-vit-large-patch14"
    - "Video_qihoo360_fg-clip2-base_d512.pkl" -> "qihoo360_fg-clip2-base"
    """
    import re
    filename = os.path.basename(pkl_path)
    basename = filename.replace('.pkl', '')
    
    remaining = basename
    
    # 尝试匹配模式后缀
    mode_match = re.search(r'_(Single|Triplet|TripletAvg|SceneNative3f|SceneNative|SceneDetect|Video)$', basename)
    if mode_match:
        remaining = basename[:mode_match.start()]
    
    # 匹配可选的维度后缀
    dim_match = re.search(r'_d\d+$', remaining)
    if dim_match:
        remaining = remaining[:dim_match.start()]
    
    # 找到第一个下划线后的部分作为模型名
    first_underscore = remaining.find('_')
    if first_underscore == -1:
        return None
    
    return remaining[first_underscore + 1:]


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
    
    # 默认视频名称格式模板
    # 支持的占位符：{镜头}, {情绪}, {场景}, {主体}, {动作}, {起始帧}, {视频解析名}, {扩展名}
    # 以及 logic_keywords.json 中定义的任何扩展大类（使用中文键名）
    DEFAULT_VIDEO_NAME_FORMAT = "{镜头}_{情绪}_{场景}_{主体}_{动作}_{起始帧}_{视频解析名}"
    
    def __init__(self,
                 io_workers: int = 8,
                 use_fp16: bool = False,
                 cache_dir: str = None,
                 prompt_template: str = None,
                 video_name_format: str = None,
                 use_chinese: bool = False):
        """
        初始化自动化搜索器（延迟加载模型）
        
        Args:
            io_workers: I/O工作线程数
            use_fp16: 是否使用FP16精度
            cache_dir: 缓存目录
            prompt_template: 自定义prompt模板，支持占位符: {情绪}, {镜头}, {主体}, {动作}, {场景}
                以及JSON中定义的任何扩展大类（使用中文键名）。默认: "A {情绪} {镜头} of a {主体} {动作} in {场景}"
            video_name_format: 自定义视频名称格式模板，支持占位符: {镜头}, {情绪}, {场景}, {主体}, {动作}, {起始帧}, {视频解析名}
                以及JSON中定义的任何扩展大类（使用中文键名）。默认: "{镜头}_{情绪}_{场景}_{主体}_{动作}_{起始帧}_{视频解析名}"
            use_chinese: 是否使用中文模式
                - False（默认）: 使用英文标签值生成prompt
                - True: 使用中文标签键名生成prompt
        """
        # 初始化路径解析器（不传参数，使用 path_resolver.py 所在目录作为项目根目录）
        self.resolver = PathResolver()
        
        # 保存中文模式设置
        self.use_chinese = use_chinese
        
        # 初始化Prompt生成器（传递 use_chinese 参数）
        self.prompt_generator = PromptGenerator(prompt_template=prompt_template, use_chinese=use_chinese)
        
        # 保存配置，延迟加载模型
        self.io_workers = io_workers
        self.use_fp16 = use_fp16
        self.truncate_dim = None  # 文搜图不需要截断维度
        
        # 视频名称格式模板
        self.video_name_format = video_name_format or self.DEFAULT_VIDEO_NAME_FORMAT
        
        # 缓存目录
        if cache_dir is None:
            cache_dir = str(self.resolver.project_root / 'temp' / 'cache')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 模型相关（延迟初始化）
        self.processor = None
    
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
                if filename.endswith('.pkl'):
                    index_files.append(os.path.join(index_dir, filename))
        
        return index_files
    
    def run_batch_search_optimized(self,
                                    index_paths: List[str] = None,
                                    similarity_threshold: float = 20.0,
                                    use_reranker: bool = False,
                                    rerank_top_k: int = 50,
                                    rerank_batch_size: int = 4,
                                    reranker_output_resolution: str = '384',
                                    candidate_batch_size: int = None,
                                    prompt_search_batch_size: int = 1024,
                                    feature_fp16: bool = True,
                                    pkl_batch_size: int = None,
                                    use_diskcache: bool = True,
                                    diskcache_dir: str = None,
                                    search_mode: int = 0,
                                    top_k: int = 50,
                                    prompt_cache_batch_size: int = 512,
                                    pkl_load_workers: int = 4,
                                    lmdb_write_batch_size: int = 1000,
                                    vector_dedup_threshold: float = None) -> Dict[str, Dict]:
        """
        优化版批量搜索（预加载 + 批量计算 + 批量Rerank + LMDB）
        
        优化点：
        1. PKL特征预加载到GPU，常驻显存
        2. 批量加载prompt向量（prompt_search_batch_size）
        3. GPU批量矩阵计算
        4. 批量Rerank（先收集所有结果，再统一重排序）
        5. 使用 LMDB 批量事务写入磁盘，避免内存溢出
        6. 支持按PKL/视频/跨PKL独立搜索模式
        7. 自动缓存管理（哈希验证，配置变化时自动重新生成）
        
        Args:
            index_paths: 索引文件路径列表，None则自动查找
            similarity_threshold: 相似度阈值
            use_reranker: 是否使用 Reranker 模式
            rerank_top_k: Reranker Top-K
            rerank_batch_size: Reranker 批处理大小
            reranker_output_resolution: Reranker 帧输出分辨率（短边像素数，推荐 384 或 512）
            prompt_search_batch_size: Prompt搜索矩阵运算批次大小
                - 搜索时每批加载的prompt向量数量
                - 推荐 1024-4096
            feature_fp16: 是否使用FP16存储特征（显存减半）
            pkl_batch_size: 每批加载的PKL数量
                - None 或 >= PKL总数: 一次性全部加载到GPU（显存占用高但搜索快）
                - < PKL总数: 分批加载，用完释放（显存占用低但稍慢）
            use_diskcache: 是否使用 LMDB 存储搜索结果（解决内存问题）
            diskcache_dir: LMDB 缓存目录，None则使用默认目录
            search_mode: 搜索模式选择
                - -1（按视频模式）: 每个视频独立搜索，每个视频返回 top_k 个结果
                - 0（按PKL模式）: 每个PKL独立搜索，每个PKL返回 top_k 个结果
                - 1（跨PKL模式）: 全局搜索，返回全局 top_k 个结果
            top_k: 每组返回的最大结果数，None则不限制
            prompt_cache_batch_size: 生成prompt向量缓存时的批处理大小
            pkl_load_workers: PKL加载线程数（全量预加载时使用）
            lmdb_write_batch_size: LMDB单事务写入批大小（分批加载时使用）
            vector_dedup_threshold: 向量去重余弦相似度阈值
                - None（默认）: 不进行向量去重
                - 0.90 ~ 0.98: 超过此阈值的同标签视频只保留一个
        
        Returns:
            去重后的最佳匹配字典 {scene_key: {result_name, similarity, video_path, start_frame, end_frame}}
        """
        # 查找索引文件
        if index_paths is None:
            index_paths = self.find_index_files()
        
        if not index_paths:
            print("[Error] 未找到索引文件")
            return {}
        
        print(f"[优化搜索] 找到 {len(index_paths)} 个索引文件")
        for path in index_paths:
            print(f"  - {os.path.basename(path)}")
        
        # 从pkl文件名提取模型名并验证一致性
        model_names = set()
        for index_path in index_paths:
            model_name = extract_model_name_from_pkl(index_path)
            if model_name:
                model_names.add(model_name)
            else:
                model_names.add("unknown")
        
        if len(model_names) > 1:
            print(f"[Error] 多索引搜索要求所有索引来自相同模型，但检测到 {len(model_names)} 个不同模型")
            return {}
        
        model_name = list(model_names)[0] if model_names else None
        if not model_name or model_name == "unknown":
            print("[Error] 无法从索引文件名中提取模型名")
            return {}
        
        print(f"[优化搜索] 检测到模型: {model_name}")
        
        # 延迟导入批量搜索引擎
        _lazy_import_batch_search()
        _lazy_import_embedding()
        
        if not _BATCH_SEARCH_AVAILABLE:
            raise RuntimeError(
                "批量搜索引擎不可用。\n"
                "请检查 batch_text_search.py 是否存在且可导入。"
            )
        
        # 检测模型类型
        model_type = detect_model_type_from_name(model_name)
        
        # 创建处理器
        processor = None
        
        # 创建批量搜索引擎（预加载PKL特征）
        batch_engine = None
        
        # 检查是否存在prompt向量缓存，并使用分批加载模式
        cache_iterator = None
        
        try:
            from A_coreUtils.prompt.prompt_vector_cache import PromptVectorCache
            
            # 创建缓存管理器（带processor，自动检测并生成缓存）
            # generate_cache() 会自动检测：
            # - 缓存文件是否存在
            # - 哈希是否匹配（prompt_template + keywords + use_chinese 变化时自动重新生成）
            # - 向量数量是否匹配
            prompt_cache = PromptVectorCache(
                processor=None,
                prompt_template=self.prompt_generator.prompt_template,
                batch_size=prompt_cache_batch_size,
                use_chinese=self.use_chinese
            )
            
            # 自动生成或使用现有缓存
            cache_valid = prompt_cache.cache_exists(model_name=model_name)
            if not cache_valid:
                print("[Batch Search] Prompt cache missing or invalid, loading CLIP to regenerate...")
            
            if not cache_valid:
                processor = EmbeddingModelProcessor(
                    model_name=model_name,
                    model_type=model_type,
                    truncate_dim=self.truncate_dim,
                    io_workers=self.io_workers,
                    use_fp16=self.use_fp16
                )
            batch_engine = BatchTextSearchEngine(
                processor=None,
                index_paths=index_paths,
                cache_dir=self.cache_dir,
                load_workers=pkl_load_workers,
                use_fp16=feature_fp16,
                pkl_batch_size=pkl_batch_size,
                video_name_format=self.video_name_format,
                search_mode=search_mode,
                top_k=top_k,
                lmdb_write_batch_size=lmdb_write_batch_size,
                truncate_dim=self.truncate_dim,
                logit_scale=100.0
            )
            
            if not cache_valid:
                prompt_cache.processor = processor
                prompt_cache.generate_cache()
            
            # 获取缓存信息
            cache_info = prompt_cache.get_cache_info(model_name=model_name)
            cached_total = cache_info['total_prompts']
            
            print(f"[优化搜索] 缓存信息: {cached_total} 个prompt, {cache_info['file_size_mb']:.1f} MB")
            
            # 使用分批加载模式（内存友好）
            # 每批加载 prompt_search_batch_size 个 prompt 向量
            cache_iterator = prompt_cache.load_cache_batched(
                model_name=model_name,
                batch_size=prompt_search_batch_size
            )
            print(f"[优化搜索] 使用分批加载模式: {cache_iterator.num_batches} 批, 每批 {prompt_search_batch_size} 个")
            
        except ImportError as e:
            raise RuntimeError(f"prompt向量缓存模块不可用: {e}")
        except RuntimeError:
            # 重新抛出 RuntimeError（来自缓存生成失败）
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"检查/生成prompt缓存时出错: {e}")
        
        # 初始化视频名称解析器（用于实时去重）
        video_name_parser = VideoNameParser()
        
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
            'feature_fp16': feature_fp16,
            'pkl_batch_size': pkl_batch_size,
            'use_diskcache': use_diskcache,
            'vector_dedup_threshold': vector_dedup_threshold,
            'prompt_template': self.prompt_generator.prompt_template,
            'use_chinese': self.use_chinese,
            'index_files': sorted([os.path.basename(p) for p in index_paths]),
        }
        config_str = json.dumps(config_payload, ensure_ascii=False, sort_keys=True)
        config_hash = _hashlib.md5(config_str.encode('utf-8')).hexdigest()
        
        # LMDB 缓存目录（内外阶段使用独立子目录，避免互相覆盖）
        diskcache_root = None
        search_cache_dir = None
        pkl_merge_cache_dir = None
        if use_diskcache:
            if diskcache_dir is None:
                diskcache_root = str(self.resolver.project_root / 'temp' / 'cache' / 'search_results')
            else:
                diskcache_root = diskcache_dir
            search_cache_dir = os.path.join(diskcache_root, 'inner_search')
            pkl_merge_cache_dir = os.path.join(diskcache_root, 'pkl_merge')
            os.makedirs(search_cache_dir, exist_ok=True)
            os.makedirs(pkl_merge_cache_dir, exist_ok=True)
        
        if cache_iterator is None:
            raise RuntimeError(
                "缓存向量不可用，且不允许回退到实时编码。\n"
                "请检查：\n"
                "  1. prompt向量缓存是否存在\n"
                "  2. logic_keywords.json 和 prompt_template 是否与缓存一致\n"
                "  3. 删除 templates/prompt_cache 目录下的缓存文件后重新运行以自动重新生成缓存"
            )

        # 批量搜索（使用分批加载模式）
        start_time = time.time()
        
        if cache_iterator is not None:
            # 计算 CLIP 初始阈值（用于候选筛选）
            # 当使用 Reranker 时，根据模型类型使用对应的纯 CLIP 阈值来召回更多候选
            clip_initial_threshold = SimilarityThresholdConfig.get_threshold(model_type, use_reranker=False) if use_reranker else similarity_threshold
            
            # 懒加载 Reranker（仅在候选处理阶段首次真正需要时加载）
            reranker_loader = None
            if use_reranker:
                reranker_model_path = self.resolver.resolve_str('models/Qwen3-VL-Reranker-2B')
                rerank_cache_dir = os.path.join(self.cache_dir, 'rerank_cache')
                print(f"[优化搜索] 已启用 Reranker，候选阶段将懒加载: {reranker_model_path}")

                def _lazy_load_reranker():
                    from A_coreUtils.qwen_models.qwen3_vl_reranker import Qwen3VLReranker
                    print(f"[优化搜索] 懒加载 Reranker 模型: {reranker_model_path}")
                    reranker_instance = Qwen3VLReranker(
                        model_name_or_path=reranker_model_path,
                        cache_dir=rerank_cache_dir,
                        torch_dtype="auto"
                    )
                    print(f"[优化搜索] Reranker 模型加载完成")
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
                reranker_weight=SimilarityThresholdConfig.RERANKER_WEIGHT,
                reranker_output_resolution=reranker_output_resolution,
                rerank_batch_size=rerank_batch_size,
                candidate_batch_size=candidate_batch_size,
                search_mode=search_mode,
                result_top_k=top_k,
                config_hash=config_hash
            )
            
            if batch_engine._preload_all:
                # 全量预加载模式：所有PKL已在构造函数中加载
                if batch_engine.all_features_gpu is None:
                    print("[优化搜索] 预加载PKL特征到GPU...")
                    batch_engine._preload_all_features()
                
                self.best_matches = batch_engine.search_with_batched_cache(**search_kwargs)
            else:
                # 分批PKL加载模式（根治版）：
                #   阶段1：每批仅做 CLIP 候选聚合，写入同一个 LMDB
                #   阶段2：全部 PKL 聚合完成后，只做一次 Reranker
                from A_coreUtils.search.batch_text_search import LMDBCache

                total_pkls = len(batch_engine.index_paths)
                pkl_bs = batch_engine.pkl_batch_size
                total_batches = (total_pkls + pkl_bs - 1) // pkl_bs

                # 生成分批PKL模式的 config_hash（包含 PKL 批次信息）
                pkl_batch_config_str = f"pkl_batch|{config_hash}|{total_pkls}|{pkl_bs}"
                pkl_batch_hash = _hashlib.md5(pkl_batch_config_str.encode()).hexdigest()

                # 候选聚合 LMDB（与第二阶段统一复用）
                pkl_batch_lmdb_dir = pkl_merge_cache_dir
                if pkl_batch_lmdb_dir is None:
                    pkl_batch_lmdb_dir = str(self.resolver.project_root / 'temp' / 'cache' / 'search_results')
                os.makedirs(pkl_batch_lmdb_dir, exist_ok=True)

                pkl_batch_lmdb = LMDBCache(pkl_batch_lmdb_dir, map_size=10 * 1024 * 1024 * 1024)

                # 检查断点续传（仅针对阶段1：PKL 批次聚合）
                start_pkl_batch = 0
                checkpoint = pkl_batch_lmdb.load_checkpoint()
                if checkpoint and checkpoint.get('config_hash') == pkl_batch_hash and checkpoint.get('phase') == 'pkl_batch':
                    start_pkl_batch = checkpoint.get('last_completed_pkl_batch', -1) + 1
                    if start_pkl_batch > 0:
                        print(f"[优化搜索] 🔄 断点续传: 从 PKL 批次 {start_pkl_batch + 1}/{total_batches} 继续")
                else:
                    # 配置变化，清空整个 temp 文件夹重新开始
                    pkl_batch_lmdb.close()
                    if checkpoint:
                        print(f"[优化搜索] 配置已变化，清空 temp 文件夹重新搜索")
                    temp_dir = str(self.resolver.project_root / 'temp')
                    if os.path.exists(temp_dir):
                        from A_coreUtils.video_processing.video_utils import cleanup_temp_folder
                        cleanup_temp_folder(temp_dir)
                    # 重建 LMDB 目录
                    pkl_batch_lmdb_dir = pkl_merge_cache_dir
                    if pkl_batch_lmdb_dir is None:
                        pkl_batch_lmdb_dir = str(self.resolver.project_root / 'temp' / 'cache' / 'search_results')
                    os.makedirs(pkl_batch_lmdb_dir, exist_ok=True)
                    pkl_batch_lmdb = LMDBCache(pkl_batch_lmdb_dir, map_size=10 * 1024 * 1024 * 1024)

                print(f"[优化搜索] 分批PKL模式: {total_pkls} 个PKL, 每批 {pkl_bs} 个, 共 {total_batches} 批")
                scene_key_to_pkl_map = {}

                # 阶段1：分批聚合 CLIP 候选（不跑 Reranker）
                for batch_idx in range(start_pkl_batch, total_batches):
                    batch_start = batch_idx * pkl_bs
                    batch_end = min(batch_start + pkl_bs, total_pkls)
                    batch_paths = batch_engine.index_paths[batch_start:batch_end]

                    print(f"\n[PKL批次 {batch_idx + 1}/{total_batches}] 加载 {len(batch_paths)} 个PKL...")

                    # 加载这一批PKL到合并张量
                    batch_engine._load_pkl_batch_to_merged(batch_paths)

                    if batch_engine.all_features_gpu is None:
                        print(f"[PKL批次 {batch_idx + 1}] 无有效数据，跳过")
                        pkl_batch_lmdb.save_checkpoint({
                            'config_hash': pkl_batch_hash,
                            'last_completed_pkl_batch': batch_idx,
                            'total_batches': total_batches,
                            'phase': 'pkl_batch',
                            'timestamp': time.time()
                        })
                        batch_engine._unload_merged_features()
                        continue

                    # 记录当前批次 scene_key -> source_pkl 映射（用于最终按PKL分组）
                    for scene_info in batch_engine.scene_map:
                        video_name = os.path.basename(scene_info['video_path']) if scene_info.get('video_path') else ''
                        scene_key = f"{scene_info['start_frame']}_{video_name}"
                        scene_key_to_pkl_map[scene_key] = scene_info.get('source_pkl', 'unknown')

                    # 重置缓存迭代器（每批PKL都需要遍历所有prompt）
                    cache_iterator.reset()

                    # 仅执行 CLIP 候选聚合：候选写入 LMDB，不做每批 Reranker
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

                    # 更新阶段1 checkpoint
                    pkl_batch_lmdb.save_checkpoint({
                        'config_hash': pkl_batch_hash,
                        'last_completed_pkl_batch': batch_idx,
                        'total_batches': total_batches,
                        'phase': 'pkl_batch',
                        'timestamp': time.time()
                    })

                    print(f"[PKL批次 {batch_idx + 1}] 候选聚合完成")

                    # 释放这批PKL的显存
                    batch_engine._unload_merged_features()

                pkl_batch_lmdb.close()

                # 阶段2：所有 PKL 合并后，仅执行一次 Reranker
                print("\n[优化搜索] 分批PKL候选聚合完成，开始统一执行 Reranker...")
                rerank_search_kwargs = dict(search_kwargs)
                rerank_search_kwargs['cache_iterator'] = None
                rerank_search_kwargs['use_diskcache'] = True
                rerank_search_kwargs['cache_dir'] = pkl_batch_lmdb_dir
                rerank_search_kwargs['skip_clip_search'] = True
                rerank_search_kwargs['result_top_k'] = None
                rerank_search_kwargs['search_mode'] = 0
                rerank_search_kwargs['config_hash'] = None
                all_batch_matches = batch_engine.search_with_batched_cache(**rerank_search_kwargs)
                print(f"[优化搜索] 统一 Reranker 完成，得到 {len(all_batch_matches)} 个候选结果")

                # 按 search_mode 分组取 Top-K
                self.best_matches = self._apply_search_mode_grouping(
                    all_batch_matches, search_mode, top_k, batch_engine, scene_key_to_pkl_map
                )
            
        search_time = time.time() - start_time
        print(f"[优化搜索] 批量搜索完成! 耗时 {search_time:.2f}s")
        
        # 提取场景特征向量（用于向量去重，在 cleanup 之前）
        # 只提取 best_matches 中存在的场景，减少内存占用
        # 同时记录每个场景的 source_pkl 用于同PKL内去重
        scene_features = {}
        scene_pkl_map = {}  # {scene_key: source_pkl}
        if vector_dedup_threshold is not None and self.best_matches and batch_engine.all_features_gpu is not None:
            print(f"[优化搜索] 提取场景特征向量用于去重...")
            import numpy as np
            
            # 构建 best_matches 中的 scene_key 集合
            best_match_keys = set(self.best_matches.keys())
            
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
            print(f"[优化搜索] 提取了 {len(scene_features)} 个场景的特征向量（共 {len(best_match_keys)} 个搜索结果）")
        
        # 释放批量搜索引擎（释放GPU显存）
        elif vector_dedup_threshold is not None and self.best_matches and not batch_engine._preload_all:
            print("[优化搜索] 提取场景特征向量用于去重...")
            import numpy as np

            best_match_keys = set(self.best_matches.keys())
            pkl_bs = batch_engine.pkl_batch_size
            total_pkls = len(batch_engine.index_paths)
            total_batches = (total_pkls + pkl_bs - 1) // pkl_bs

            for batch_idx in range(total_batches):
                batch_start = batch_idx * pkl_bs
                batch_end = min(batch_start + pkl_bs, total_pkls)
                batch_paths = batch_engine.index_paths[batch_start:batch_end]
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

            print(f"[优化搜索] 提取了 {len(scene_features)} 个场景的特征向量（共 {len(best_match_keys)} 个搜索结果）")

        batch_engine.cleanup()
        del batch_engine
        del processor
        _cleanup_gpu_memory()
        
        # 向量去重（如果启用）- 同PKL内去重
        if vector_dedup_threshold is not None and scene_features:
            print(f"\n[向量去重] 开始向量去重，阈值={vector_dedup_threshold}，模式=同PKL内去重")
            from .batch_text_search import deduplicate_by_vector_similarity
            self.best_matches = deduplicate_by_vector_similarity(
                best_matches=self.best_matches,
                scene_features=scene_features,
                video_name_format=self.video_name_format,
                similarity_threshold=vector_dedup_threshold,
                scene_pkl_map=scene_pkl_map  # 传递 PKL 映射用于同PKL内去重
            )
        
        # 分批加载模式直接返回结果
        total_time = time.time() - start_time
        
        print("=" * 60)
        print(f"[优化搜索] 完成! (分批加载模式)")
        print(f"  - 总耗时: {total_time:.2f}s")
        print(f"  - 去重后场景数: {len(self.best_matches)}")
        
        return self.best_matches
    
    @staticmethod
    def _apply_search_mode_grouping(
        all_matches: Dict,
        search_mode: int,
        top_k: int,
        batch_engine,
        scene_key_to_pkl_map: Dict = None
    ) -> Dict:
        """
        对合并后的搜索结果按 search_mode 分组取 Top-K
        
        Args:
            all_matches: 合并后的所有搜索结果 {scene_key: data}
            search_mode: 搜索模式 (-1=按视频, 0=按PKL, 1=跨PKL)
            top_k: 每组返回的最大结果数
            batch_engine: BatchTextSearchEngine 实例（用于获取 scene_map 信息）
        
        Returns:
            分组过滤后的结果字典
        """
        from collections import defaultdict
        
        if top_k is None:
            return all_matches

        if search_mode == -1:
            video_groups = defaultdict(list)
            for scene_key, data in all_matches.items():
                video_path = data['video_path']
                video_groups[video_path].append((scene_key, data))

            filtered = {}
            for video_path, scenes in video_groups.items():
                scenes.sort(key=lambda x: x[1]['similarity'], reverse=True)
                for scene_key, data in scenes[:top_k]:
                    filtered[scene_key] = data

            print(f"[分组过滤] 按视频模式: {len(video_groups)} 个视频, 每视频最多 {top_k} 个场景, 共 {len(filtered)} 个")
            return filtered

        if search_mode == 0:
            pkl_groups = defaultdict(list)

            scene_key_to_pkl = scene_key_to_pkl_map or {}
            if not scene_key_to_pkl:
                for scene_info in batch_engine.scene_map:
                    video_name = os.path.basename(scene_info['video_path']) if scene_info.get('video_path') else ''
                    scene_key = f"{scene_info['start_frame']}_{video_name}"
                    scene_key_to_pkl[scene_key] = scene_info.get('source_pkl', 'unknown')

            for scene_key, data in all_matches.items():
                pkl_path = scene_key_to_pkl.get(scene_key, 'unknown')
                pkl_groups[pkl_path].append((scene_key, data))

            filtered = {}
            for pkl_path, scenes in pkl_groups.items():
                scenes.sort(key=lambda x: x[1]['similarity'], reverse=True)
                for scene_key, data in scenes[:top_k]:
                    filtered[scene_key] = data

            print(f"[分组过滤] 按PKL模式: {len(pkl_groups)} 个PKL, 每PKL最多 {top_k} 个场景, 共 {len(filtered)} 个")
            return filtered

        sorted_matches = sorted(all_matches.items(), key=lambda x: x[1]['similarity'], reverse=True)
        filtered = {}
        for scene_key, data in sorted_matches[:top_k]:
            filtered[scene_key] = data

        print(f"[分组过滤] 跨PKL模式: 全局最多 {top_k} 个场景, 共 {len(filtered)} 个")
        return filtered

    def _format_video_name(self, combo: Dict, start_frame: int, video_parsed_name: str) -> str:
        """
        根据格式模板生成视频名称（动态占位符处理）
        
        Args:
            combo: 关键词组合字典，包含各大类的中文值（如 lens_cn, mood_cn 等）
            start_frame: 起始帧号
            video_parsed_name: 解析后的视频名称
        
        Returns:
            格式化后的视频名称（不含扩展名）
        
        支持的占位符：
            - {镜头}: 镜头类型（中文）
            - {情绪}: 情绪类型（中文）
            - {场景}: 场景类型（中文）
            - {主体}: 主体类型（中文）
            - {动作}: 动作类型（中文）
            - {起始帧}: 起始帧号
            - {视频解析名}: 解析后的视频名称
            - 以及 logic_keywords.json 中定义的任何扩展大类（使用中文键名）
        """
        import re
        
        # 1. 从 video_name_format 中动态解析所有占位符
        placeholders = re.findall(r'\{(\w+)\}', self.video_name_format)
        
        # 2. 为所有占位符预设空字符串默认值
        format_dict = {ph: "" for ph in placeholders}
        
        # 3. 填充系统占位符
        format_dict['起始帧'] = str(start_frame)
        format_dict['视频解析名'] = _sanitize_filename(video_parsed_name)
        
        # 4. 填充标准大类（从 combo 的 xxx_cn 字段）
        pl_category_mapping = {
            '镜头': 'lens_cn',
            '情绪': 'mood_cn',
            '场景': 'scene_cn',
            '主体': 'subject_cn',
            '动作': 'action_cn',
        }
        for category_name, combo_key in pl_category_mapping.items():
            if category_name in format_dict:
                format_dict[category_name] = _sanitize_filename(combo.get(combo_key, ""))
        
        # 5. 填充扩展大类的中文值（从 combo 的 xxx_cn 字段）
        for key, value in combo.items():
            if key.endswith('_cn'):
                category_name = key[:-3]
                if category_name in format_dict:
                    format_dict[category_name] = _sanitize_filename(value)
        
        # 6. 使用格式模板生成名称（所有占位符都已预设默认值，不会 KeyError）
        result_name = self.video_name_format.format(**format_dict)
        
        # 7. 动态清理连续分隔符（从格式模板中提取分隔符）
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


# ============================================================
#  相似度阈值配置（从config.json读取）
# ============================================================

class SimilarityThresholdConfig:
    """
    相似度阈值配置（从config.json读取）
    
    不同模型和模式的推荐阈值：
    - CLIP Large: 21
    - FG-CLIP2: 14
    - Reranker: 51
    - 混合模式: clip_threshold * (1 - N) + 51 * N, N = 0.6
    
    配置文件: config.json -> similarity_thresholds
    """
    
    # 默认值（当config.json不存在或读取失败时使用）
    _DEFAULT_CLIP_LARGE = 21.0
    _DEFAULT_FGCLIP2 = 14.0
    _DEFAULT_RERANKER = 51.0
    _DEFAULT_RERANKER_WEIGHT = 0.6
    
    # 缓存的配置值
    _config_loaded = False
    _clip_large = None
    _fgclip2 = None
    _reranker = None
    _reranker_weight = None
    
    @classmethod
    def _load_config(cls):
        """从config.json加载阈值配置"""
        if cls._config_loaded:
            return
        
        try:
            resolver = PathResolver()
            config_path = str(resolver.project_root / 'config.json')
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            thresholds = config.get('similarity_thresholds', {})
            cls._clip_large = thresholds.get('clip_large', cls._DEFAULT_CLIP_LARGE)
            cls._fgclip2 = thresholds.get('fgclip2', cls._DEFAULT_FGCLIP2)
            cls._reranker = thresholds.get('reranker', cls._DEFAULT_RERANKER)
            cls._reranker_weight = thresholds.get('reranker_weight', cls._DEFAULT_RERANKER_WEIGHT)
            
            # 静默：配置加载信息
        except Exception as e:
            print(f"[错误] 无法从 config.json 加载阈值配置: {e}，使用默认值")
            cls._clip_large = cls._DEFAULT_CLIP_LARGE
            cls._fgclip2 = cls._DEFAULT_FGCLIP2
            cls._reranker = cls._DEFAULT_RERANKER
            cls._reranker_weight = cls._DEFAULT_RERANKER_WEIGHT
        
        cls._config_loaded = True
    
    @classmethod
    @property
    def CLIP_LARGE_THRESHOLD(cls) -> float:
        cls._load_config()
        return cls._clip_large
    
    @classmethod
    @property
    def FGCLIP2_THRESHOLD(cls) -> float:
        cls._load_config()
        return cls._fgclip2
    
    @classmethod
    @property
    def RERANKER_THRESHOLD(cls) -> float:
        cls._load_config()
        return cls._reranker
    
    @classmethod
    @property
    def RERANKER_WEIGHT(cls) -> float:
        cls._load_config()
        return cls._reranker_weight
    
    @classmethod
    def get_threshold(cls, model_type: str = 'clip', use_reranker: bool = False) -> float:
        """
        获取推荐的相似度阈值
        
        Args:
            model_type: 模型类型 ('clip', 'fgclip2', 'auto')
            use_reranker: 是否使用reranker
        
        Returns:
            推荐的相似度阈值
        """
        cls._load_config()
        
        # 获取基础模型阈值
        if model_type.lower() in ['fgclip2', 'fg-clip2', 'fg_clip2']:
            base_threshold = cls._fgclip2
        else:
            base_threshold = cls._clip_large
        
        if use_reranker:
            # 混合模式: base * (1 - N) + reranker * N
            N = cls._reranker_weight
            return base_threshold * (1 - N) + cls._reranker * N
        else:
            return base_threshold
    
    @classmethod
    def get_all_thresholds(cls) -> Dict:
        """获取所有阈值配置"""
        cls._load_config()
        return {
            "clip_large": cls._clip_large,
            "fgclip2": cls._fgclip2,
            "reranker": cls._reranker,
            "reranker_weight": cls._reranker_weight,
            "clip_with_reranker": cls.get_threshold('clip', True),
            "fgclip2_with_reranker": cls.get_threshold('fgclip2', True)
        }


# ============================================================
#  相邻片段合并
# ============================================================

def merge_adjacent_scenes(
    best_matches: Dict[str, Dict],
    adjacent_merge_frames: int,
    video_name_format: str = None
) -> Dict[str, Dict]:
    """
    合并相邻的场景片段
    
    当两个片段的 endframe 和 startframe 相差 ≤ N 帧时，合并为一个片段。
    合并规则：
    - 时间戳：使用第一个片段的 startframe，最后一个片段的 endframe
    - 标签去重：同类标签用 "_" 连接
    - result_name：根据合并后的标签重新生成
    
    Args:
        best_matches: 去重后的搜索结果字典
            格式: {scene_key: {result_name, similarity, video_path, start_frame, end_frame, ...}}
        adjacent_merge_frames: 相邻帧阈值 N，endframe 和 startframe 差值 ≤ N 时合并
        video_name_format: 视频名称格式模板，用于重新生成 result_name
    
    Returns:
        合并后的结果字典
    """
    if adjacent_merge_frames is None or adjacent_merge_frames < 0:
        return best_matches
    
    if not best_matches:
        return best_matches
    
    print(f"\n[相邻合并] 开始合并相邻片段，阈值={adjacent_merge_frames}帧")
    original_count = len(best_matches)
    
    # 按视频路径分组
    video_groups = {}  # {video_path: [(scene_key, scene_data), ...]}
    for scene_key, scene_data in best_matches.items():
        video_path = scene_data.get('video_path', '')
        if video_path not in video_groups:
            video_groups[video_path] = []
        video_groups[video_path].append((scene_key, scene_data))
    
    # 对每个视频组进行合并
    merged_results = {}
    
    for video_path, scenes in video_groups.items():
        # 按 start_frame 排序
        scenes.sort(key=lambda x: x[1].get('start_frame', 0))
        
        # 合并相邻片段
        merged_scenes = []
        current_group = [scenes[0]] if scenes else []
        
        for i in range(1, len(scenes)):
            prev_scene_key, prev_data = current_group[-1]
            curr_scene_key, curr_data = scenes[i]
            
            prev_end = prev_data.get('end_frame', 0)
            curr_start = curr_data.get('start_frame', 0)
            
            # 检查是否相邻（差值 ≤ N）
            if curr_start - prev_end <= adjacent_merge_frames:
                # 相邻，加入当前组
                current_group.append(scenes[i])
            else:
                # 不相邻，保存当前组并开始新组
                merged_scenes.append(current_group)
                current_group = [scenes[i]]
        
        # 保存最后一组
        if current_group:
            merged_scenes.append(current_group)
        
        # 处理每个合并组
        for group in merged_scenes:
            if len(group) == 1:
                # 单个片段，直接保留
                scene_key, scene_data = group[0]
                merged_results[scene_key] = scene_data
            else:
                # 多个片段需要合并
                merged_data = _merge_scene_group(group, video_name_format)
                # 使用新的 scene_key（基于合并后的 start_frame）
                video_name = os.path.basename(merged_data.get('video_path', ''))
                new_scene_key = f"{merged_data['start_frame']}_{video_name}"
                merged_results[new_scene_key] = merged_data
    
    merged_count = len(merged_results)
    print(f"[相邻合并] 完成: {original_count} -> {merged_count} 个片段 (合并了 {original_count - merged_count} 个)")
    
    return merged_results


def _merge_scene_group(
    group: List[Tuple[str, Dict]],
    video_name_format: str = None
) -> Dict:
    """
    合并一组相邻的场景片段
    
    Args:
        group: [(scene_key, scene_data), ...] 需要合并的片段列表
        video_name_format: 视频名称格式模板
    
    Returns:
        合并后的场景数据字典
    """
    if not group:
        return {}
    
    if len(group) == 1:
        return group[0][1]
    
    # 提取所有片段的数据
    first_data = group[0][1]
    last_data = group[-1][1]
    
    # 基础字段：使用第一个片段的 start_frame，最后一个片段的 end_frame
    merged = {
        'start_frame': first_data.get('start_frame', 0),
        'end_frame': last_data.get('end_frame', 0),
        'video_path': first_data.get('video_path', ''),
        'fps': first_data.get('fps'),
        'source_pkl': first_data.get('source_pkl', ''),
    }
    
    # 收集所有标签（用于去重和合并）
    # 标签字段格式: {大类}_cn, {大类}_en, subject_cn, action_cn 等
    label_fields = {}  # {field_name: [values]}
    similarity_sum = 0
    
    for scene_key, scene_data in group:
        similarity_sum += scene_data.get('similarity', 0)
        
        for key, value in scene_data.items():
            # 跳过非标签字段
            if key in ['start_frame', 'end_frame', 'video_path', 'fps', 'source_pkl', 
                       'similarity', 'result_name', 'prompt']:
                continue
            
            if key not in label_fields:
                label_fields[key] = []
            
            if value and value not in label_fields[key]:
                label_fields[key].append(value)
    
    # 合并标签（同类标签用 "_" 连接，去重）
    for field_name, values in label_fields.items():
        if len(values) == 1:
            merged[field_name] = values[0]
        else:
            # 多个值，用 "_" 连接
            merged[field_name] = "_".join(str(v) for v in values)
    
    # 计算平均相似度
    merged['similarity'] = similarity_sum / len(group)
    
    # 重新生成 result_name
    if video_name_format:
        merged['result_name'] = _generate_merged_result_name(merged, video_name_format)
    else:
        # 使用默认格式
        merged['result_name'] = _generate_merged_result_name(merged, None)
    
    return merged


def _generate_merged_result_name(merged_data: Dict, video_name_format: str = None) -> str:
    """
    根据合并后的数据生成 result_name（动态占位符处理）
    
    Args:
        merged_data: 合并后的场景数据
        video_name_format: 视频名称格式模板
    
    Returns:
        生成的 result_name
    """
    import re
    
    # 初始化视频名称解析器
    video_name_parser = VideoNameParser()
    
    # 解析视频名称
    video_path = merged_data.get('video_path', '')
    video_basename = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(video_basename)[0]
    parsed_name = video_name_parser.parse(video_name_no_ext)
    
    # 使用默认格式（如果未指定）
    if video_name_format is None:
        video_name_format = "{主体}_{动作}_{场景}_{情绪}_{起始帧}_{视频解析名}"
    
    # 1. 从 video_name_format 中动态解析所有占位符
    placeholders = re.findall(r'\{(\w+)\}', video_name_format)
    
    # 2. 为所有占位符预设空字符串默认值
    format_dict = {ph: "" for ph in placeholders}
    
    # 3. 填充系统占位符
    format_dict['起始帧'] = str(merged_data.get('start_frame', 0))
    format_dict['视频解析名'] = _sanitize_filename(parsed_name)
    
    # 4. 填充标准大类（从 merged_data 的 xxx_cn 字段）
    pl_category_mapping = {
        '镜头': ['镜头_cn', 'lens_cn'],
        '情绪': ['情绪_cn', 'mood_cn'],
        '场景': ['场景_cn', 'scene_cn'],
        '主体': ['主体_cn', 'subject_cn'],
        '动作': ['动作_cn', 'action_cn'],
    }
    for category_name, keys in pl_category_mapping.items():
        if category_name in format_dict:
            for key in keys:
                if merged_data.get(key):
                    format_dict[category_name] = _sanitize_filename(merged_data.get(key, ""))
                    break
    
    # 5. 填充扩展大类的中文值（从 merged_data 的 xxx_cn 字段）
    for key, value in merged_data.items():
        if key.endswith('_cn'):
            category_name = key[:-3]
            if category_name in format_dict and not format_dict[category_name]:
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
#  视频导出器
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
        """获取视频文件扩展名"""
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
        deduplicated_data: Dict[str, Dict],
        output_dir: str,
        progress_callback: Callable[[int, int, str], None] = None,
        debug_similarity: bool = False
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
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计信息
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'failed_files': []
        }
        
        # 收集所有需要导出的视频
        export_tasks = []
        for scene_key, item in deduplicated_data.items():
            result_name = item.get('result_name', scene_key)
            similarity = item.get('similarity', 0)
            
            video_path = item.get('video_path')
            if not video_path or not os.path.exists(video_path):
                stats['skipped'] += 1
                continue
            
            # 获取视频扩展名
            ext = self._get_video_extension(video_path)
            
            # 使用去重后的结果名作为文件名
            # 调试模式：在文件名前添加相似度
            if self._debug_similarity:
                output_filename = f"相似度{similarity:.1f}_{_sanitize_filename(result_name)}{ext}"
            else:
                output_filename = f"{_sanitize_filename(result_name)}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            # 检查是否已存在
            if os.path.exists(output_path):
                stats['skipped'] += 1
                continue
            
            export_tasks.append({
                'video_path': video_path,
                'output_path': output_path,
                'start_frame': item.get('start_frame', 0),
                'end_frame': item.get('end_frame', 0),
                'fps': item.get('fps'),
                'result_name': result_name,
                'similarity': similarity
            })
        
        stats['total'] = len(export_tasks)
        
        if stats['total'] == 0:
            print("  没有需要导出的视频")
            return stats
        
        print(f"\n📹 开始导出 {stats['total']} 个去重后的视频片段...")
        
        # 执行导出
        for i, task in enumerate(export_tasks):
            output_filename = os.path.basename(task['output_path'])
            
            if progress_callback:
                progress_callback(i + 1, stats['total'], output_filename)
            elif self.verbose:
                print(f"  [{i+1}/{stats['total']}] {output_filename}")
            
            success = self.export_single(
                video_path=task['video_path'],
                output_path=task['output_path'],
                start_frame=task['start_frame'],
                end_frame=task['end_frame'],
                fps=task['fps']
            )
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
                stats['failed_files'].append(output_filename)
        
        return stats


# ============================================================
#  交互式搜索入口
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
    best_matches: Dict[str, Dict],
    resolver: PathResolver,
    output_directory: str,
    video_output_directory: str = None,
    video_copy_mode: bool = False,
    start_frame_offset: int = None,
    end_frame_offset: int = None,
    debug_similarity: bool = False
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
        deduplicated_data=best_matches,
        output_dir=resolved_output_dir,
        progress_callback=None,
        debug_similarity=debug_similarity
    )
    return export_stats, resolved_output_dir


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
    use_reranker: bool = False,
    rerank_top_k: int = 50,
    rerank_batch_size: int = 4,
    reranker_output_resolution: str = '384',
    candidate_batch_size: int = None,
    # 视频导出参数
    video_output_directory: str = None,
    video_copy_mode: bool = True,
    # 视频帧偏移参数
    start_frame_offset: int = None,
    end_frame_offset: int = None,
    # 优化模式参数（use_optimized_search 已移除，强制开启）
    prompt_search_batch_size: int = 1024,  # Prompt搜索矩阵运算批次大小
    feature_fp16: Optional[bool] = None,
    pkl_batch_size: int = None,
    # diskcache 参数
    use_diskcache: bool = True,
    diskcache_dir: str = None,
    # prompt模板参数
    prompt_template: str = None,
    # 视频名称格式参数
    video_name_format: str = None,
    # 调试参数
    debug_similarity: bool = False,
    # 搜索模式参数
    search_mode: int = 0,
    top_k: int = 50,
    # prompt向量缓存参数
    prompt_cache_batch_size: int = 512,
    # 中文模式参数
    use_chinese: bool = False,
    # 线程配置参数
    pkl_load_workers: int = 4,
    lmdb_write_batch_size: int = 1000,
    # 向量去重参数
    vector_dedup_threshold: float = None,
    # 相邻片段合并参数
    adjacent_merge_frames: int = None
) -> Dict:
    """
    交互式自动化场景搜索入口
    
    遍历所有合理的关键词组合进行文搜图，基于 logic_Prompt.py 的逻辑约束。
    模型自动从pkl文件名获取。
    
    Args:
        index_directory: 索引文件目录（包含 .pkl 文件），None则使用默认 'indexes' 目录
        output_directory: 输出目录（保存搜索结果），None则使用默认 'output' 目录
        use_fp16: FP16 精度 - 开启后可节省约50%显存
        use_reranker: 是否使用 Reranker 模式
        rerank_top_k: Reranker Top-K - 从初始召回结果中选取前K个进行重排序，推荐 30-100
        rerank_batch_size: Reranker 批处理大小 - 每批处理的图像数量，推荐 2-8
        reranker_output_resolution: Reranker 帧输出分辨率（短边像素数，推荐 384 或 512）
        video_output_directory: 视频导出目录，None则使用 output/videos
        video_copy_mode: True使用copy模式（快速），False使用精确切割模式
        start_frame_offset: 起始帧偏移量（负数向前，正数向后）
            - None: 使用默认值（copy模式=0，精确切割=-2）
            - 例如：-2 表示起始帧向前偏移2帧
        end_frame_offset: 结束帧偏移量（负数向前，正数向后）
            - None: 使用默认值（copy模式=2，精确切割=2）
            - 例如：2 表示结束帧向后偏移2帧
        prompt_search_batch_size: Prompt搜索矩阵运算批次大小
            - 搜索时每批加载的prompt向量数量
            - 推荐 1024-4096
        feature_fp16: 是否使用FP16存储特征向量（显存减半）
        pkl_batch_size: 每批加载的PKL数量
            - None 或 >= PKL总数: 一次性全部加载到GPU（显存占用高但搜索快）
            - < PKL总数: 分批加载，用完释放（显存占用低但稍慢）
        use_diskcache: 是否使用 LMDB 存储搜索结果（解决内存问题）
        diskcache_dir: LMDB 缓存目录，None则使用默认目录
        prompt_template: 自定义prompt模板，支持占位符: {mood}, {lens}, {subject}, {action}, {scene}
            以及JSON中定义的任何扩展大类。默认: "A {mood} {lens} of a {subject} {action} in {scene}"
        video_name_format: 自定义视频名称格式模板，支持占位符: {镜头}, {情绪}, {场景}, {主体}, {动作}, {起始帧}, {视频解析名}
            以及JSON中定义的任何扩展大类（使用中文键名）。默认: "{镜头}_{情绪}_{场景}_{主体}_{动作}_{起始帧}_{视频解析名}"
        debug_similarity: 调试模式，在文件名前添加相似度
        search_mode: 搜索模式选择
            - -1（按视频模式）: 每个视频独立搜索，每个视频返回 top_k 个结果
            - 0（按PKL模式）: 每个PKL文件独立搜索，每个PKL返回 top_k 个结果
            - 1（跨PKL模式）: 全局搜索，返回全局 top_k 个结果
        top_k: 每组返回的最大结果数，None则不限制
        prompt_cache_batch_size: 生成prompt向量缓存时的批处理大小，推荐 256-1024
        use_chinese: 是否使用中文模式
            - False（默认）: 使用英文标签值生成prompt，适合英文CLIP模型
            - True: 使用中文标签键名生成prompt，适合中文CLIP模型（如 FG-CLIP2）
        pkl_load_workers: PKL加载线程数
            - 全量预加载时（pkl_batch_size=None）使用
            - 推荐值: 4-8
        lmdb_write_batch_size: LMDB单事务写入批大小
            - 分批加载时（pkl_batch_size<PKL总数）使用
            - 推荐值: 4-8
        vector_dedup_threshold: 向量去重余弦相似度阈值
            - None（默认）: 不进行向量去重
            - 0.90 ~ 0.98: 超过此阈值的同标签视频只保留一个
            - 优先规则: 同标签组内如有 OP/ED 视频，只保留 OP/ED 视频
        adjacent_merge_frames: 相邻片段合并帧阈值
            - None（默认）: 不进行相邻片段合并
            - N（正整数）: 当片段A的endframe与片段B的startframe差值≤N时合并
            - 合并规则: 同类标签用"_"连接并去重，时间戳使用第一个片段的startframe
    
    自动配置（无需传参）：
        - similarity_threshold: 根据模型类型自动选择（CLIP Large: 21, FG-CLIP2: 14）
        - categories: 遍历所有类别
    
    Returns:
        搜索结果列表
    """
    # 初始化路径解析器（不传参数，使用 path_resolver.py 所在目录作为项目根目录）
    resolver = PathResolver()
    
    # 设置默认目录
    if index_directory is None:
        index_directory = str(resolver.project_root / 'indexes')
    elif not os.path.isabs(index_directory):
        index_directory = str(resolver.resolve(index_directory))
    
    if output_directory is None:
        output_directory = str(resolver.project_root / 'output')
    elif not os.path.isabs(output_directory):
        output_directory = str(resolver.resolve(output_directory))

    # 默认保持模型计算精度与特征存储精度一致；如需分离可显式传入 feature_fp16
    if feature_fp16 is None:
        feature_fp16 = use_fp16
    
    # 静默：标题和目录信息
    
    # 检查索引目录
    if not os.path.exists(index_directory):
        print(f"\n❌ 错误: 索引目录不存在 - {index_directory}")
        return []
    
    # 创建输出目录
    os.makedirs(output_directory, exist_ok=True)
    
    # 查找索引文件
    index_paths = []
    for filename in os.listdir(index_directory):
        if filename.endswith('.pkl'):
            index_paths.append(os.path.join(index_directory, filename))
    
    if not index_paths:
        print(f"\n❌ 错误: 索引目录中没有 .pkl 文件")
        return []
    
    # 静默：索引文件列表
    
    # 从pkl文件名提取模型名
    model_names = set()
    for index_path in index_paths:
        model_name = extract_model_name_from_pkl(index_path)
        if model_name:
            model_names.add(model_name)
        else:
            model_names.add("unknown")
    
    if len(model_names) > 1:
        print(f"\n❌ 错误: 多索引搜索要求所有索引来自相同模型")
        print(f"   检测到 {len(model_names)} 个不同模型: {', '.join(model_names)}")
        return []
    
    detected_model_name = list(model_names)[0] if model_names else None
    if not detected_model_name or detected_model_name == "unknown":
        print("\n❌ 错误: 无法从索引文件名中提取模型名")
        return []
    
    # 检测模型类型
    detected_model_type = detect_model_type_from_name(detected_model_name)
    
    # 打印语言模式
    lang_mode = "中文" if use_chinese else "英文"
    print(f"[Prompt搜索] 语言模式: {lang_mode}")
    
    # 初始化关键词生成器（传递 use_chinese 参数）
    generator = PromptGenerator(prompt_template=prompt_template, use_chinese=use_chinese)
    
    # 获取所有大类名称（用于验证）
    all_categories = list(generator._simple_categories.keys())
    
    # 验证 prompt_template（如果提供）
    if prompt_template is not None:
        is_valid, error_msg = validate_prompt_template(prompt_template, all_categories)
        if not is_valid:
            raise ValueError(f"[Prompt搜索] {error_msg}")
        print(f"[Prompt搜索] 使用自定义 prompt_template: {prompt_template}")
    
    # 验证或生成 video_name_format
    if video_name_format is None:
        video_name_format = generate_default_prompt_video_name_format(all_categories, prefix="")
        print(f"[Prompt搜索] 使用默认 video_name_format: {video_name_format}")
    else:
        is_valid, error_msg = validate_prompt_video_name_format(video_name_format, all_categories)
        if not is_valid:
            raise ValueError(f"[Prompt搜索] {error_msg}")
        print(f"[Prompt搜索] 使用自定义 video_name_format: {video_name_format}")
    
    # 静默：关键词统计信息
    
    total = generator.count_total_combinations(None)  # 遍历所有类别
    # 静默：配置信息
    
    # 从 config.json 自动获取相似度阈值
    final_threshold = SimilarityThresholdConfig.get_threshold(detected_model_type, use_reranker)
    
    # 静默：阈值配置信息
    
    # 开始搜索
    print("\n🚀 开始批量搜索")
    
    start_time = time.time()
    
    # 初始化搜索器（不指定模型，由pkl文件名决定）
    try:
        searcher = AutoSceneSearcher(
            use_fp16=use_fp16,
            prompt_template=prompt_template,
            video_name_format=video_name_format,
            use_chinese=use_chinese
        )
    except Exception as e:
        print(f"❌ 错误: 初始化搜索器失败 - {e}")
        import traceback
        traceback.print_exc()
        return []
    
    # 执行搜索（边搜索边去重）- 强制使用优化模式
    try:
        # 使用优化版批量搜索（预加载 + 批量计算 + diskcache）
        # 静默：优化模式配置信息
        best_matches = searcher.run_batch_search_optimized(
            index_paths=index_paths,
            similarity_threshold=final_threshold,
            use_reranker=use_reranker,
            rerank_top_k=rerank_top_k,
            rerank_batch_size=rerank_batch_size,
            reranker_output_resolution=reranker_output_resolution,
            prompt_search_batch_size=prompt_search_batch_size,
            feature_fp16=feature_fp16,
            pkl_batch_size=pkl_batch_size,
            use_diskcache=use_diskcache,
            diskcache_dir=diskcache_dir,
            search_mode=search_mode,
            top_k=top_k,
            prompt_cache_batch_size=prompt_cache_batch_size,
            pkl_load_workers=pkl_load_workers,
            lmdb_write_batch_size=lmdb_write_batch_size,
            vector_dedup_threshold=vector_dedup_threshold,
            candidate_batch_size=candidate_batch_size
        )
    except Exception as e:
        print(f"❌ 错误: 搜索执行失败 - {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    # 显示统计
    total_time = time.time() - start_time
    
    print(f"\n📊 搜索完成: {len(best_matches):,} 个场景, 耗时 {total_time:.2f}s")
    
    # 视频导出（必须执行）
    if best_matches:
        # 相邻片段合并（视频导出前）
        if adjacent_merge_frames is not None and adjacent_merge_frames >= 0:
            best_matches = merge_adjacent_scenes(
                best_matches=best_matches,
                adjacent_merge_frames=adjacent_merge_frames,
                video_name_format=video_name_format
            )
        
        print(f"\n📹 视频导出: {len(best_matches)} 个场景")
        
        # 设置视频输出目录

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

        print(f"  导出统计: 成功 {export_stats['success']}, 失败 {export_stats['failed']}, 跳过 {export_stats['skipped']}")

        if export_stats['failed_files']:
            print(f"  导出失败文件: {len(export_stats['failed_files'])} 个")

        cleanup_temp_after_export(resolver)
    
    print("\n✅ 搜索完成!")
    
    return best_matches


# ============================================================
#  测试入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("自动化场景搜索器 - 模块测试")
    print("基于 logic_Prompt.py 的逻辑约束")
    print("=" * 60)
    
    # 初始化路径解析器（不传参数，使用 path_resolver.py 所在目录作为项目根目录）
    resolver = PathResolver()
    print(f"项目根目录: {resolver.get_project_root()}")
    
    # 初始化Prompt生成器并显示统计信息
    generator = PromptGenerator()
    
    print("\n[关键词统计] (从JSON动态读取)")
    for category in generator.CATEGORY_LIST:
        subjects = generator.get_subjects(category)
        actions = generator.get_actions(category)
        print(f"  {category}: {len(subjects)} 主体 × {len(actions)} 动作")
    
    print(f"  场景: {len(generator.get_scenes())}")
    print(f"  情绪: {len(generator.get_moods())}")
    print(f"  镜头: {len(generator.get_lenses())}")
    
    total = generator.count_total_combinations()
    print(f"\n[总组合数] {total:,}")
    
    # 显示示例组合
    print("\n[示例组合] (前5个)")
    for i, combo in enumerate(generator.iterate_all_combinations()):
        if i >= 5:
            break
        print(f"  {i+1}. {combo['prompt']}")
    
    # 显示阈值配置
    print("\n[相似度阈值配置]")
    thresholds = SimilarityThresholdConfig.get_all_thresholds()
    print(f"  CLIP Large: {thresholds['clip_large']}")
    print(f"  FG-CLIP2: {thresholds['fgclip2']}")
    print(f"  Reranker: {thresholds['reranker']}")
    print(f"  CLIP + Reranker: {thresholds['clip_with_reranker']:.1f}")
    print(f"  FG-CLIP2 + Reranker: {thresholds['fgclip2_with_reranker']:.1f}")
    
    print("\n" + "=" * 60)
    print("使用方法:")
    print("  运行主入口: python prompt_output_app.py")
    print("")
    print("  或使用API:")
    print("    from auto_scene_search import run_interactive_search")
    print("    results = run_interactive_search()")
    print("=" * 60)
