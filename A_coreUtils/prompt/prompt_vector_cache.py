# prompt_vector_cache.py
# Prompt向量缓存器 - 预计算所有prompt组合的归一化向量
# v1.0: 初始版本
# v2.0: 流式写入，低内存占用
# v3.0: 回退 pkl 格式，保留流式写入优化
# v3.1: 移除 memmap，改用预分配数组（减少无意义的磁盘 I/O）
# v3.2: 添加分批加载支持（PromptVectorBatchIterator），避免大规模缓存内存溢出
# v4.0: 分离存储优化
#       - 向量: .npy 文件（支持 mmap 内存映射）
#       - prompts: .pkl 文件（一次性加载，约42MB）
#       - prompt_metadata: LMDB 存储（按需读取，避免内存溢出）
# v5.0: memmap 流式写入优化
#       - 向量: .dat 文件（memmap 流式写入，内存峰值 ~10MB）
#       - 向量元信息: .meta.json 文件（存储 shape 和 dtype）
#       - prompts: .pkl 文件（一次性加载，约42MB）
#       - prompt_metadata: LMDB 存储（按需读取，避免内存溢出）
#
# 功能：
# 1. 遍历 logic_keywords.json 中所有可能的prompt组合
# 2. 使用指定模型编码为归一化向量
# 3. 分离存储到 templates 文件夹
# 4. 搜索时直接加载缓存，无需重复编码和归一化
# 5. 支持分批加载，适用于大规模缓存（70万+ prompt）
#
# 缓存文件命名格式:
#   - prompt_cache_{model_name}_{lang}_vectors.dat (向量，memmap)
#   - prompt_cache_{model_name}_{lang}_vectors.meta.json (向量元信息)
#   - prompt_cache_{model_name}_{lang}_prompts.pkl (prompts + 基础元数据)
#   - prompt_cache_{model_name}_{lang}_metadata_lmdb/ (LMDB目录，存储prompt_metadata)

import os
import sys
import pickle
import hashlib
import json
import time
import struct
import numpy as np
import torch
import lmdb
from typing import List, Dict, Tuple, Generator, Optional
from datetime import datetime

# ============================================================
#  路径设置 - 确保能找到项目根目录的模块
# ============================================================
_current_file = os.path.abspath(__file__)
_prompt_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_prompt_dir)
_cut_detect_scene_dir = os.path.dirname(_a_core_utils_dir)
if _cut_detect_scene_dir not in sys.path:
    sys.path.insert(0, _cut_detect_scene_dir)

# 导入路径解析器
from path_resolver import PathResolver

# 全局设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PromptVectorCache:
    """
    Prompt向量缓存器 (v3.2 - pkl 格式，分批加载)
    
    预计算所有prompt组合的归一化向量，存储到 templates 文件夹。
    搜索时直接加载缓存，无需重复编码和归一化。
    
    使用方式：
    ```python
    # 初始化
    cache = PromptVectorCache(processor, prompt_template="A {mood} {lens} of a {subject} {action} in {scene}")
    
    # 检查缓存是否存在，不存在则生成
    if not cache.cache_exists():
        cache.generate_cache()
    
    # 分批加载缓存（内存友好）
    for batch_vectors, batch_prompts, batch_metadata, batch_info in cache.load_cache_batched(batch_size=10000):
        # 处理每批数据
        ...
    ```
    """
    
    def __init__(self, 
                 processor=None,
                 prompt_template: str = None,
                 keywords_path: str = None,
                 cache_dir: str = None,
                 batch_size: int = 512,
                 use_chinese: bool = False):
        """
        初始化Prompt向量缓存器
        
        Args:
            processor: EmbeddingModelProcessor 实例（生成缓存时需要）
            prompt_template: prompt模板，None则使用默认模板
            keywords_path: logic_keywords.json 路径，None则使用默认路径
            cache_dir: 缓存目录，None则使用 Cut_DetectScene/templates/prompt_cache
            batch_size: 批量编码大小
            use_chinese: 是否使用中文模式
                - False（默认）: 使用英文标签值生成prompt
                - True: 使用中文标签键名生成prompt
        """
        # 初始化路径解析器
        self.resolver = PathResolver()
        
        # 保存中文模式设置
        self.use_chinese = use_chinese
        
        # 设置默认路径
        if keywords_path is None:
            keywords_path = str(self.resolver.project_root / 'logic_keywords.json')
        self.keywords_path = keywords_path
        
        if cache_dir is None:
            cache_dir = str(self.resolver.project_root / 'templates' / 'prompt_cache')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 设置prompt模板
        self.prompt_template = prompt_template or "A {mood} {lens} of a {subject} {action} in {scene}"
        
        # 处理器和批量大小
        self.processor = processor
        self.batch_size = batch_size
        
        # 加载关键词数据
        with open(keywords_path, 'r', encoding='utf-8') as f:
            self.keywords_data = json.load(f)
        
        # 计算缓存文件名
        self._cache_hash = self._compute_cache_hash()
        self._model_name = None
        if processor is not None:
            self._model_name = self._get_model_name()
    
    def _get_model_name(self) -> str:
        """获取模型名称（用于缓存文件命名）"""
        if self.processor is None:
            return "unknown"
        
        # 尝试从processor获取模型名
        if hasattr(self.processor, 'model_name'):
            return self.processor.model_name
        elif hasattr(self.processor, '_model_name'):
            return self.processor._model_name
        else:
            return "unknown"
    
    def _compute_cache_hash(self) -> str:
        """
        计算缓存哈希值
        
        基于 prompt_template、logic_keywords.json 内容和 use_chinese 设置
        """
        # 读取关键词文件内容
        with open(self.keywords_path, 'r', encoding='utf-8') as f:
            keywords_content = f.read()
        
        # 组合内容（包含 use_chinese 设置）
        lang_mode = "chinese" if self.use_chinese else "english"
        combined = f"{self.prompt_template}\n{keywords_content}\n{lang_mode}"
        
        # 计算MD5哈希（取前8位）
        hash_obj = hashlib.md5(combined.encode('utf-8'))
        return hash_obj.hexdigest()[:8]
    
    def get_cache_path(self, model_name: str = None) -> str:
        """
        获取缓存基础路径（不带扩展名）
        
        v4.0: 返回基础路径，各文件通过辅助方法获取：
        - get_vectors_path(): .npy 向量文件
        - get_prompts_path(): .pkl prompts文件
        - get_metadata_path(): LMDB 目录
        
        v4.1: 根据 use_chinese 区分缓存文件：
        - 英文模式: prompt_cache_{model_name}_en
        - 中文模式: prompt_cache_{model_name}_cn
        
        Args:
            model_name: 模型名称，None则使用当前processor的模型名
        
        Returns:
            缓存基础路径（不带扩展名）
        """
        if model_name is None:
            model_name = self._model_name or "unknown"
        
        # 清理模型名中的特殊字符
        safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        # 基础路径：prompt_cache_{model_name}_{lang}
        lang_suffix = "cn" if self.use_chinese else "en"
        base_name = f"prompt_cache_{safe_model_name}_{lang_suffix}"
        return os.path.join(self.cache_dir, base_name)
    
    def get_vectors_path(self, model_name: str = None) -> str:
        """获取向量文件路径 (.dat，memmap格式)"""
        return self.get_cache_path(model_name) + "_vectors.dat"
    
    def get_vectors_meta_path(self, model_name: str = None) -> str:
        """获取向量元信息文件路径 (.meta.json)"""
        return self.get_cache_path(model_name) + "_vectors.meta.json"
    
    def get_prompts_path(self, model_name: str = None) -> str:
        """获取prompts文件路径 (.pkl，一次性加载)"""
        return self.get_cache_path(model_name) + "_prompts.pkl"
    
    def get_metadata_path(self, model_name: str = None) -> str:
        """获取prompt_metadata LMDB目录路径"""
        return self.get_cache_path(model_name) + "_metadata_lmdb"
    
    def cache_exists(self, model_name: str = None) -> bool:
        """
        检查缓存是否存在且有效（v5.0 memmap 格式）
        
        检查四个文件/目录是否都存在：
        - vectors.dat: memmap 向量文件
        - vectors.meta.json: 向量元信息文件
        - prompts.pkl: prompts + 基础元数据
        - metadata_lmdb/: LMDB目录
        
        同时验证哈希值：如果 prompt_template 或 logic_keywords.json 变化，
        缓存将被视为过期。
        
        Args:
            model_name: 模型名称，None则使用当前processor的模型名
        
        Returns:
            True 如果缓存存在且有效（哈希匹配）
        """
        vectors_path = self.get_vectors_path(model_name)
        vectors_meta_path = self.get_vectors_meta_path(model_name)
        prompts_path = self.get_prompts_path(model_name)
        metadata_path = self.get_metadata_path(model_name)
        
        # 检查所有文件是否存在
        if not os.path.exists(vectors_path):
            return False
        if not os.path.exists(vectors_meta_path):
            return False
        if not os.path.exists(prompts_path):
            return False
        if not os.path.isdir(metadata_path):
            return False
        
        # 验证prompts文件有效性
        try:
            with open(prompts_path, 'rb') as f:
                data = pickle.load(f)
            
            # 检查必要字段
            required_fields = ['prompts', 'metadata']
            for field in required_fields:
                if field not in data:
                    print(f"[缓存无效] prompts文件缺少字段: {field}")
                    return False
            
            # 验证哈希值（检测 prompt_template 或 keywords 是否变化）
            cached_hash = data['metadata'].get('cache_hash', '')
            current_hash = self._cache_hash
            if cached_hash != current_hash:
                print(f"[缓存过期] 配置已变化 (缓存哈希: {cached_hash}, 当前哈希: {current_hash})")
                print(f"  -> prompt_template 或 logic_keywords.json 已修改，需要重新生成缓存")
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
            if len(vectors_mmap) != len(data['prompts']):
                print(f"[缓存无效] 向量数量({len(vectors_mmap)})与prompts数量({len(data['prompts'])})不匹配")
                del vectors_mmap
                return False
            del vectors_mmap
            
            return True
            
        except Exception as e:
            print(f"[缓存损坏] {e}")
            return False
    
    def _generate_all_prompts(self) -> Generator[Tuple[str, Dict], None, None]:
        """
        生成所有prompt组合
        
        Yields:
            (prompt_text, metadata_dict) 元组
        """
        # 导入PromptGenerator
        from A_coreUtils.search.auto_scene_search import PromptGenerator
        
        # 创建生成器（传递 use_chinese 参数）
        generator = PromptGenerator(
            keywords_path=self.keywords_path,
            prompt_template=self.prompt_template,
            use_chinese=self.use_chinese
        )
        
        # 遍历所有组合
        for combo in generator.iterate_all_combinations():
            yield combo['prompt'], combo
    
    def generate_cache(self, progress_callback=None) -> str:
        """
        生成prompt向量缓存（v6.0 memmap + 生产者-消费者并行模式）
        
        v6.0: memmap 流式写入 + 生产者-消费者并行
        - 生产者线程：GPU 编码向量
        - 消费者线程：memmap 写入 + LMDB 写入
        - 向量: .dat 文件（memmap 流式写入，内存峰值 ~10MB）
        - 向量元信息: .meta.json 文件（存储 shape 和 dtype）
        - prompts: .pkl 文件（一次性加载，约42MB）
        - prompt_metadata: LMDB 存储（按索引读取，避免内存溢出）
        
        自动检测缓存有效性：
        - 检查四个文件是否存在
        - 验证哈希值（prompt_template + keywords 变化时自动重新生成）
        - 验证向量数量与prompts数量匹配
        
        Args:
            progress_callback: 进度回调函数 (current, total, message)
        
        Returns:
            缓存基础路径
        """
        import gc as python_gc
        import shutil
        from queue import Queue
        from threading import Thread, Event
        
        if self.processor is None:
            raise RuntimeError("需要提供 processor 才能生成缓存")
        
        # 更新模型名
        self._model_name = self._get_model_name()
        cache_base_path = self.get_cache_path()
        vectors_path = self.get_vectors_path()
        vectors_meta_path = self.get_vectors_meta_path()
        prompts_path = self.get_prompts_path()
        metadata_path = self.get_metadata_path()
        
        # 检查缓存是否有效（包含哈希验证）
        if self.cache_exists():
            print(f"[缓存有效] {cache_base_path}")
            return cache_base_path
        
        # 缓存无效或不存在，删除旧文件后重新生成
        for path in [vectors_path, vectors_meta_path, prompts_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.isdir(metadata_path):
            shutil.rmtree(metadata_path)
        
        print("=" * 70)
        print("📝 Prompt向量缓存生成器 (v6.0 memmap + 生产者-消费者并行)")
        print("=" * 70)
        print(f"  模型: {self._model_name}")
        print(f"  模板: {self.prompt_template}")
        print(f"  缓存目录: {self.cache_dir}")
        print(f"  批量大小: {self.batch_size}")
        print(f"  存储格式:")
        print(f"    - 向量: {os.path.basename(vectors_path)} (memmap 流式写入)")
        print(f"    - 向量元信息: {os.path.basename(vectors_meta_path)} (json)")
        print(f"    - Prompts: {os.path.basename(prompts_path)} (pkl)")
        print(f"    - Metadata: {os.path.basename(metadata_path)}/ (LMDB)")
        
        # 第一遍：统计总数（使用数学计算，避免遍历生成器导致内存累积）
        print("\n[步骤1] 统计prompt总数...")
        from A_coreUtils.search.auto_scene_search import PromptGenerator
        temp_generator = PromptGenerator(
            keywords_path=self.keywords_path,
            prompt_template=self.prompt_template,
            use_chinese=self.use_chinese
        )
        total_prompts = temp_generator.count_total_combinations()
        del temp_generator
        print(f"  总计: {total_prompts:,} 个prompt")
        
        # 获取向量维度（通过编码一个样本）
        first_prompt = next(self._generate_all_prompts())[0]
        sample_vector = self.processor.encode_text([first_prompt])
        vector_dim = sample_vector.shape[1]
        print(f"  向量维度: {vector_dim}")
        
        # 创建 memmap 文件（流式写入，极低内存占用）
        print(f"\n[步骤2] 创建 memmap 文件...")
        vectors_mmap = np.memmap(
            vectors_path,
            dtype='float32',
            mode='w+',
            shape=(total_prompts, vector_dim)
        )
        print(f"  memmap 文件: {total_prompts:,} x {vector_dim} (float32)")
        estimated_size = total_prompts * vector_dim * 4 / (1024 * 1024 * 1024)
        print(f"  预计向量大小: {estimated_size:.2f} GB")
        
        # 保存向量元信息
        vectors_meta = {
            'shape': [total_prompts, vector_dim],
            'dtype': 'float32',
            'total_prompts': total_prompts,
            'vector_dim': vector_dim
        }
        with open(vectors_meta_path, 'w', encoding='utf-8') as f:
            json.dump(vectors_meta, f, indent=2)
        print(f"  向量元信息已保存: {os.path.basename(vectors_meta_path)}")
        
        # 创建 LMDB 环境（用于存储 prompt_metadata）
        print(f"\n[步骤3] 初始化 LMDB 存储...")
        os.makedirs(metadata_path, exist_ok=True)
        # 预估 LMDB 大小：每个 metadata 约 500 字节，留 2 倍余量
        lmdb_map_size = total_prompts * 1000
        lmdb_env = lmdb.open(metadata_path, map_size=lmdb_map_size, max_dbs=0)
        print(f"  LMDB 目录: {metadata_path}")
        print(f"  预分配大小: {lmdb_map_size / (1024 * 1024):.1f} MB")
        
        # 生产者-消费者并行模式
        print(f"\n[步骤4] 生产者-消费者并行编码 (batch_size={self.batch_size})...")
        start_time = time.time()
        
        # 队列和事件
        write_queue = Queue(maxsize=4)  # 写入队列，限制大小避免内存溢出
        stop_event = Event()
        
        # 消费者线程：负责 memmap 写入 + LMDB 写入
        def writer_thread():
            lmdb_txn = lmdb_env.begin(write=True)
            lmdb_batch_count = 0
            lmdb_commit_interval = 10000
            
            while not stop_event.is_set() or not write_queue.empty():
                try:
                    item = write_queue.get(timeout=0.1)
                    if item is None:
                        break
                    
                    idx, batch_vectors, batch_metadata = item
                    
                    # 写入 memmap
                    vectors_mmap[idx:idx + len(batch_vectors)] = batch_vectors
                    vectors_mmap.flush()
                    
                    # 写入 LMDB
                    for i, meta in enumerate(batch_metadata):
                        key = struct.pack('>I', idx + i)
                        value = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
                        lmdb_txn.put(key, value)
                        lmdb_batch_count += 1
                    
                    # 定期提交 LMDB 事务
                    if lmdb_batch_count >= lmdb_commit_interval:
                        lmdb_txn.commit()
                        lmdb_txn = lmdb_env.begin(write=True)
                        lmdb_batch_count = 0
                    
                    write_queue.task_done()
                except:
                    continue
            
            # 提交最后的事务
            lmdb_txn.commit()
        
        # 启动写入线程
        writer = Thread(target=writer_thread, daemon=True)
        writer.start()
        
        # 显存清理间隔
        gc_interval = max(1, 50 // max(1, self.batch_size // 512))
        
        # 生产者：编码并放入队列
        prompt_generator = self._generate_all_prompts()
        current_idx = 0
        batch_prompts = []
        batch_metadata = []
        
        for prompt_text, prompt_meta in prompt_generator:
            # 收集到批次
            batch_prompts.append(prompt_text)
            batch_metadata.append(prompt_meta)
            
            # 批次满了，编码并放入队列
            if len(batch_prompts) >= self.batch_size:
                # 编码向量（GPU计算）
                batch_vectors = self.processor.encode_text(batch_prompts)
                
                # 放入写入队列（消费者线程负责写入）
                write_queue.put((current_idx, batch_vectors, batch_metadata))
                
                current_idx += len(batch_prompts)
                batch_prompts = []
                batch_metadata = []
                
                # 定期清理显存
                batch_idx = current_idx // self.batch_size
                if batch_idx % gc_interval == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    python_gc.collect()
                
                # 进度显示
                display_interval = 1 if self.batch_size >= 1024 else 10
                if progress_callback:
                    progress_callback(current_idx, total_prompts, f"处理中... {current_idx}/{total_prompts}")
                elif batch_idx % display_interval == 0:
                    elapsed = time.time() - start_time
                    speed = current_idx / elapsed if elapsed > 0 else 0
                    eta = (total_prompts - current_idx) / speed if speed > 0 else 0
                    print(f"  [{current_idx:,}/{total_prompts:,}] {speed:.1f} prompts/s, ETA: {eta:.1f}s")
        
        # 处理最后一个不完整的批次
        if batch_prompts:
            batch_vectors = self.processor.encode_text(batch_prompts)
            write_queue.put((current_idx, batch_vectors, batch_metadata))
            current_idx += len(batch_prompts)
        
        # 等待写入完成
        write_queue.put(None)
        stop_event.set()
        writer.join()
        
        # 关闭 LMDB 和 memmap
        lmdb_env.close()
        del vectors_mmap
        
        elapsed_time = time.time() - start_time
        print(f"\n  编码完成! 耗时: {elapsed_time:.2f}s")
        print(f"  向量形状: ({total_prompts}, {vector_dim})")
        print(f"  向量已归一化: ✓")
        print(f"  实际处理: {current_idx:,} 个prompt")
        
        # 验证数量一致
        if current_idx != total_prompts:
            raise RuntimeError(f"数量不一致! 预期 {total_prompts}, 实际 {current_idx}")
        
        # 保存 prompts + 基础元数据到 .pkl 文件（使用轻量方法，避免内存爆炸）
        print(f"\n[步骤5] 保存 prompts 文件...")
        from A_coreUtils.search.auto_scene_search import PromptGenerator
        temp_generator = PromptGenerator(
            keywords_path=self.keywords_path,
            prompt_template=self.prompt_template,
            use_chinese=self.use_chinese
        )
        regenerated_prompts = list(temp_generator.iterate_prompts_only())
        del temp_generator
        prompts_data = {
            'prompts': regenerated_prompts,
            'metadata': {
                'model_name': self._model_name,
                'prompt_template': self.prompt_template,
                'cache_hash': self._cache_hash,
                'total_prompts': total_prompts,
                'vector_dim': vector_dim,
                'created_at': datetime.now().isoformat(),
                'keywords_path': self.keywords_path,
                'normalized': True,
                'format_version': '6.0',
            }
        }
        with open(prompts_path, 'wb') as f:
            pickle.dump(prompts_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        del regenerated_prompts, prompts_data
        python_gc.collect()
        prompts_size = os.path.getsize(prompts_path) / (1024 * 1024)
        print(f"    -> {os.path.basename(prompts_path)}: {prompts_size:.2f} MB")
        
        # 获取文件大小
        vectors_size = os.path.getsize(vectors_path) / (1024 * 1024)
        print(f"    -> {os.path.basename(vectors_path)}: {vectors_size:.2f} MB")
        
        # 获取 LMDB 目录大小
        lmdb_size = sum(
            os.path.getsize(os.path.join(metadata_path, f))
            for f in os.listdir(metadata_path)
            if os.path.isfile(os.path.join(metadata_path, f))
        ) / (1024 * 1024)
        print(f"    -> {os.path.basename(metadata_path)}/: {lmdb_size:.2f} MB")
        
        total_size = vectors_size + prompts_size + lmdb_size
        print(f"\n  总大小: {total_size:.2f} MB")
        print(f"    - 向量 (memmap): {vectors_size:.2f} MB ({vectors_size/total_size*100:.1f}%)")
        print(f"    - Prompts (pkl): {prompts_size:.2f} MB ({prompts_size/total_size*100:.1f}%)")
        print(f"    - Metadata (LMDB): {lmdb_size:.2f} MB ({lmdb_size/total_size*100:.1f}%)")
        
        print("\n" + "=" * 70)
        print("✅ 缓存生成完成!")
        print("=" * 70)
        
        return cache_base_path
    
    def load_cache_batched(self, model_name: str = None, batch_size: int = 10000) -> 'PromptVectorBatchIterator':
        """
        分批加载缓存（v5.0 memmap 格式）
        
        返回一个迭代器，每次返回一批向量和对应的元数据。
        适用于大规模 prompt 缓存（如70万+），避免一次性加载到内存。
        
        v5.0 优化：
        - 向量使用 memmap 内存映射，按需加载
        - prompts 一次性加载（约42MB）
        - prompt_metadata 使用 LMDB 按索引读取
        
        Args:
            model_name: 模型名称
            batch_size: 每批加载的 prompt 数量
        
        Returns:
            PromptVectorBatchIterator 迭代器
        """
        vectors_path = self.get_vectors_path(model_name)
        vectors_meta_path = self.get_vectors_meta_path(model_name)
        prompts_path = self.get_prompts_path(model_name)
        metadata_path = self.get_metadata_path(model_name)
        
        # 检查文件是否存在
        if not os.path.exists(vectors_path):
            raise FileNotFoundError(f"向量文件不存在: {vectors_path}")
        if not os.path.exists(vectors_meta_path):
            raise FileNotFoundError(f"向量元信息文件不存在: {vectors_meta_path}")
        if not os.path.exists(prompts_path):
            raise FileNotFoundError(f"Prompts文件不存在: {prompts_path}")
        if not os.path.isdir(metadata_path):
            raise FileNotFoundError(f"Metadata LMDB目录不存在: {metadata_path}")
        
        return PromptVectorBatchIterator(
            vectors_path=vectors_path,
            vectors_meta_path=vectors_meta_path,
            prompts_path=prompts_path,
            metadata_path=metadata_path,
            batch_size=batch_size
        )
    
    def get_cache_info(self, model_name: str = None) -> Dict:
        """
        获取缓存信息（v5.0 memmap 格式，不加载向量数据）
        
        Args:
            model_name: 模型名称
        
        Returns:
            缓存元数据字典
        """
        vectors_path = self.get_vectors_path(model_name)
        vectors_meta_path = self.get_vectors_meta_path(model_name)
        prompts_path = self.get_prompts_path(model_name)
        metadata_path = self.get_metadata_path(model_name)
        
        if not os.path.exists(prompts_path):
            raise FileNotFoundError(f"Prompts文件不存在: {prompts_path}")
        
        # 加载 prompts 文件获取元数据
        with open(prompts_path, 'rb') as f:
            data = pickle.load(f)
        
        # 从 meta.json 获取向量维度
        if os.path.exists(vectors_meta_path):
            with open(vectors_meta_path, 'r', encoding='utf-8') as f:
                vectors_meta = json.load(f)
            vector_dim = vectors_meta.get('vector_dim', 0)
        else:
            vector_dim = 0
        
        # 计算总文件大小
        vectors_size = os.path.getsize(vectors_path) / (1024 * 1024) if os.path.exists(vectors_path) else 0
        prompts_size = os.path.getsize(prompts_path) / (1024 * 1024)
        lmdb_size = sum(
            os.path.getsize(os.path.join(metadata_path, f))
            for f in os.listdir(metadata_path)
            if os.path.isfile(os.path.join(metadata_path, f))
        ) / (1024 * 1024) if os.path.isdir(metadata_path) else 0
        
        return {
            'total_prompts': len(data['prompts']),
            'vector_dim': vector_dim,
            'metadata': data['metadata'],
            'vectors_path': vectors_path,
            'prompts_path': prompts_path,
            'metadata_path': metadata_path,
            'file_size_mb': vectors_size + prompts_size + lmdb_size,
            'vectors_size_mb': vectors_size,
            'prompts_size_mb': prompts_size,
            'lmdb_size_mb': lmdb_size
        }


class PromptVectorBatchIterator:
    """
    Prompt向量分批迭代器（v5.0 memmap 格式）
    
    用于分批加载大规模 prompt 缓存，避免一次性加载到内存导致溢出。
    
    v5.0 优化：
    - 向量使用 memmap 内存映射，按需加载到内存
    - prompts 一次性加载（约42MB，可接受）
    - prompt_metadata 使用 LMDB 按索引读取，避免内存溢出
    
    使用方式：
    ```python
    cache = PromptVectorCache(...)
    for batch_vectors, batch_prompts, batch_metadata, batch_info in cache.load_cache_batched(batch_size=10000):
        # batch_vectors: torch.Tensor [batch_size, dim] GPU张量
        # batch_prompts: List[str] prompt文本列表
        # batch_metadata: List[Dict] prompt元数据列表
        # batch_info: Dict 批次信息 {batch_idx, start_idx, end_idx, total_prompts}
        ...
    ```
    """
    
    def __init__(self,
                 vectors_path: str,
                 vectors_meta_path: str,
                 prompts_path: str,
                 metadata_path: str,
                 batch_size: int = 10000,
                 use_fp16: bool = True):
        """
        初始化分批迭代器（v5.0 memmap 格式）
        
        Args:
            vectors_path: 向量文件路径 (.dat)
            vectors_meta_path: 向量元信息文件路径 (.meta.json)
            prompts_path: prompts文件路径 (.pkl)
            metadata_path: LMDB目录路径
            batch_size: 每批加载的 prompt 数量
            use_fp16: 是否使用 FP16 精度
        """
        self.vectors_path = vectors_path
        self.vectors_meta_path = vectors_meta_path
        self.prompts_path = prompts_path
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        
        # 1. 加载向量元信息
        with open(vectors_meta_path, 'r', encoding='utf-8') as f:
            vectors_meta = json.load(f)
        
        # 2. 向量使用 memmap 内存映射（按需加载）
        self._vectors = np.memmap(
            vectors_path,
            dtype=vectors_meta['dtype'],
            mode='r',
            shape=tuple(vectors_meta['shape'])
        )
        
        # 3. prompts 一次性加载（约42MB）
        with open(prompts_path, 'rb') as f:
            prompts_data = pickle.load(f)
        self._prompts = prompts_data['prompts']
        self._metadata = prompts_data['metadata']
        
        # 4. LMDB 环境（延迟打开，按需读取）
        self._lmdb_env = None
        
        self.total_prompts = len(self._prompts)
        self.num_batches = (self.total_prompts + batch_size - 1) // batch_size
        self._current_batch = 0
    
    def _ensure_lmdb_open(self):
        """确保 LMDB 环境已打开"""
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(self.metadata_path, readonly=True, lock=False)
    
    def _get_metadata_batch(self, start_idx: int, end_idx: int) -> List[Dict]:
        """
        从 LMDB 批量读取 prompt_metadata
        
        Args:
            start_idx: 起始索引
            end_idx: 结束索引（不包含）
        
        Returns:
            List[Dict]: metadata 列表
        """
        self._ensure_lmdb_open()
        
        batch_metadata = []
        with self._lmdb_env.begin() as txn:
            for idx in range(start_idx, end_idx):
                key = struct.pack('>I', idx)
                value = txn.get(key)
                if value is not None:
                    meta = pickle.loads(value)
                    batch_metadata.append(meta)
                else:
                    # 如果找不到，返回空字典
                    batch_metadata.append({})
        
        return batch_metadata
    
    def __iter__(self):
        self._current_batch = 0
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, List[str], List[Dict], Dict]:
        if self._current_batch >= self.num_batches:
            raise StopIteration
        
        start_idx = self._current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_prompts)
        
        # 1. 提取当前批次的向量（mmap 按需加载）
        batch_vectors_np = self._vectors[start_idx:end_idx]
        
        # 转换为 GPU 张量（需要复制，因为 mmap 是只读的）
        batch_vectors_np = np.array(batch_vectors_np)  # 复制到内存
        batch_vectors = torch.from_numpy(batch_vectors_np).to(DEVICE)
        if self.use_fp16 and DEVICE == "cuda":
            batch_vectors = batch_vectors.half()
        
        # 2. 提取当前批次的 prompts（已在内存中）
        batch_prompts = self._prompts[start_idx:end_idx]
        
        # 3. 从 LMDB 读取当前批次的 metadata
        batch_metadata = self._get_metadata_batch(start_idx, end_idx)
        
        batch_info = {
            'batch_idx': self._current_batch,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'total_prompts': self.total_prompts,
            'num_batches': self.num_batches
        }
        
        self._current_batch += 1
        
        return batch_vectors, batch_prompts, batch_metadata, batch_info
    
    def __len__(self) -> int:
        return self.num_batches
    
    @property
    def metadata(self) -> Dict:
        """获取缓存元数据"""
        return self._metadata
    
    def reset(self):
        """重置迭代器"""
        self._current_batch = 0
    
    def close(self):
        """关闭 LMDB 环境"""
        if self._lmdb_env is not None:
            self._lmdb_env.close()
            self._lmdb_env = None
    
    def __del__(self):
        """析构时关闭 LMDB"""
        self.close()
