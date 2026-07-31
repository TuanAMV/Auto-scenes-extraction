# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
# embedding_model.py
# 嵌入模型处理器 - GPU优化版本 (并行I/O加载)
# 支持多种模型格式:
#   - open_clip 格式: CLIP (需要 open_clip_config.json)
#   - transformers 格式: CLIP, FG-CLIP2 (需要 config.json)
# [简化] 移除 Chinese CLIP 支持，只保留三元组模式
# 自动检测模型类型，完全离线模式
# v3.6: 简化逻辑，移除 Chinese-CLIP 和单帧/三帧平均模式
# v3.5: PKL文件名使用完整模型名，去掉Cosine后缀和模型类型标签
# v2.9: 移除欧氏距离选项，仅保留余弦相似度

import os
import sys
import logging
logging.getLogger().setLevel(logging.ERROR)  # 只显示错误，不显示警告

# ============================================================
#  路径设置 - 确保能找到项目根目录的模块
# ============================================================
_current_file = os.path.abspath(__file__)
_embedding_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_embedding_dir)
_cut_detect_scene_dir = os.path.dirname(_a_core_utils_dir)
if _cut_detect_scene_dir not in sys.path:
    sys.path.insert(0, _cut_detect_scene_dir)

# ============================================================
#  关键修复: 在所有导入之前设置离线环境变量
# ============================================================
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import time
import torch
import torch.nn.functional as F
import numpy as np
import shutil
import traceback
import re
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from PIL import Image

# [简化] 移除 Chinese CLIP 支持
try:
    from transformers import CLIPModel, CLIPImageProcessor
    HAS_TRANSFORMERS_CLIP = True
except ImportError:
    HAS_TRANSFORMERS_CLIP = False

# 导入模块（从同级video_processing目录）
from A_coreUtils.video_processing.video_utils import (
    FrameExtractorThread, 
    VideoMetaHelper, 
    SceneBoundaryDetector,
    sanitize_name,
    cleanup_temp_folder,
    TEMP_DIR
)

# 导入 Lance 索引读写工具
from A_coreUtils.lance_index_io import (
    scenes_to_lance_rows,
    get_indexed_video_paths,
    read_lance_index_raw,
    AsyncLanceWriter,
)

# 导入路径解析器
from path_resolver import PathResolver, get_project_root

# ============================================================
#  全局配置
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 初始化路径解析器（不传参数，使用 path_resolver.py 所在目录作为项目根目录）
_path_resolver = PathResolver()


class VectorizerThread:
    """GPU线程: 批量读取帧并向量化
    
    修复内容：
    1. 修复预加载导致帧丢失的bug - 预加载结果与当前batch不匹配时会丢帧
    2. 正确统计实际处理帧数
    3. 只删除实际处理过的文件
    4. v3.1: 在预加载时同时进行黑帧检测，避免额外IO开销
    """
    
    def __init__(self, extractor: FrameExtractorThread, model, preprocess,
                 batch_size: int = 128, truncate_dim: int = None,
                 model_type: str = 'fgclip2', io_workers: int = 8,
                 prefetch: bool = True,
                 brightness_threshold: float = None, black_pixel_ratio: float = 98,
                 white_threshold: float = None, white_pixel_ratio: float = 90):
        self.extractor = extractor
        self.model = model
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim
        self.model_type = model_type
        self.io_workers = io_workers
        self.prefetch = prefetch
        self.features_list = []
        self.frame_indices = []
        self.vectorization_done = Event()
        self.error = None
        self._io_executor = ThreadPoolExecutor(max_workers=io_workers)
        self._prefetch_queue = Queue(maxsize=2)
        # v3.1: 黑帧/白帧检测参数
        self.brightness_threshold = brightness_threshold
        self.black_pixel_ratio = black_pixel_ratio
        self.white_threshold = white_threshold
        self.white_pixel_ratio = white_pixel_ratio
        self._black_frames_filtered = 0
        self._white_frames_filtered = 0
        
    def start(self):
        thread = Thread(target=self._vectorize_worker, daemon=True)
        thread.start()
        return self
    
    def _is_invalid_frame(self, img: Image.Image) -> bool:
        """检测黑帧/白帧 - 在预加载时调用，复用已加载图像，无额外IO开销"""
        try:
            gray = img.convert('L')
            pixels = np.array(gray)
            total = pixels.size

            # 黑帧检测
            if self.brightness_threshold is not None and self.brightness_threshold > 0:
                black_ratio = np.sum(pixels < self.brightness_threshold) / total
                if black_ratio >= (self.black_pixel_ratio / 100.0):
                    self._black_frames_filtered += 1
                    return True

            # 白帧检测（复用同一张灰度图）
            if self.white_threshold is not None and self.white_threshold > 0:
                white_ratio = np.sum(pixels > self.white_threshold) / total
                if white_ratio >= (self.white_pixel_ratio / 100.0):
                    self._white_frames_filtered += 1
                    return True
        except Exception:
            pass
        return False
    
    def _load_single_image(self, filepath: str):
        try:
            if not os.path.exists(filepath):
                return None
            img = Image.open(filepath).convert('RGB')
            
            # v3.1: 在加载时检测黑帧/白帧，跳过无效帧
            if self._is_invalid_frame(img):
                os.remove(filepath)
                return None
            
            tensor = self.preprocess(img)
            filename = os.path.basename(filepath)
            frame_num = int(filename.replace('frame_', '').replace('.jpg', '')) - 1
            return (frame_num, tensor, filepath)
        except Exception as e:
            return None
    
    def _load_images_parallel(self, filepaths: list) -> tuple:
        futures = {self._io_executor.submit(self._load_single_image, fp): fp 
                   for fp in filepaths}
        results = []
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
        if not results:
            return [], [], []
        results.sort(key=lambda x: x[0])
        indices = [r[0] for r in results]
        tensors = [r[1] for r in results]
        paths = [r[2] for r in results]
        return indices, tensors, paths
    
    def _prefetch_worker(self, filepaths: list):
        return self._load_images_parallel(filepaths)
    
    def _vectorize_worker(self):
        """单实例模式：双缓冲流水线 - I/O与GPU并行执行"""
        try:
            local_processed = set()  # 本线程处理过的文件（用于本地去重）
            actual_frames_processed = 0  # 实际有效处理的帧数
            batch_number = 0
            prefetch_future = None
            prefetch_target_files = None  # 记录预加载对应的文件列表
            
            print(f"[GPU] 向量化线程启动 (I/O并行: {self.io_workers}线程, 预加载: {'开启' if self.prefetch else '关闭'})")
            
            while True:
                available_files = self.extractor.get_frame_files()
                files_to_process = [f for f in available_files if f not in local_processed]
                
                if len(files_to_process) >= self.batch_size:
                    files_to_process.sort(key=lambda f: int(os.path.basename(f).replace('frame_', '').replace('.jpg', '')))
                    batch_files = files_to_process[:self.batch_size]
                    next_batch_files = files_to_process[self.batch_size:self.batch_size*2] if self.prefetch else []
                    batch_number += 1
                    
                    # 双缓冲：_process_batch_optimized 内部在 I/O 完成后、GPU 之前提交预取
                    # 返回 (有效帧数, 新预取future, 新预取文件列表)
                    valid_count, prefetch_future, prefetch_target_files = self._process_batch_optimized(
                        batch_files,
                        batch_number,
                        prefetch_future=prefetch_future,
                        prefetch_target_files=prefetch_target_files,
                        next_batch_files=next_batch_files
                    )
                    actual_frames_processed += valid_count
                    local_processed.update(batch_files)
                    
                elif self.extractor.is_done():
                    if files_to_process:
                        files_to_process.sort(key=lambda f: int(os.path.basename(f).replace('frame_', '').replace('.jpg', '')))
                        batch_number += 1
                        print(f"[GPU] 处理最后一批 {len(files_to_process)} 帧...")
                        valid_count, _, _ = self._process_batch_optimized(
                            files_to_process,
                            batch_number,
                            prefetch_future=prefetch_future,
                            prefetch_target_files=prefetch_target_files
                        )
                        actual_frames_processed += valid_count
                    break
                else:
                    time.sleep(0.05)
            
            print(f"[GPU] ✔ 向量化完成,共处理 {actual_frames_processed} 帧,分 {batch_number} 批")
            
        except Exception as e:
            self.error = f"向量化线程异常: {e}"
            traceback.print_exc()
        finally:
            self._io_executor.shutdown(wait=False)
            self.vectorization_done.set()
    
    def _process_batch_optimized(self, batch_files: list, batch_number: int,
                                  prefetch_future=None, prefetch_target_files=None,
                                  next_batch_files=None) -> tuple:
        """双缓冲流水线：I/O完成后立即提交下一批预取，再执行GPU推理
        
        返回: (实际处理帧数, 新预取future, 新预取文件列表)
        
        流水线时序：
        1. 阻塞等待上一轮预取结果（或同步加载首批）
        2. 提交下一批预取到I/O线程池
        3. GPU推理当前batch（此时下一批I/O在并行执行）
        """
        prefix = f"[GPU] 批次 #{batch_number}"
        batch_start = time.time()
        io_start = time.time()
        
        # ① 获取当前batch数据：阻塞等待预取结果，或同步加载
        use_prefetch = False
        if prefetch_future is not None:
            if prefetch_target_files is not None:
                if set(prefetch_target_files) == set(batch_files):
                    use_prefetch = True
        
        if use_prefetch:
            try:
                valid_indices, image_tensors, valid_paths = prefetch_future.result(timeout=30.0)
                io_source = "预加载"
            except Exception:
                valid_indices, image_tensors, valid_paths = self._load_images_parallel(batch_files)
                io_source = "并行加载(预取异常)"
        else:
            valid_indices, image_tensors, valid_paths = self._load_images_parallel(batch_files)
            io_source = "并行加载"
        
        io_time = time.time() - io_start
        
        if not image_tensors:
            print(f"{prefix}: 无有效图像,跳过 (提交 {len(batch_files)} 文件)")
            # 即使当前batch为空，也要提交下一批预取
            new_prefetch_future = None
            new_prefetch_target_files = None
            if self.prefetch and next_batch_files:
                new_prefetch_future = self._io_executor.submit(self._load_images_parallel, next_batch_files)
                new_prefetch_target_files = list(next_batch_files)
            return 0, new_prefetch_future, new_prefetch_target_files
        
        # ② GPU之前提交下一批预取（关键：让I/O与GPU并行）
        new_prefetch_future = None
        new_prefetch_target_files = None
        if self.prefetch and next_batch_files:
            new_prefetch_future = self._io_executor.submit(self._load_images_parallel, next_batch_files)
            new_prefetch_target_files = list(next_batch_files)
        
        # ③ GPU推理（此时下一批I/O在并行执行）
        gpu_start = time.time()
        with torch.no_grad():
            if self.model_type == 'fgclip2':
                # FG-CLIP2: 逐个处理，因为每个输入是 BatchFeature 字典
                all_features = []
                for img_input in image_tensors:
                    feat = self.model.encode_image(img_input)
                    all_features.append(feat)
                features = torch.cat(all_features, dim=0)
            else:
                batch_tensor = torch.stack(image_tensors).to(DEVICE)
                # Align input dtype with model weight dtype (fixes Float/Half mismatch on CUDA).
                model_ref = getattr(self.model, 'model', self.model)
                try:
                    target_dtype = next(model_ref.parameters()).dtype
                    if batch_tensor.dtype != target_dtype:
                        batch_tensor = batch_tensor.to(dtype=target_dtype)
                except (StopIteration, AttributeError, TypeError):
                    pass
                features = self.model.encode_image(batch_tensor)
        gpu_time = time.time() - gpu_start
        
        features = _truncate_and_normalize(features, self.truncate_dim, log=(batch_number == 1))
        
        total_time = time.time() - batch_start
        fps = len(valid_indices) / total_time if total_time > 0 else 0
        self.features_list.append(features)
        self.frame_indices.extend(valid_indices)
        dim_info = f"dim={features.shape[1]}"
        
        # ✅ v3.0: 显示有效帧数/提交文件数
        print(f"{prefix}: {len(valid_indices)}/{len(batch_files)}帧 [{io_source}] | I/O={io_time:.2f}s GPU={gpu_time:.2f}s | {fps:.1f} 帧/秒 | {dim_info} | 总进度: {len(self.frame_indices)} 帧")
        
        # ✅ v3.0: 只删除实际处理过的文件 (valid_paths)
        for filepath in valid_paths:
            try:
                os.remove(filepath)
            except:
                pass
        
        return len(valid_indices), new_prefetch_future, new_prefetch_target_files
    
    def get_results(self) -> tuple:
        if self.features_list:
            all_features = torch.cat(self.features_list, dim=0)
            if len(self.frame_indices) > 1:
                is_sorted = all(self.frame_indices[i] <= self.frame_indices[i+1] for i in range(len(self.frame_indices)-1))
                if not is_sorted:
                    print("[GPU] ⚠️ 检测到帧顺序不一致，进行全局排序...")
                    sorted_indices = sorted(range(len(self.frame_indices)), key=lambda i: self.frame_indices[i])
                    self.frame_indices = [self.frame_indices[i] for i in sorted_indices]
                    all_features = all_features[sorted_indices]
                    print("[GPU] ✔ 排序完成")
            return self.frame_indices, all_features
        return [], torch.tensor([])
    
    def is_done(self) -> bool:
        return self.vectorization_done.is_set()


def _truncate_and_normalize(features, truncate_dim, log=True):
    """截断维度并 L2 归一化（统一入口）"""
    if truncate_dim is not None and features.shape[1] > truncate_dim:
        if log:
            print(f"[Info] 截断向量维度: {features.shape[1]} -> {truncate_dim}")
        features = features[:, :truncate_dim]
    features = F.normalize(features, dim=-1)
    return features


class _SimpleProcessor:
    """轻量 processor 容器，持有 tokenizer 和 image_processor"""
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor


class _TransformersModelWrapper:
    """统一的 transformers 模型包装器，支持 CLIP 和 FG-CLIP2"""
    def __init__(self, model, fp16=False, is_fgclip2=False):
        self.model = model
        self.fp16 = fp16
        self.is_fgclip2 = is_fgclip2
        self.logit_scale = model.logit_scale if hasattr(model, 'logit_scale') else None
        self.logit_bias = model.logit_bias if hasattr(model, 'logit_bias') else None

    def encode_image(self, image_inputs):
        """编码图像 - 支持 tensor 和 dict 两种输入"""
        with torch.no_grad():
            if isinstance(image_inputs, torch.Tensor):
                if image_inputs.dim() == 3:
                    image_inputs = image_inputs.unsqueeze(0)
                if self.fp16 and image_inputs.dtype != torch.float16:
                    image_inputs = image_inputs.half()
                output = self.model.get_image_features(pixel_values=image_inputs)
            elif isinstance(image_inputs, dict) or hasattr(image_inputs, 'keys'):
                image_inputs = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in image_inputs.items()}
                output = self.model.get_image_features(**image_inputs)
            else:
                raise RuntimeError(f"不支持的图像输入类型: {type(image_inputs)}")
            features = output.pooler_output if hasattr(output, 'pooler_output') else output
            return features

    def encode_text(self, text_inputs, walk_type="short"):
        """编码文本 - FG-CLIP2 支持 walk_type 参数，CLIP 忽略"""
        with torch.no_grad():
            if isinstance(text_inputs, dict) or hasattr(text_inputs, 'keys'):
                text_inputs = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in text_inputs.items()}
                if self.is_fgclip2:
                    output = self.model.get_text_features(**text_inputs, walk_type=walk_type)
                else:
                    output = self.model.get_text_features(**text_inputs)
            else:
                if self.is_fgclip2:
                    output = self.model.get_text_features(input_ids=text_inputs, walk_type=walk_type)
                else:
                    output = self.model.get_text_features(input_ids=text_inputs)
            features = output.pooler_output if hasattr(output, 'pooler_output') else output
            return features


class EmbeddingModelProcessor:
    """嵌入模型处理器 - 支持 CLIP、FG-CLIP2 模型的通用接口"""
    
    def __init__(self, model_name: str = None, model_type: str = 'auto',
                 batch_size: int = 128, truncate_dim: int = None,
                 io_workers: int = 8, model_path: str = None,
                 use_fp16: bool = True):
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim
        self.io_workers = io_workers
        self.model_default_dim = None
        self.use_fp16 = use_fp16
        self.device = DEVICE
        if model_path is not None and model_name is None:
            model_name = model_path
        if model_name is None:
            model_name = 'qihoo360-fg-clip2-base'
        self.model_type = model_type
        if os.path.isabs(model_name) or os.path.exists(model_name):
            self.model_path = model_name
        else:
            self.model_path = _path_resolver.join_str('models', model_name)
        # 保存模型名称（用于缓存文件命名）
        self.model_name = os.path.basename(self.model_path)
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._processor = None
        self._model_raw = None
        # [简化] 移除 Chinese CLIP 支持
        self._load_model()
    
    def _apply_fp16(self, model):
        """应用 FP16 精度转换"""
        if self.use_fp16 and DEVICE == 'cuda':
            return model.half()
        return model
    
    def _load_model(self):
        if not os.path.isdir(self.model_path):
            raise RuntimeError(f"模型目录不存在: {self.model_path}")
        model_name_lower = os.path.basename(self.model_path).lower()
        open_clip_config_path = os.path.join(self.model_path, 'open_clip_config.json')
        config_json_path = os.path.join(self.model_path, 'config.json')
        has_open_clip_config = os.path.exists(open_clip_config_path)
        has_config_json = os.path.exists(config_json_path)
        # [简化] 移除 Chinese CLIP 检测
        is_fgclip2 = 'fg-clip2' in model_name_lower or 'fg_clip2' in model_name_lower or 'fgclip2' in model_name_lower
        if has_config_json and not has_open_clip_config:
            try:
                import json
                with open(config_json_path, 'r') as f:
                    config = json.load(f)
                # 检测 FG-CLIP2 (通过 auto_map 或 architectures)
                if config.get('auto_map', {}).get('AutoModelForCausalLM'):
                    is_fgclip2 = True
                architectures = config.get('architectures', [])
                if any('fgclip' in arch.lower() for arch in architectures):
                    is_fgclip2 = True
            except Exception as e:
                print(f"[Warning] 解析 config.json 时出错: {e}")
        
        if has_open_clip_config:
            self._load_openclip_model()
        elif is_fgclip2 and has_config_json:
            self._load_fgclip2_model()
        elif has_config_json:
            self._load_transformers_clip_model()
        else:
            raise RuntimeError(f"无法识别模型格式。模型目录 {self.model_path} 中需要以下文件之一:\n  - open_clip_config.json (open_clip 格式)\n  - config.json (transformers 格式)")
    
    def _load_fgclip2_model(self):
        """加载 FG-CLIP2 模型 - v3.2 支持多实例"""
        if not HAS_TRANSFORMERS_CLIP:
            raise RuntimeError("FG-CLIP2 需要 transformers 库，请安装: pip install transformers")
        try:
            print(f"[Info] 检测到 FG-CLIP2 模型: {self.model_path}")
            print(f"[Info] 使用 transformers 库加载 (trust_remote_code=True)...")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
            
            # 加载 tokenizer 和 image_processor
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)
            self._image_processor = AutoImageProcessor.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)
            
            # FG-CLIP2 的 max_num_patches 计算函数
            def determine_max_patches(image):
                w, h = image.size
                max_val = (w // 16) * (h // 16)
                if max_val > 784:
                    return 1024
                elif max_val > 576:
                    return 784
                elif max_val > 256:
                    return 576
                elif max_val > 128:
                    return 256
                else:
                    return 128
            
            # 图像预处理函数
            image_processor = self._image_processor
            def preprocess_image(image):
                if isinstance(image, str):
                    image = Image.open(image).convert('RGB')
                max_patches = determine_max_patches(image)
                inputs = image_processor(images=image, max_num_patches=max_patches, return_tensors="pt")
                return inputs  # 返回完整的输入字典，不仅仅是 pixel_values
            
            self._preprocess = preprocess_image
            # [简化] 移除 Chinese CLIP 支持
            self.model_type = 'fgclip2'
            
            # 加载单个模型实例（与 CLIP transformers 一致）
            # 加载单个模型实例（与 CLIP transformers 一致）
            self._model_raw = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )

            # 直接修复 Fgclip2TextEmbeddings 中可能损坏的 position_ids/mask1/mask2
            text_cfg = getattr(self._model_raw.config, 'text_config', None) or self._model_raw.config
            keep_len = text_cfg.keep_len
            longtext_len = text_cfg.longtext_len
            for sub in self._model_raw.modules():
                if sub.__class__.__name__ == 'Fgclip2TextEmbeddings':
                    sub.position_ids = torch.arange(longtext_len).expand((1, -1))
                    m1 = torch.zeros([longtext_len, 1])
                    m1[:keep_len] = 1
                    sub.mask1 = m1
                    m2 = torch.zeros([longtext_len, 1])
                    m2[keep_len:] = 1
                    sub.mask2 = m2
                    print(f"[Fix] Fgclip2TextEmbeddings buffers reset: position_ids=arange({longtext_len}), mask1/2 with keep_len={keep_len}")

            self._model_raw = self._model_raw.to(DEVICE)
            self._model_raw = self._apply_fp16(self._model_raw)
            self._model_raw.eval()
            
            # 获取模型默认维度（必须在模型加载之后）
            try:
                config = self._model_raw.config
                if hasattr(config, 'projection_dim'):
                    self.model_default_dim = config.projection_dim
                elif hasattr(config, 'vision_config') and hasattr(config.vision_config, 'hidden_size'):
                    self.model_default_dim = config.vision_config.hidden_size
                elif hasattr(config, 'hidden_size'):
                    self.model_default_dim = config.hidden_size
                else:
                    self.model_default_dim = 768
                    print(f"[Warning] 无法从配置获取维度，使用默认值: {self.model_default_dim}")
            except Exception as e:
                self.model_default_dim = 768
                print(f"[Warning] 获取维度失败: {e}，使用默认值: {self.model_default_dim}")
            
            print(f"[Info] 模型默认维度: {self.model_default_dim}")
            
            self._model = _TransformersModelWrapper(self._model_raw, self.use_fp16, is_fgclip2=True)
            self._preprocess = preprocess_image
            
            self._processor = _SimpleProcessor(self._tokenizer, self._image_processor)
            
            print(f"[Info] FG-CLIP2 模型加载完成，设备: {DEVICE}, FP16: {self.use_fp16}")
            
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"FG-CLIP2 模型加载失败: {e}")
    
    def _load_transformers_clip_model(self):
        if not HAS_TRANSFORMERS_CLIP:
            raise RuntimeError("CLIP (transformers格式) 需要 transformers 库，请安装: pip install transformers")
        try:
            self._model_raw = CLIPModel.from_pretrained(self.model_path, local_files_only=True).to(DEVICE)
            self._model_raw = self._apply_fp16(self._model_raw)
            self._model_raw.eval()
            self._image_processor = CLIPImageProcessor.from_pretrained(self.model_path, local_files_only=True)
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            except:
                from transformers import CLIPTokenizer
                self._tokenizer = CLIPTokenizer.from_pretrained(self.model_path, local_files_only=True)
            def preprocess_image(image):
                if isinstance(image, str):
                    image = Image.open(image).convert('RGB')
                inputs = self._image_processor(images=image, return_tensors="pt")
                return inputs['pixel_values'].squeeze(0)
            self._preprocess = preprocess_image
            # [简化] 移除 Chinese CLIP 支持
            self.model_type = 'clip'
            self._processor = _SimpleProcessor(self._tokenizer, self._image_processor)
            self.model_default_dim = self._model_raw.config.projection_dim
            print(f"[Info] 模型默认维度: {self.model_default_dim}")
            self._model = _TransformersModelWrapper(self._model_raw, self.use_fp16, is_fgclip2=False)
        except Exception as e:
            raise RuntimeError(f"CLIP 模型加载失败: {e}")
    
    # [简化] 移除 _load_chinese_clip_model 函数，不再支持 Chinese CLIP
    
    def _load_openclip_model(self):
        """加载 open_clip 格式的模型（本地离线加载）"""
        import open_clip
        import json
        
        config_path = os.path.join(self.model_path, 'open_clip_config.json')
        model_path_bin = os.path.join(self.model_path, 'open_clip_pytorch_model.bin')
        model_path_safetensors = os.path.join(self.model_path, 'open_clip_model.safetensors')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 确定模型文件路径
        if os.path.exists(model_path_bin):
            pretrained_path = model_path_bin
        elif os.path.exists(model_path_safetensors):
            pretrained_path = model_path_safetensors
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path_bin} 或 {model_path_safetensors}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_cfg = config.get('model_cfg', {})
        model_name = model_cfg.get('model_name', None)
        
        # 从文件夹名推断模型架构（比配置文件更可靠）
        folder_name = os.path.basename(self.model_path)
        inferred_model_name = self._infer_openclip_arch_from_name(folder_name)
        
        if inferred_model_name:
            if model_name and model_name != inferred_model_name:
                print(f"[Warning] 配置文件 model_name={model_name} 与文件夹名不匹配，使用推断值: {inferred_model_name}")
            model_name = inferred_model_name
        elif model_name is None:
            raise RuntimeError(f"无法从文件夹名 '{folder_name}' 推断模型架构，且配置文件中无 model_name")
        
        if self.model_type == 'auto':
            self.model_type = 'clip'
        
        # 读取配置
        text_cfg = model_cfg.get('text_cfg', {})
        vision_cfg = model_cfg.get('vision_cfg', {})
        self._context_length = text_cfg.get('context_length', 77)
        embed_dim = model_cfg.get('embed_dim', 768)
        self.model_default_dim = embed_dim
        
        image_size = vision_cfg.get('image_size', 224)
        if isinstance(image_size, list):
            image_size = image_size[0]
        
        print(f"[Info] 加载 open_clip 模型: {model_name}, embed_dim={embed_dim}, image_size={image_size}")
        
        # 创建模型（传入完整配置以确保架构正确）
        self._model = open_clip.create_model(
            model_name,
            pretrained=None,
            device=DEVICE,
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            text_cfg=text_cfg,
        )
        
        # 加载权重
        state_dict = torch.load(pretrained_path, map_location=DEVICE, weights_only=True)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        self._model.load_state_dict(state_dict, strict=True)
        
        # 创建预处理函数
        try:
            from open_clip import image_transform_v2, PreprocessCfg
            self._preprocess = image_transform_v2(PreprocessCfg(size=image_size, mode='RGB'), is_train=False)
        except (ImportError, AttributeError):
            from torchvision import transforms
            self._preprocess = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                   std=(0.26862954, 0.26130258, 0.27577711)),
            ])
        
        self._model = self._apply_fp16(self._model)
        self._model.eval()
        self._model_raw = self._model
        self._tokenizer = self._load_tokenizer_local(model_name, config)
        print(f"[Info] open_clip 模型加载完成, FP16: {self.use_fp16}")
    
    def _infer_openclip_arch_from_name(self, folder_name: str) -> str:
        """从文件夹名推断 open_clip 模型架构
        
        例如：
        - laion_CLIP-ViT-B-32-256x256-DataComp-s34B-b86K -> ViT-B-32
        - laion_CLIP-ViT-L-14-DataComp.XL-s13B-b90K -> ViT-L-14
        - openai_clip-vit-large-patch14 -> ViT-L-14
        """
        folder_lower = folder_name.lower()
        
        # 常见的 ViT 架构模式
        patterns = [
            # ViT-B-32, ViT-L-14 等格式
            (r'vit-b-32|vit_b_32|vitb32', 'ViT-B-32'),
            (r'vit-b-16|vit_b_16|vitb16', 'ViT-B-16'),
            (r'vit-l-14|vit_l_14|vitl14', 'ViT-L-14'),
            (r'vit-l-16|vit_l_16|vitl16', 'ViT-L-16'),
            (r'vit-h-14|vit_h_14|vith14', 'ViT-H-14'),
            (r'vit-g-14|vit_g_14|vitg14', 'ViT-g-14'),
            (r'vit-bigg-14|vit_bigg_14|vitbigg14', 'ViT-bigG-14'),
            # Hugging Face 格式: vit-base-patch32, vit-large-patch14
            (r'vit-base-patch32|vit_base_patch32', 'ViT-B-32'),
            (r'vit-base-patch16|vit_base_patch16', 'ViT-B-16'),
            (r'vit-large-patch14|vit_large_patch14', 'ViT-L-14'),
            (r'vit-large-patch16|vit_large_patch16', 'ViT-L-16'),
            (r'vit-huge-patch14|vit_huge_patch14', 'ViT-H-14'),
            # RN (ResNet) 格式
            (r'rn50x64|rn-50x64', 'RN50x64'),
            (r'rn50x16|rn-50x16', 'RN50x16'),
            (r'rn50x4|rn-50x4', 'RN50x4'),
            (r'rn101|rn-101', 'RN101'),
            (r'rn50|rn-50', 'RN50'),
            # ConvNeXt 格式
            (r'convnext-base|convnext_base', 'convnext_base'),
            (r'convnext-large|convnext_large', 'convnext_large'),
            (r'convnext-xxlarge|convnext_xxlarge', 'convnext_xxlarge'),
        ]
        
        for pattern, arch_name in patterns:
            if re.search(pattern, folder_lower):
                return arch_name
        
        return None
    
    def _load_tokenizer_local(self, model_name: str, config: dict):
        """加载 open_clip 模型的 tokenizer"""
        import open_clip
        context_length = getattr(self, '_context_length', 77)
        tokenizer_config_path = os.path.join(self.model_path, 'tokenizer_config.json')
        tokenizer_path = os.path.join(self.model_path, 'tokenizer.json')
        if os.path.exists(tokenizer_path) or os.path.exists(tokenizer_config_path):
            try:
                from open_clip.tokenizer import HFTokenizer
                print(f"[Info] 从本地加载 HFTokenizer: {self.model_path}")
                hf_tokenizer = HFTokenizer(self.model_path, context_length=context_length)
                return hf_tokenizer
            except Exception as e:
                print(f"[Warning] HFTokenizer 加载失败: {e}, 尝试备用方法")
        try:
            print(f"[Info] 使用 open_clip 内置 tokenizer: {model_name} (context_length={context_length})")
            tokenizer = open_clip.get_tokenizer(model_name)
            def tokenize_with_context(texts):
                if isinstance(texts, str):
                    texts = [texts]
                tokens = tokenizer(texts)
                if tokens.shape[1] != context_length:
                    if tokens.shape[1] > context_length:
                        tokens = tokens[:, :context_length]
                    else:
                        padding = torch.zeros(tokens.shape[0], context_length - tokens.shape[1], dtype=tokens.dtype)
                        tokens = torch.cat([tokens, padding], dim=1)
                return tokens
            return tokenize_with_context
        except Exception as e:
            print(f"[Warning] open_clip.get_tokenizer 失败: {e}")
        try:
            from open_clip.tokenizer import SimpleTokenizer
            print(f"[Info] 使用 SimpleTokenizer (CLIP BPE)")
            simple_tokenizer = SimpleTokenizer()
            def tokenize_clip(texts):
                if isinstance(texts, str):
                    texts = [texts]
                all_tokens = []
                for text in texts:
                    sot_token = simple_tokenizer.encoder["<|startoftext|>"]
                    eot_token = simple_tokenizer.encoder["<|endoftext|>"]
                    tokens = [sot_token] + simple_tokenizer.encode(text) + [eot_token]
                    if len(tokens) > context_length:
                        tokens = tokens[:context_length]
                        tokens[-1] = eot_token
                    else:
                        tokens = tokens + [0] * (context_length - len(tokens))
                    all_tokens.append(tokens)
                return torch.tensor(all_tokens, dtype=torch.long)
            return tokenize_clip
        except Exception as e:
            print(f"[Warning] SimpleTokenizer 失败: {e}")
        spm_path = os.path.join(self.model_path, 'spiece.model')
        if os.path.exists(spm_path):
            try:
                import sentencepiece as spm
                print(f"[Info] 从本地加载 SentencePiece: {spm_path}")
                sp = spm.SentencePieceProcessor()
                sp.Load(spm_path)
                def tokenize_fn(texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    tokens = []
                    for text in texts:
                        encoded = sp.Encode(text, out_type=int)
                        if len(encoded) > context_length - 2:
                            encoded = encoded[:context_length - 2]
                        encoded = [1] + encoded + [2]
                        encoded = encoded + [0] * (context_length - len(encoded))
                        tokens.append(encoded)
                    return torch.tensor(tokens, dtype=torch.long)
                return tokenize_fn
            except Exception as e:
                print(f"[Warning] SentencePiece 加载失败: {e}")
        try:
            from transformers import AutoTokenizer
            print(f"[Info] 尝试使用 transformers AutoTokenizer (离线): {self.model_path}")
            hf_tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=False)
            def tokenize_fn(texts):
                if isinstance(texts, str):
                    texts = [texts]
                encoded = hf_tokenizer(texts, padding='max_length', truncation=True, max_length=context_length, return_tensors='pt')
                return encoded['input_ids']
            return tokenize_fn
        except Exception as e:
            print(f"[Warning] transformers AutoTokenizer 加载失败: {e}")
        raise RuntimeError(f"无法加载 tokenizer。请确保以下文件之一存在于 {self.model_path}:\n  - tokenizer.json\n  - tokenizer_config.json\n  - spiece.model\n或者模型是 open_clip 支持的内置模型。")
    
    def get_logit_scale(self) -> torch.Tensor:
        """获取模型的 logit_scale 参数"""
        with torch.no_grad():
            if hasattr(self._model_raw, 'logit_scale'):
                return self._model_raw.logit_scale.exp()
            elif hasattr(self._model, 'logit_scale'):
                return self._model.logit_scale.exp()
            else:
                return torch.tensor(100.0, device=DEVICE)
    
    def _encode_text_raw(self, texts: list) -> torch.Tensor:
        """文本编码核心逻辑 - 返回原始特征张量（未截断/归一化）
        
        统一 tokenization 分支，供 encode_text 等方法复用
        """
        is_transformers_format = hasattr(self, '_processor') and self._processor is not None
        if is_transformers_format:
            if hasattr(self._processor, 'tokenizer'):
                tokenizer = self._processor.tokenizer
            else:
                tokenizer = self._tokenizer
            if self.model_type == 'fgclip2':
                text_inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=196).to(DEVICE)
                return self._model.encode_text(text_inputs, walk_type="long")
            else:
                text_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEVICE)
                return self._model.encode_text(text_inputs)
        else:
            text_tokens = self._tokenizer(texts).to(DEVICE)
            return self._model.encode_text(text_tokens)
    
    def encode_text(self, texts: list, task: str = "retrieval", prompt_name: str = "query") -> np.ndarray:
        """文本编码 - 获取文本的特征向量（截断 + L2 归一化后转 numpy）"""
        if self._model is None:
            raise RuntimeError("模型未加载")
        
        features = None
        try:
            with torch.no_grad():
                features = self._encode_text_raw(texts)
                features = _truncate_and_normalize(features, self.truncate_dim)
                return features.detach().cpu().numpy()
        finally:
            del features

    def _extract_and_vectorize_parallel(self, video_path: str, sample_interval: int, output_resolution: str = '256', brightness_threshold: float = None, black_pixel_ratio: float = 98, white_threshold: float = None, white_pixel_ratio: float = 90) -> tuple:
        """单实例模式：CPU-GPU 流水线并行"""
        # v3.1: 不在 ffmpeg 中做黑帧检测，改为在 VectorizerThread 预加载时检测
        extractor = FrameExtractorThread(video_path, sample_interval, output_resolution)
        extractor.start()
        time.sleep(0.5)
        
        # 单模型实例模式
        vectorizer = VectorizerThread(
            extractor, self._model, self._preprocess, self.batch_size, self.truncate_dim,
            self.model_type, io_workers=self.io_workers, prefetch=True,
            brightness_threshold=brightness_threshold, black_pixel_ratio=black_pixel_ratio,
            white_threshold=white_threshold, white_pixel_ratio=white_pixel_ratio
        )
        vectorizer.start()
        extractor.extraction_done.wait()
        vectorizer.vectorization_done.wait()
        if extractor.error:
            raise RuntimeError(f"提取失败: {extractor.error}")
        if vectorizer.error:
            raise RuntimeError(f"向量化失败: {vectorizer.error}")
        frame_indices, features = vectorizer.get_results()
        # v3.1: 打印黑帧过滤统计
        if vectorizer._black_frames_filtered > 0:
            print(f"[GPU] 黑帧检测: 过滤了 {vectorizer._black_frames_filtered} 个黑帧")
        if vectorizer._white_frames_filtered > 0:
            print(f"[GPU] 白帧检测: 过滤了 {vectorizer._white_frames_filtered} 个白帧")
        extractor.cleanup()
        actual_frame_indices = [i * sample_interval for i in frame_indices]
        return actual_frame_indices, features

    def process_video(self, video_path: str, sample_interval: int = 5,
                  localmax_order: int = 2,
                  output_resolution: str = '512', min_scene_length: int = 7,
                  cosine_similarity_threshold: float = None,
                  brightness_threshold: float = 32,
                  black_pixel_ratio: float = 98,
                  white_threshold: float = None,
                  white_pixel_ratio: float = 90) -> list:
        """
        处理视频，检测场景边界并提取特征
        
        [简化] v3.6: 只保留三元组模式，移除 single_frame_mode 和 triplet_average_mode
        v2.9: 移除欧氏距离选项，仅保留余弦相似度
        
        Args:
            video_path: 视频文件路径
            sample_interval: 采样间隔
            localmax_order: 局部最大值检测的阶数
            output_resolution: 输出分辨率
            min_scene_length: 最小场景长度
            cosine_similarity_threshold: 余弦相似度阈值
            brightness_threshold: 黑场检测亮度阈值 (0-255)
            black_pixel_ratio: 黑色像素占比阈值 (0-100)
            white_threshold: 白场检测亮度阈值 (0-255)，None=禁用
            white_pixel_ratio: 白色像素占比阈值 (0-100)
        
        Returns:
            场景列表
        """
        # [简化] 只保留三元组模式
        print(f"[处理] {os.path.basename(video_path)} (模式: 三元组, 检测: 余弦相似度)")
        
        frame_indices, features = self._extract_and_vectorize_parallel(video_path, sample_interval, output_resolution, brightness_threshold=brightness_threshold, black_pixel_ratio=black_pixel_ratio, white_threshold=white_threshold, white_pixel_ratio=white_pixel_ratio)
        if not len(frame_indices):
            print(f"[Error] 未提取到帧: {video_path}")
            return []
        print(f"[完成] 提取+向量化: {len(frame_indices)} 帧 (GPU tensor)")
        
        boundary_frames = SceneBoundaryDetector.find_scene_boundaries(
            frame_indices, features, localmax_order, cosine_similarity_threshold
        )
        
        fps, total_frames, _, _ = VideoMetaHelper.get_video_meta(video_path)
        scenes = []
        for i in range(len(boundary_frames)):
            start = boundary_frames[i]
            # 修改: 使用采样帧作为场景结束帧，而不是连续帧
            if i + 1 < len(boundary_frames):
                # 找到下一个边界帧之前的最后一个采样帧
                end = SceneBoundaryDetector.find_prev_sample_frame(boundary_frames[i + 1], frame_indices, sample_interval)
            else:
                # 最后一个场景使用最后一个采样帧
                end = frame_indices[-1]
            if end > start:
                scenes.append((start, end))
        
        if min_scene_length > 1:
            scenes = SceneBoundaryDetector.merge_short_scenes(
                scenes, frame_indices, features, min_scene_length
            )
        
        # 提取每个场景的三元组特征并构建结果
        results = []
        for start, end in scenes:
            feats = SceneBoundaryDetector.get_scene_features((start, end), frame_indices, features)
            scene_data = {'start_frame': start, 'end_frame': end, 'features': []}
            for f in feats:
                if isinstance(f, torch.Tensor):
                    scene_data['features'].append(f.cpu())
                else:
                    scene_data['features'].append(torch.tensor(f, dtype=torch.float16))
            results.append(scene_data)
        
        print(f"[结果] {os.path.basename(video_path)}: {len(results)} 个场景")
        return results


class SemanticSearchEngine:
    """语义搜索引擎 - 使用模型原生 logit_scale 计算相似度（Lance 直读版）
    
    v4.0: 使用 Lance 列式存储直接读取，无需 NPZ 缓存
    v2.9: 归一化优化 - 索引中的向量已归一化，搜索时直接使用
    """
    
    def __init__(self, processor: EmbeddingModelProcessor, cache_dir: str):
        self.processor = processor
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _load_features_from_lance(self, lance_path: str) -> tuple:
        """
        从 Lance 索引加载特征（直接读取，无需缓存层）
        
        Returns:
            tuple: (features_np, scene_map, feature_counts)
        """
        try:
            features_np, scene_map, feature_counts = read_lance_index_raw(lance_path)
            if features_np is None:
                return None, None, None
            print(f"    [Lance读取] {os.path.basename(lance_path)}: {len(scene_map)} 场景, {features_np.shape[0]} 向量")
            return features_np, scene_map, feature_counts
        except Exception as e:
            print(f"    [Error] 无法加载 Lance 索引: {e}")
            return None, None, None
    
    def search_by_text(self, query_text: str, index_paths: list, similarity_threshold: float = 20.0) -> list:
        """
        文本语义搜索 - 使用模型原生 logit_scale * (text @ image.T) 计算相似度
        
        v3.6: 只保留三元组模式
        - 每个场景3个向量都参与匹配，取最大相似度
        
        v2.9: 归一化优化
        - 文本向量: encode_text 内部已归一化（1次）
        - 图片向量: pkl 中已归一化，直接使用（0次）
        """
        valid_index_paths = []
        for path in index_paths:
            if os.path.isdir(path) and path.endswith('.lance'):
                valid_index_paths.append(path)
            elif os.path.isfile(path):
                print(f"[Warning] 跳过非 Lance 文件: {path}")
            else:
                print(f"[Warning] 无效路径: {path}")
        if not valid_index_paths:
            raise ValueError("没有有效的索引文件")
        
        truncate_dim = self.processor.truncate_dim
        if truncate_dim is None:
            for path in valid_index_paths:
                filename = os.path.basename(path)
                match = re.search(r'_d(\d+)(?:_|\.|$)', filename)
                if match:
                    truncate_dim = int(match.group(1))
                    self.processor.truncate_dim = truncate_dim
                    print(f"[Info] 从文件名自动检测到 truncate_dim={truncate_dim}")
                    break
        
        print("[步骤A] 提取文本特征...")
        # ✅ v2.9: encode_text 内部已完成归一化
        query_vector = self.processor.encode_text([query_text], task="retrieval", prompt_name="query")
        if query_vector is None:
            raise ValueError("无法为查询文本生成特征向量")
        print(f"[Info] 查询向量维度: {query_vector.shape[1]}")
        
        # 获取 logit_scale
        logit_scale = self.processor.get_logit_scale()
        print(f"[Info] 使用 logit_scale: {logit_scale.item():.2f}")
        print(f"[Info] 相似度阈值: {similarity_threshold}")
        
        # 转换为 tensor
        query_tensor = torch.tensor(query_vector, device=DEVICE, dtype=torch.float16)
        
        print(f"[步骤B] 扫描 {len(valid_index_paths)} 个索引库...")
        all_results = []
        
        for i, path in enumerate(valid_index_paths):
            print(f"  [{i+1}/{len(valid_index_paths)}] {os.path.basename(path)}")
            try:
                # ✅ v4.0: 从 Lance 索引直接读取特征
                features_np, scene_map, feature_counts = self._load_features_from_lance(path)
                
                if features_np is None or scene_map is None:
                    print(f"    [Warning] 无场景数据: {os.path.basename(path)}")
                    continue
                
                # ✅ v2.9: 只在维度不匹配时截断并归一化
                if self.processor.truncate_dim is not None and features_np.shape[1] > self.processor.truncate_dim:
                    print(f"    [Info] 截断维度: {features_np.shape[1]} -> {self.processor.truncate_dim}")
                    features_np = features_np[:, :self.processor.truncate_dim]
                    # 截断后需要重新归一化
                    norms = np.linalg.norm(features_np, axis=1, keepdims=True)
                    features_np = features_np / (norms + 1e-8)
                
                # 检查维度匹配
                if features_np.shape[1] != query_vector.shape[1]:
                    print(f"    [Warning] 维度不匹配: 索引={features_np.shape[1]}, 查询={query_vector.shape[1]}, 跳过此索引")
                    continue
                
                # 转换为 tensor 送入 GPU
                image_features = torch.tensor(features_np, device=DEVICE, dtype=torch.float16)
                
                # 使用 logit_scale * (text_features @ image_features.T) 计算相似度
                with torch.no_grad():
                    similarities = (logit_scale * query_tensor.to(logit_scale.dtype) @ image_features.to(logit_scale.dtype).T).squeeze(0)
                
                # 转换为 numpy 处理结果
                similarities_np = similarities.cpu().numpy()
                
                # ✅ v3.0: 根据 feature_counts 分组，每个场景取最大相似度
                feat_idx = 0
                for scene_idx, count in enumerate(feature_counts):
                    # 获取该场景所有向量的相似度
                    scene_sims = similarities_np[feat_idx:feat_idx + count]
                    # 取最大值作为该场景的匹配分数
                    max_sim = float(np.max(scene_sims))
                    feat_idx += count
                    
                    if max_sim >= similarity_threshold:
                        original_scene = scene_map[scene_idx]
                        video_path = original_scene["video_path"]
                        fps = original_scene.get("fps") or VideoMetaHelper.get_fps_cached(video_path)
                        start_frame = original_scene["start_frame"]
                        end_frame = original_scene["end_frame"]
                        duration_frames = end_frame - start_frame
                        duration_seconds = round(VideoMetaHelper.frame_to_seconds(duration_frames, fps), 2)
                        video_name = os.path.basename(video_path)
                        duration_display = VideoMetaHelper.format_duration(duration_seconds)
                        all_results.append({
                            "video_path": video_path,
                            "video_name": video_name,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "duration_frames": duration_frames,
                            "duration_seconds": duration_seconds,
                            "duration_display": duration_display,
                            "fps": round(fps, 2),
                            "start": VideoMetaHelper.seconds_to_timecode(VideoMetaHelper.frame_to_seconds(start_frame, fps)),
                            "end": VideoMetaHelper.seconds_to_timecode(VideoMetaHelper.frame_to_seconds(end_frame, fps)),
                            "similarity": round(max_sim, 4)
                        })
                
                # 清理 GPU 内存
                del image_features
                del similarities
                
            except Exception as e:
                print(f"    [Error] 处理失败 {os.path.basename(path)}: {e}")
                traceback.print_exc()
                continue
        
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        print(f"[完成] 找到 {len(all_results)} 个结果")
        
        return all_results
    


def create_video_index(source_paths: list, output_dir: str, sample_interval: int = 5,
                       localmax_order: int = 2, output_resolution: str = '256',
                       min_scene_length: int = 7, model_name: str = None,
                       model_path: str = None, model_type: str = 'auto',
                       batch_size: int = 128, truncate_dim: int = None,
                       io_workers: int = 8, workers: int = 2,
                       cosine_similarity_threshold: float = None,
                       brightness_threshold: float = 32,
                       black_pixel_ratio: float = 98,
                       white_threshold: float = None,
                       white_pixel_ratio: float = 90,
                       resume_processing: bool = True,
                       use_fp16: bool = True):
    """
    创建视频索引（Lance 格式，异步写入）
    """
    if model_path is not None and model_name is None:
        model_name = model_path
    supported_ext = ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.flv')
    all_videos = set()
    for path in source_paths:
        if os.path.isfile(path) and path.lower().endswith(supported_ext):
            all_videos.add(path)
        elif os.path.isdir(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path) and item.lower().endswith(supported_ext):
                    all_videos.add(item_path)
    videos_to_process = sorted(list(all_videos))
    if not videos_to_process:
        print("[Error] 未找到视频文件")
        return None
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    print("[Info] 清理临时文件夹...")
    cleanup_temp_folder(TEMP_DIR)
    ref_path = source_paths[0].rstrip(os.sep)
    if os.path.isdir(ref_path):
        source_folder_name = os.path.basename(ref_path)
    else:
        source_folder_name = f"{os.path.basename(os.path.dirname(ref_path))}_{os.path.splitext(os.path.basename(ref_path))[0]}"
    clean_name = sanitize_name(source_folder_name) or "Untitled"
    dim_suffix = f"_d{truncate_dim}" if truncate_dim else ""
    if model_name:
        model_name_clean = os.path.basename(model_name.rstrip(os.sep))
    else:
        model_name_clean = "clip"
    model_name_clean = sanitize_name(model_name_clean) or "clip"
    # 默认按模型分目录存放索引: <output_dir>/<model_name_clean>/*.lance
    # 如果 output_dir 已经是该模型目录，则不重复嵌套。
    output_dir_base = os.path.basename(os.path.normpath(output_dir))
    if os.path.normcase(output_dir_base) == os.path.normcase(model_name_clean):
        model_output_dir = output_dir
    else:
        model_output_dir = os.path.join(output_dir, model_name_clean)
    os.makedirs(model_output_dir, exist_ok=True)

    final_index_file = os.path.join(model_output_dir, f"{clean_name}_{model_name_clean}{dim_suffix}.lance")
    # 非增量模式：删除同名 .lance 索引，避免 append 导致数据重复
    if not resume_processing and os.path.exists(final_index_file):
        print(f"[Info] resume_processing=False，删除同名索引重建: {os.path.basename(final_index_file)}")
        shutil.rmtree(final_index_file, ignore_errors=True)
    # 增量处理：获取已索引的视频路径
    indexed_videos = set()
    if resume_processing and os.path.exists(final_index_file):
        try:
            indexed_videos = get_indexed_video_paths(final_index_file)
        except Exception as e:
            print(f"[Error] 现有索引损坏,将重建: {e}")
            shutil.rmtree(final_index_file, ignore_errors=True)
    new_videos = [v for v in videos_to_process if v not in indexed_videos]
    if not new_videos:
        print(f"[结果] 所有 {len(videos_to_process)} 个视频均已索引")
        return final_index_file
    processor = EmbeddingModelProcessor(model_name, model_type, batch_size, truncate_dim, io_workers, use_fp16=use_fp16)
    async_writer = AsyncLanceWriter(final_index_file, max_workers=workers)
    total_time = 0
    for i, video_path in enumerate(new_videos):
        t_start = time.time()
        try:
            scenes = processor.process_video(
                video_path,
                sample_interval=sample_interval,
                localmax_order=localmax_order,
                output_resolution=output_resolution,
                min_scene_length=min_scene_length,
                cosine_similarity_threshold=cosine_similarity_threshold,
                brightness_threshold=brightness_threshold,
                black_pixel_ratio=black_pixel_ratio,
                white_threshold=white_threshold,
                white_pixel_ratio=white_pixel_ratio
            )
            if scenes:
                fps, total_frames, _, _ = VideoMetaHelper.get_video_meta(video_path)
                lance_rows = scenes_to_lance_rows(video_path, scenes, fps)
                if lance_rows:
                    async_writer.submit(lance_rows)
            total_time += time.time() - t_start
        except Exception as e:
            print(f"[Error] 处理失败 {video_path}: {e}")
            traceback.print_exc()
    # 等待所有异步写入完成
    async_writer.shutdown()
    cleanup_temp_folder(TEMP_DIR)
    total_indexed = len(indexed_videos) + async_writer.write_count
    print(f"[结果] 索引完成: {total_indexed} 个视频, 耗时 {total_time:.2f}s, 文件 {final_index_file}")
    return final_index_file


def batch_create_video_index(input_dir: str, output_dir: str, **kwargs):
    """
    批量创建视频索引
    
    [简化] v3.6: 只保留三元组模式
    v2.9: 移除欧氏距离选项，仅保留余弦相似度
    """
    if not os.path.exists(input_dir):
        print(f"[Error] 输入目录不存在: {input_dir}")
        return False
    subfolders = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)
    if not subfolders:
        print(f"[Error] 输入目录中没有子文件夹")
        return False
    os.makedirs(output_dir, exist_ok=True)
    print("[Info] 清理临时文件夹...")
    cleanup_temp_folder(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    errors = []
    for i, folder_path in enumerate(subfolders):
        print(f"\n{'='*60}")
        print(f"[批处理] [{i+1}/{len(subfolders)}] {os.path.basename(folder_path)}")
        print(f"{'='*60}")
        try:
            create_video_index([folder_path], output_dir, **kwargs)
        except Exception as e:
            errors.append(f"{os.path.basename(folder_path)}: {e}")
            print(f"[Error] {os.path.basename(folder_path)} 处理失败: {e}")
    print(f"\n{'='*60}")
    print(f"[结果] 批处理完成: 成功 {len(subfolders) - len(errors)}/{len(subfolders)}")
    if errors:
        print(f"[失败列表]:")
        for err in errors:
            print(f"  - {err}")
    print(f"{'='*60}")
    cleanup_temp_folder(TEMP_DIR)
    return len(errors) == 0


# ============================================================
#  测试入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("嵌入模型处理器 - CLIP/FG-CLIP2 通用版 (离线模式)")
    print("v3.6: 简化逻辑，移除 Chinese-CLIP 和单帧/三帧平均模式")
    print("v3.5: PKL文件名使用完整模型名，去掉Cosine后缀")
    print("v2.9: 移除欧氏距离选项，仅保留余弦相似度")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE', '未设置')}")
    print(f"transformers CLIP 支持: {'是' if HAS_TRANSFORMERS_CLIP else '否'}")
    # [简化] 移除 Chinese CLIP 支持
    print(f"默认模型: qihoo360-fg-clip2-base")
    print("支持的模型类型: clip, fgclip2, auto")
    # [简化] 只保留三元组模式
    print("\n存储模式: 三元组模式 (保存完整三帧向量)")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print("=" * 60)
