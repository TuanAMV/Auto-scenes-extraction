# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
import os
import sys
import hashlib

# ============================================================
#  路径设置 - 确保能找到项目根目录的模块
# ============================================================
_current_file = os.path.abspath(__file__)
_qwen_models_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_qwen_models_dir)
_cut_detect_scene_dir = os.path.dirname(_a_core_utils_dir)
if _cut_detect_scene_dir not in sys.path:
    sys.path.insert(0, _cut_detect_scene_dir)

# Force offline loading (no remote download)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import numpy as np
import logging
import unicodedata
import pickle

from PIL import Image
from scipy import special
from typing import List, Union, Optional, Dict, Tuple
from urllib.parse import urlparse
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)

# Default configuration constants
MAX_LENGTH = 10240
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR  # 4 tokens
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR  # 1800 tokens
FPS = 1
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS  # 7680 tokens


def is_image_path(path: str) -> bool:
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
    
    if path.startswith(('http://', 'https://')):
        # Parse URL to remove query parameters
        parsed_url = urlparse(path)
        clean_path = parsed_url.path
    else:
        clean_path = path
    
    # Check file extension
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions


def is_video_input(video) -> bool:
    if isinstance(video, str):
        return True
    
    if isinstance(video, list) and len(video) > 0:
        # Check first element to determine the type
        first_elem = video[0]
        
        if isinstance(first_elem, Image.Image):
            return True
        
        if isinstance(first_elem, str):
            return is_image_path(first_elem)
    
    return False


def sample_frames(
    frames: List[Union[str, Image.Image]], 
    max_segments: int
) -> List[Union[str, Image.Image]]:
    duration = len(frames)
    if duration <= max_segments:
        return frames

    frame_id_array = np.linspace(0, duration - 1, max_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    sampled_frames = [frames[frame_idx] for frame_idx in frame_id_list]
    return sampled_frames


class Qwen3VLReranker():
    
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Given a search query, retrieve relevant candidates that answer the query.",
        cache_dir: str = None,
        **kwargs,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen3-VL Reranker only supports GPU execution. CUDA is not available."
            )
        self.device = torch.device("cuda")

        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction
        
        # 图像预处理缓存
        self.cache_dir = cache_dir
        self._image_cache: Dict[str, Dict] = {}  # {image_key: preprocessed_data}
        self._pil_cache: Dict[str, Image.Image] = {}  # {image_key: PIL Image} 用于标签中心模式

        # Enforce local-only loading unless caller overrides
        if "local_files_only" not in kwargs:
            kwargs["local_files_only"] = True

        # Load the language model
        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **kwargs
        ).to(self.device)

        self.model = lm.model
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side='left',
            **kwargs
        )
        self.model.eval()
        self._lm = lm  # 保存引用以便后续释放

        # Initialize binary classification head for yes/no scoring
        token_true_id = self.processor.tokenizer.get_vocab()["yes"]
        token_false_id = self.processor.tokenizer.get_vocab()["no"]
        self.score_linear = self.get_binary_linear(lm, token_true_id, token_false_id)
        self.score_linear.eval()
        self.score_linear.to(self.device).to(self.model.dtype)

    def cleanup(self):
        """释放模型占用的显存"""
        import gc
        try:
            # 清理图像缓存
            if hasattr(self, '_image_cache'):
                self._image_cache.clear()
            if hasattr(self, '_pil_cache'):
                self._pil_cache.clear()
            if hasattr(self, 'score_linear') and self.score_linear is not None:
                del self.score_linear
                self.score_linear = None
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            if hasattr(self, '_lm') and self._lm is not None:
                del self._lm
                self._lm = None
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
                self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Qwen3VLReranker 模型已释放")
        except Exception as e:
            logger.warning(f"释放 Qwen3VLReranker 模型时出错: {e}")
    
    # ============================================================
    #  图像预处理缓存方法
    # ============================================================
    
    def _get_image_key(self, image_path: str) -> str:
        """生成图像的唯一键"""
        # 使用文件路径和修改时间生成键
        if os.path.exists(image_path):
            mtime = os.path.getmtime(image_path)
            key_str = f"{image_path}_{mtime}_{self.min_pixels}_{self.max_pixels}"
        else:
            key_str = f"{image_path}_{self.min_pixels}_{self.max_pixels}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def preprocess_image(self, image_path: str) -> Optional[Dict]:
        """
        预处理单张图像并缓存
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            预处理后的数据字典，包含 pixel_values 等
        """
        image_key = self._get_image_key(image_path)
        
        # 检查内存缓存
        if image_key in self._image_cache:
            return self._image_cache[image_key]
        
        try:
            # 构建图像内容
            image_content = image_path if image_path.startswith(('http://', 'https://')) else 'file://' + image_path
            
            # 处理图像
            images, _, _ = process_vision_info(
                [{'role': 'user', 'content': [{'type': 'image', 'image': image_content, 
                                               'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels}]}],
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )
            
            if images:
                # 使用 processor 处理图像
                processed = self.processor(
                    images=images,
                    return_tensors="pt",
                    do_resize=False
                )
                
                # 缓存结果
                cache_data = {
                    'pixel_values': processed.get('pixel_values'),
                    'image_grid_thw': processed.get('image_grid_thw'),
                }
                self._image_cache[image_key] = cache_data
                return cache_data
                
        except Exception as e:
            logger.warning(f"预处理图像失败 {image_path}: {e}")
        
        return None
    
    def preprocess_images_batch(self, image_paths: List[str]) -> Dict[str, Dict]:
        """
        批量预处理图像
        
        Args:
            image_paths: 图像路径列表
        
        Returns:
            {image_path: preprocessed_data}
        """
        results = {}
        new_paths = []
        
        # 检查哪些需要处理
        for path in image_paths:
            image_key = self._get_image_key(path)
            if image_key in self._image_cache:
                results[path] = self._image_cache[image_key]
            else:
                new_paths.append(path)
        
        # 处理新图像
        if new_paths:
            logger.info(f"[图像缓存] 预处理 {len(new_paths)} 张新图像 (缓存命中 {len(results)})")
            for path in new_paths:
                data = self.preprocess_image(path)
                if data:
                    results[path] = data
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            'cached_images': len(self._image_cache),
            'cache_dir': self.cache_dir
        }
    
    def clear_cache_batch(self, image_paths: List[str]) -> int:
        """
        批量清理指定图像的缓存
        
        用于流式 Reranker 场景：推理完一批后立即清理缓存，控制内存占用
        
        Args:
            image_paths: 图像路径列表
        
        Returns:
            实际清理的缓存数量
        """
        cleared = 0
        for path in image_paths:
            image_key = self._get_image_key(path)
            if image_key in self._image_cache:
                del self._image_cache[image_key]
                cleared += 1
        return cleared

    def get_binary_linear(self, model, token_yes: int, token_no: int) -> torch.nn.Linear:
        lm_head_weights = model.lm_head.weight.data

        weight_yes = lm_head_weights[token_yes]
        weight_no = lm_head_weights[token_no]

        D = weight_yes.size()[0]
        linear_layer = torch.nn.Linear(D, 1, bias=False)
        with torch.no_grad():
            linear_layer.weight[0] = weight_yes - weight_no
        return linear_layer

    @torch.no_grad()
    def compute_scores(self, inputs: Dict) -> List[float]:
        batch_scores = self.model(**inputs).last_hidden_state[:, -1]
        scores = self.score_linear(batch_scores)
        scores = torch.sigmoid(scores).squeeze(-1).cpu().detach().tolist()
        return scores

    def truncate_tokens_optimized(
        self,
        tokens: List[str],
        max_length: int,
        special_tokens: List[str]
    ) -> List[str]:
        if len(tokens) <= max_length:
            return tokens

        special_tokens_set = set(special_tokens)

        # Calculate budget: how many non-special tokens we can keep
        num_special = sum(1 for token in tokens if token in special_tokens_set)
        num_non_special_to_keep = max_length - num_special

        # Build final list according to budget
        final_tokens = []
        non_special_kept_count = 0
        for token in tokens:
            if token in special_tokens_set:
                final_tokens.append(token)
            elif non_special_kept_count < num_non_special_to_keep:
                final_tokens.append(token)
                non_special_kept_count += 1

        return final_tokens

    def tokenize(self, pairs: List[Dict], **kwargs) -> Dict:
        max_length = self.max_length
        text = self.processor.apply_chat_template(pairs, tokenize=False, add_generation_prompt=True)
        
        try:
            images, videos, video_kwargs = process_vision_info(
                pairs,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            videos = None
            video_kwargs = {'do_sample_frames': False}
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}],
                add_generation_prompt=True,
                tokenize=False
            )
        
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
            
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            truncation=False,
            padding=False,
            do_resize=False,
            **video_kwargs
        )
        
        # Truncate input IDs while preserving special tokens
        for i, ele in enumerate(inputs['input_ids']):
            truncated = self.truncate_tokens_optimized(
                ele[:-5],
                max_length,
                self.processor.tokenizer.all_special_ids
            ) + ele[-5:]
            inputs['input_ids'][i] = truncated

            # 同步截断 mm_token_type_ids，保持与 input_ids 长度一致
            if 'mm_token_type_ids' in inputs:
                new_len = len(truncated)
                mm_ids = inputs['mm_token_type_ids'][i]
                if new_len < len(mm_ids):
                    front = mm_ids[:new_len - 5]
                    back = mm_ids[-5:]
                    if isinstance(mm_ids, torch.Tensor):
                        inputs['mm_token_type_ids'][i] = torch.cat([front, back])
                    else:
                        inputs['mm_token_type_ids'][i] = front + back
            
        # Apply padding using tokenizer's pad() method
        # 使用 tokenizer 的 pad() 方法进行 padding
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*fast tokenizer.*")
            temp_inputs = self.processor.tokenizer.pad(
                {'input_ids': inputs['input_ids']},
                padding=True,
                return_tensors="pt"
            )
        for key in temp_inputs:
            inputs[key] = temp_inputs[key]

        # 补齐 mm_token_type_ids 使其与 padded input_ids 长度一致
        if 'mm_token_type_ids' in inputs and isinstance(inputs['mm_token_type_ids'], list):
            mm_list = inputs['mm_token_type_ids']
            from torch.nn.utils.rnn import pad_sequence
            # 确保每个元素都是 tensor
            as_tensors = [t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in mm_list]
            inputs['mm_token_type_ids'] = pad_sequence(as_tensors, batch_first=True, padding_value=0)

        return inputs

    def format_mm_content(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        prefix: str = 'Query:',
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict]:
        content = []
        content.append({'type': 'text', 'text': prefix})
        
        # Normalize text input to list
        if text is None:
            texts = []
        elif isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Normalize image input to list
        if image is None:
            images = []
        elif not isinstance(image, list):
            images = [image]
        else:
            images = image
        
        # Normalize video input to list
        if video is None:
            videos = []
        elif is_video_input(video):
            videos = [video]
        else:
            # Assume it's a list of videos
            videos = video

        if not texts and not images and not videos:
            content.append({'type': 'text', 'text': "NULL"})
            return content

        # Process each video
        for vid in videos:
            video_content = None
            video_kwargs = {'total_pixels': self.total_pixels}
            
            if isinstance(vid, list):
                # Video as frame sequence
                video_content = vid
                if self.max_frames is not None:
                    video_content = sample_frames(video_content, self.max_frames)
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele)
                    for ele in video_content
                ]
            elif isinstance(vid, str):
                # Video as file path
                video_content = vid if vid.startswith(('http://', 'https://')) else 'file://' + vid
                video_kwargs = {'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames}
            else:
                raise TypeError(f"Unrecognized video type: {type(vid)}")

            # Add video input to content
            if video_content:
                content.append({
                    'type': 'video',
                    'video': video_content,
                    **video_kwargs
                })

        # Process each image
        for img in images:
            image_content = None
            
            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = img if img.startswith(('http://', 'https://')) else 'file://' + img
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")

            # Add image input to content
            if image_content:
                content.append({
                    'type': 'image',
                    'image': image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })

        # Process each text
        for txt in texts:
            content.append({'type': 'text', 'text': txt})
        
        return content

    def format_mm_instruction(
        self,
        query_text: Optional[Union[str, tuple]] = None,
        query_image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        query_video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        doc_text: Optional[Union[List[str], str]] = None,
        doc_image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        doc_video: Optional[Union[List[Union[str, List[Union[str, Image.Image]]]], str, List[Union[str, Image.Image]]]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict]:
        inputs = []
        inputs.append({
            "role": "system",
            "content": [{
                "type": "text",
                "text": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            }]
        })
        
        # Handle query_text as tuple containing (instruction, text)
        if isinstance(query_text, tuple):
            instruct, query_text = query_text
        else:
            instruct = instruction
            
        contents = []
        contents.append({
            "type": "text",
            "text": '<Instruct>: ' + (instruct or self.default_instruction)
        })
        
        # Format query content
        query_content = self.format_mm_content(
            query_text, query_image, query_video,
            prefix='<Query>:',
            fps=fps,
            max_frames=max_frames
        )
        contents.extend(query_content)
        
        # Format document content
        doc_content = self.format_mm_content(
            doc_text, doc_image, doc_video,
            prefix='\n<Document>:',
            fps=fps,
            max_frames=max_frames
        )
        contents.extend(doc_content)
        
        inputs.append({
            "role": "user",
            "content": contents
        })
        
        return inputs

    def process(
        self,
        inputs: Dict,
        batch_size: int = 8,
    ) -> List[float]:
        """
        批量处理 query-document 对，计算相关性分数
        
        Args:
            inputs: 包含 query 和 documents 的字典
                - query: {'text': str, 'image': str, 'video': str}
                - documents: [{'text': str, 'image': str, 'video': str}, ...]
            batch_size: 批量推理大小，控制每次 GPU 推理的 pair 数量
        
        Returns:
            每个 document 的相关性分数列表
        """
        instruction = inputs.get('instruction', self.default_instruction)

        query = inputs.get("query", {})
        documents = inputs.get("documents", [])
        
        if not query or not documents:
            return []

        # 1. 一次性构建所有 (query, document) pairs
        pairs = [
            self.format_mm_instruction(
                query.get('text', None),
                query.get('image', None),
                query.get('video', None),
                document.get('text', None),
                document.get('image', None),
                document.get('video', None),
                instruction=instruction,
                fps=inputs.get('fps', self.fps),
                max_frames=inputs.get('max_frames', self.max_frames)
            )
            for document in documents
        ]

        # 2. 批量推理（按 batch_size 分批）
        final_scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            # 批量 tokenize（一次处理 batch_size 个 pairs）
            tokenized_inputs = self.tokenize(batch_pairs)
            tokenized_inputs = tokenized_inputs.to(self.model.device)
            
            # 批量推理（一次 GPU 前向传播）
            batch_scores = self.compute_scores(tokenized_inputs)
            final_scores.extend(batch_scores)
            
        return final_scores

    def compute_score(
        self,
        query_text: str,
        image_path: str,
        instruction: Optional[str] = None
    ) -> float:
        """
        便捷方法：计算单个查询文本与图像的相关性分数
        
        Args:
            query_text: 查询文本（如标签名称）
            image_path: 图像文件路径
            instruction: 可选的指令文本
        
        Returns:
            相关性分数 (0-1 之间的浮点数)
        """
        inputs = {
            'instruction': instruction or self.default_instruction,
            'query': {
                'text': query_text
            },
            'documents': [
                {
                    'image': image_path
                }
            ]
        }
        
        scores = self.process(inputs)
        
        if scores:
            return scores[0]
        return 0.0
    
    def clear_pil_cache(self) -> int:
        """
        清理 PIL 缓存
        
        Returns:
            清理的图像数量
        """
        count = len(self._pil_cache)
        self._pil_cache.clear()
        return count
