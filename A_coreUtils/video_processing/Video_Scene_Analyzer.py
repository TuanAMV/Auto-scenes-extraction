# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
"""
视频场景分析器 v5 - 使用FFmpeg替代cv2进行帧提取
HuggingFace模型：MiniCPM-V-4_5, MiniCPM-V-4_5-int4（使用transformers）
GGUF模型：*.gguf文件（使用llama-mtmd-cli）

核心改进:
- 使用 FFmpegPrecisionCutter 进行高精度帧提取，替代 cv2
- 更准确的帧定位和更好的兼容性
"""

import os
import sys
import torch
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import re
import json
import subprocess

# ============================================================
#  路径设置 - 确保能找到项目根目录的模块
# ============================================================
_current_file = os.path.abspath(__file__)
_video_processing_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_video_processing_dir)
_cut_detect_scene_dir = os.path.dirname(_a_core_utils_dir)
if _cut_detect_scene_dir not in sys.path:
    sys.path.insert(0, _cut_detect_scene_dir)

# 导入路径解析器
from path_resolver import PathResolver

# 导入 FFmpegPrecisionCutter（从同一目录）
from .ffmpeg_precision_cutter import FFmpegPrecisionCutter, VideoInfo


class VideoSceneAnalyzer:
    """视频场景分析器 - 支持HuggingFace和GGUF格式模型"""
    
    # 预设模型别名
    MODEL_ALIASES = {
        'hf': 'models/MiniCPM-V-4_5',
        'hf-int4': 'models/MiniCPM-V-4_5-int4',
        'gguf-f16': 'models/MiniCPM-V-4_5-F16.gguf',
        'gguf-q4': 'models/MiniCPM-V-4_5-Q4_K_M.gguf',
    }
    
    def __init__(
        self, 
        config_file: Optional[str] = None,
        model_path: Optional[str] = None,
        model_type: str = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        device: str = "cuda",
        trust_remote_code: bool = True,
        compression_ratio: float = 0.5,
        n_ctx: int = 4096,
        n_gpu_layers: int = 99,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 100,
        repeat_penalty: float = 1.05
    ):
        """
        初始化分析器
        
        参数:
            model_path: 模型路径或别名
            compression_ratio: 图片压缩比例（用于缩放提取的帧）
        """
        # 初始化路径解析器（不传参数，使用 path_resolver.py 所在目录作为项目根目录）
        self.path_resolver = PathResolver()
        self.script_dir = self.path_resolver.get_project_root()
        
        self.config = self._load_config(config_file)
        
        # 解析模型路径
        if model_path:
            actual_model_path = self.MODEL_ALIASES.get(model_path, model_path)
        else:
            actual_model_path = self.config.get('model_path', 'models/MiniCPM-V-4_5')
        
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.compression_ratio = compression_ratio
        
        self.temp_dir = self.path_resolver.join(self.config.get('temp_dir', 'temp'))
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        self.model_path = self._resolve_path(actual_model_path)
        
        # 自动检测模型类型
        if model_type == "auto":
            if str(self.model_path).endswith('.gguf'):
                self.model_type = "gguf"
            else:
                self.model_type = "huggingface"
        else:
            self.model_type = model_type
        
        # GGUF CLI参数
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.cli_temperature = temperature
        self.cli_top_p = top_p
        self.cli_top_k = top_k
        self.cli_repeat_penalty = repeat_penalty
        
        # HuggingFace参数
        self.dtype = self._parse_dtype(dtype)
        self.attn_implementation = attn_implementation
        
        # 获取CLI路径（仅GGUF模式需要）
        if self.model_type == "gguf":
            self.cli_path = self._get_cli_path()
            self.mmproj_path = self._get_mmproj_path()
        else:
            self.cli_path = None
            self.mmproj_path = None
        
        # 加载提示词
        self.prompts = self._load_prompts()
        
        # 初始化 FFmpegPrecisionCutter（用于帧提取）
        self.ffmpeg_cutter = FFmpegPrecisionCutter(
            temp_dir=str(self.temp_dir),
            verbose=False
        )
        
        self._load_model()
    
    def _load_config(self, config_file: Optional[str] = None) -> dict:
        """加载配置文件"""
        default_config = {
            'model_path': 'MiniCPM-V-4_5',
            'llama_cli_path': 'models/llama/llama-mtmd-cli.exe',
            'mmproj_path': 'mmproj-model-f16.gguf',
            'default_video_dir': '.',
            'temp_dir': 'temp',
            'prompt_file': 'prompt.txt'
        }
        
        if config_file is None:
            config_path = self.script_dir / "config.json"
        else:
            config_path = self._resolve_path(config_file)
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"错误：配置文件加载失败 - {e}")
        
        return default_config
    
    def _load_prompts(self) -> dict:
        """从prompt.txt加载提示词"""
        prompts = {
            'analysis_prompt': '',
            'scoring_prompt': '',
            'compare_subjects_prompt': ''
        }
        
        prompt_file = self.config.get('prompt_file', 'prompt.txt')
        prompt_path = self._resolve_path(prompt_file)
        
        if not prompt_path.exists():
            print(f"错误：提示词文件不存在 - {prompt_path}")
            return prompts
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            sections = re.split(r'\[(\w+_prompt)\]', content)
            
            current_key = None
            for section in sections:
                section = section.strip()
                if section.endswith('_prompt'):
                    current_key = section
                elif current_key and section:
                    prompts[current_key] = section.strip()
                    current_key = None
                    
        except Exception as e:
            print(f"错误：提示词文件加载失败 - {e}")
        
        return prompts
    
    def _resolve_path(self, path: str) -> Path:
        """智能解析路径（使用统一的PathResolver）"""
        return self.path_resolver.resolve(path, base_dir=self.script_dir)
    
    def _get_cli_path(self) -> Path:
        """获取llama-mtmd-cli路径"""
        cli_path_str = self.config.get('llama_cli_path', 'models/llama/llama-mtmd-cli.exe')
        cli_path = self._resolve_path(cli_path_str)
        
        if not cli_path.exists():
            raise FileNotFoundError(f"错误：llama-mtmd-cli未找到 - {cli_path}")
        
        return cli_path
    
    def _get_mmproj_path(self) -> Path:
        """获取mmproj路径"""
        mmproj_path_str = self.config.get('mmproj_path', 'mmproj-model-f16.gguf')
        mmproj_path = self._resolve_path(mmproj_path_str)
        
        if not mmproj_path.exists():
            raise FileNotFoundError(f"错误：mmproj文件未找到 - {mmproj_path}")
        
        return mmproj_path
    
    def _parse_dtype(self, dtype: str):
        """解析数据类型字符串"""
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
            'bf16': torch.bfloat16,
            'fp16': torch.float16,
            'fp32': torch.float32,
        }
        return dtype_map.get(dtype.lower(), torch.bfloat16)
    
    def _load_model(self):
        """加载模型"""
        try:
            if self.model_type == "gguf":
                pass  # CLI模式无需预加载
            else:
                self._load_huggingface_model()
        except Exception as e:
            print(f"错误：模型加载失败 - {e}")
            raise
    
    def _load_huggingface_model(self):
        """加载HuggingFace格式模型"""
        from transformers import AutoModel, AutoTokenizer
        
        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            trust_remote_code=self.trust_remote_code,
            attn_implementation=self.attn_implementation,
            dtype=self.dtype
        )
        
        if self.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.eval().cuda()
        else:
            self.model = self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=self.trust_remote_code
        )
    
    def _call_cli(
        self,
        image_paths: List[str],
        prompt: str,
        max_tokens: int = 512
    ) -> str:
        """调用llama-mtmd-cli（仅用于GGUF模型）"""
        images_str = ','.join(str(p) for p in image_paths)
        
        cmd = [
            str(self.cli_path),
            '-m', str(self.model_path),
            '--mmproj', str(self.mmproj_path),
            '-c', str(self.n_ctx),
            '--temp', str(self.cli_temperature),
            '--top-p', str(self.cli_top_p),
            '--top-k', str(self.cli_top_k),
            '--repeat-penalty', str(self.cli_repeat_penalty),
            '-ngl', str(self.n_gpu_layers),
            '-n', str(max_tokens),
            '--image', images_str,
            '-p', prompt
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=300
            )
            
            if result.returncode != 0:
                print(f"错误：CLI执行失败 - {result.stderr}")
                raise RuntimeError(f"CLI执行失败，返回码: {result.returncode}")
            
            output = result.stdout
            lines = output.strip().split('\n')
            
            response_lines = []
            for line in lines:
                if any(x in line for x in ['llm_load', 'llama_model', 'llama_new_context', 
                                            'ggml_', 'system_info:', 'sampling:']):
                    continue
                if 'tokens per second' in line or 'generated' in line:
                    continue
                if line.strip():
                    response_lines.append(line)
            
            response = '\n'.join(response_lines).strip()
            
            if not response:
                response = output.strip()
            
            return response
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("错误：CLI执行超时（5分钟）")
        except Exception as e:
            raise RuntimeError(f"错误：CLI调用失败 - {e}")
    
    def compress_image(self, image_path: str, output_path: str, ratio: Optional[float] = None):
        """压缩图片"""
        ratio = ratio if ratio is not None else self.compression_ratio
        
        img = Image.open(image_path)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        img_resized.save(output_path, 'JPEG', quality=95, optimize=True)
    
    def _get_real_frame_count(self, video_path: str) -> int:
        """
        获取视频真实帧数（使用FFmpeg，替代cv2）
        """
        video_path_resolved = self._resolve_path(video_path)
        return self.ffmpeg_cutter.get_real_frame_count(str(video_path_resolved))
    
    def extract_middle_frame(
        self,
        video_path: str,
        output_path: str,
        compress: bool = True,
        ratio: Optional[float] = None
    ) -> str:
        """
        提取视频中间帧（使用FFmpeg，替代cv2）
        """
        video_path_resolved = self._resolve_path(video_path)
        
        if not video_path_resolved.exists():
            raise FileNotFoundError(f"错误：视频文件不存在 - {video_path_resolved}")
        
        # 计算缩放尺寸
        scale = None
        if compress:
            ratio = ratio if ratio is not None else self.compression_ratio
            info = self.ffmpeg_cutter.get_video_info(str(video_path_resolved))
            if info and info.width > 0 and info.height > 0:
                scale = (int(info.width * ratio), int(info.height * ratio))
        
        # 使用临时路径提取
        temp_path = str(self.temp_dir / "temp_middle_frame.jpg")
        
        try:
            result = self.ffmpeg_cutter.extract_middle_frame(
                str(video_path_resolved),
                temp_path,
                quality=2,
                scale=scale
            )
            
            # 移动到目标路径
            if result and os.path.exists(temp_path):
                if temp_path != output_path:
                    import shutil
                    shutil.move(temp_path, output_path)
                return output_path
            else:
                raise ValueError("无法提取中间帧")
                
        except Exception as e:
            raise ValueError(f"提取中间帧失败: {str(e)}")
    
    def extract_frames(
        self, 
        video_path: str,
        start_frame_index: int = 3,
        end_frame_offset: int = 3,
        compress: bool = False,
        ratio: Optional[float] = None
    ) -> Tuple[str, str]:
        """
        从视频中提取首尾帧（使用FFmpeg，替代cv2）
        
        参数:
            video_path: 视频路径
            start_frame_index: 起始帧索引（从1开始）
            end_frame_offset: 结束帧偏移量
            compress: 是否压缩（通过缩放实现）
            ratio: 压缩比例
            
        返回:
            (起始帧路径, 结束帧路径)
        """
        video_path_resolved = self._resolve_path(video_path)
        
        if not video_path_resolved.exists():
            raise FileNotFoundError(f"错误：视频文件不存在 - {video_path_resolved}")
        
        # 计算缩放尺寸
        scale = None
        if compress:
            ratio = ratio if ratio is not None else self.compression_ratio
            info = self.ffmpeg_cutter.get_video_info(str(video_path_resolved))
            if info and info.width > 0 and info.height > 0:
                scale = (int(info.width * ratio), int(info.height * ratio))
        
        # 使用 FFmpegPrecisionCutter 提取首尾帧
        start_frame_path, end_frame_path = self.ffmpeg_cutter.extract_start_end_frames(
            str(video_path_resolved),
            start_frame_index=start_frame_index,
            end_frame_offset=end_frame_offset,
            output_dir=str(self.temp_dir),
            quality=2,
            scale=scale
        )
        
        # 重命名为期望的文件名
        final_start_path = str(self.temp_dir / "start_frame.jpg")
        final_end_path = str(self.temp_dir / "end_frame.jpg")
        
        import shutil
        if start_frame_path != final_start_path:
            shutil.move(start_frame_path, final_start_path)
        if end_frame_path != final_end_path:
            shutil.move(end_frame_path, final_end_path)
        
        return final_start_path, final_end_path
    
    def extract_three_frames(
        self,
        video_path: str,
        start_offset: int = 3,
        end_offset: int = 3,
        compress: bool = False,
        ratio: Optional[float] = None
    ) -> Tuple[str, str, str]:
        """
        提取首中尾三帧（使用FFmpeg）
        
        返回:
            (起始帧路径, 中间帧路径, 结束帧路径)
        """
        video_path_resolved = self._resolve_path(video_path)
        
        if not video_path_resolved.exists():
            raise FileNotFoundError(f"错误：视频文件不存在 - {video_path_resolved}")
        
        # 计算缩放尺寸
        scale = None
        if compress:
            ratio = ratio if ratio is not None else self.compression_ratio
            info = self.ffmpeg_cutter.get_video_info(str(video_path_resolved))
            if info and info.width > 0 and info.height > 0:
                scale = (int(info.width * ratio), int(info.height * ratio))
        
        # 使用 FFmpegPrecisionCutter 提取三帧
        frames, frame_nums = self.ffmpeg_cutter.extract_key_frames(
            str(video_path_resolved),
            start_offset=start_offset,
            end_offset=end_offset,
            include_middle=True,
            output_dir=str(self.temp_dir),
            quality=2,
            scale=scale
        )
        
        if len(frames) < 3:
            raise ValueError("无法提取三帧")
        
        return frames[0], frames[1], frames[2]
    
    def _build_analysis_prompt(self) -> str:
        """从prompts获取分析提示词"""
        return self.prompts.get('analysis_prompt', '')
    
    def _build_scoring_prompt(self) -> str:
        """从prompts获取评分提示词"""
        return self.prompts.get('scoring_prompt', '')
    
    def _get_compare_subjects_prompt(self) -> str:
        """从prompts获取主体比较提示词"""
        return self.prompts.get('compare_subjects_prompt', '')
    
    def _parse_score(self, answer: str) -> int:
        """解析评分结果"""
        try:
            score_match = re.search(r'\d+', answer.strip())
            if score_match:
                score = int(score_match.group())
                return max(1, min(10, score))
            else:
                return 4
        except:
            return 4
    
    def _parse_answer(self, answer: str) -> Dict[str, str]:
        """解析AI回答"""
        result = {
            'shot_scale': '',
            'emotion': '',
            'subject': '',
            'action': '',
            'background': '',
            'camera_movement': ''
        }
        
        patterns = {
            'shot_scale': (r'景别[：:]\s*([^\n]+)', 6),
            'emotion': (r'情绪[：:]\s*([^\n]+)', 4),
            'subject': (r'主体[：:]\s*([^\n]+)', 2),
            'action': (r'动作[：:]\s*([^\n]+)', 2),
            'background': (r'背景[：:]\s*([^\n]+)', 2),
            'camera_movement': (r'镜头[：:]\s*([^\n]+)', 6)
        }
        
        for key, (pattern, max_len) in patterns.items():
            match = re.search(pattern, answer)
            if match:
                value = match.group(1).strip()
                value = re.sub(r'[【】\[\]（）()]', '', value)
                result[key] = value[:max_len]
        
        return result
    
    def _format_output(self, parsed_result: Dict[str, str]) -> str:
        """格式化输出为 xxx_xxx 格式"""
        parts = []
        
        for key in ['shot_scale', 'emotion', 'subject', 'action', 'background', 'camera_movement']:
            value = parsed_result.get(key, '').strip('_')
            if value and value != '无':
                parts.append(value)
        
        return '_'.join(parts) if parts else '未知'
    
    def analyze_frames(
        self,
        start_frame_path: str,
        end_frame_path: str,
        enable_thinking: bool = False,
        enable_scoring: bool = False,
        temperature: float = 0.3,
        max_new_tokens: int = 512,
        top_p: float = 0.8,
        top_k: int = 100,
        repetition_penalty: float = 1.05
    ) -> Dict:
        """分析两帧画面"""
        if self.model_type == "gguf" and enable_thinking:
            enable_thinking = False
        
        prompt = self._build_analysis_prompt()
        
        if self.model_type == "gguf":
            answer = self._call_cli(
                image_paths=[start_frame_path, end_frame_path],
                prompt=prompt,
                max_tokens=max_new_tokens
            )
        else:
            img1 = self._load_image(start_frame_path)
            img2 = self._load_image(end_frame_path)
            
            msgs = [{'role': 'user', 'content': [img1, img2, prompt]}]
            
            answer = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                enable_thinking=enable_thinking
            )
        
        parsed_result = self._parse_answer(answer)
        formatted_output = self._format_output(parsed_result)
        
        result = {
            'raw_answer': answer,
            'parsed_result': parsed_result,
            'formatted_output': formatted_output,
            'score': None
        }
        
        if enable_scoring:
            score_prompt = self._build_scoring_prompt()
            
            if self.model_type == "gguf":
                combined_prompt = f"之前的分析结果：\n{answer}\n\n{score_prompt}"
                score_answer = self._call_cli(
                    image_paths=[start_frame_path, end_frame_path],
                    prompt=combined_prompt,
                    max_tokens=128
                )
            else:
                msgs.append({'role': 'assistant', 'content': answer})
                msgs.append({'role': 'user', 'content': score_prompt})
                
                score_answer = self.model.chat(
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    sampling=True,
                    temperature=temperature,
                    max_new_tokens=128,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    enable_thinking=False
                )
            
            score = self._parse_score(score_answer)
            result['score'] = score
        
        return result
    
    def analyze_video(
        self,
        video_path: str,
        start_frame_index: int = 3,
        end_frame_offset: int = 3,
        compress_frames: bool = True,
        enable_thinking: bool = False,
        enable_scoring: bool = False,
        temperature: float = 0.3,
        max_new_tokens: int = 512,
        top_p: float = 0.8,
        top_k: int = 100,
        repetition_penalty: float = 1.05,
        cleanup: bool = True
    ) -> Dict:
        """分析视频（使用FFmpeg提取帧）"""
        start_frame_path, end_frame_path = self.extract_frames(
            video_path,
            start_frame_index=start_frame_index,
            end_frame_offset=end_frame_offset,
            compress=compress_frames
        )
        
        result = self.analyze_frames(
            start_frame_path,
            end_frame_path,
            enable_thinking=enable_thinking,
            enable_scoring=enable_scoring,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        
        if cleanup:
            try:
                os.remove(start_frame_path)
                os.remove(end_frame_path)
            except Exception:
                pass
        
        return result
    
    def analyze_and_rename_folder(
        self,
        folder_path: str,
        video_extensions: Optional[List[str]] = None,
        enable_thinking: bool = False,
        enable_scoring: bool = False,
        compress_frames: bool = True,
        gpu_clear_interval: int = 10,
        cleanup: bool = True,
        temperature: float = 0.3,
        max_new_tokens: int = 512,
        top_p: float = 0.8,
        top_k: int = 100,
        repetition_penalty: float = 1.05
    ) -> List[Dict]:
        """批量分析文件夹中的视频并重命名"""
        folder = self._resolve_path(folder_path)
        
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"错误：文件夹不存在 - {folder}")
        
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.m4v']
        
        video_files = []
        for ext in video_extensions:
            video_files.extend(folder.glob(f'*{ext}'))
            video_files.extend(folder.glob(f'*{ext.upper()}'))
        
        video_files = sorted(set(video_files), key=lambda x: x.name)
        
        if not video_files:
            print(f"结果：文件夹中没有找到视频文件")
            return []
        
        results = []
        
        for idx, video_file in enumerate(video_files, 1):
            try:
                result = self.analyze_video(
                    str(video_file),
                    enable_thinking=enable_thinking,
                    enable_scoring=enable_scoring,
                    compress_frames=compress_frames,
                    cleanup=cleanup,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens
                )
                
                original_name = video_file.stem
                extension = video_file.suffix
                scene_name = result['formatted_output']
                
                if result.get('score') is not None:
                    new_filename = f"{scene_name}_分值{result['score']}_{original_name}{extension}"
                else:
                    new_filename = f"{scene_name}_{original_name}{extension}"
                
                new_video_path = video_file.parent / new_filename
                video_file.rename(new_video_path)
                
                result['original_path'] = str(video_file)
                result['new_path'] = str(new_video_path)
                results.append(result)
                
                print(f"结果：[{idx}/{len(video_files)}] {new_video_path.name}")
                
                if self.model_type != "gguf" and idx % gpu_clear_interval == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"错误：处理 {video_file.name} 失败 - {e}")
                continue
        
        return results
    
    def _load_image(self, image_path: str) -> Image.Image:
        """加载图片"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"错误：图片文件不存在 - {image_path}")
        
        img = Image.open(image_path)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        return img
    
    def compare_subjects(
        self,
        frame1_path: str,
        frame2_path: str,
        temperature: float = 0.3,
        max_new_tokens: int = 512
    ) -> bool:
        """比较两帧画面的主体是否相同"""
        prompt = self._get_compare_subjects_prompt()
        
        if self.model_type == "gguf":
            answer = self._call_cli(
                image_paths=[frame1_path, frame2_path],
                prompt=prompt,
                max_tokens=max_new_tokens
            )
        else:
            img1 = self._load_image(frame1_path)
            img2 = self._load_image(frame2_path)
            
            msgs = [{'role': 'user', 'content': [img1, img2, prompt]}]
            
            answer = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                enable_thinking=False
            )
        
        answer_clean = answer.strip().replace(' ', '')
        is_same = '相同' in answer_clean and '不同' not in answer_clean
        
        return is_same
    
    def analyze_and_rename_video(
        self,
        video_path: str,
        start_frame_index: int = 3,
        end_frame_offset: int = 3,
        compress_frames: bool = True,
        enable_thinking: bool = False,
        enable_scoring: bool = True,
        temperature: float = 0.3,
        max_new_tokens: int = 512,
        cleanup: bool = True
    ) -> Dict:
        """分析单个视频并重命名"""
        video_file = self._resolve_path(video_path)
        
        if not video_file.exists():
            raise FileNotFoundError(f"错误：视频文件不存在 - {video_file}")
        
        result = self.analyze_video(
            str(video_file),
            start_frame_index=start_frame_index,
            end_frame_offset=end_frame_offset,
            compress_frames=compress_frames,
            enable_thinking=enable_thinking,
            enable_scoring=enable_scoring,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            cleanup=cleanup
        )
        
        original_name = video_file.stem
        extension = video_file.suffix
        scene_name = result['formatted_output']
        
        if result.get('score') is not None:
            new_filename = f"{scene_name}_分值{result['score']}_{original_name}{extension}"
        else:
            new_filename = f"{scene_name}_{original_name}{extension}"
        
        new_video_path = video_file.parent / new_filename
        video_file.rename(new_video_path)
        
        result['original_path'] = str(video_file)
        result['new_path'] = str(new_video_path)
        result['original_name'] = original_name
        result['new_filename'] = new_filename
        
        if result.get('score') is not None:
            print(f"结果：评分 {result['score']}")
        print(f"结果：重命名为 {new_filename}")
        
        return result


if __name__ == "__main__":
    
    # 初始化分析器
    try:
        analyzer = VideoSceneAnalyzer(model_path='hf-int4')
    except Exception as e:
        print(f"错误：{e}")
        exit(1)
    
    # 示例：分析文件夹中的视频
    try:
        result = analyzer.analyze_and_rename_folder(
            folder_path=r"C:\Users\Admin\Desktop\test\cuts",
            enable_scoring=True
        )
    except Exception as e:
        print(f"错误：{e}")