# reranker_frame_extractor.py
# Reranker 专用单线程帧提取器
# v1.0: 参考 video_utils.FrameExtractorThread 的单线程设计
# v1.1: 添加 seek 优化（减少解码量）
# v1.2: 添加 decord 后端（帧号直接索引，零进程开销）
#
# 设计原则：
# 1. 单个 FFmpeg 进程（避免多线程 CPU 竞争）
# 2. 批量提取多个帧（使用 select 滤镜，一次 FFmpeg 调用）
# 3. 图像预处理缓存（类似 VectorizerThread 的预加载机制）
# 4. 与 Reranker GPU 推理配合
# 5. seek 优化：提前 seek 到目标帧附近，减少解码量
# 6. decord 后端：帧号精确索引，批量随机访问，不写临时文件

import os
import sys
import time
import hashlib
import subprocess
import glob
import shutil
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from threading import Thread, Lock

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

# 导入视频工具的辅助函数（只读取，不修改）
from A_coreUtils.video_processing.video_utils import (
    VideoMetaHelper,
    TEMP_DIR,
    FFMPEG_PATH,
    FFPROBE_PATH,
)

# 检测 decord 是否可用
try:
    from decord import VideoReader as _DecordVideoReader
    from decord import cpu as _decord_cpu
    import numpy as np
    from PIL import Image
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False


def _get_video_dimensions(video_path: str) -> Tuple[int, int]:
    """获取视频宽高"""
    try:
        cmd = [FFPROBE_PATH, '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                               encoding='utf-8', errors='ignore')
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
    except:
        pass
    return 0, 0


def _build_scale_filter(width: int, height: int, output_resolution: str) -> Optional[str]:
    """
    构建短边缩放滤镜
    
    Args:
        width: 视频宽度
        height: 视频高度
        output_resolution: 目标短边像素数（如 '384'）
    
    Returns:
        str: scale 滤镜字符串，如 "scale=512:384"
    """
    if not output_resolution or output_resolution == 'original':
        return None
    
    if output_resolution.isdigit():
        target_short = int(output_resolution)
        if width >= height:
            # 横向视频：高度为短边
            out_height = target_short
            out_width = int(width * target_short / height)
        else:
            # 纵向视频：宽度为短边
            out_width = target_short
            out_height = int(height * target_short / width)
        # 确保偶数（FFmpeg 要求）
        out_width = out_width - (out_width % 2)
        out_height = out_height - (out_height % 2)
        return f"scale={out_width}:{out_height}"
    
    return None


class RerankerFrameExtractor:
    """
    Reranker 专用单线程帧提取器
    
    参考 video_utils.FrameExtractorThread 的设计：
    - 单个 FFmpeg 进程（避免多线程 CPU 竞争）
    - 批量提取（一次 FFmpeg 调用提取多帧）
    - 帧缓存（避免重复提取）
    
    与 parallel_frame_extractor.py 的区别：
    - 不使用 ThreadPoolExecutor 多线程并行
    - 按视频顺序处理，减少 CPU 峰值
    
    使用方式：
    ```python
    extractor = RerankerFrameExtractor(output_resolution='384')
    
    # 批量提取场景中间帧
    scenes = [
        {'video_path': 'video1.mp4', 'start_frame': 100, 'end_frame': 200},
        {'video_path': 'video2.mp4', 'start_frame': 50, 'end_frame': 150},
    ]
    frame_paths = extractor.extract_batch(scenes)
    # frame_paths = {scene_key: frame_path}
    
    # 清理临时文件
    extractor.cleanup()
    ```
    """
    
    def __init__(self,
                 output_resolution: str = '384',
                 cache_dir: str = None,
                 backend: str = 'auto'):
        """
        初始化单线程帧提取器
        
        Args:
            output_resolution: 输出分辨率（短边像素数）
            cache_dir: 帧缓存目录
            backend: 帧提取后端
                - 'auto': 优先 decord，不可用时回退 ffmpeg
                - 'decord': 强制使用 decord（不可用时报错）
                - 'ffmpeg': 强制使用 ffmpeg
        """
        self.output_resolution = output_resolution
        
        # 确定后端
        if backend == 'auto':
            self.backend = 'decord' if _HAS_DECORD else 'ffmpeg'
        elif backend == 'decord':
            if not _HAS_DECORD:
                raise RuntimeError("decord 不可用，请安装: pip install decord")
            self.backend = 'decord'
        else:
            self.backend = 'ffmpeg'
        
        # decord VideoReader 缓存 {video_path: VideoReader}
        self._decord_cache: Dict[str, object] = {}
        
        # 缓存目录
        if cache_dir is None:
            self.cache_dir = os.path.join(TEMP_DIR, 'rerank_frames_single')
        else:
            self.cache_dir = cache_dir
        
        # 仅确保缓存目录存在；缓存清理由外部在阶段结束时统一触发
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 帧缓存 {scene_key: frame_path}
        self.frame_cache: Dict[str, str] = {}
        self._cache_lock = Lock()
        
        # FPS 缓存
        self._fps_cache: Dict[str, float] = {}
        
        # 视频尺寸缓存
        self._dimension_cache: Dict[str, Tuple[int, int]] = {}
    
    def _clear_cache_dir(self):
        """清空缓存目录（初始化时调用）"""
        if os.path.exists(self.cache_dir):
            try:
                for item in os.listdir(self.cache_dir):
                    item_path = os.path.join(self.cache_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except (PermissionError, OSError) as e:
                        print(f"[帧提取] 清理文件失败: {item} - {e}")
                print(f"[帧提取-单线程] 已清空缓存目录: {self.cache_dir}")
            except Exception as e:
                print(f"[帧提取] 清空缓存目录失败: {e}")
    
    @staticmethod
    def get_scene_key(video_path: str, start_frame: int) -> str:
        """
        生成场景唯一标识
        
        格式: {start_frame}_{video_filename}
        """
        video_name = os.path.basename(video_path)
        return f"{start_frame}_{video_name}"
    
    @staticmethod
    def get_mid_frame(start_frame: int, end_frame: int) -> int:
        """计算中间帧号"""
        return (start_frame + end_frame) // 2
    
    def _get_frame_path(self, scene_key: str) -> str:
        """生成帧图像保存路径"""
        # 使用 hash 避免文件名过长
        key_hash = hashlib.md5(scene_key.encode('utf-8')).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{key_hash}.jpg")
    
    def _get_fps(self, video_path: str) -> float:
        """获取视频 FPS（带缓存）"""
        if video_path not in self._fps_cache:
            self._fps_cache[video_path] = VideoMetaHelper.get_fps_cached(video_path, self._fps_cache)
        return self._fps_cache.get(video_path, 25.0)
    
    def _get_dimensions(self, video_path: str) -> Tuple[int, int]:
        """获取视频尺寸（带缓存）"""
        if video_path not in self._dimension_cache:
            self._dimension_cache[video_path] = _get_video_dimensions(video_path)
        return self._dimension_cache.get(video_path, (0, 0))
    
    def _extract_single_frame(self,
                               video_path: str,
                               frame_number: int,
                               fps: float,
                               output_path: str) -> bool:
        """
        提取单帧（用于单个场景）
        
        参考 video_utils.extract_single_frame_rerank
        
        使用 input seeking（-ss 在 -i 之前）：
        - FFmpeg 会 seek 到最近的关键帧（I帧）
        - 然后只解码从关键帧到目标帧之间的帧
        - 不需要解码整个视频
        """
        try:
            # 提前 1 秒 seek，确保不跳过目标帧
            # -ss 在 -i 之前是 input seeking，会 seek 到最近的 I 帧
            approx_time = max(0, (frame_number / fps) - 1.0)
            seek_frame = int(approx_time * fps)
            relative_frame = frame_number - seek_frame
            
            cmd = [FFMPEG_PATH, '-y']
            cmd.extend([
                '-ss', str(approx_time),
                '-i', video_path,
            ])
            
            # 构建滤镜链：精确帧选择 + 可选缩放
            vf_parts = [f"select='eq(n,{relative_frame})'"]
            
            # 使用短边缩放
            if self.output_resolution and self.output_resolution != 'original':
                width, height = self._get_dimensions(video_path)
                if width > 0 and height > 0:
                    scale_filter = _build_scale_filter(width, height, self.output_resolution)
                    if scale_filter:
                        vf_parts.append(scale_filter)
            
            cmd.extend([
                '-vf', ','.join(vf_parts),
                '-vframes', '1',
                '-q:v', '2',
                '-vsync', 'vfr',
                '-loglevel', 'error',
                output_path
            ])
            
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                          encoding='utf-8', errors='ignore')
            return os.path.exists(output_path)
        except (subprocess.SubprocessError, OSError):
            return False
    
    def _extract_video_frames_batch(self,
                                     video_path: str,
                                     scenes: List[Dict]) -> Dict[str, str]:
        """
        批量提取单个视频的多个场景帧
        
        根据 self.backend 选择 decord 或 FFmpeg 后端。
        
        Args:
            video_path: 视频路径
            scenes: 该视频的场景列表
        
        Returns:
            {scene_key: frame_path} 字典
        """
        results = {}
        
        # 检查缓存，过滤已提取的场景
        scenes_to_extract = []
        for scene in scenes:
            scene_key = self.get_scene_key(scene['video_path'], scene['start_frame'])
            with self._cache_lock:
                if scene_key in self.frame_cache:
                    cached_path = self.frame_cache[scene_key]
                    if os.path.exists(cached_path):
                        results[scene_key] = cached_path
                        continue
            scenes_to_extract.append(scene)
        
        if not scenes_to_extract:
            return results
        
        # 计算每个场景的目标帧号（根据 target_frame_idx 选择首/中/尾帧）
        target_frames_info = []  # [(target_frame, scene_key, scene), ...]
        for scene in scenes_to_extract:
            start_frame = scene['start_frame']
            end_frame = scene['end_frame']
            # 获取目标帧索引：0=首帧, 1=中帧, 2=尾帧（默认中帧）
            target_frame_idx = scene.get('target_frame_idx', 1)
            if target_frame_idx == 0:
                target_frame = start_frame
            elif target_frame_idx == 2:
                target_frame = end_frame
            else:
                target_frame = self.get_mid_frame(start_frame, end_frame)
            scene_key = self.get_scene_key(scene['video_path'], start_frame)
            target_frames_info.append((target_frame, scene_key, scene))
        
        # 按目标帧号排序（顺序读取减少 seek）
        target_frames_info.sort(key=lambda x: x[0])
        
        # ---- decord 后端 ----
        if self.backend == 'decord':
            return self._extract_video_frames_decord(video_path, target_frames_info, results)
        
        # ---- FFmpeg 后端（原逻辑） ----
        # 如果只有一帧，使用单帧提取（更快）
        if len(target_frames_info) == 1:
            target_frame, scene_key, scene = target_frames_info[0]
            fps = self._get_fps(video_path)
            frame_path = self._get_frame_path(scene_key)
            
            success = self._extract_single_frame(video_path, target_frame, fps, frame_path)
            if success:
                with self._cache_lock:
                    self.frame_cache[scene_key] = frame_path
                results[scene_key] = frame_path
            return results
        
        # 多帧批量提取
        sorted_target_frames = [info[0] for info in target_frames_info]
        extracted_files = self._batch_extract_frames_single_thread(video_path, sorted_target_frames)
        
        # 建立映射：目标帧号 -> 输出文件路径
        for target_frame, scene_key, scene in target_frames_info:
            if target_frame in extracted_files:
                frame_path = extracted_files[target_frame]
                # 移动到最终缓存路径
                final_path = frame_path
                try:
                    if os.path.exists(frame_path):
                        with self._cache_lock:
                            self.frame_cache[scene_key] = final_path
                        results[scene_key] = final_path
                except Exception as e:
                    print(f"[帧提取] 移动文件失败: {scene_key} - {e}")
        
        return results
    
    # ================================================================
    # decord 后端方法
    # ================================================================
    
    def _get_decord_reader(self, video_path: str):
        """获取或创建 decord VideoReader（带缓存）"""
        if video_path not in self._decord_cache:
            self._decord_cache[video_path] = _DecordVideoReader(video_path, ctx=_decord_cpu(0))
        return self._decord_cache[video_path]
    
    def _resize_frame_pil(self, frame_np, target_short: int):
        """
        使用 PIL 对帧进行短边缩放（与 FFmpeg scale 滤镜等效）
        
        Args:
            frame_np: numpy 数组 [H, W, C] RGB
            target_short: 目标短边像素数
        
        Returns:
            缩放后的 PIL Image
        """
        img = Image.fromarray(frame_np)
        w, h = img.size
        if w <= 0 or h <= 0:
            return img
        
        short_side = min(w, h)
        if short_side <= target_short:
            return img
        
        scale = target_short / short_side
        new_w = int(w * scale + 0.5)
        new_h = int(h * scale + 0.5)
        # 确保偶数（视频处理惯例）
        new_w = new_w if new_w % 2 == 0 else new_w + 1
        new_h = new_h if new_h % 2 == 0 else new_h + 1
        
        return img.resize((new_w, new_h), Image.LANCZOS)
    
    def _extract_video_frames_decord(self,
                                      video_path: str,
                                      target_frames_info: List[Tuple],
                                      results: Dict[str, str]) -> Dict[str, str]:
        """
        使用 decord 批量提取帧
        
        Args:
            video_path: 视频路径
            target_frames_info: [(target_frame, scene_key, scene), ...] 已排序
            results: 已有的缓存结果（会被更新）
        
        Returns:
            {scene_key: frame_path} 字典
        """
        try:
            vr = self._get_decord_reader(video_path)
            total_frames = len(vr)
            
            # 收集帧号（去重 + 裁剪到有效范围）
            frame_to_keys = defaultdict(list)  # {frame_number: [scene_key, ...]}
            valid_frame_numbers = []
            seen_frames = set()
            
            for target_frame, scene_key, scene in target_frames_info:
                # 确保帧号在有效范围内
                clamped = max(0, min(target_frame, total_frames - 1))
                frame_to_keys[clamped].append(scene_key)
                if clamped not in seen_frames:
                    seen_frames.add(clamped)
                    valid_frame_numbers.append(clamped)
            
            if not valid_frame_numbers:
                return results
            
            # 批量提取（decord 内部优化 seek）
            frames_batch = vr.get_batch(valid_frame_numbers).asnumpy()  # [N, H, W, C] RGB
            
            # 解析缩放参数
            need_resize = (self.output_resolution
                          and self.output_resolution != 'original'
                          and self.output_resolution.isdigit())
            target_short = int(self.output_resolution) if need_resize else 0
            
            # 保存为 jpg
            for i, frame_number in enumerate(valid_frame_numbers):
                frame_np = frames_batch[i]
                
                # 缩放
                if need_resize:
                    img = self._resize_frame_pil(frame_np, target_short)
                else:
                    img = Image.fromarray(frame_np)
                
                # 为该帧号对应的所有 scene_key 保存
                for scene_key in frame_to_keys[frame_number]:
                    frame_path = self._get_frame_path(scene_key)
                    img.save(frame_path, 'JPEG', quality=95)
                    
                    with self._cache_lock:
                        self.frame_cache[scene_key] = frame_path
                    results[scene_key] = frame_path
            
            return results
            
        except Exception as e:
            print(f"[帧提取-decord] 失败: {os.path.basename(video_path)} - {e}, 回退到 FFmpeg")
            # 回退到 FFmpeg
            return self._extract_video_frames_batch_ffmpeg_fallback(
                video_path, target_frames_info, results)
    
    def _extract_video_frames_batch_ffmpeg_fallback(self,
                                                     video_path: str,
                                                     target_frames_info: List[Tuple],
                                                     results: Dict[str, str]) -> Dict[str, str]:
        """decord 失败时回退到 FFmpeg 提取"""
        if len(target_frames_info) == 1:
            target_frame, scene_key, scene = target_frames_info[0]
            fps = self._get_fps(video_path)
            frame_path = self._get_frame_path(scene_key)
            success = self._extract_single_frame(video_path, target_frame, fps, frame_path)
            if success:
                with self._cache_lock:
                    self.frame_cache[scene_key] = frame_path
                results[scene_key] = frame_path
            return results
        
        sorted_target_frames = [info[0] for info in target_frames_info]
        extracted_files = self._batch_extract_frames_single_thread(video_path, sorted_target_frames)
        
        for target_frame, scene_key, scene in target_frames_info:
            if target_frame in extracted_files:
                frame_path = extracted_files[target_frame]
                try:
                    if os.path.exists(frame_path):
                        with self._cache_lock:
                            self.frame_cache[scene_key] = frame_path
                        results[scene_key] = frame_path
                except Exception as e:
                    print(f"[帧提取] 移动文件失败: {scene_key} - {e}")
        
        return results
    
    def _batch_extract_frames_single_thread(self, 
                                             video_path: str, 
                                             frame_numbers: List[int]) -> Dict[int, str]:
        """
        批量提取指定帧号的帧 - 单线程版本
        
        参考 video_utils.FrameExtractorThread 的设计：
        - 单个 FFmpeg 进程
        - 使用后台线程读取 stderr 防止死锁
        - 使用 filter_script 文件避免命令行长度限制
        
        Args:
            video_path: 视频路径
            frame_numbers: 帧号列表（已排序）
        
        Returns:
            {frame_number: temp_file_path} 字典
        """
        if not frame_numbers:
            return {}
        
        # 创建临时目录
        batch_temp_dir = os.path.join(self.cache_dir, f"batch_{int(time.time() * 1000)}")
        os.makedirs(batch_temp_dir, exist_ok=True)
        
        try:
            # 获取视频尺寸
            width, height = self._get_dimensions(video_path)
            
            # 构建 select 表达式（使用绝对帧号，不做 seek）
            select_expr = '+'.join([f'eq(n\\,{f})' for f in frame_numbers])
            
            # 构建滤镜链
            filter_parts = [f"select='{select_expr}'"]
            
            # 添加缩放滤镜
            if self.output_resolution and self.output_resolution != 'original':
                if width > 0 and height > 0:
                    scale_filter = _build_scale_filter(width, height, self.output_resolution)
                    if scale_filter:
                        filter_parts.append(scale_filter)
            
            filter_chain = ','.join(filter_parts)
            
            # 写入滤镜脚本文件（避免命令行长度限制）
            script_path = os.path.join(batch_temp_dir, 'filter_script.txt')
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(filter_chain)
            
            # 构建 FFmpeg 命令（不使用 -ss，从头读取，用绝对帧号匹配）
            batch_output_prefix = f"batch_frame_{os.getpid()}_{time.time_ns()}"
            output_pattern = os.path.join(self.cache_dir, f"{batch_output_prefix}_%06d.jpg")
            
            cmd = [FFMPEG_PATH, '-y']
            cmd.extend([
                '-i', video_path,
                '-filter_script:v', script_path,
                '-vsync', 'vfr',
                '-q:v', '2',
                '-loglevel', 'error',
                output_pattern
            ])
            
            # 执行 FFmpeg（单线程，使用后台线程读取 stderr 防止死锁）
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # 后台线程读取 stderr（参考 FrameExtractorThread）
            stderr_chunks = []
            def stderr_reader():
                try:
                    for line in process.stderr:
                        stderr_chunks.append(line)
                except (IOError, OSError):
                    pass
            
            reader_thread = Thread(target=stderr_reader, daemon=True)
            reader_thread.start()
            
            # 等待完成
            process.wait()
            reader_thread.join(timeout=5.0)
            
            if process.returncode != 0:
                stderr = ''.join(stderr_chunks)
                print(f"[帧提取] FFmpeg 批量提取失败: {stderr}")
                return {}
            
            # 建立帧号到文件的映射
            # FFmpeg -vsync vfr 输出连续编号，与排序后的帧号一一对应
            extracted_files = glob.glob(os.path.join(self.cache_dir, f"{batch_output_prefix}_*.jpg"))
            extracted_files.sort()  # 确保按编号排序
            
            result = {}
            for i, file_path in enumerate(extracted_files):
                if i < len(frame_numbers):
                    result[frame_numbers[i]] = file_path
            
            return result
            
        except Exception as e:
            print(f"[帧提取] 批量提取异常: {e}")
            return {}
    
    def extract_batch(self, scenes: List[Dict], show_progress: bool = True) -> Dict[str, str]:
        """
        批量提取场景帧 - 单线程顺序处理
        
        支持指定提取哪一帧（首/中/尾），通过 target_frame_idx 参数控制：
        - 0: 首帧
        - 1: 中间帧（默认）
        - 2: 尾帧
        
        与 parallel_frame_extractor.py 的区别：
        - 不使用 ThreadPoolExecutor 多线程并行
        - 按视频顺序处理，减少 CPU 峰值
        
        Args:
            scenes: 场景列表，每个包含 video_path, start_frame, end_frame, 可选 target_frame_idx
            show_progress: 是否显示进度
        
        Returns:
            {scene_key: frame_path} 字典
        """
        if not scenes:
            return {}
        
        start_time = time.time()
        
        # 去重：同一场景只提取一次
        unique_scenes = {}
        for scene in scenes:
            scene_key = self.get_scene_key(scene['video_path'], scene['start_frame'])
            if scene_key not in unique_scenes:
                unique_scenes[scene_key] = scene
        
        # 检查缓存，过滤已提取的
        scenes_to_extract = []
        cached_results = {}
        
        with self._cache_lock:
            for scene_key, scene in unique_scenes.items():
                if scene_key in self.frame_cache:
                    cached_path = self.frame_cache[scene_key]
                    if os.path.exists(cached_path):
                        cached_results[scene_key] = cached_path
                        continue
                scenes_to_extract.append(scene)
        
        backend_tag = f"[帧提取-{self.backend}]"
        if show_progress:
            print(f"{backend_tag} 需要提取 {len(scenes_to_extract)} 个场景 (缓存命中 {len(cached_results)} 个)")
        
        if not scenes_to_extract:
            return cached_results
        
        # 按视频分组
        video_scenes = defaultdict(list)
        for scene in scenes_to_extract:
            video_path = scene['video_path']
            video_scenes[video_path].append(scene)
        
        if show_progress:
            print(f"{backend_tag} 分布在 {len(video_scenes)} 个视频中")
        
        # 顺序处理每个视频（单线程，避免 CPU 竞争）
        results = dict(cached_results)
        completed = 0
        
        for video_path, video_scene_list in video_scenes.items():
            try:
                video_results = self._extract_video_frames_batch(video_path, video_scene_list)
                results.update(video_results)
                completed += 1
                
                if show_progress:
                    video_name = os.path.basename(video_path)
                    print(f"    [{completed}/{len(video_scenes)}] {video_name}: {len(video_results)} 帧")
                    
            except Exception as e:
                print(f"[帧提取] 视频处理失败: {os.path.basename(video_path)} - {e}")
        
        elapsed = time.time() - start_time
        if show_progress:
            fps_rate = len(results) / elapsed if elapsed > 0 else 0
            print(f"{backend_tag} 完成! 共 {len(results)} 帧, 耗时 {elapsed:.2f}s, 速度 {fps_rate:.1f} 帧/s")
        
        return results
    
    def cleanup(self, remove_files: bool = True):
        """清理临时文件和缓存
        
        Args:
            remove_files: 是否删除缓存帧文件，默认为 True
        """
        try:
            # 释放 decord VideoReader 缓存
            self._decord_cache.clear()
            
            # remove_files=False: 仅清理 batch_* 临时目录（中间批次）
            # remove_files=True: 额外清理缓存帧文件（任务结束）
            if remove_files:
                with self._cache_lock:
                    for frame_path in self.frame_cache.values():
                        try:
                            if os.path.exists(frame_path):
                                os.remove(frame_path)
                        except (PermissionError, OSError):
                            pass
                    self.frame_cache.clear()
            
            # 仅清理批量提取临时目录，保留缓存根目录
            if os.path.exists(self.cache_dir):
                for item in os.listdir(self.cache_dir):
                    if item.startswith('batch_'):
                        item_path = os.path.join(self.cache_dir, item)
                        try:
                            shutil.rmtree(item_path)
                        except (PermissionError, OSError):
                            pass
        except Exception as e:
            print(f"[帧提取] 清理失败: {e}")
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        with self._cache_lock:
            return {
                'cached_frames': len(self.frame_cache),
                'cache_dir': self.cache_dir
            }
