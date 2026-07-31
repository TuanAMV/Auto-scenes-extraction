# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
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
import io
import json
import hashlib
import subprocess
import glob
import shutil
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import cv2
import numpy as np

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
    FrameExtractorThread,
    TEMP_DIR,
    FFMPEG_PATH,
    FFPROBE_PATH,
)

# 检测 decord 是否可用
try:
    from decord import VideoReader as _DecordVideoReader
    from decord import cpu as _decord_cpu
    from PIL import Image
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False


def _get_video_dimensions(video_path: str) -> Tuple[int, int]:
    """获取视频宽高（委托给 VideoMetaHelper）"""
    _, _, width, height = VideoMetaHelper.get_video_meta(video_path)
    return width, height


def _build_scale_filter(width: int, height: int, output_resolution: str) -> Optional[str]:
    """构建短边缩放滤镜（委托给 FrameExtractorThread._build_scale_filter）"""
    if not output_resolution or output_resolution == 'original':
        return None
    scale_filter, _, _ = FrameExtractorThread._build_scale_filter(width, height, output_resolution)
    return scale_filter


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
                 backend: str = 'auto',
                 decord_decode_batch_size: int = 30,
                 decord_save_workers: int = 2,
                 decord_max_pending_tasks: int = 8,
                 jpeg_quality: int = 95):
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
        # decord reader 输出尺寸缓存（若 VideoReader 构造时指定了 width/height）
        # {video_path: (out_w, out_h) or None}
        self._decord_reader_output_size: Dict[str, Optional[Tuple[int, int]]] = {}
        
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
        # 仅恒定帧率视频可安全用 frame/fps 驱动 FFmpeg 输入 seek。
        self._ffmpeg_seek_fps_cache: Dict[str, Optional[float]] = {}
        
        # 视频尺寸缓存
        self._dimension_cache: Dict[str, Tuple[int, int]] = {}
        
        # decord 分块/写盘参数，控制内存峰值（非法输入回退默认值）
        try:
            self.decord_decode_batch_size = max(1, int(decord_decode_batch_size))
        except (TypeError, ValueError):
            self.decord_decode_batch_size = 30
        try:
            self.decord_save_workers = max(1, int(decord_save_workers))
        except (TypeError, ValueError):
            self.decord_save_workers = 2
        try:
            pending = int(decord_max_pending_tasks)
        except (TypeError, ValueError):
            pending = 8
        self.decord_max_pending_tasks = max(self.decord_save_workers, pending)
        try:
            self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        except (TypeError, ValueError):
            self.jpeg_quality = 95
    
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
    def get_scene_key(video_path: str, start_frame: int, target_frame_idx: int = 1) -> str:
        """
        生成场景唯一标识
        
        格式: {start_frame}_{target_frame_idx}_{video_filename}
        """
        video_name = os.path.basename(video_path)
        return f"{start_frame}_{target_frame_idx}_{video_name}"
    
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

    def _prepare_memory_frame(self, frame_rgb: np.ndarray, output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """将解码后的 RGB 帧转换为分析阶段使用的 BGR 数组，不经过图片文件。"""
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if output_size is not None:
            out_w, out_h = output_size
            if frame_bgr.shape[1] != out_w or frame_bgr.shape[0] != out_h:
                frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return frame_bgr

    def _get_memory_output_size(self, video_path: str) -> Tuple[int, int]:
        """计算内存帧的输出尺寸，与原有短边缩放规则一致。"""
        width, height = self._get_dimensions(video_path)
        if width <= 0 or height <= 0:
            return width, height
        if not self.output_resolution or self.output_resolution == 'original':
            return width, height
        scale_filter, out_w, out_h = FrameExtractorThread._build_scale_filter(
            width, height, self.output_resolution
        )
        return out_w, out_h

    def _iter_decord_memory_frames(self, video_path: str, frame_numbers: List[int]):
        """使用一个 Decord reader 分块读取并流式返回 BGR 数组。

        每个 ``get_batch`` 请求最多 ``decord_decode_batch_size`` 个目标帧。
        目标帧已经按帧号排序，适合多个独立场景的稀疏中间帧/光流帧请求。
        """
        reader = self._get_decord_reader(video_path)
        total_frames = len(reader)
        valid_numbers = [int(frame_no) for frame_no in frame_numbers
                         if 0 <= int(frame_no) < total_frames]
        if not valid_numbers:
            return

        decord_out_size = self._decord_reader_output_size.get(video_path)
        resize_size = None
        if decord_out_size is None:
            source_size = self._get_memory_output_size(video_path)
            if source_size[0] > 0 and source_size[1] > 0:
                resize_size = source_size

        batch_size = max(1, int(self.decord_decode_batch_size))
        for batch_start in range(0, len(valid_numbers), batch_size):
            frame_chunk = valid_numbers[batch_start:batch_start + batch_size]
            frames_batch = reader.get_batch(frame_chunk).asnumpy()
            try:
                for index, frame_number in enumerate(frame_chunk):
                    yield frame_number, self._prepare_memory_frame(frames_batch[index], resize_size)
            finally:
                del frames_batch

    def _get_ffmpeg_seek_fps(self, video_path: str) -> Optional[float]:
        """返回可安全做帧号 seek 的恒定帧率；VFR 或探测失败返回 None。"""
        if video_path in self._ffmpeg_seek_fps_cache:
            return self._ffmpeg_seek_fps_cache[video_path]

        seek_fps = None
        try:
            cmd = [
                FFPROBE_PATH, '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
                '-of', 'json', video_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding='utf-8',
                errors='replace', timeout=15,
            )
            result.check_returncode()
            streams = json.loads(result.stdout).get('streams', [])
            stream = streams[0] if streams else {}

            def parse_rate(value):
                numerator, denominator = str(value or '0/1').split('/', 1)
                denominator_value = float(denominator)
                if denominator_value == 0:
                    return 0.0
                return float(numerator) / denominator_value

            nominal_fps = parse_rate(stream.get('r_frame_rate'))
            average_fps = parse_rate(stream.get('avg_frame_rate'))
            tolerance = max(1e-6, nominal_fps * 1e-6)
            if (nominal_fps > 0 and average_fps > 0
                    and abs(nominal_fps - average_fps) <= tolerance):
                seek_fps = average_fps
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, json.JSONDecodeError):
            seek_fps = None

        self._ffmpeg_seek_fps_cache[video_path] = seek_fps
        return seek_fps

    def _iter_ffmpeg_memory_frames(self, video_path: str, frame_numbers: List[int]):
        """使用一个 FFmpeg rawvideo 管道返回选中帧，避免 JPG 中转。"""
        if not frame_numbers:
            return

        width, height = self._get_dimensions(video_path)
        if width <= 0 or height <= 0:
            return
        _, out_w, out_h = FrameExtractorThread._build_scale_filter(
            width, height, self.output_resolution
        )
        if out_w <= 0 or out_h <= 0:
            out_w, out_h = width, height

        unique_numbers = sorted(set(int(frame_no) for frame_no in frame_numbers if int(frame_no) >= 0))
        if not unique_numbers:
            return

        fps = self._get_ffmpeg_seek_fps(video_path)
        seek_frame = 0
        if fps is not None and fps > 0:
            # 与旧版单帧提取保持一致：先跳到目标窗口前约一秒，
            # 再按相对帧号筛选，兼顾快速定位和帧号精度。
            seek_padding_frames = max(1, int(round(fps)))
            seek_frame = max(0, unique_numbers[0] - seek_padding_frames)
        seek_time = (seek_frame / fps) if seek_frame > 0 and fps else 0.0
        relative_numbers = [frame_no - seek_frame for frame_no in unique_numbers]

        select_expr = '+'.join(f'eq(n\\,{frame_no})' for frame_no in relative_numbers)
        filter_parts = [f"select='{select_expr}'"]
        if self.output_resolution and self.output_resolution != 'original':
            scale_filter = _build_scale_filter(width, height, self.output_resolution)
            if scale_filter:
                filter_parts.append(scale_filter)

        cmd = [FFMPEG_PATH, '-hide_banner', '-loglevel', 'error']
        if seek_time > 0:
            cmd.extend(['-ss', f'{seek_time:.9f}'])
        cmd.extend([
            '-i', video_path,
            '-vf', ','.join(filter_parts),
            '-an', '-sn', '-vsync', '0',
            '-frames:v', str(len(unique_numbers)),
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1',
        ])
        process = None
        stderr_chunks = []
        reader_thread = None
        frame_size = int(out_w) * int(out_h) * 3
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            def stderr_reader():
                try:
                    for chunk in iter(process.stderr.readline, b''):
                        stderr_chunks.append(chunk)
                except (IOError, OSError):
                    pass

            reader_thread = Thread(target=stderr_reader, daemon=True)
            reader_thread.start()

            def read_exact_frame():
                chunks = []
                remaining = frame_size
                while remaining > 0:
                    chunk = process.stdout.read(remaining)
                    if not chunk:
                        return None
                    chunks.append(chunk)
                    remaining -= len(chunk)
                return b''.join(chunks)

            for frame_number in unique_numbers:
                raw_frame = read_exact_frame()
                if raw_frame is None:
                    break
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((out_h, out_w, 3)).copy()
                yield frame_number, frame

            return_code = process.wait()
            if return_code != 0:
                stderr = b''.join(stderr_chunks).decode('utf-8', errors='replace')
                raise RuntimeError(f"FFmpeg 内存抽帧失败: {stderr}")
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            if reader_thread is not None:
                reader_thread.join(timeout=2.0)

    def iter_memory_frames(
        self,
        video_path: str,
        frame_numbers: List[int],
        release_reader: bool = True,
    ):
        """按数值帧号流式返回 ``(frame_number, BGR ndarray)``。

        ``release_reader=False`` 用于同一视频的连续场景处理，让多个场景
        复用一个 Decord reader；调用方完成该视频后应调用
        :meth:`release_video_reader`。
        """
        if not frame_numbers:
            return

        requested = sorted(set(int(frame_no) for frame_no in frame_numbers))
        if self.backend == 'decord':
            try:
                yield from self._iter_decord_memory_frames(video_path, requested)
                return
            except Exception as exc:
                print(f"[帧提取-内存-decord] {os.path.basename(video_path)} 失败: {exc}，回退 FFmpeg")
            finally:
                if release_reader:
                    self.release_video_reader(video_path)

        try:
            yield from self._iter_ffmpeg_memory_frames(video_path, requested)
        finally:
            if release_reader:
                self.release_video_reader(video_path)

    def release_video_reader(self, video_path: str):
        """释放指定视频的 Decord reader 和关联尺寸缓存。"""
        self._decord_cache.pop(video_path, None)
        self._decord_reader_output_size.pop(video_path, None)
    
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
            scene_key = self.get_scene_key(scene['video_path'], scene['start_frame'], scene.get('target_frame_idx', 1))
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
            scene_key = self.get_scene_key(scene['video_path'], start_frame, scene.get('target_frame_idx', 1))
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
            out_w = -1
            out_h = -1

            # 尽量在解码阶段就做缩放：显著降低峰值内存与后续 JPEG 编码开销
            if (self.output_resolution
                    and self.output_resolution != 'original'
                    and self.output_resolution.isdigit()):
                target_short = int(self.output_resolution)
                src_w, src_h = self._get_dimensions(video_path)
                if src_w > 0 and src_h > 0:
                    short_side = min(src_w, src_h)
                    if short_side > target_short:
                        scale = target_short / float(short_side)
                        out_w = int(src_w * scale + 0.5)
                        out_h = int(src_h * scale + 0.5)
                        # 与 _resize_frame_pil 保持一致：四舍五入后确保偶数
                        out_w = out_w if out_w % 2 == 0 else out_w + 1
                        out_h = out_h if out_h % 2 == 0 else out_h + 1

            try:
                if out_w > 0 and out_h > 0:
                    self._decord_cache[video_path] = _DecordVideoReader(
                        video_path, ctx=_decord_cpu(0), width=out_w, height=out_h
                    )
                    self._decord_reader_output_size[video_path] = (out_w, out_h)
                else:
                    self._decord_cache[video_path] = _DecordVideoReader(video_path, ctx=_decord_cpu(0))
                    self._decord_reader_output_size[video_path] = None
            except TypeError:
                # 兼容旧版 decord：不支持 width/height 参数
                self._decord_cache[video_path] = _DecordVideoReader(video_path, ctx=_decord_cpu(0))
                self._decord_reader_output_size[video_path] = None
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
    
    def _save_decord_frame(self,
                           frame_np,
                           scene_keys: Tuple[str, ...],
                           need_resize: bool,
                           target_short: int) -> List[Tuple[str, str]]:
        """
        Save one frame for one or many scene keys.
        For duplicated frame numbers, JPEG is encoded once then copied to each target path.
        """
        if not scene_keys:
            return []
        
        img = self._resize_frame_pil(frame_np, target_short) if need_resize else Image.fromarray(frame_np)
        try:
            if len(scene_keys) == 1:
                scene_key = scene_keys[0]
                frame_path = self._get_frame_path(scene_key)
                img.save(frame_path, 'JPEG', quality=self.jpeg_quality)
                return [(scene_key, frame_path)]
            
            # Same frame shared by multiple scenes: encode JPEG once.
            buffer = io.BytesIO()
            img.save(buffer, 'JPEG', quality=self.jpeg_quality)
            jpeg_data = buffer.getvalue()
        finally:
            try:
                img.close()
            except Exception:
                pass
        
        saved = []
        for scene_key in scene_keys:
            frame_path = self._get_frame_path(scene_key)
            with open(frame_path, 'wb') as f:
                f.write(jpeg_data)
            saved.append((scene_key, frame_path))
        
        return saved
    
    def _extract_video_frames_decord(self,
                                      video_path: str,
                                      target_frames_info: List[Tuple],
                                      results: Dict[str, str]) -> Dict[str, str]:
        """
        Use decord to batch extract frames, but decode/save in chunks to avoid memory spikes.
        """
        try:
            vr = self._get_decord_reader(video_path)
            total_frames = len(vr)

            # Collect and deduplicate valid frame numbers
            frame_to_keys = defaultdict(list)  # {frame_number: [scene_key, ...]}
            valid_frame_numbers = []
            seen_frames = set()

            for target_frame, scene_key, scene in target_frames_info:
                clamped = max(0, min(target_frame, total_frames - 1))
                frame_to_keys[clamped].append(scene_key)
                if clamped not in seen_frames:
                    seen_frames.add(clamped)
                    valid_frame_numbers.append(clamped)

            if not valid_frame_numbers:
                return results

            # 若 VideoReader 构造时已指定 width/height（解码阶段缩放），则无需再做 PIL 缩放
            decord_out_size = self._decord_reader_output_size.get(video_path)
            need_resize = (
                decord_out_size is None
                and self.output_resolution
                and self.output_resolution != 'original'
                and self.output_resolution.isdigit()
            )
            target_short = int(self.output_resolution) if need_resize else 0

            # 动态限制 decord decode chunk 与 pending 任务数量，控制峰值内存
            decode_batch_size = self.decord_decode_batch_size
            max_pending = self.decord_max_pending_tasks
            est_w, est_h = decord_out_size or self._get_dimensions(video_path)
            if est_w > 0 and est_h > 0:
                bytes_per_frame = max(1, int(est_w) * int(est_h) * 3)  # RGB uint8
                # 控制单次 get_batch().asnumpy() 的峰值（约 <= 128MB）
                auto_decode_cap = max(1, (128 * 1024 * 1024) // bytes_per_frame)
                decode_batch_size = max(1, min(int(decode_batch_size), int(auto_decode_cap)))
                # 控制保存线程 pending 帧拷贝的峰值（约 <= 256MB）
                auto_pending_cap = max(self.decord_save_workers, (256 * 1024 * 1024) // bytes_per_frame)
                max_pending = max(self.decord_save_workers, min(int(max_pending), int(auto_pending_cap)))

            # Single-thread fallback: chunked decode + immediate save
            if self.decord_save_workers <= 1:
                for batch_start in range(0, len(valid_frame_numbers), decode_batch_size):
                    frame_chunk = valid_frame_numbers[batch_start:batch_start + decode_batch_size]
                    frames_batch = vr.get_batch(frame_chunk).asnumpy()  # [B, H, W, C] uint8 RGB

                    for i, frame_number in enumerate(frame_chunk):
                        saved_items = self._save_decord_frame(
                            frames_batch[i],
                            tuple(frame_to_keys[frame_number]),
                            need_resize,
                            target_short,
                        )
                        for scene_key, frame_path in saved_items:
                            with self._cache_lock:
                                self.frame_cache[scene_key] = frame_path
                            results[scene_key] = frame_path

                    del frames_batch

                return results

            # Dual pipeline: main thread decodes chunks, worker threads save to disk.
            pending = []
            with ThreadPoolExecutor(max_workers=self.decord_save_workers) as executor:
                for batch_start in range(0, len(valid_frame_numbers), decode_batch_size):
                    frame_chunk = valid_frame_numbers[batch_start:batch_start + decode_batch_size]
                    frames_batch = vr.get_batch(frame_chunk).asnumpy()  # [B, H, W, C] uint8 RGB

                    for i, frame_number in enumerate(frame_chunk):
                        # Copy to standalone array to avoid keeping whole chunk alive in futures.
                        frame_np = np.ascontiguousarray(frames_batch[i])
                        scene_keys = tuple(frame_to_keys[frame_number])
                        future = executor.submit(
                            self._save_decord_frame,
                            frame_np,
                            scene_keys,
                            need_resize,
                            target_short,
                        )
                        pending.append(future)

                        if len(pending) >= max_pending:
                            done, not_done = wait(pending, return_when=FIRST_COMPLETED)
                            for done_future in done:
                                saved_items = done_future.result()
                                for scene_key, frame_path in saved_items:
                                    with self._cache_lock:
                                        self.frame_cache[scene_key] = frame_path
                                    results[scene_key] = frame_path
                            pending = list(not_done)

                    del frames_batch

                if pending:
                    done, _ = wait(pending)
                    for done_future in done:
                        saved_items = done_future.result()
                        for scene_key, frame_path in saved_items:
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
                text=True,
                encoding='utf-8',
                errors='replace'
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
    
    def extract_batch(self, scenes: List[Dict], show_progress: bool = True,
                      progress_label: str = "帧提取", progress_unit: str = "场景") -> Dict[str, str]:
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
            progress_label: 进度日志标签
            progress_unit: 进度数量单位
        
        Returns:
            {scene_key: frame_path} 字典
        """
        if not scenes:
            return {}
        
        start_time = time.time()
        
        # 去重：同一场景只提取一次
        unique_scenes = {}
        for scene in scenes:
            scene_key = self.get_scene_key(scene['video_path'], scene['start_frame'], scene.get('target_frame_idx', 1))
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
        
        backend_tag = f"[{progress_label}-{self.backend}]"
        if show_progress:
            print(f"{backend_tag} 需要提取 {len(scenes_to_extract)} 个{progress_unit} (缓存命中 {len(cached_results)} 个)")
        
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
                    print(f"    {backend_tag} [{completed}/{len(video_scenes)}] {video_name}: {len(video_results)} 帧")
                    
            except Exception as e:
                print(f"{backend_tag} 视频处理失败: {os.path.basename(video_path)} - {e}")
            finally:
                # 及时释放 decord VideoReader，避免多视频 reader 同时驻留内存
                self._decord_cache.pop(video_path, None)
                self._decord_reader_output_size.pop(video_path, None)
        
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
            self._decord_reader_output_size.clear()
            
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

