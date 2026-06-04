# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
# video_processing_utils.py
# 视频处理系统工具类 - GPU优化版本 (修复版)
# v3.0: 新增 SceneFeatureExtractor.extract_all_from_scene 支持三元组完整匹配
# v2.9: 移除欧氏距离选项，仅保留余弦相似度

import os
import sys
import subprocess
import json
import shutil
import time
import glob
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Event

# ============================================================
#  路径设置 - 确保能找到项目根目录的模块
# ============================================================
_current_file = os.path.abspath(__file__)
_video_processing_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_video_processing_dir)
_cut_detect_scene_dir = os.path.dirname(_a_core_utils_dir)
if _cut_detect_scene_dir not in sys.path:
    sys.path.insert(0, _cut_detect_scene_dir)

# ============================================================
#  全局配置
# ============================================================

# 导入统一路径解析器（从项目根目录）
from path_resolver import PathResolver

# 初始化路径解析器（不传参数，使用 path_resolver.py 所在目录作为项目根目录）
_path_resolver = PathResolver()
TEMP_DIR = _path_resolver.join_str('temp')

# FFmpeg工具路径（使用 PathResolver 获取 models/ffmpeg/bin）
_ffmpeg_bin_dir = _path_resolver.join('models', 'ffmpeg', 'bin')

_ffmpeg_exe = _ffmpeg_bin_dir / 'ffmpeg.exe'
if not _ffmpeg_exe.exists():
    _ffmpeg_exe = _ffmpeg_bin_dir / 'ffmpeg'
if not _ffmpeg_exe.exists():
    raise FileNotFoundError(
        f"未找到 ffmpeg，请确保存在于: {_ffmpeg_bin_dir}\n"
        f"需要文件: ffmpeg.exe (Windows) 或 ffmpeg (Linux/Mac)"
    )
FFMPEG_PATH = str(_ffmpeg_exe)

_ffprobe_exe = _ffmpeg_bin_dir / 'ffprobe.exe'
if not _ffprobe_exe.exists():
    _ffprobe_exe = _ffmpeg_bin_dir / 'ffprobe'
if not _ffprobe_exe.exists():
    raise FileNotFoundError(
        f"未找到 ffprobe，请确保存在于: {_ffmpeg_bin_dir}\n"
        f"需要文件: ffprobe.exe (Windows) 或 ffprobe (Linux/Mac)"
    )
FFPROBE_PATH = str(_ffprobe_exe)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 临时文件夹管理
# ============================================================
def cleanup_temp_folder(folder_path: str = None):
    if folder_path is None:
        folder_path = TEMP_DIR
    if os.path.isdir(folder_path):
        try:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except PermissionError:
                    print(f"[Warning] 文件被占用，跳过: {filename}")
                except OSError as e:
                    print(f"[Warning] 删除 {filename} 失败: {e}")
        except OSError as e:
            print(f"[Warning] 清理文件夹时出错: {e}")


# ============================================================
# 单帧提取工具 (纯CPU解码 - 禁用GPU加速避免与模型推理冲突)
# ============================================================

def extract_single_frame(video_path: str, frame_number: int, fps: float, output_path: str) -> bool:
    """
    精确按帧号提取缩略图（用于 HTML 预览，固定宽度 320px）
    方案3: 快速seek + 精确定位
    """
    try:
        # 提前2秒seek，确保不跳过目标帧
        approx_time = max(0, (frame_number / fps) - 2.0)
        seek_frame = int(approx_time * fps)
        relative_frame = frame_number - seek_frame
        
        cmd = [FFMPEG_PATH, '-y',
            '-ss', str(approx_time),
            '-i', video_path,
            '-vf', f"select='eq(n,{relative_frame})',scale=320:-1",
            '-vframes', '1', 
            '-q:v', '3',
            '-vsync', 'vfr',
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True, 
                      encoding='utf-8', errors='ignore')
        return os.path.exists(output_path)
    except (subprocess.SubprocessError, OSError):
        return False


def extract_single_frame_rerank(
    video_path: str, 
    frame_number: int, 
    fps: float, 
    output_path: str,
    output_resolution: str = '384'
) -> bool:
    """
    精确按帧号提取帧图像（用于 Reranker，支持短边缩放）
    方案3: 快速seek + 精确定位
    
    Args:
        video_path: 视频文件路径
        frame_number: 目标帧号
        fps: 视频帧率
        output_path: 输出图像路径
        output_resolution: 输出分辨率（短边像素数，如 '384'，或 'original' 保持原始）
    
    Returns:
        bool: 是否成功提取
    """
    try:
        # 提前2秒seek，确保不跳过目标帧
        approx_time = max(0, (frame_number / fps) - 2.0)
        seek_frame = int(approx_time * fps)
        relative_frame = frame_number - seek_frame
        
        cmd = [FFMPEG_PATH, '-y',
            '-ss', str(approx_time),
            '-i', video_path,
        ]
        
        # 构建滤镜链：精确帧选择 + 可选缩放
        vf_parts = [f"select='eq(n,{relative_frame})'"]
        
        # 使用短边缩放（需要先获取视频尺寸）
        if output_resolution and output_resolution != 'original':
            # 获取视频尺寸
            width, height = _get_video_dimensions(video_path)
            if width > 0 and height > 0:
                scale_filter = _build_scale_filter_for_rerank(width, height, output_resolution)
                if scale_filter:
                    vf_parts.append(scale_filter)
        
        cmd.extend([
            '-vf', ','.join(vf_parts),
            '-vframes', '1', 
            '-q:v', '2',  # 稍高质量用于 reranker
            '-vsync', 'vfr',
            output_path
        ])
        
        subprocess.run(cmd, check=True, capture_output=True, text=True, 
                      encoding='utf-8', errors='ignore')
        return os.path.exists(output_path)
    except (subprocess.SubprocessError, OSError):
        return False


def _get_video_dimensions(video_path: str) -> tuple:
    """获取视频宽高（复用 VideoMetaHelper.get_video_meta）"""
    _, _, width, height = VideoMetaHelper.get_video_meta(video_path)
    return width, height


def _build_scale_filter_for_rerank(width: int, height: int, output_resolution: str) -> str:
    """构建短边缩放滤镜（用于 Reranker），委托给 FrameExtractorThread._build_scale_filter"""
    if not output_resolution:
        return None
    scale_filter, _, _ = FrameExtractorThread._build_scale_filter(width, height, output_resolution)
    return scale_filter


# ============================================================
# 批量帧提取器 - 修复stderr管道死锁
# ============================================================
class FrameExtractorThread:
    """CPU线程: ffmpeg提取帧到temp文件夹
    
    v3.1: 黑帧检测移至 VectorizerThread 的预加载阶段，与图像加载合并，无额外IO开销
    """
    
    def __init__(self, video_path: str, sample_interval: int, output_resolution: str = '384'):
        self.video_path = video_path
        self.sample_interval = sample_interval
        self.output_resolution = output_resolution
        self.temp_dir = os.path.join(TEMP_DIR, f"extract_{int(time.time() * 1000)}")
        self.total_frames = 0
        self.extraction_done = Event()
        self.error = None
        
    def start(self):
        os.makedirs(self.temp_dir, exist_ok=True)
        thread = Thread(target=self._extract_worker, daemon=True)
        thread.start()
        return self
    
    def _extract_worker(self):
        """提取工作线程 - 修复版本：使用后台线程读取stderr防止死锁"""
        try:
            fps, total_frames, width, height = VideoMetaHelper.get_video_meta(self.video_path)
            if total_frames == 0:
                self.error = "无法获取视频元数据"
                self.extraction_done.set()
                return
            
            expected_frames = (total_frames // self.sample_interval) + 1
            
            # 构建滤镜链 - 只做采样和缩放，黑帧检测在 VectorizerThread 预加载时进行
            filters = [f"select=not(mod(n\\,{self.sample_interval}))"]
            scale_filter, out_width, out_height = self._build_scale_filter(width, height, self.output_resolution)
            if scale_filter:
                filters.append(scale_filter)
            
            output_pattern = os.path.join(self.temp_dir, "frame_%06d.jpg")
            cmd = [FFMPEG_PATH, '-i', self.video_path, '-vf', ','.join(filters),
                   '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', output_pattern]
            
            print(f"[CPU] 开始流式提取帧 (预计 {expected_frames} 帧)")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # ✅ 修复：后台线程异步读取stderr，防止管道阻塞导致死锁
            stderr_chunks = []
            def stderr_reader():
                try:
                    for line in process.stderr:
                        stderr_chunks.append(line)
                except (IOError, OSError):
                    pass
            
            reader_thread = Thread(target=stderr_reader, daemon=True)
            reader_thread.start()
            
            # 主线程轮询进度
            # ✅ 修复：使用最高水位线统计，因为GPU会边处理边删除文件
            last_reported = 0
            max_frame_number_seen = 0  # 记录看到的最大帧编号
            last_print_time = time.time()
            import re
            frame_pattern = re.compile(r'frame_(\d+)\.jpg$')
            
            while process.poll() is None:
                current_files = glob.glob(os.path.join(self.temp_dir, "frame_*.jpg"))
                
                # ✅ 关键：追踪最大帧编号（水位线），而非当前文件数
                for f in current_files:
                    match = frame_pattern.search(f)
                    if match:
                        frame_num = int(match.group(1))
                        if frame_num > max_frame_number_seen:
                            max_frame_number_seen = frame_num
                
                current_time = time.time()
                if max_frame_number_seen != last_reported and current_time - last_print_time > 1.0:
                    progress = (max_frame_number_seen / expected_frames * 100) if expected_frames > 0 else 0
                    print(f"[CPU] 提取进度: {max_frame_number_seen}/{expected_frames} 帧 ({progress:.1f}%)")
                    last_reported = max_frame_number_seen
                    last_print_time = current_time
                time.sleep(0.3)
            
            # 等待读取线程结束
            reader_thread.join(timeout=5.0)
            stderr = ''.join(stderr_chunks)
            
            if process.returncode != 0:
                self.error = f"ffmpeg提取失败: {stderr}"
            else:
                # ✅ 修复：最后再扫描一次更新水位线
                final_files = glob.glob(os.path.join(self.temp_dir, "frame_*.jpg"))
                for f in final_files:
                    match = frame_pattern.search(f)
                    if match:
                        frame_num = int(match.group(1))
                        if frame_num > max_frame_number_seen:
                            max_frame_number_seen = frame_num
                
                # ✅ 使用水位线作为实际提取帧数（最大帧编号=总帧数）
                self.total_frames = max_frame_number_seen
                print(f"[CPU] 帧提取完成,共 {self.total_frames} 帧")
                
        except Exception as e:
            self.error = f"提取线程异常: {e}"
        finally:
            self.extraction_done.set()
    
    @staticmethod
    def _build_scale_filter(width: int, height: int, output_resolution: str) -> tuple:
        if output_resolution == 'original':
            return None, width, height
        if output_resolution.endswith('%'):
            percent = float(output_resolution.rstrip('%')) / 100
            out_width = int(width * percent)
            out_height = int(height * percent)
            out_width = out_width - (out_width % 2)
            out_height = out_height - (out_height % 2)
            return f"scale={out_width}:{out_height}", out_width, out_height
        if output_resolution.isdigit():
            target_short = int(output_resolution)
            if width >= height:
                out_height = target_short
                out_width = int(width * target_short / height)
            else:
                out_width = target_short
                out_height = int(height * target_short / width)
            out_width = out_width - (out_width % 2)
            out_height = out_height - (out_height % 2)
            return f"scale={out_width}:{out_height}", out_width, out_height
        return None, width, height

    def get_frame_files(self) -> list:
        """获取当前已提取的所有帧文件路径"""
        return glob.glob(os.path.join(self.temp_dir, "frame_*.jpg"))
    
    def is_done(self) -> bool:
        """检查提取是否完成"""
        return self.extraction_done.is_set()

    def cleanup(self):
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# ============================================================
# 视频元数据工具
# ============================================================
class VideoMetaHelper:
    @staticmethod
    def get_video_meta(video_path: str) -> tuple:
        cmd = [FFPROBE_PATH, '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate,nb_frames,width,height',
               '-of', 'json', video_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                   encoding='utf-8', errors='ignore', timeout=30)
            data = json.loads(result.stdout)
            stream = data['streams'][0]
            
            fps_str = stream.get('r_frame_rate', '30/1')
            fps_parts = fps_str.split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            
            nb_frames = stream.get('nb_frames', '0')
            total_frames = int(nb_frames) if nb_frames and nb_frames != 'N/A' else 0
            
            if total_frames == 0:
                duration = VideoMetaHelper.get_duration(video_path)
                if duration > 0 and fps > 0:
                    total_frames = int(duration * fps)
                    print(f"[Info] 使用时长估算帧数: {total_frames} (时长: {duration:.2f}s, FPS: {fps:.2f})")
                else:
                    total_frames = VideoMetaHelper._count_frames_via_decode(video_path)
            
            width = int(stream.get('width', 1920))
            height = int(stream.get('height', 1080))
            
            return fps, total_frames, width, height
        except Exception as e:
            print(f"[Warning] 获取视频元数据失败: {e}")
            return 30.0, 0, 1920, 1080

    @staticmethod
    def _count_frames_via_decode(video_path: str) -> int:
        try:
            cmd = [FFPROBE_PATH, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                   '-show_entries', 'stream=nb_read_frames', '-of', 'json', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                   encoding='utf-8', errors='ignore', timeout=120)
            data = json.loads(result.stdout)
            return int(data['streams'][0].get('nb_read_frames', 0))
        except Exception:
            return 0

    @staticmethod
    def get_duration(video_path: str) -> float:
        cmd = [FFPROBE_PATH, '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'json', video_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                   encoding='utf-8', errors='ignore', timeout=30)
            data = json.loads(result.stdout)
            return float(data['format'].get('duration', 0))
        except Exception:
            return 0.0
        
    @staticmethod
    def frame_to_seconds(frame: int, fps: float) -> float:
        """将帧号转换为秒数"""
        if fps <= 0:
            fps = 25.0
        return frame / fps

    @staticmethod
    def get_fps_cached(video_path: str, cache: dict = None) -> float:
        """获取视频FPS（带缓存）"""
        if cache is not None and video_path in cache:
            return cache[video_path]
        fps, _, _, _ = VideoMetaHelper.get_video_meta(video_path)
        if cache is not None:
            cache[video_path] = fps
        return fps
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """将秒数格式化为 MM:SS 或 HH:MM:SS"""
        if seconds < 0:
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    
    @staticmethod
    def seconds_to_timecode(seconds: float) -> str:
        """将秒数转换为时间码 HH:MM:SS.mmm"""
        if seconds < 0:
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

# ============================================================
# 场景边界检测器
# ============================================================
class SceneBoundaryDetector:
    @staticmethod
    def _strict_local_minima_indices(similarities, order: int = 1):
        """
        获取严格局部最小值索引（仿 argrelmin 严格比较）。
        平顶（plateau）不算极值。
        """
        try:
            order = int(order)
        except (TypeError, ValueError):
            order = 1
        if order < 1:
            order = 1
        
        if isinstance(similarities, torch.Tensor):
            if similarities.ndim != 1:
                similarities = similarities.reshape(-1)
            window_size = 2 * order + 1
            if similarities.numel() < window_size:
                return torch.empty(0, dtype=torch.long, device=similarities.device)
            windows = similarities.unfold(0, window_size, 1)
            center = windows[:, order:order + 1]
            is_strict_min = (center < windows[:, :order]).all(dim=1) & (center < windows[:, order + 1:]).all(dim=1)
            return torch.where(is_strict_min)[0] + order
        
        similarities_np = np.asarray(similarities).reshape(-1)
        window_size = 2 * order + 1
        if similarities_np.size < window_size:
            return np.array([], dtype=np.int64)
        
        local_min_indices = []
        for i in range(order, similarities_np.size - order):
            center = similarities_np[i]
            left = similarities_np[i - order:i]
            right = similarities_np[i + 1:i + order + 1]
            if np.all(center < left) and np.all(center < right):
                local_min_indices.append(i)
        
        return np.array(local_min_indices, dtype=np.int64)

    @staticmethod
    def find_scene_boundaries(frame_indices: list, features, localmax_order: int = 2,
                              cosine_similarity_threshold: float = None) -> list:
        """
        检测场景边界，返回边界帧索引列表
        
        Args:
            frame_indices: 采样帧的原始帧号列表
            features: 特征向量 (torch.Tensor 或 np.ndarray)
            localmax_order: 局部最大值检测的阶数
            cosine_similarity_threshold: 余弦相似度阈值（默认0.85，相似度低于此值则分割）
        
        Returns:
            边界帧索引列表（原始帧号）
        """
        if len(frame_indices) < 2:
            return [frame_indices[0]] if frame_indices else [0]
        
        # 设置默认阈值
        if cosine_similarity_threshold is None:
            cosine_similarity_threshold = 0.85
        
        try:
            localmax_order = int(localmax_order)
        except (TypeError, ValueError):
            localmax_order = 2
        if localmax_order < 1:
            localmax_order = 1
        
        is_tensor = isinstance(features, torch.Tensor)
        
        # 计算相邻帧间的相似度（向量化）
        if is_tensor:
            similarities = (features[:-1] * features[1:]).sum(dim=1)
            if similarities.numel() == 0:
                return [frame_indices[0]]
        else:
            features_np = np.asarray(features)
            similarities = np.einsum('ij,ij->i', features_np[:-1], features_np[1:])
            if similarities.size == 0:
                return [frame_indices[0]]
        
        # 找局部极值作为候选边界
        boundaries = [frame_indices[0]]  # 起始帧
        
        if len(similarities) > 2 * localmax_order:
            # 找严格局部最小值（平顶不算极值）
            local_min_indices = SceneBoundaryDetector._strict_local_minima_indices(
                similarities, order=localmax_order
            )
            
            if is_tensor:
                if local_min_indices.numel() > 0:
                    valid_indices = local_min_indices[
                        similarities[local_min_indices] < cosine_similarity_threshold
                    ].detach().cpu().tolist()
                    for idx in valid_indices:
                        boundaries.append(frame_indices[idx + 1])
            else:
                for idx in local_min_indices:
                    if similarities[idx] < cosine_similarity_threshold:
                        boundaries.append(frame_indices[idx + 1])
        
        # 如果没找到边界，检查是否有超过阈值的点
        if len(boundaries) == 1:
            if is_tensor:
                valid_indices = torch.where(similarities < cosine_similarity_threshold)[0].detach().cpu().tolist()
                for idx in valid_indices:
                    boundaries.append(frame_indices[idx + 1])
            else:
                for i, sim in enumerate(similarities):
                    if sim < cosine_similarity_threshold:
                        boundaries.append(frame_indices[i + 1])
        
        # 确保边界是排序的且唯一的
        boundaries = sorted(set(boundaries))
        
        return boundaries

    @staticmethod
    def find_prev_sample_frame(boundary_frame: int, frame_indices: list, sample_interval: int = 3) -> int:
        """
        找到边界帧之前的最后一个采样帧
        
        逻辑：
        1. end_frame = 切割点 - 采样间隔
        2. 如果 end_frame 不在 frame_indices 中（被黑帧过滤），继续往前减
        3. 返回第一个在 frame_indices 中的帧
        
        Args:
            boundary_frame: 边界帧号（切割点）
            frame_indices: 采样帧列表（已排序，黑帧已过滤）
            sample_interval: 采样间隔
        
        Returns:
            边界帧之前的最后一个采样帧号
        """
        if not frame_indices:
            return boundary_frame - sample_interval
        
        # 转为 set 加速查找
        frame_set = set(frame_indices)
        
        # 从切割点往前找
        end_frame = boundary_frame - sample_interval
        
        # 如果不在 frame_indices 中（黑帧被过滤），继续往前
        while end_frame > 0 and end_frame not in frame_set:
            end_frame -= sample_interval
        
        # 如果找到了有效帧，返回它；否则返回第一个采样帧
        return end_frame if end_frame in frame_set else frame_indices[0]

    @staticmethod
    def get_scene_features(scene: tuple, frame_indices: list, features) -> list:
        """
        获取场景的特征向量（起始帧、中间帧、结束帧）
        
        始终返回三元组列表 [start_feature, mid_feature, end_feature]
        """
        start_frame, end_frame = scene
        
        # 找到场景内的采样帧索引
        scene_sample_indices = [i for i, fidx in enumerate(frame_indices) 
                               if start_frame <= fidx <= end_frame]
        
        is_tensor = isinstance(features, torch.Tensor)
        
        if not scene_sample_indices:
            if is_tensor:
                dim = features.shape[1]
                zero_vec = torch.zeros(dim, device=features.device)
            else:
                dim = features.shape[1] if len(features.shape) > 1 else 512
                zero_vec = np.zeros(dim)
            return [zero_vec, zero_vec, zero_vec]
        
        start_idx = scene_sample_indices[0]
        start_feature = features[start_idx]
        
        end_idx = scene_sample_indices[-1]
        end_feature = features[end_idx]
        
        if len(scene_sample_indices) >= 3:
            mid_idx = scene_sample_indices[len(scene_sample_indices) // 2]
            mid_feature = features[mid_idx]
        else:
            mid_feature = (start_feature + end_feature) / 2
        
        return [start_feature, mid_feature, end_feature]
    
    @staticmethod
    def merge_short_scenes(scenes: list, frame_indices: list, features, 
                          min_scene_length: int = 7) -> list:
        """
        合并短场景到相邻最相似的场景
        
        迭代合并，直到没有短场景可合并
        """
        if not scenes or min_scene_length <= 1:
            return scenes
        
        is_tensor = isinstance(features, torch.Tensor)
        use_gpu = is_tensor and features.is_cuda
        
        max_iterations = len(scenes)
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 检查是否还有短场景
            short_scene_exists = any(
                (end - start) < min_scene_length 
                for start, end in scenes
            )
            
            if not short_scene_exists:
                break
            
            # 为当前场景列表计算代表向量(中间帧)
            scene_vectors = []
            for start, end in scenes:
                scene_feat = SceneBoundaryDetector.get_scene_features((start, end), frame_indices, features)
                scene_vectors.append(scene_feat[1])
            
            merged_scenes = []
            skip_indices = set()
            merged_any = False
            
            i = 0
            while i < len(scenes):
                if i in skip_indices:
                    i += 1
                    continue
                
                start, end = scenes[i]
                scene_length = end - start
                
                # 如果是短场景，尝试合并
                if scene_length < min_scene_length:
                    best_neighbor = -1
                    best_sim = -1.0
                    
                    # 检查前一个场景
                    if i > 0 and (i - 1) not in skip_indices:
                        if use_gpu:
                            val = torch.dot(scene_vectors[i], scene_vectors[i - 1]).item()
                        else:
                            val = float(np.dot(scene_vectors[i], scene_vectors[i - 1]))
                        if val > best_sim:
                            best_sim = val
                            best_neighbor = i - 1
                    
                    # 检查后一个场景
                    if i < len(scenes) - 1 and (i + 1) not in skip_indices:
                        if use_gpu:
                            val = torch.dot(scene_vectors[i], scene_vectors[i + 1]).item()
                        else:
                            val = float(np.dot(scene_vectors[i], scene_vectors[i + 1]))
                        if val > best_sim:
                            best_sim = val
                            best_neighbor = i + 1
                    
                    if best_neighbor >= 0:
                        merged_any = True
                        if best_neighbor < i:
                            if merged_scenes:
                                prev_start, prev_end = merged_scenes[-1]
                                merged_scenes[-1] = (prev_start, end)
                        else:
                            next_start, next_end = scenes[best_neighbor]
                            merged_scenes.append((start, next_end))
                            skip_indices.add(best_neighbor)
                    else:
                        merged_scenes.append((start, end))
                else:
                    merged_scenes.append((start, end))
                
                i += 1
            
            scenes = merged_scenes
            
            if not merged_any:
                break
        
        return scenes


# ============================================================
# 场景特征提取工具
# ============================================================
class SceneFeatureExtractor:
    @staticmethod
    def extract_all_from_scene(scene: dict) -> list:
        """
        从场景中提取所有特征向量
        
        Returns:
            list: 三元组特征向量列表 [start_feat, mid_feat, end_feat]
        """
        result = []
        features_list = scene.get('features', [])
        if isinstance(features_list, list):
            for feat in features_list:
                if feat is not None:
                    if isinstance(feat, torch.Tensor):
                        result.append(feat.cpu().numpy())
                    elif isinstance(feat, np.ndarray):
                        result.append(feat)
                    else:
                        result.append(np.array(feat, dtype=np.float32))
        return result
    
# ============================================================
# 工具函数
# ============================================================
def sanitize_name(base_name):
    return "".join([c for c in base_name if c.isalnum() or c in (' ', '.', '_', '-')]).strip()
