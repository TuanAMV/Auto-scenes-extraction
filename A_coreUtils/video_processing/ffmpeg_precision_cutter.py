# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
"""
FFmpeg 高精度视频处理器 (v5 - 精确切割优化版)
============================================

核心修复:
---------
1. 编码模式下自动检测音频流，无音频时跳过aselect
2. 精简输出，只保留错误和结果
3. 支持从 config.json 读取视频输出参数（路径索引与 hybrid 一致）
4. Copy模式优化：先快速切割原视频，再对切割后的片段强力清洗
   - 优势：copy切割速度快（不重新编码）
   - 优势：小片段清洗速度快
   - 优势：总体效率更高

新增功能 (v5 - Hybrid集成):
-------------------------
5. 添加 get_video_meta_via_ffprobe() - 快速获取视频元数据（fps, total_frames）
6. 添加 clean_video_for_detection() - 为场景检测器准备清洗版本
7. 添加 split_video_into_segments() - 视频分段功能（支持多线程检测）
8. 所有新增方法与原有精确切割方法完全隔离，互不干扰

v5.1 精确切割优化:
-----------------
9. 精确切割模式改为"先copy再精确切割"
   - 步骤1: 用copy模式快速切割出扩展片段（前后各多5秒）
   - 步骤2: 对扩展片段进行精确编码切割
   - 帧号偏移确保与原视频对应
   - 优势：copy极快 + 小片段编码快 = 总体效率大幅提升
"""

import subprocess
import os
import sys
import json
import tempfile
import shutil
import glob
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import numpy as np

# ============================================================
#  路径设置 - 确保能找到项目根目录的模块
# ============================================================
_current_file = os.path.abspath(__file__)
_video_processing_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_video_processing_dir)
_project_root_dir = os.path.dirname(_a_core_utils_dir)
if _project_root_dir not in sys.path:
    sys.path.insert(0, _project_root_dir)

# 导入路径解析器
from path_resolver import PathResolver


@dataclass
class VideoInfo:
    fps: float
    fps_num: int
    fps_den: int
    time_base_num: int
    time_base_den: int
    total_frames: int
    duration: float
    width: int
    height: int
    codec_name: str
    has_audio: bool = True  # 新增：是否有音频流


class FFmpegPrecisionCutter:
    """FFmpeg 高精度视频处理器"""
    
    HWACCEL_MAP = {
        'nvidia': 'cuda', 'cuda': 'cuda',
        'intel': 'qsv', 'qsv': 'qsv',
        'amd': 'amf', 'amf': 'amf',
        'apple': 'videotoolbox', 'videotoolbox': 'videotoolbox',
        'none': None
    }
    
    HWACCEL_CONFIGS = {
        'cuda': {
            'decode': 'cuda',
            'h264_encoder': 'h264_nvenc',
            'hevc_encoder': 'hevc_nvenc',
            'preset_map': {
                'ultrafast': 'p1', 'superfast': 'p2', 'veryfast': 'p3',
                'faster': 'p4', 'fast': 'p5', 'medium': 'p5',
                'slow': 'p6', 'slower': 'p7', 'veryslow': 'p7'
            }
        },
        'qsv': {
            'decode': 'qsv',
            'h264_encoder': 'h264_qsv',
            'hevc_encoder': 'hevc_qsv',
            'preset_map': {
                'ultrafast': 'veryfast', 'superfast': 'veryfast',
                'veryfast': 'veryfast', 'faster': 'faster',
                'fast': 'fast', 'medium': 'medium',
                'slow': 'slow', 'slower': 'slower', 'veryslow': 'veryslow'
            }
        },
        'amf': {
            'decode': 'dxva2',
            'h264_encoder': 'h264_amf',
            'hevc_encoder': 'hevc_amf',
            'preset_map': {
                'ultrafast': 'speed', 'superfast': 'speed',
                'veryfast': 'speed', 'faster': 'speed',
                'fast': 'balanced', 'medium': 'balanced',
                'slow': 'quality', 'slower': 'quality', 'veryslow': 'quality'
            }
        },
        'videotoolbox': {
            'decode': 'videotoolbox',
            'h264_encoder': 'h264_videotoolbox',
            'hevc_encoder': 'hevc_videotoolbox',
            'preset_map': {},
            'bitrate_map': {18: '8M', 20: '6M', 23: '4M', 26: '3M', 28: '2M'}
        }
    }
    
    def _load_config(self, config_file: Optional[str] = None, require_video_output: bool = True) -> dict:
        """
        加载配置文件（路径索引与 VideoSceneCutterExport 保持一致）
        
        配置文件查找顺序：
        1. 指定的 config_file 路径
        2. Cut_DetectScene/config.json（项目根目录）
        3. 当前脚本目录下的 config.json（兼容旧版）
        
        Args:
            config_file: 配置文件路径（可选）
            require_video_output: 是否要求 video_output 配置项存在（便捷函数设为 False）
        
        Raises:
            FileNotFoundError: 配置文件不存在时抛出
            json.JSONDecodeError: 配置文件格式错误时抛出
            KeyError: require_video_output=True 且缺少 video_output 时抛出
        """
        script_dir = Path(__file__).parent.absolute()
        # Cut_DetectScene 目录（项目根目录）
        cut_detect_scene_dir = script_dir.parent.parent
        
        # 确定配置文件路径
        if config_file is not None:
            config_path = Path(config_file).expanduser()
            if not config_path.is_absolute():
                config_path = (script_dir / config_path).resolve()
        else:
            # 优先使用项目根目录的 config.json
            config_path = cut_detect_scene_dir / "config.json"
            if not config_path.exists():
                # 兼容旧版：尝试当前脚本目录
                config_path = script_dir / "config.json"
        
        # 配置文件必须存在
        if not config_path.exists():
            raise FileNotFoundError(
                f"配置文件不存在: {config_path}\n"
                f"请确保 Cut_DetectScene/config.json 存在，或通过 config_file 参数指定配置文件路径"
            )
        
        # 读取配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"配置文件格式错误: {config_path}",
                e.doc, e.pos
            )
        
        # 验证必要的配置项（仅在需要时验证）
        if require_video_output and 'video_output' not in config:
            raise KeyError(
                f"配置文件缺少 'video_output' 配置项: {config_path}\n"
                f"请参考项目文档添加 video_output 配置"
            )
        
        return config
    
    def __init__(
        self,
        # 配置文件
        config_file: Optional[str] = None,
        # 视频输出参数（None 表示从 config.json 读取）
        copy_mode: Optional[bool] = None,
        video_codec: Optional[str] = None,
        audio_codec: Optional[str] = None,
        crf: Optional[int] = None,
        preset: Optional[str] = None,
        audio_bitrate: Optional[str] = None,
        hwaccel: Optional[str] = None,
        hwaccel_device: int = 0,
        force_clean: Optional[bool] = None,
        # 新增：像素格式和 MP4 优化参数
        pixel_format: Optional[str] = None,
        movflags: Optional[str] = None,
        # 黑场检测参数
        enable_black_detection: Optional[bool] = None,
        black_threshold: Optional[int] = None,
        temp_dir: Optional[str] = None,
        verbose: bool = False,
        # 内部参数：是否需要 video_output 配置（便捷函数设为 False）
        _require_video_output: bool = True
    ):
        """
        Args:
            config_file: 配置文件路径（None 则使用当前目录下的 config.json）
            copy_mode: 使用 copy 模式（无损快速），None 则从 config.json 读取
            video_codec: 视频编码器，None 则从 config.json 读取
            audio_codec: 音频编码器，None 则从 config.json 读取
            crf: 视频质量，None 则从 config.json 读取
            preset: 编码预设，None 则从 config.json 读取
            audio_bitrate: 音频码率，None 则从 config.json 读取
            hwaccel: 硬件加速类型，None 则从 config.json 读取
            hwaccel_device: 硬件加速设备号
            force_clean: 强制清理视频流，None 则从 config.json 读取
            pixel_format: 像素格式（如 yuv420p），None 则从 config.json 读取
            movflags: MP4 优化标志（如 +faststart），None 则从 config.json 读取
            enable_black_detection: 启用黑场检测，None 则从 config.json 读取
            black_threshold: 黑场检测阈值，None 则从 config.json 读取
            temp_dir: 临时目录路径
            verbose: 是否输出详细信息
            _require_video_output: 内部参数，是否要求 video_output 配置存在
        """
        # 加载配置文件
        config = self._load_config(config_file, require_video_output=_require_video_output)
        video_config = config.get('video_output', {})
        
        # 使用配置文件中的默认值（如果参数为 None）
        if copy_mode is None:
            copy_mode = video_config.get('copy_mode', False)
        if video_codec is None:
            video_codec = video_config.get('video_codec', 'libx265')
        if audio_codec is None:
            audio_codec = video_config.get('audio_codec', 'aac')
        if crf is None:
            crf = video_config.get('crf', 18)
        if preset is None:
            preset = video_config.get('preset', 'medium')
        if audio_bitrate is None:
            audio_bitrate = video_config.get('audio_bitrate', '192k')
        if hwaccel is None:
            hwaccel = video_config.get('hwaccel', 'cuda')
        if force_clean is None:
            force_clean = video_config.get('force_clean', True)
        if pixel_format is None:
            pixel_format = video_config.get('pixel_format', 'yuv420p')
        if movflags is None:
            movflags = video_config.get('movflags', '+faststart')
        if enable_black_detection is None:
            enable_black_detection = video_config.get('enable_black_detection', True)
        if black_threshold is None:
            black_threshold = video_config.get('black_threshold', 15)
        
        # 读取 NVENC 高级参数
        nvenc_config = video_config.get('nvenc', {})
        self.nvenc_rc_mode = nvenc_config.get('rc_mode', 'vbr')
        self.nvenc_spatial_aq = nvenc_config.get('spatial_aq', True)
        self.nvenc_temporal_aq = nvenc_config.get('temporal_aq', True)
        self.nvenc_rc_lookahead = nvenc_config.get('rc_lookahead', 20)
        self.nvenc_qmin_offset = nvenc_config.get('qmin_offset', 0)
        self.nvenc_qmax_offset = nvenc_config.get('qmax_offset', 2)
        
        # 黑场检测
        self.enable_black_detection = enable_black_detection
        self.black_threshold = black_threshold
        
        # 基础参数
        self.copy_mode = copy_mode
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.crf = crf
        self.preset = preset
        self.audio_bitrate = audio_bitrate
        self.force_clean = force_clean
        self.pixel_format = pixel_format
        self.movflags = movflags
        self.verbose = verbose
        
        # 硬件加速
        self.hwaccel = self.HWACCEL_MAP.get(hwaccel.lower() if hwaccel else 'none')
        self.hwaccel_device = hwaccel_device
        
        # 临时目录 - 使用 PathResolver 获取项目根目录的 temp 文件夹
        _resolver = PathResolver()
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            temp_base = config.get('temp_dir', 'temp')
            self.temp_dir = _resolver.join(temp_base)
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg工具路径（使用 PathResolver 获取项目根目录）
        project_ffmpeg_dir = _resolver.join('models', 'ffmpeg', 'bin')
        
        # 查找 ffmpeg
        if (project_ffmpeg_dir / 'ffmpeg.exe').exists():
            self.ffmpeg_path = str(project_ffmpeg_dir / 'ffmpeg.exe')
        elif (project_ffmpeg_dir / 'ffmpeg').exists():
            self.ffmpeg_path = str(project_ffmpeg_dir / 'ffmpeg')
        else:
            raise FileNotFoundError(
                f"未找到 ffmpeg，请确保存在于: {project_ffmpeg_dir}\n"
                f"需要文件: ffmpeg.exe (Windows) 或 ffmpeg (Linux/Mac)"
            )
        
        # 查找 ffprobe
        if (project_ffmpeg_dir / 'ffprobe.exe').exists():
            self.ffprobe_path = str(project_ffmpeg_dir / 'ffprobe.exe')
        elif (project_ffmpeg_dir / 'ffprobe').exists():
            self.ffprobe_path = str(project_ffmpeg_dir / 'ffprobe')
        else:
            raise FileNotFoundError(
                f"未找到 ffprobe，请确保存在于: {project_ffmpeg_dir}\n"
                f"需要文件: ffprobe.exe (Windows) 或 ffprobe (Linux/Mac)"
            )
    
    def has_audio_stream(self, video_path: str) -> bool:
        """检测视频是否包含音频流"""
        cmd = [
            self.ffprobe_path, '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                                   encoding='utf-8', errors='ignore')
            return result.stdout.strip() == 'audio'
        except Exception:
            return False
    
    def get_video_info(self, video_path: str) -> Optional[VideoInfo]:
        """获取视频完整信息"""
        cmd = [
            self.ffprobe_path, '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,avg_frame_rate,time_base,nb_frames,duration,width,height,codec_name',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True,
                                   encoding='utf-8', errors='ignore')
            data = json.loads(result.stdout)
            
            stream = data.get('streams', [{}])[0]
            format_info = data.get('format', {})
            
            fps_str = stream.get('r_frame_rate') or stream.get('avg_frame_rate', '30/1')
            if '/' in fps_str:
                fps_num, fps_den = map(int, fps_str.split('/'))
                fps = fps_num / fps_den if fps_den != 0 else 30.0
            else:
                fps = float(fps_str) if fps_str else 30.0
                fps_num, fps_den = int(fps * 1000), 1000
            
            time_base = stream.get('time_base', '1/90000')
            tb_parts = time_base.split('/')
            tb_num = int(tb_parts[0]) if len(tb_parts) > 0 else 1
            tb_den = int(tb_parts[1]) if len(tb_parts) > 1 else 90000
            
            total_frames = int(stream.get('nb_frames', 0) or 0)
            duration = float(stream.get('duration', 0) or format_info.get('duration', 0) or 0)
            
            if total_frames == 0 and duration > 0:
                total_frames = int(duration * fps)
            
            has_audio = self.has_audio_stream(video_path)
            
            return VideoInfo(
                fps=fps,
                fps_num=fps_num,
                fps_den=fps_den,
                time_base_num=tb_num,
                time_base_den=tb_den,
                total_frames=total_frames,
                duration=duration,
                width=int(stream.get('width', 0) or 0),
                height=int(stream.get('height', 0) or 0),
                codec_name=stream.get('codec_name', 'unknown'),
                has_audio=has_audio
            )
        except Exception as e:
            print(f"错误: 获取视频信息失败 - {e}")
            return None
    
    def get_real_frame_count(self, video_path: str, fps: float = None) -> int:
        """
        获取视频的实际帧数（精确统计）
        
        优先级：
        1. nb_frames（容器元数据，最快）
        2. count_frames（实际统计，准确但慢）
        3. duration * fps（估算，保底）
        
        Args:
            video_path: 视频路径
            fps: 帧率（可选，用于方法3的估算）
        """
        # 方法1：尝试从容器元数据读取
        try:
            cmd = [
                self.ffprobe_path, '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_frames',
                '-of', 'csv=p=0',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5,
                                   encoding='utf-8', errors='ignore')
            if result.returncode == 0 and result.stdout.strip():
                nb_frames = int(result.stdout.strip())
                if nb_frames > 0:
                    return nb_frames
        except Exception:
            pass
        
        # 方法2：实际统计帧数（准确但稍慢）
        try:
            cmd = [
                self.ffprobe_path, '-v', 'error',
                '-select_streams', 'v:0',
                '-count_frames',
                '-show_entries', 'stream=nb_read_frames',
                '-of', 'csv=p=0',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                   encoding='utf-8', errors='ignore')
            if result.returncode == 0 and result.stdout.strip():
                nb_frames = int(result.stdout.strip())
                if nb_frames > 0:
                    return nb_frames
        except Exception:
            pass
        
        # 方法3：使用时长估算（保底方案）
        try:
            cmd = [
                self.ffprobe_path, '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration',
                '-of', 'csv=p=0',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5,
                                   encoding='utf-8', errors='ignore')
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                # 如果没有提供fps，尝试获取
                if fps is None:
                    video_info = self.get_video_info(video_path)
                    if video_info:
                        fps = video_info.fps
                if fps and fps > 0:
                    return int(duration * fps)
        except Exception:
            pass
        
        return 0
    
    # ============== 新增：Hybrid集成方法 ==============
    
    def get_video_meta_via_ffprobe(self, video_path: str) -> Tuple[float, int]:
        """
        使用 ffprobe 获取视频元数据（便捷方法，兼容Hybrid）
        
        Returns:
            (fps, total_frames)
        """
        video_info = self.get_video_info(video_path)
        if video_info:
            return video_info.fps, video_info.total_frames
        
        # 回退方案：尝试使用 ffmpeg -i 解析 stderr
        try:
            cmd = [self.ffmpeg_path, '-i', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding='utf-8', errors='ignore')
            out = result.stderr
            
            import re
            fps = 0
            fps_match = re.search(r'(\d+(?:\.\d+)?) fps', out)
            if fps_match:
                fps = float(fps_match.group(1))
            
            duration = 0
            dur_match = re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', out)
            if dur_match:
                h, m, s = map(float, dur_match.groups())
                duration = h * 3600 + m * 60 + s
            
            total_frames = int(duration * fps) if fps > 0 else 0
            return fps, total_frames
        except Exception:
            return 0.0, 0
    
    def _build_clean_command(
        self,
        input_video: str,
        output_path: str,
        remove_audio: bool = True,
        for_detection: bool = False
    ) -> List[str]:
        """
        构建清洗命令（内部方法）
        
        Args:
            input_video: 输入视频
            output_path: 输出路径
            remove_audio: 是否移除音频
            for_detection: 是否用于场景检测（True时强制移除音频并启用确定性输出）
        
        Returns:
            FFmpeg命令列表
        """
        cmd = [self.ffmpeg_path, '-y', '-loglevel', 'error', '-i', input_video]
        
        # 视频流：copy模式
        cmd.extend(['-map', '0:v:0', '-c:v', 'copy'])
        
        # 音频流处理
        if for_detection or remove_audio:
            cmd.append('-an')  # 移除音频
        else:
            cmd.extend(['-map', '0:a?', '-c:a', 'copy'])  # 保留音频
        
        # 移除其他流和元数据
        cmd.extend([
            '-sn',              # 移除字幕
            '-dn',              # 移除数据流
            '-map_metadata', '-1',  # 移除元数据
            '-map_chapters', '-1'   # 移除章节
        ])
        
        # 检测模式：确保确定性输出
        if for_detection:
            cmd.extend(['-fflags', '+bitexact'])
        
        cmd.append(output_path)
        return cmd
    
    def clean_video_for_detection(self, input_video: str, output_video: str) -> bool:
        """
        为场景检测器准备清洗版本（强力清洗：去除元数据、字幕、音频等）
        
        Args:
            input_video: 原始视频
            output_video: 清洗后视频
        
        Returns:
            是否成功
        """
        cmd = self._build_clean_command(input_video, output_video, for_detection=True)
        
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60
            )
            return os.path.exists(output_video) and os.path.getsize(output_video) > 0
        except Exception:
            return False
    
    def split_video_into_segments(
        self, 
        video_path: str, 
        output_dir: str, 
        segment_frames: int,
        fps: float
    ) -> List[Dict]:
        """
        将视频分割成多个片段（用于分段多线程检测）
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            segment_frames: 每段帧数
            fps: 帧率
        
        Returns:
            List of dict: [{'path': str, 'frame_offset': int, 'frame_count': int, 'fps': float}, ...]
        """
        # 计算每段秒数
        segment_seconds = int(segment_frames / fps)
        
        # === 第一步：获取原始视频元数据 ===
        fps_actual, total_frames = self.get_video_meta_via_ffprobe(video_path)
        if fps_actual <= 0:
            raise ValueError(f"无法获取视频帧率: {video_path}")
        
        if total_frames == 0:
            raise ValueError(f"无法获取视频帧数（包括估算）: {video_path}")
        
        if self.verbose:
            print(f"  原视频元数据: FPS={fps_actual:.2f}, 总帧数={total_frames}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        seg_pattern = os.path.join(output_dir, "segment_%04d.mkv")
        
        # === 第二步：使用 ffmpeg 分段 ===
        cmd = [
            self.ffmpeg_path, '-y', '-i', video_path,
            '-c:v', 'copy', '-map', '0:v:0',
            '-sn', '-dn', '-map_metadata', '-1',
            '-f', 'segment', '-segment_time', str(segment_seconds),
            '-reset_timestamps', '1', seg_pattern
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            # 如果 copy 失败，尝试重编码
            cmd[4:6] = ['-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p']
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 获取所有分段文件
        segment_files = sorted(glob.glob(os.path.join(output_dir, "segment_*.mkv")))
        if not segment_files:
            raise RuntimeError("分段失败：未生成任何分段文件")
        
        # === 第三步：基于原视频元数据计算每段帧偏移（统一使用理论计算） ===
        segments = []
        segment_frame_count = segment_frames  # 使用传入的帧数
        
        for i, seg_path in enumerate(segment_files):
            frame_offset = i * segment_frame_count
            
            # 最后一段可能不满
            if i == len(segment_files) - 1:
                frame_count = total_frames - frame_offset
            else:
                frame_count = segment_frame_count
            
            segments.append({
                'path': seg_path,
                'frame_offset': frame_offset,
                'frame_count': frame_count,
                'fps': fps_actual
            })
        
        return segments
    
    # ============== 以下为原有精确切割方法，保持不变 ==============
    
    def _frame_to_time_precise(self, frame: int, fps_num: int, fps_den: int) -> str:
        """帧号转精确时间"""
        frac = Fraction(frame * fps_den, fps_num)
        # [修复] FFmpeg -ss 不接受分数格式(如"0/1")，需转为秒数格式
        return f"{float(frac):.9f}"
    
    def _get_hwaccel_params(self):
        """获取硬件加速参数"""
        if self.hwaccel is None:
            return None, self.video_codec, self.preset
        
        config = self.HWACCEL_CONFIGS.get(self.hwaccel, {})
        hwaccel_decode = config.get('decode')
        
        # 编码器选择
        if self.video_codec.lower() in ['libx264', 'h264']:
            video_encoder = config.get('h264_encoder', 'libx264')
        elif self.video_codec.lower() in ['libx265', 'hevc', 'h265']:
            video_encoder = config.get('hevc_encoder', 'libx265')
        else:
            video_encoder = self.video_codec
        
        # 预设映射
        preset_map = config.get('preset_map', {})
        encoder_preset = preset_map.get(self.preset, self.preset)
        
        return hwaccel_decode, video_encoder, encoder_preset
    
    def _get_encoder_quality_params(self, hwaccel_decode, video_encoder):
        """获取编码器质量参数（从 config.json 读取 NVENC 高级参数）"""
        params = []
        
        if self.hwaccel == 'cuda' and 'nvenc' in video_encoder:
            # 使用从 config.json 读取的 NVENC 参数
            qmin = self.crf + self.nvenc_qmin_offset
            qmax = self.crf + self.nvenc_qmax_offset
            params.extend(['-rc:v', self.nvenc_rc_mode, '-cq:v', str(self.crf), '-qmin', str(qmin), '-qmax', str(qmax)])
            spatial_aq_val = '1' if self.nvenc_spatial_aq else '0'
            temporal_aq_val = '1' if self.nvenc_temporal_aq else '0'
            params.extend(['-spatial_aq', spatial_aq_val, '-temporal_aq', temporal_aq_val, '-rc-lookahead', str(self.nvenc_rc_lookahead)])
        elif self.hwaccel == 'qsv' and 'qsv' in video_encoder:
            params.extend(['-global_quality', str(self.crf), '-look_ahead', '1'])
        elif self.hwaccel == 'amf' and 'amf' in video_encoder:
            params.extend(['-rc', 'cqp', '-qp_i', str(self.crf), '-qp_p', str(self.crf), '-qp_b', str(self.crf)])
        elif self.hwaccel == 'videotoolbox' and 'videotoolbox' in video_encoder:
            bitrate_map = self.HWACCEL_CONFIGS['videotoolbox'].get('bitrate_map', {})
            bitrate = bitrate_map.get(self.crf, '4M')
            params.extend(['-b:v', bitrate])
        else:
            params.extend(['-crf', str(self.crf)])
        
        return params
    
    def _escape_path_for_lavfi(self, path: str) -> str:
        """
        为 FFmpeg lavfi 滤镜转义路径
        
        FFmpeg lavfi 滤镜（如 movie=）需要特殊的路径转义：
        1. 反斜杠 \\ -> /（Windows路径转换）
        2. 单引号 ' -> '\\''（shell转义）
        3. 冒号 : -> \\:（lavfi特殊字符）
        4. 分号 ; -> \\;（lavfi特殊字符）
        5. 方括号 [] -> \\[\\]（lavfi特殊字符）
        6. 逗号 , -> \\,（lavfi特殊字符）
        7. 最后用单引号包裹整个路径
        
        Args:
            path: 原始文件路径
        
        Returns:
            转义后的路径字符串（已包含单引号）
        """
        # Step 1: Windows 路径转换为正斜杠
        escaped = path.replace('\\', '/')
        
        # Step 2: 转义 lavfi 特殊字符（顺序很重要）
        # 注意：反斜杠本身也需要转义，所以用 \\\\ 表示一个反斜杠
        special_chars = [
            ("'", "'\\''"),      # 单引号转义（shell风格）
            (':', '\\:'),        # 冒号
            (';', '\\;'),        # 分号
            ('[', '\\['),        # 左方括号
            (']', '\\]'),        # 右方括号
            (',', '\\,'),        # 逗号
        ]
        
        for char, escaped_char in special_chars:
            escaped = escaped.replace(char, escaped_char)
        
        # Step 3: 用单引号包裹
        return f"'{escaped}'"
    
    def check_black_video(self, video_path: str, threshold: int = None) -> bool:
        """检查视频是否为黑场"""
        if threshold is None:
            threshold = self.black_threshold
        
        cmd = [
            self.ffprobe_path, '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'frame=pkt_pts_time,pict_type',
            '-read_intervals', '%+#1',
            '-of', 'csv=p=0',
            video_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True,
                          encoding='utf-8', errors='ignore')
        except Exception:
            return False
        
        # 【修复】路径转义：处理空格和特殊字符
        # FFmpeg lavfi 滤镜需要完整转义：反斜杠、单引号、冒号、特殊字符等
        escaped_path = self._escape_path_for_lavfi(video_path)
        
        cmd_black = [
            self.ffprobe_path, '-v', 'error',
            '-f', 'lavfi',
            '-i', f"movie={escaped_path},blackdetect=d=0.1:pic_th=0.{threshold:02d}",
            '-show_entries', 'tags=lavfi.black_start,lavfi.black_end',
            '-of', 'csv=p=0'
        ]
        
        try:
            result = subprocess.run(cmd_black, capture_output=True, text=True, timeout=30,
                                   encoding='utf-8', errors='ignore')
            lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
            
            if len(lines) > 0:
                first_line = lines[0].strip()
                parts = first_line.split(',')
                if len(parts) >= 2:
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        return start < 0.1 and end > 0.5
                    except (ValueError, TypeError):
                        pass
            
            return False
        except Exception:
            return False
    
    def clean_video(self, input_video: str, output_path: Optional[str] = None, remove_audio: bool = True) -> Optional[str]:
        """
        清洗视频（可自动生成临时文件）
        
        Args:
            input_video: 输入视频
            output_path: 输出路径（None则自动生成临时文件）
            remove_audio: 是否移除音频
        
        Returns:
            输出文件路径（失败返回None）
        """
        # 自动生成临时文件
        if output_path is None:
            suffix = Path(input_video).suffix or '.mkv'
            fd, output_path = tempfile.mkstemp(suffix=suffix, prefix='cleaned_')
            os.close(fd)
        
        # 使用共享的命令构建方法
        cmd = self._build_clean_command(input_video, output_path, remove_audio=remove_audio, for_detection=False)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding='utf-8', errors='ignore')
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            return None
        except Exception:
            return None
    
    def cut_by_frames(
        self,
        input_video: str,
        output_file: str,
        start_frame: int,
        end_frame: int,
        fps: Optional[float] = None,
        video_info: Optional[VideoInfo] = None,
        end_inclusive: bool = True,
        start_frame_offset: int = 0,
        end_frame_offset: int = 0
    ) -> bool:
        """
        基于帧数进行高精度切割
        
        Args:
            input_video: 输入视频路径
            output_file: 输出文件路径
            start_frame: 起始帧号（包含）
            end_frame: 结束帧号
            fps: 帧率（可选，自动获取）
            video_info: 视频信息（可选，自动获取）
            end_inclusive: 结束帧是否包含在内
                - True: 闭区间 [start_frame, end_frame]，适用于帧号区间
                  例如：to_scenes() 返回的 (0, 99) 表示帧 0-99 都需要包含
                - False: 半开区间 [start_frame, end_frame)，适用于时间区间
                  例如：cut_by_time(0.0, 4.0) 内部转换后使用
            start_frame_offset: 起始帧偏移量（负数向前，正数向后），默认0
                - 例如：-2 表示起始帧向前偏移2帧
            end_frame_offset: 结束帧偏移量（负数向前，正数向后），默认0
                - 例如：2 表示结束帧向后偏移2帧
        
        Returns:
            bool: 切割是否成功
        
        Note:
            - 使用场景检测器返回的帧区间时，应设置 end_inclusive=True
            - 使用时间区间时，应设置 end_inclusive=False（cut_by_time 已自动处理）
            - start_frame_offset/end_frame_offset 由调用方控制，用于场景边界微调
        """
        # 应用调用方指定的帧偏移
        adjusted_start = max(0, start_frame + start_frame_offset)
        adjusted_end = end_frame + end_frame_offset
        
        if video_info is None:
            video_info = self.get_video_info(input_video)
        
        if fps is None:
            fps = video_info.fps if video_info else 30.0
        
        fps_num = video_info.fps_num if video_info else int(fps * 1000)
        fps_den = video_info.fps_den if video_info else 1000
        
        # 【修正】获取音频流信息
        # Copy模式：从原视频获取（先切割后清洗，切割时需要音频信息）
        # Encode模式：如果 force_clean=True，跳过音频（因为会在编码前清洗）
        if self.copy_mode:
            # Copy模式：直接从原视频获取音频信息
            has_audio = video_info.has_audio if video_info else self.has_audio_stream(input_video)
        else:
            # Encode模式：如果 force_clean=True，跳过音频处理
            if self.force_clean:
                has_audio = False
            else:
                has_audio = video_info.has_audio if video_info else self.has_audio_stream(input_video)
        
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            if self.copy_mode:
                return self._cut_copy_mode(input_video, output_file, adjusted_start, adjusted_end,
                                          fps, fps_num, fps_den, end_inclusive, has_audio)
            else:
                return self._cut_encode_mode(input_video, output_file, adjusted_start, adjusted_end,
                                            fps, fps_num, fps_den, end_inclusive, has_audio)
        except Exception as e:
            print(f"错误: 切割失败 - {e}")
            return False
    
    def cut_by_time(self, input_video: str, output_file: str, start_time: float, end_time: float) -> bool:
        """
        基于时间进行切割
        
        Args:
            input_video: 输入视频路径
            output_file: 输出文件路径
            start_time: 起始时间（秒，包含）
            end_time: 结束时间（秒，不包含）
        
        Returns:
            bool: 切割是否成功
        
        Note:
            时间区间为半开区间 [start_time, end_time)
            内部自动设置 end_inclusive=False
        """
        video_info = self.get_video_info(input_video)
        fps = video_info.fps if video_info else 30.0
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        return self.cut_by_frames(input_video, output_file, start_frame, end_frame, fps, video_info, end_inclusive=False)
    
    def _cut_copy_mode(self, input_video: str, output_file: str, start_frame: int, end_frame: int, 
                       fps: float, fps_num: int, fps_den: int, end_inclusive: bool = False,
                       has_audio: bool = True) -> bool:
        """复制模式切割 - 先快速切割，再强力清洗"""
        start_time_precise = self._frame_to_time_precise(start_frame, fps_num, fps_den)
        
        # 计算实际结束帧（帧偏移已在 cut_by_frames 中应用）
        if end_inclusive:
            actual_end_frame = end_frame + 1
        else:
            actual_end_frame = end_frame
        
        end_time_precise = self._frame_to_time_precise(actual_end_frame, fps_num, fps_den)
        
        # Step 1: 快速切割（copy模式）
        temp_cut = None
        try:
            suffix = Path(output_file).suffix or '.mkv'
            fd, temp_cut = tempfile.mkstemp(suffix=suffix, prefix='cut_', dir=self.temp_dir)
            os.close(fd)
            
            cmd = [
                self.ffmpeg_path, '-y', '-loglevel', 'error',
                '-ss', start_time_precise,
                '-to', end_time_precise,
                '-i', input_video,
                '-c:v', 'copy', '-avoid_negative_ts', 'make_zero'
            ]
            
            if has_audio and self.audio_codec == 'copy':
                cmd.extend(['-c:a', 'copy'])
            else:
                cmd.append('-an')
            
            cmd.extend(['-sn', temp_cut])
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding='utf-8', errors='ignore')
            
            if result.returncode != 0 or not os.path.exists(temp_cut):
                if self.verbose:
                    print(f"错误: Copy切割失败 - {result.stderr}")
                return False
            
            # Step 2: 强力清洗（移除元数据、字幕等）
            if self.force_clean:
                clean_cmd = [
                    self.ffmpeg_path, '-y', '-loglevel', 'error',
                    '-i', temp_cut,
                    '-map', '0:v:0', '-c:v', 'copy'
                ]
                
                if has_audio and self.audio_codec == 'copy':
                    clean_cmd.extend(['-map', '0:a?', '-c:a', 'copy'])
                else:
                    clean_cmd.append('-an')
                
                clean_cmd.extend([
                    '-sn', '-dn',
                    '-map_metadata', '-1',
                    '-map_chapters', '-1',
                    output_file
                ])
                
                result = subprocess.run(clean_cmd, capture_output=True, text=True,
                                       encoding='utf-8', errors='ignore')
                
                if result.returncode != 0:
                    if self.verbose:
                        print(f"错误: 清洗失败 - {result.stderr}")
                    if os.path.exists(temp_cut):
                        shutil.move(temp_cut, output_file)
                    return os.path.exists(output_file)
            else:
                if os.path.exists(temp_cut):
                    shutil.move(temp_cut, output_file)
            
            return os.path.exists(output_file) and os.path.getsize(output_file) > 0
            
        except Exception as e:
            if self.verbose:
                print(f"错误: Copy模式切割失败 - {e}")
            return False
        finally:
            if temp_cut and os.path.exists(temp_cut):
                try:
                    os.remove(temp_cut)
                except OSError:
                    pass
    
    def _cut_encode_mode(self, input_video: str, output_file: str, start_frame: int, end_frame: int,
                        fps: float, fps_num: int, fps_den: int, end_inclusive: bool = False,
                        has_audio: bool = True) -> bool:
        """
        编码模式切割 - 先copy再精确切割（优化版v2）
        
        步骤：
        1. 用copy模式快速切割出片段（从目标起始帧到结束帧，时间戳归0）
           - Copy会自动往前找I帧，所以实际起始帧 <= 请求的起始帧
           - 不需要后扩展，因为结束帧是精确的
        2. 用ffprobe获取copy片段的总帧数，通过帧数差计算偏移
        3. 对copy片段进行精确编码切割，使用偏移后的帧号
        
        关键公式：
        - 请求帧数 = end_frame - start_frame
        - copy后总帧数 = 实际帧数（包含I帧前移的额外帧）
        - 偏移帧数 = copy后总帧数 - 请求帧数
        - 精确切割范围 = [偏移帧数, copy后总帧数-1]
        
        优势：
        - copy切割极快（不重新编码）
        - 小片段精确切割速度快
        - 通过帧数差计算偏移，避免时间戳精度问题
        """
        # 计算实际结束帧（帧偏移已在 cut_by_frames 中应用）
        if end_inclusive:
            actual_end_frame = end_frame + 1
        else:
            actual_end_frame = end_frame
        
        # 请求的帧数范围
        requested_frame_count = actual_end_frame - start_frame
        
        temp_copy = None
        try:
            # Step 1: 用copy模式快速切割，时间戳归0
            suffix = Path(output_file).suffix or '.mkv'
            fd, temp_copy = tempfile.mkstemp(suffix=suffix, prefix='precopy_', dir=self.temp_dir)
            os.close(fd)
            
            copy_start_time = self._frame_to_time_precise(start_frame, fps_num, fps_den)
            copy_end_time = self._frame_to_time_precise(actual_end_frame, fps_num, fps_den)
            
            # 时间戳归0，不使用 -copyts
            copy_cmd = [
                self.ffmpeg_path, '-y', '-loglevel', 'error',
                '-ss', copy_start_time,
                '-to', copy_end_time,
                '-i', input_video,
                '-c:v', 'copy',
                '-avoid_negative_ts', 'make_zero',  # 时间戳归0
            ]
            
            # Copy时保留音频（如果有）
            if has_audio:
                copy_cmd.extend(['-c:a', 'copy'])
            else:
                copy_cmd.append('-an')
            
            copy_cmd.extend(['-sn', temp_copy])
            
            result = subprocess.run(copy_cmd, capture_output=True, text=True,
                                   encoding='utf-8', errors='ignore')
            
            if result.returncode != 0 or not os.path.exists(temp_copy):
                if self.verbose:
                    print(f"[Info] Copy预切割失败，回退到直接编码模式")
                # 回退到直接编码模式
                return self._cut_encode_mode_direct(input_video, output_file, start_frame, actual_end_frame,
                                                   fps, fps_num, fps_den, has_audio)
            
            # Step 2: 获取copy片段的总帧数，通过帧数差计算偏移
            copy_total_frames = self.get_real_frame_count(temp_copy, fps)
            if copy_total_frames <= 0:
                if self.verbose:
                    print(f"[Warning] 无法获取copy片段帧数，回退到直接编码模式")
                return self._cut_encode_mode_direct(input_video, output_file, start_frame, actual_end_frame,
                                                   fps, fps_num, fps_den, has_audio)
            
            # 计算帧偏移：copy后总帧数 - 请求帧数 = I帧前移导致的额外帧数
            frame_offset = copy_total_frames - requested_frame_count
            
            # 【修复】帧偏移负值处理：如果 copy 后帧数少于请求帧数，说明切割异常
            # 此时应回退到直接编码模式，而不是简单置0
            if frame_offset < 0:
                if self.verbose:
                    print(f"[Warning] Copy帧数({copy_total_frames}) < 请求帧数({requested_frame_count})，回退到直接编码模式")
                return self._cut_encode_mode_direct(input_video, output_file, start_frame, actual_end_frame,
                                                   fps, fps_num, fps_den, has_audio)
            
            # 在copy片段中的目标帧号
            local_start_frame = frame_offset
            local_end_frame = copy_total_frames - 1
            
            # Step 3: 对copy片段进行精确编码切割
            hwaccel_decode, video_encoder, encoder_preset = self._get_hwaccel_params()
            
            output_ext = os.path.splitext(output_file)[1].lower()
            is_mp4_compatible = output_ext in ['.mp4', '.mov', '.m4v']
            
            # 使用计算后的本地帧号
            video_filter = f"select='between(n\\,{local_start_frame}\\,{local_end_frame})',setpts=N/FRAME_RATE/TB"
            
            if has_audio:
                audio_start_time = self._frame_to_time_precise(local_start_frame, fps_num, fps_den)
                audio_end_time = self._frame_to_time_precise(local_end_frame + 1, fps_num, fps_den)
                audio_filter = f"atrim=start={audio_start_time}:end={audio_end_time},asetpts=PTS-STARTPTS"
                filter_complex = f"[0:v]{video_filter}[v];[0:a]{audio_filter}[a]"
            else:
                # 【修复】无音频时也需要输出标签 [v]，否则 -map [v] 会失败
                filter_complex = f"[0:v]{video_filter}[v]"
            
            cmd = [self.ffmpeg_path, '-y', '-loglevel', 'error']
            
            if hwaccel_decode:
                cmd.extend(['-hwaccel', hwaccel_decode])
                if hwaccel_decode == 'cuda':
                    cmd.extend(['-hwaccel_device', str(self.hwaccel_device)])
            
            # 输入是copy后的小片段
            cmd.extend(['-i', temp_copy])
            cmd.extend(['-filter_complex', filter_complex])
            
            if has_audio:
                cmd.extend(['-map', '[v]', '-map', '[a]'])
            else:
                cmd.extend(['-map', '[v]'])
            
            cmd.extend(['-c:v', video_encoder])
            
            if encoder_preset and self.hwaccel in ['cuda', 'qsv']:
                cmd.extend(['-preset', encoder_preset])
            elif encoder_preset and self.hwaccel == 'amf':
                cmd.extend(['-quality', encoder_preset])
            elif self.hwaccel is None:
                cmd.extend(['-preset', self.preset])
            
            cmd.extend(self._get_encoder_quality_params(hwaccel_decode, video_encoder))
            
            # 添加像素格式参数
            if self.pixel_format:
                cmd.extend(['-pix_fmt', self.pixel_format])
            
            if has_audio:
                audio_codec = 'aac' if self.audio_codec == 'copy' else self.audio_codec
                cmd.extend(['-c:a', audio_codec, '-b:a', self.audio_bitrate])
                cmd.append('-shortest')
            else:
                cmd.append('-an')
            
            cmd.extend(['-sn'])
            
            # 使用配置的 movflags 参数
            if is_mp4_compatible and self.movflags:
                cmd.extend(['-movflags', self.movflags])
            
            cmd.append(output_file)
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding='utf-8', errors='ignore')
            
            if result.returncode != 0:
                if self.verbose:
                    print(f"错误: 精确切割失败 - {result.stderr}")
                # 备用方案：纯视频模式
                return self._cut_encode_mode_video_only(temp_copy, output_file, local_start_frame, local_end_frame,
                                                        fps, fps_num, fps_den, video_encoder, encoder_preset, is_mp4_compatible)
            
            return os.path.exists(output_file) and os.path.getsize(output_file) > 0
            
        except Exception as e:
            if self.verbose:
                print(f"错误: 编码模式切割失败 - {e}")
            return False
        finally:
            # 清理临时文件
            if temp_copy and os.path.exists(temp_copy):
                try:
                    os.remove(temp_copy)
                except OSError:
                    pass
    
    def _cut_encode_mode_direct(self, input_video: str, output_file: str, start_frame: int, end_frame: int,
                               fps: float, fps_num: int, fps_den: int, has_audio: bool = True) -> bool:
        """
        直接编码模式切割（回退方案）
        
        当copy预切割失败时使用，直接对原视频进行编码切割
        """
        hwaccel_decode, video_encoder, encoder_preset = self._get_hwaccel_params()
        
        output_ext = os.path.splitext(output_file)[1].lower()
        is_mp4_compatible = output_ext in ['.mp4', '.mov', '.m4v']
        
        video_end_frame = max(start_frame, end_frame - 1)
        video_filter = f"select='between(n\\,{start_frame}\\,{video_end_frame})',setpts=N/FRAME_RATE/TB"
        
        if has_audio:
            audio_start_time = self._frame_to_time_precise(start_frame, fps_num, fps_den)
            audio_end_time = self._frame_to_time_precise(end_frame, fps_num, fps_den)
            audio_filter = f"atrim=start={audio_start_time}:end={audio_end_time},asetpts=PTS-STARTPTS"
            filter_complex = f"[0:v]{video_filter}[v];[0:a]{audio_filter}[a]"
        else:
            # 【修复】无音频时也需要输出标签 [v]，否则 -map [v] 会失败
            filter_complex = f"[0:v]{video_filter}[v]"
        
        cmd = [self.ffmpeg_path, '-y', '-loglevel', 'error']
        
        if hwaccel_decode:
            cmd.extend(['-hwaccel', hwaccel_decode])
            if hwaccel_decode == 'cuda':
                cmd.extend(['-hwaccel_device', str(self.hwaccel_device)])
        
        cmd.extend(['-i', input_video])
        cmd.extend(['-filter_complex', filter_complex])
        
        if has_audio:
            cmd.extend(['-map', '[v]', '-map', '[a]'])
        else:
            cmd.extend(['-map', '[v]'])
        
        cmd.extend(['-c:v', video_encoder])
        
        if encoder_preset and self.hwaccel in ['cuda', 'qsv']:
            cmd.extend(['-preset', encoder_preset])
        elif encoder_preset and self.hwaccel == 'amf':
            cmd.extend(['-quality', encoder_preset])
        elif self.hwaccel is None:
            cmd.extend(['-preset', self.preset])
        
        cmd.extend(self._get_encoder_quality_params(hwaccel_decode, video_encoder))
        
        # 添加像素格式参数
        if self.pixel_format:
            cmd.extend(['-pix_fmt', self.pixel_format])
        
        if has_audio:
            audio_codec = 'aac' if self.audio_codec == 'copy' else self.audio_codec
            cmd.extend(['-c:a', audio_codec, '-b:a', self.audio_bitrate])
            cmd.append('-shortest')
        else:
            cmd.append('-an')
        
        cmd.extend(['-sn'])
        
        # 使用配置的 movflags 参数
        if is_mp4_compatible and self.movflags:
            cmd.extend(['-movflags', self.movflags])
        
        cmd.append(output_file)
        
        result = subprocess.run(cmd, capture_output=True, text=True,
                               encoding='utf-8', errors='ignore')
        
        if result.returncode != 0:
            if self.verbose:
                print(f"错误: FFmpeg - {result.stderr}")
            return self._cut_encode_mode_video_only(input_video, output_file, start_frame, end_frame,
                                                    fps, fps_num, fps_den, video_encoder, encoder_preset, is_mp4_compatible)
        
        return os.path.exists(output_file) and os.path.getsize(output_file) > 0
    
    def _cut_encode_mode_video_only(self, input_video: str, output_file: str, start_frame: int, 
                                    end_frame: int, fps: float, fps_num: int, fps_den: int,
                                    video_encoder: str, encoder_preset: str, is_mp4_compatible: bool) -> bool:
        """编码模式：仅视频"""
        video_filter = f"select='between(n\\,{start_frame}\\,{end_frame})',setpts=N/FRAME_RATE/TB"
        
        cmd = [self.ffmpeg_path, '-y', '-loglevel', 'error']
        cmd.extend(['-i', input_video])
        cmd.extend(['-vf', video_filter])
        cmd.extend(['-c:v', video_encoder])
        
        if encoder_preset and self.hwaccel in ['cuda', 'qsv']:
            cmd.extend(['-preset', encoder_preset])
        elif encoder_preset and self.hwaccel == 'amf':
            cmd.extend(['-quality', encoder_preset])
        elif self.hwaccel is None:
            cmd.extend(['-preset', self.preset])
        
        cmd.extend(self._get_encoder_quality_params(None, video_encoder))
        
        # 添加像素格式参数
        if self.pixel_format:
            cmd.extend(['-pix_fmt', self.pixel_format])
        
        cmd.extend(['-an', '-sn'])
        
        # 使用配置的 movflags 参数
        if is_mp4_compatible and self.movflags:
            cmd.extend(['-movflags', self.movflags])
        
        cmd.append(output_file)
        
        result = subprocess.run(cmd, capture_output=True, text=True,
                               encoding='utf-8', errors='ignore')
        return result.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0
    
    def batch_cut(self, input_video: str, output_dir: str, scenes: list, filename_generator=None,
                  fps: Optional[float] = None, end_inclusive: bool = False) -> Tuple[int, int]:
        """批量切割视频"""
        os.makedirs(output_dir, exist_ok=True)
        
        video_info = self.get_video_info(input_video)
        if fps is None:
            fps = video_info.fps if video_info else 30.0
        
        success_count = 0
        fail_count = 0
        
        video_name = Path(input_video).stem
        video_ext = Path(input_video).suffix
        
        for i, (start_frame, end_frame) in enumerate(scenes):
            if filename_generator:
                filename = filename_generator(input_video, start_frame, i)
            else:
                filename = f"{video_name}_{start_frame}{video_ext}"
            
            output_file = os.path.join(output_dir, filename)
            
            if self.cut_by_frames(input_video, output_file, start_frame, end_frame, fps, video_info, end_inclusive):
                success_count += 1
            else:
                fail_count += 1
        
        return success_count, fail_count
    
    def extract_frames_by_numbers(self, video_path: str, frame_numbers: List[int],
                                   output_dir: Optional[str] = None, quality: int = 2,
                                   output_resolution: str = None) -> List[str]:
        """
        提取指定帧号的图片
        
        Args:
            video_path: 视频文件路径
            frame_numbers: 要提取的帧号列表
            output_dir: 输出目录（默认使用 temp_dir）
            quality: JPEG 质量（1-31，越小越好）
            output_resolution: 输出分辨率（短边像素数，如 '384'，或 'original' 保持原始）
        
        Returns:
            List[str]: 提取的帧文件路径列表
        """
        if output_dir is None:
            output_dir = str(self.temp_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        video_info = self.get_video_info(video_path)
        if not video_info:
            return []
        
        fps = video_info.fps
        width = video_info.width
        height = video_info.height
        output_files = []
        
        # 构建 scale filter（参考 video_utils._build_scale_filter）
        scale_filter = None
        if output_resolution and output_resolution != 'original':
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
                scale_filter = f"scale={out_width}:{out_height}"
        
        for frame_num in frame_numbers:
            timestamp = frame_num / fps
            output_file = os.path.join(output_dir, f"frame_{frame_num:08d}.jpg")
            
            cmd = [
                self.ffmpeg_path, '-y',
                '-ss', f'{timestamp:.6f}',
                '-i', video_path,
                '-frames:v', '1',
            ]
            
            # 添加 scale filter
            if scale_filter:
                cmd.extend(['-vf', scale_filter])
            
            cmd.extend(['-q:v', str(quality), output_file])
            
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=10)
                if os.path.exists(output_file):
                    output_files.append(output_file)
            except (subprocess.SubprocessError, OSError):
                pass
        
        return output_files
    
    def cleanup_temp(self):
        """清理临时目录（包括所有临时文件）"""
        if self.temp_dir.exists():
            # 【修复】清理所有临时文件类型，包括视频文件
            temp_patterns = [
                '*.jpg', '*.png',           # 图片文件
                'cut_*.mkv', 'cut_*.mp4',   # copy模式临时文件
                'precopy_*.mkv', 'precopy_*.mp4',  # 编码模式临时文件
                'cleaned_*.mkv', 'cleaned_*.mp4',  # 清洗临时文件
                'segment_*.mkv',            # 分段临时文件
            ]
            for pattern in temp_patterns:
                for f in self.temp_dir.glob(pattern):
                    try:
                        f.unlink()
                    except OSError:
                        pass

# 便捷函数
def cut_video_by_frames(input_video: str, output_file: str, start_frame: int, end_frame: int,
                        fps: float = None, copy_mode: bool = True, end_inclusive: bool = False,
                        start_frame_offset: int = 0, end_frame_offset: int = 0, **kwargs) -> bool:
    """
    便捷函数：基于帧数切割视频
    
    Args:
        input_video: 输入视频路径
        output_file: 输出文件路径
        start_frame: 起始帧号
        end_frame: 结束帧号
        fps: 帧率（可选）
        copy_mode: 是否使用copy模式（快速但可能有关键帧偏移）
        end_inclusive: 结束帧是否包含在内
        start_frame_offset: 起始帧偏移量（负数向前，正数向后），默认0
        end_frame_offset: 结束帧偏移量（负数向前，正数向后），默认0
        **kwargs: 传递给 FFmpegPrecisionCutter 的其他参数
    
    Returns:
        bool: 切割是否成功
    """
    cutter = FFmpegPrecisionCutter(copy_mode=copy_mode, **kwargs)
    return cutter.cut_by_frames(input_video, output_file, start_frame, end_frame, fps,
                                end_inclusive=end_inclusive,
                                start_frame_offset=start_frame_offset,
                                end_frame_offset=end_frame_offset)


def export_video_clip(input_video: str, output_file: str, start_frame: int, end_frame: int,
                      copy_mode: bool = True, start_frame_offset: int = 0, end_frame_offset: int = 0,
                      **kwargs) -> bool:
    """
    便捷函数：导出视频片段（推荐用于场景导出）
    
    这是 cut_video_by_frames 的封装，默认使用 end_inclusive=True（闭区间），
    适用于场景检测器返回的帧区间。
    
    Args:
        input_video: 输入视频路径
        output_file: 输出文件路径
        start_frame: 起始帧号
        end_frame: 结束帧号
        copy_mode: 是否使用copy模式（快速但可能有关键帧偏移）
        start_frame_offset: 起始帧偏移量（负数向前，正数向后），默认0
            - 例如：-2 表示起始帧向前偏移2帧
        end_frame_offset: 结束帧偏移量（负数向前，正数向后），默认0
            - 例如：2 表示结束帧向后偏移2帧
        **kwargs: 传递给 FFmpegPrecisionCutter 的其他参数
    
    Returns:
        bool: 导出是否成功
    
    Example:
        # Copy模式导出，尾帧+2
        export_video_clip(video, output, 0, 100, copy_mode=True, end_frame_offset=2)
        
        # 精确模式导出，起始帧-2，尾帧+2
        export_video_clip(video, output, 0, 100, copy_mode=False,
                         start_frame_offset=-2, end_frame_offset=2)
    """
    return cut_video_by_frames(
        input_video=input_video,
        output_file=output_file,
        start_frame=start_frame,
        end_frame=end_frame,
        copy_mode=copy_mode,
        end_inclusive=True,  # 场景导出使用闭区间
        start_frame_offset=start_frame_offset,
        end_frame_offset=end_frame_offset,
        **kwargs
    )



def get_video_info(video_path: str) -> Optional[VideoInfo]:
    # 【修复】只需要获取视频信息，不需要 video_output 配置
    cutter = FFmpegPrecisionCutter(_require_video_output=False)
    return cutter.get_video_info(video_path)


def extract_frames(video_path: str, frame_numbers: List[int], output_dir: str = None,
                   quality: int = 2, output_resolution: str = None) -> List[str]:
    """
    便捷函数：提取指定帧号的图片
    
    Args:
        video_path: 视频文件路径
        frame_numbers: 要提取的帧号列表
        output_dir: 输出目录
        quality: JPEG 质量（1-31，越小越好）
        output_resolution: 输出分辨率（短边像素数，如 '384'，或 'original' 保持原始）
    """
    # 【修复】只需要提取帧，不需要 video_output 配置
    cutter = FFmpegPrecisionCutter(temp_dir=output_dir, _require_video_output=False) if output_dir else FFmpegPrecisionCutter(_require_video_output=False)
    return cutter.extract_frames_by_numbers(video_path, frame_numbers, output_dir, quality, output_resolution)


def check_black_video(video_path: str, threshold: int = 15) -> bool:
    # 【修复】只需要检测黑场，不需要 video_output 配置
    cutter = FFmpegPrecisionCutter(_require_video_output=False)
    return cutter.check_black_video(video_path, threshold)


# ============== Hybrid便捷函数 ==============

def get_video_meta_via_ffprobe(video_path: str) -> Tuple[float, int]:
    """
    便捷函数：获取视频元数据
    
    Returns:
        (fps, total_frames)
    """
    # 【修复】只需要获取元数据，不需要 video_output 配置
    cutter = FFmpegPrecisionCutter(_require_video_output=False)
    return cutter.get_video_meta_via_ffprobe(video_path)


def clean_video_for_detection(input_video: str, output_video: str) -> bool:
    """
    便捷函数：为检测器清洗视频
    
    强力清洗模式：移除音频、字幕、元数据等所有干扰元素
    
    Args:
        input_video: 输入视频
        output_video: 输出视频路径
    
    Returns:
        是否成功
    """
    # 【修复】清洗视频使用 copy 模式，不需要 video_output 配置
    cutter = FFmpegPrecisionCutter(_require_video_output=False)
    return cutter.clean_video_for_detection(input_video, output_video)


def clean_video(input_video: str, output_path: Optional[str] = None, remove_audio: bool = True) -> Optional[str]:
    """
    便捷函数：清洗视频（可选保留音频）
    
    Args:
        input_video: 输入视频
        output_path: 输出路径（None则自动生成临时文件）
        remove_audio: 是否移除音频
    
    Returns:
        输出文件路径（失败返回None）
    """
    # 【修复】清洗视频使用 copy 模式，不需要 video_output 配置
    cutter = FFmpegPrecisionCutter(_require_video_output=False)
    return cutter.clean_video(input_video, output_path, remove_audio)


def split_video_into_segments(video_path: str, output_dir: str, segment_frames: int, fps: float) -> List[Dict]:
    """
    便捷函数：分割视频为多个片段
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
        segment_frames: 每段帧数
        fps: 帧率
    
    Returns:
        [{'path': str, 'frame_offset': int, 'frame_count': int, 'fps': float}, ...]
    """
    # 【修复】分割视频使用 copy 模式，不需要 video_output 配置
    cutter = FFmpegPrecisionCutter(_require_video_output=False)
    return cutter.split_video_into_segments(video_path, output_dir, segment_frames, fps)


def get_real_frame_count(video_path: str, fps: float = None) -> int:
    """
    便捷函数：获取视频实际帧数（三种方法）
    
    优先级：
    1. nb_frames（容器元数据，最快）
    2. count_frames（实际统计，准确但慢）
    3. duration * fps（估算，保底）
    
    Args:
        video_path: 视频路径
        fps: 帧率（可选，用于方法3的估算）
    
    Returns:
        视频帧数
    """
    # 【修复】只需要获取帧数，不需要 video_output 配置
    cutter = FFmpegPrecisionCutter(_require_video_output=False)
    return cutter.get_real_frame_count(video_path, fps)
