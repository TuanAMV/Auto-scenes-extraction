# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
"""
视频场景合并器 - 负责主体判断和视频合并
"""

import os
import sys
import gc
import re
import subprocess
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

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

# FFmpeg路径（使用 PathResolver 获取 models/ffmpeg/bin）
_merger_resolver = PathResolver()
_merger_ffmpeg_bin = _merger_resolver.join('models', 'ffmpeg', 'bin')
_merger_ffmpeg_exe = _merger_ffmpeg_bin / 'ffmpeg.exe'
if not _merger_ffmpeg_exe.exists():
    _merger_ffmpeg_exe = _merger_ffmpeg_bin / 'ffmpeg'
if not _merger_ffmpeg_exe.exists():
    raise FileNotFoundError(
        f"未找到 ffmpeg，请确保存在于: {_merger_ffmpeg_bin}\n"
        f"需要文件: ffmpeg.exe (Windows) 或 ffmpeg (Linux/Mac)"
    )
FFMPEG_PATH = str(_merger_ffmpeg_exe)

# 运行时导入 VideoSceneAnalyzer
try:
    from .Video_Scene_Analyzer import VideoSceneAnalyzer
except ImportError:
    VideoSceneAnalyzer = None
    print("警告: Video_Scene_Analyzer 模块未找到，部分功能可能不可用")


class VideoSceneMerger:
    """
    视频场景合并器 - 专注于主体判断和合并
    
    工作流程：
    1. 读取已重命名的视频（由Analyzer重命名）
    2. 按起始帧数排序，依次比较相邻视频的主体是否相同
    3. 合并同主体视频，从文件名提取标签去重和最高分值
    
    文件名格式：景别_情绪_主体_动作_背景_镜头_分值X_剧名_剧集_起始帧数.扩展名
    或：景别_情绪_主体_动作_背景_镜头_分值X_剧名_剧集_起始帧数_1/2/3.扩展名（去掉末尾数字）
    合并时同类标签放在一起，分值取最高，起始帧取最早
    """
    
    # 标签分类字典
    TAG_CATEGORIES = {
        'shot_scale': {
            '远景', '全景', '中景', '近景', '特写', '大特写',
            '极远景', '中近景', '中全景', '过肩镜头'
        },
        'emotion': {
            '孤独', '紧张', '恐惧', '悲伤', '喜悦', '愤怒', '平静', '焦虑',
            '绝望', '希望', '忧郁', '兴奋', '惊讶', '厌恶', '温馨', '压抑',
            '轻松', '沉重', '神秘', '浪漫', '伤感', '欢快', '阴郁', '明朗'
        },
        'subject': {
            '人物', '人群', '女孩', '男孩', '男人', '女人', '老人', '孩子',
            '动物', '建筑', '车辆', '物品', '风景', '天空', '大海', '山脉',
            '树木', '花草', '街道', '房间', '城市', '乡村'
        },
        'action': {
            '站立', '行走', '奔跑', '静止', '坐着', '躺着', '跳跃', '飞行',
            '游泳', '驾驶', '交谈', '拥抱', '打斗', '舞蹈', '工作', '休息',
            '穿梭', '等待', '观望', '思考', '哭泣', '微笑', '转身', '离开'
        },
        'background': {
            '室内', '室外', '白天', '夜晚', '黄昏', '黎明', '雨天', '晴天',
            '雪景', '雾气', '都市', '郊外', '海边', '山区', '森林', '沙漠',
            '办公室', '家庭', '学校', '医院', '餐厅', '商场', '公园', '车站'
        },
        'camera_movement': {
            '推', '拉', '摇', '移', '跟', '升', '降', '甩', '晃',
            '固定', '手持', '航拍', '旋转', '环绕', '俯拍', '仰拍',
            '平移', '变焦', '聚焦', '虚焦'
        }
    }
    
    # 类别顺序
    CATEGORY_ORDER = ['shot_scale', 'emotion', 'subject', 'action', 'background', 'camera_movement']
    
    def __init__(
        self,
        analyzer: Optional[Any] = None,
        model_path: str = 'hf-int4',
        compression_ratio: float = 0.5,
        video_extensions: Optional[List[str]] = None,
        max_context_count: int = 200
    ):
        """
        初始化合并器
        
        参数:
            analyzer: VideoSceneAnalyzer实例，如果为None则自动创建
            model_path: 模型路径或别名
            compression_ratio: 图片尺寸压缩比例 (0-1)
            video_extensions: 支持的视频格式列表
            max_context_count: 最大上下文调用次数，超过后重新加载模型
        """
        self.model_path = model_path
        self.compression_ratio = compression_ratio
        self.max_context_count = max_context_count
        self.context_count = 0
        
        self.analyzer = analyzer if analyzer else VideoSceneAnalyzer(
            model_path=model_path,
            compression_ratio=compression_ratio
        )
        
        if video_extensions is None:
            self.video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.m4v'}
        else:
            self.video_extensions = set(video_extensions)
        
        self.script_dir = Path(__file__).parent.absolute()
        # 使用 PathResolver 获取项目根目录的 temp 文件夹
        _resolver = PathResolver()
        self.temp_dir = _resolver.join("temp")
        self.temp_dir.mkdir(exist_ok=True, parents=True)
    
    def clear_cache(self):
        """清理GPU和系统缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def reload_model(self):
        """重新加载分析器模型"""
        if hasattr(self.analyzer, 'model'):
            del self.analyzer.model
        if hasattr(self.analyzer, 'tokenizer'):
            del self.analyzer.tokenizer
        
        self.clear_cache()
        
        self.analyzer = VideoSceneAnalyzer(
            model_path=self.model_path,
            compression_ratio=self.compression_ratio
        )
        
        self.context_count = 0
    
    def check_and_reload_if_needed(self):
        """检查上下文计数，必要时重新加载模型"""
        if self.context_count >= self.max_context_count:
            self.reload_model()
    
    def get_sorted_videos(self, folder_path: str) -> List[Path]:
        """
        获取文件夹中按起始帧数排序的视频文件列表
        
        参数:
            folder_path: 文件夹路径
            
        返回:
            按起始帧数排序后的视频文件路径列表
        """
        folder = Path(folder_path)
        
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"文件夹不存在: {folder_path}")
        
        video_files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in self.video_extensions
        ]
        
        def extract_start_frame(video_path: Path) -> int:
            """
            从文件名提取起始帧数用于排序
            
            文件名格式：..._剧名_剧集_起始帧数.ext
            或：..._剧名_剧集_起始帧数_1/2/3.ext（带后缀）
            
            判断逻辑：
            1. 后缀 _1/_2/_3 特点：通常是1位数字(1-9)，用于避免文件重名
            2. 起始帧特点：可以是任意长度的数字，代表视频中的帧位置
            3. 如果最后部分是1位小数字(1-9)，且倒数第二部分也是纯数字，
               则最后部分是后缀，倒数第二部分是起始帧
            """
            filename = video_path.stem
            parts = filename.split('_')
            
            if len(parts) < 1:
                return 0
            
            # 从后往前查找起始帧
            last_part = parts[-1]
            
            # 检查最后部分是否是纯数字
            if last_part.isdigit():
                # 判断是否为后缀（1位数字1-9，且倒数第二部分也是数字）
                is_likely_suffix = (
                    len(last_part) == 1 and 
                    1 <= int(last_part) <= 9 and
                    len(parts) >= 2 and 
                    parts[-2].isdigit()
                )
                
                if is_likely_suffix:
                    # 最后部分是后缀，倒数第二部分是起始帧
                    try:
                        return int(parts[-2])
                    except ValueError:
                        return 0
                else:
                    # 最后部分就是起始帧
                    try:
                        return int(last_part)
                    except ValueError:
                        return 0
            
            # 最后部分不是数字，往前找
            for i in range(len(parts) - 2, -1, -1):
                if parts[i].isdigit():
                    try:
                        return int(parts[i])
                    except ValueError:
                        continue
            
            return 0
        
        video_files.sort(key=extract_start_frame)
        
        return video_files
    
    def classify_tag(self, tag: str) -> str:
        """
        将标签分类到对应类别
        
        参数:
            tag: 标签字符串
            
        返回:
            类别名称，如果无法分类则返回 'other'
        """
        for category, keywords in self.TAG_CATEGORIES.items():
            if tag in keywords:
                return category
        return 'other'
    
    def parse_filename(self, filename: str) -> Dict[str, any]:
        """
        从文件名解析标签和分值
        
        文件名格式：标签1_标签2_..._分值X_剧名_剧集_起始帧数.扩展名
        或：标签1_标签2_..._分值X_剧名_剧集_起始帧数_1/2/3.扩展名
        
        参数:
            filename: 文件名（不含扩展名）
            
        返回:
            包含剧名、剧集、起始帧数、分类标签和分值的字典
        """
        result = {
            'series_name': '',
            'episode': '',
            'start_frame': '',
            'tags': [],
            'categorized_tags': {
                'shot_scale': [],
                'emotion': [],
                'subject': [],
                'action': [],
                'background': [],
                'camera_movement': [],
                'other': []
            },
            'score': None
        }
        
        parts = filename.split('_')
        
        if len(parts) < 3:
            result['tags'] = parts
            for tag in parts:
                category = self.classify_tag(tag)
                result['categorized_tags'][category].append(tag)
            return result
        
        # 检查并去掉末尾的数字后缀（_1/_2/_3等）
        # 后缀特点：1位小数字(1-9)，且倒数第二部分也是数字（起始帧）
        if (len(parts) >= 2 and 
            parts[-1].isdigit() and 
            len(parts[-1]) == 1 and 
            1 <= int(parts[-1]) <= 9 and
            parts[-2].isdigit()):
            # 去掉末尾的数字后缀
            parts = parts[:-1]
        
        # 解析文件名：标签1_标签2_..._分值X_剧名_剧集_起始帧数
        result['start_frame'] = parts[-1]
        result['episode'] = parts[-2]
        result['series_name'] = parts[-3]
        
        # 剩余部分是标签和分值
        tag_parts = parts[:-3]
        
        for part in tag_parts:
            if not part:
                continue
            
            # 查找分值
            score_match = re.match(r'^分值(\d+)$', part)
            if score_match:
                try:
                    result['score'] = int(score_match.group(1))
                except:
                    pass
            elif re.match(r'^score(\d+)$', part, re.IGNORECASE):
                try:
                    score_str = re.search(r'\d+', part)
                    if score_str:
                        result['score'] = int(score_str.group())
                except:
                    pass
            elif re.match(r'^\d+$', part) and len(part) <= 2:
                try:
                    result['score'] = int(part)
                except:
                    pass
            else:
                # 标签进行分类
                result['tags'].append(part)
                category = self.classify_tag(part)
                result['categorized_tags'][category].append(part)
        
        return result
    
    def generate_merged_name(
        self,
        video_infos: List[Dict[str, any]]
    ) -> str:
        """
        生成合并后的文件名
        
        规则：
        1. 同类标签放在一起
        2. 分值取最高
        3. 起始帧取最早
        4. 标签去重
        
        参数:
            video_infos: 视频信息列表
            
        返回:
            合并后的文件名（不含扩展名）
        """
        if not video_infos:
            return "merged_video"
        
        series_name = video_infos[0].get('series_name', '')
        episode = video_infos[0].get('episode', '')
        
        # 收集所有起始帧，取最早的
        start_frames = []
        for info in video_infos:
            frame = info.get('start_frame', '')
            if frame:
                try:
                    start_frames.append(int(frame))
                except ValueError:
                    pass
        earliest_frame = str(min(start_frames)) if start_frames else ''
        
        # 按类别收集标签（去重）
        categorized_tags = {
            'shot_scale': set(),
            'emotion': set(),
            'subject': set(),
            'action': set(),
            'background': set(),
            'camera_movement': set(),
            'other': set()
        }
        
        scores = []
        
        for info in video_infos:
            cat_tags = info.get('categorized_tags', {})
            for category in categorized_tags:
                for tag in cat_tags.get(category, []):
                    if tag and tag not in ['无', '']:
                        categorized_tags[category].add(tag)
            
            if info.get('score') is not None:
                scores.append(info['score'])
        
        # 按类别顺序构建标签列表
        filename_parts = []
        
        for category in self.CATEGORY_ORDER:
            tags = categorized_tags.get(category, set())
            if tags:
                sorted_tags = sorted(tags)
                filename_parts.extend(sorted_tags)
        
        # 添加未分类的标签
        other_tags = categorized_tags.get('other', set())
        if other_tags:
            filename_parts.extend(sorted(other_tags))
        
        # 添加最高分值
        if scores:
            max_score = max(scores)
            filename_parts.append(f"分值{max_score}")
        
        # 添加基础信息
        if series_name:
            filename_parts.append(series_name)
        if episode:
            filename_parts.append(episode)
        if earliest_frame:
            filename_parts.append(earliest_frame)
        
        filename = '_'.join(filename_parts)
        
        return filename
    
    def merge_videos(
        self,
        video_paths: List[Path],
        output_path: str
    ) -> bool:
        """
        合并多个视频文件
        
        参数:
            video_paths: 视频路径列表
            output_path: 输出路径
            
        返回:
            是否成功
        """
        if len(video_paths) < 2:
            return False
        
        list_file = self.temp_dir / "concat_list.txt"
        
        with open(list_file, 'w', encoding='utf-8') as f:
            for video_path in video_paths:
                abs_path = video_path.absolute()
                f.write(f"file '{abs_path}'\n")
        
        cmd = [
            FFMPEG_PATH,
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_file),
            '-c', 'copy',
            '-fflags', '+genpts',
            '-avoid_negative_ts', 'make_zero',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        try:
            os.remove(list_file)
        except:
            pass
        
        return result.returncode == 0
    
    def merge_same_subject_videos(
        self,
        folder_path: str,
        output_folder: Optional[str] = None,
        keep_original: bool = False
    ) -> List[Tuple[str, List[str]]]:
        """
        合并同主体视频
        
        参数:
            folder_path: 输入文件夹路径
            output_folder: 输出文件夹路径，如果为None则输出到输入文件夹
            keep_original: 是否保留原始文件
            
        返回:
            合并结果列表 [(output_path, source_names), ...]
        """
        folder = Path(folder_path)
        
        # 设置输出文件夹
        if output_folder is None:
            output_dir = folder
        else:
            output_dir = Path(output_folder)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        video_files = self.get_sorted_videos(folder_path)
        
        if len(video_files) == 0:
            print(f"错误: 未找到视频文件")
            return []
        
        results = []
        i = 0
        
        while i < len(video_files):
            current_group = [video_files[i]]
            
            # 与后续视频比较
            j = i + 1
            while j < len(video_files):
                self.check_and_reload_if_needed()
                
                # 提取两个视频的中间帧
                frame1_path = str(self.temp_dir / "compare_frame1.jpg")
                frame2_path = str(self.temp_dir / "compare_frame2.jpg")
                
                self.analyzer.extract_middle_frame(
                    str(current_group[-1]),
                    frame1_path,
                    compress=True,
                    ratio=self.compression_ratio
                )
                
                self.analyzer.extract_middle_frame(
                    str(video_files[j]),
                    frame2_path,
                    compress=True,
                    ratio=self.compression_ratio
                )
                
                # 判断主体是否相同
                is_same = self.analyzer.compare_subjects(frame1_path, frame2_path)
                self.context_count += 1
                
                # 清理临时帧
                try:
                    os.remove(frame1_path)
                    os.remove(frame2_path)
                except:
                    pass
                
                if is_same:
                    current_group.append(video_files[j])
                    j += 1
                else:
                    break
            
            # 处理当前组
            if len(current_group) == 1:
                # 单个视频
                source_video = current_group[0]
                
                if output_dir != folder:
                    output_file = output_dir / source_video.name
                    try:
                        import shutil
                        shutil.copy2(str(source_video), str(output_file))
                        source_names = [source_video.name]
                        results.append((str(output_file), source_names))
                    except Exception as e:
                        print(f"错误: 复制文件失败 - {source_video.name}: {e}")
                        source_names = [source_video.name]
                        results.append((str(source_video), source_names))
                else:
                    source_names = [source_video.name]
                    results.append((str(source_video), source_names))
            else:
                # 合并多个视频
                video_infos = []
                for video in current_group:
                    info = self.parse_filename(video.stem)
                    video_infos.append(info)
                
                # 生成合并后的文件名
                merged_name = self.generate_merged_name(video_infos)
                output_ext = current_group[0].suffix
                output_file = output_dir / f"{merged_name}{output_ext}"
                print(f'当前合并：{output_ext}')
                # 合并视频
                success = self.merge_videos(current_group, str(output_file))
                
                if success:
                    source_names = [v.name for v in current_group]
                    results.append((str(output_file), source_names))
                    print(source_names)
                    # 根据keep_original参数决定是否删除源视频
                    if not keep_original:
                        for video in current_group:
                            try:
                                os.remove(video)
                            except:
                                pass
                else:
                    print(f"错误: 合并失败 - {[v.name for v in current_group]}")
            
            i = j
        
        self.clear_cache()
        
        print(f"完成: 共生成 {len(results)} 个视频")
        
        return results


def main():
    """使用示例"""
    input_folder = r"D:\3_11CodeProject\Cut_DetectScene\videos\cuts"
    output_folder = r"D:\3_11CodeProject\Cut_DetectScene\videos\cuts"
    
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹不存在")
        return
    
    merger = VideoSceneMerger(
        model_path='hf-int4',
        compression_ratio=0.5,
        max_context_count=20
    )
    
    results = merger.merge_same_subject_videos(
        folder_path=input_folder,
        output_folder=output_folder,
        keep_original=True
    )


if __name__ == '__main__':
    main()