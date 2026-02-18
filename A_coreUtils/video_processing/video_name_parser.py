# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
"""
视频文件名解析器 - 使用 guessit + 正则回退
v2.0: 简化重构版
"""

import os
import sys
import re
from pathlib import Path
import datetime

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

# 尝试导入 guessit
try:
    from guessit import guessit
    HAS_GUESSIT = True
except ImportError:
    HAS_GUESSIT = False
    print("警告: guessit 模块未安装，将使用正则解析")


class VideoNameParser:
    """
    视频文件名分析器 - 使用 guessit + 正则回退
    
    输出格式: 剧名_剧集
    例如: "MyShow_S01E01", "AnimeTitle_OP", "MovieName_PV202501011234"
    """
    
    # 需要排除的数字（分辨率、年份等）
    EXCLUDED_NUMBERS = {480, 720, 1080, 2160, 4320}
    
    def __init__(self):
        pass
    
    def _parse_episode_info(self, filename: str) -> str:
        """
        解析剧集信息
        
        优先级：
        1. guessit 解析
        2. 正则回退链：
           - SxxExx / Season / Episode
           - OP/ED/NCOP/NCED
           - OVA/OAD/SP/Special
           - Movie/Film
           - PV/MV/Trailer/CM
           - 独立数字
           - 保底 PV+时间戳
        """
        # ========== 第一优先级：guessit ==========
        if HAS_GUESSIT:
            try:
                guess = guessit(filename)
                season = guess.get('season')
                episode = guess.get('episode')
                
                # 处理列表情况
                if isinstance(season, list):
                    season = season[0]
                if isinstance(episode, list):
                    episode = episode[0]
                
                if season is not None and episode is not None:
                    return f"S{int(season):02d}E{int(episode):02d}"
                elif episode is not None:
                    # 支持小数集数
                    if isinstance(episode, float):
                        return f"E{episode}"
                    return f"E{int(episode):02d}"
                
                # 检查 guessit 的 other 字段是否有 OP/ED
                other = guess.get('other')
                if other:
                    other_str = str(other).upper() if not isinstance(other, list) else ' '.join(str(x).upper() for x in other)
                    oped_match = re.search(r'(NC)?(OP|ED)(\d*)', other_str)
                    if oped_match:
                        return f"{oped_match.group(1) or ''}{oped_match.group(2)}{oped_match.group(3) or ''}"
            except Exception:
                pass
        
        # ========== 第二优先级：正则回退链 ==========
        
        # ① 标准剧集：SxxExx / Season / Episode
        # 支持 S1, S01, Season 1, Season_1, season-1
        season_match = re.search(r'(?:S|Season)[\s_\-]*(\d+)', filename, re.IGNORECASE)
        # 支持 E1, E01, E_01, EP1, Episode 1, Episode_1
        ep_match = re.search(r'(?:E|EP|Episode)[\s_\-]*(\d+)', filename, re.IGNORECASE)
        
        if season_match and ep_match:
            return f"S{int(season_match.group(1)):02d}E{int(ep_match.group(1)):02d}"
        if ep_match:
            return f"E{int(ep_match.group(1)):02d}"
        
        # ② OP/ED/NCOP/NCED
        oped = re.search(r'\b(NC)?(OP|ED)(\d*)\b', filename, re.IGNORECASE)
        if oped:
            nc = (oped.group(1) or '').upper()
            type_ = oped.group(2).upper()
            num = oped.group(3) or ''
            return f"{nc}{type_}{num}"
        
        # Opening/Ending 全拼
        if re.search(r'\bOpening\b', filename, re.IGNORECASE):
            return "OP"
        if re.search(r'\bEnding\b', filename, re.IGNORECASE):
            return "ED"
        
        # ③ OVA/OAD/SP/Special
        special = re.search(r'\b(OVA|OAD|SP|Special)[\s_\-]*(\d*)\b', filename, re.IGNORECASE)
        if special:
            type_ = "SP" if special.group(1).upper() == "SPECIAL" else special.group(1).upper()
            num = special.group(2) or ''
            return f"{type_}{num}"
        
        # ④ Movie/Film
        movie = re.search(r'\b(Movie|Film)[\s_\-]*(\d*)\b', filename, re.IGNORECASE)
        if movie:
            num = movie.group(2) or ''
            return f"Movie{num}"
        
        # ⑤ PV/MV/Trailer/Preview/CM（所有 PV/MV 类型都加时间戳）
        pv = re.search(r'\b(PV|MV|Trailer|Preview|CM)(\d*)\b', filename, re.IGNORECASE)
        if pv:
            type_ = pv.group(1).upper()
            # 统一 Trailer/Preview/CM 为 PV
            if type_ in ["TRAILER", "PREVIEW", "CM"]:
                type_ = "PV"
            # PV/MV 类型统一加时间戳（无论有无数字）
            if type_ in ["PV", "MV"]:
                return f"{type_}{datetime.datetime.now().strftime('%Y%m%d%H%M')}"
        
        # ⑥ 独立数字（如 "- 05", "[01]"）
        # 匹配前面有分隔符的1-3位数字
        standalone = re.search(r'[\s\-_\[]+(\d{1,3})(?:[\s\-_\.\]\)]|$)', filename)
        if standalone:
            num = int(standalone.group(1))
            # 排除分辨率和年份
            if num not in self.EXCLUDED_NUMBERS and num < 500:
                return f"E{num:02d}"
        
        # ⑦ 保底：PV+时间戳
        return f"PV{datetime.datetime.now().strftime('%Y%m%d%H%M')}"
    
    def _parse_title(self, filename: str, episode_str: str) -> str:
        """
        解析剧名
        
        优先级：
        1. guessit.title
        2. 清理文件名
        """
        title = None
        
        # ========== 第一优先级：guessit ==========
        if HAS_GUESSIT:
            try:
                guess = guessit(filename)
                guessit_title = guess.get('title')
                if guessit_title and isinstance(guessit_title, str):
                    title = guessit_title
            except Exception:
                pass
        
        # ========== 第二优先级：清理文件名 ==========
        if not title:
            title = filename
        
        # 清理剧名
        # 移除方括号内容（如 [Judas], [1080p]）
        title = re.sub(r'\[.*?\]', '', title).strip()
        
        # 移除剧集信息（使用单词边界 \b 确保完整匹配）
        # 移除 SxxExx 格式（包括版本号）
        title = re.sub(r'[\s\-_]*\bS\d+E\d+(?:v\d+)?\b', '', title, flags=re.IGNORECASE).strip()
        # 移除单独的 Sxx 或 Exx（包括 E_01 这种格式）
        title = re.sub(r'[\s\-_]*\bS[\s_\-]*\d+\b', '', title, flags=re.IGNORECASE).strip()
        title = re.sub(r'[\s\-_]*\bE[\s_\-]*\d+\b', '', title, flags=re.IGNORECASE).strip()
        title = re.sub(r'[\s\-_]*\bEP[\s_\-]*\d+\b', '', title, flags=re.IGNORECASE).strip()
        # 移除 Season/Episode 全拼（包括 season_1 这种格式）
        title = re.sub(r'[\s\-_]*\bSeason[\s_\-]*\d+\b', '', title, flags=re.IGNORECASE).strip()
        title = re.sub(r'[\s\-_]*\bEpisode[\s_\-]*\d+\b', '', title, flags=re.IGNORECASE).strip()
        # 移除 OP/ED/NCOP/NCED（使用单词边界）
        title = re.sub(r'[\s\-_]*\b(?:NC)?(?:OP|ED)\d*\b', '', title, flags=re.IGNORECASE).strip()
        # 移除 Opening/Ending（使用单词边界）
        title = re.sub(r'[\s\-_]*\bOpening\b[\s\-_]*\bTheme\b', '', title, flags=re.IGNORECASE).strip()
        title = re.sub(r'[\s\-_]*\bEnding\b[\s\-_]*\bTheme\b', '', title, flags=re.IGNORECASE).strip()
        title = re.sub(r'[\s\-_]*\bOpening\b', '', title, flags=re.IGNORECASE).strip()
        title = re.sub(r'[\s\-_]*\bEnding\b', '', title, flags=re.IGNORECASE).strip()
        # 移除 OVA/OAD/SP/Special（使用单词边界）
        title = re.sub(r'[\s\-_]*\bSpecial[\s_\-]*\d*\b', '', title, flags=re.IGNORECASE).strip()
        title = re.sub(r'[\s\-_]*\b(?:OVA|OAD|SP)[\s_\-]*\d*\b', '', title, flags=re.IGNORECASE).strip()
        # 移除 Film（使用单词边界，但保留 Movie 作为剧名的一部分）
        title = re.sub(r'[\s\-_]*\bFilm[\s_\-]*\d*\b', '', title, flags=re.IGNORECASE).strip()
        # 移除 PV/MV/Trailer/CM（使用单词边界）
        title = re.sub(r'[\s\-_]*\b(?:PV|MV|Trailer|Preview|CM)\d*\b', '', title, flags=re.IGNORECASE).strip()
        # 移除独立数字（前面有分隔符的1-3位数字）
        title = re.sub(r'[\s\-_]+\d{1,3}(?=[\s\-_\.]|$)', '', title).strip()
        
        # 移除编码信息
        title = re.sub(r'\b(?:x26[45]|HEVC|H\.?264|AVC|10bit|Hi10P)\b', '', title, flags=re.IGNORECASE).strip()
        title = re.sub(r'\b(?:FLAC|AAC|AC3|DTS|MP3|Opus)\b', '', title, flags=re.IGNORECASE).strip()
        title = re.sub(r'\b(?:1080p|720p|480p|2160p|4K|BD|BluRay|WEB-DL|HDR)\b', '', title, flags=re.IGNORECASE).strip()
        
        # 移除年份（如 (2024)）
        title = re.sub(r'\s*[\(\[]\d{4}[\)\]]', '', title).strip()
        
        # 清理多余的分隔符
        title = re.sub(r'[\s\-_]+', ' ', title).strip()
        title = re.sub(r'^[\s\-_]+|[\s\-_]+$', '', title).strip()
        
        # 如果剧名为空，使用原文件名
        if not title:
            title = filename
        
        return title
    
    def parse_filename(self, video_path: str, title: str = None) -> str:
        """
        解析视频文件名
        
        参数:
            video_path: 视频文件路径
            title: 剧名（可选），如果提供则直接使用，否则从文件名解析
            
        返回:
            格式化的字符串 "剧名_剧集"
        """
        filename = os.path.basename(video_path)
        filename_without_ext = os.path.splitext(filename)[0]
        
        # 解析剧集信息
        episode_str = self._parse_episode_info(filename_without_ext)
        
        # 解析剧名
        if title:
            title_str = title
        else:
            title_str = self._parse_title(filename_without_ext, episode_str)
        
        # 组合输出
        return f"{title_str}_{episode_str}"
    
    def process_video(self, video_path: str, title: str = None) -> str:
        """
        处理单个视频文件
        
        参数:
            video_path: 视频文件路径
            title: 剧名（可选），如果提供则直接使用，否则从文件名解析
            
        返回:
            格式化的字符串 "剧名_剧集"
        """
        if not os.path.exists(video_path):
            return None
        
        return self.parse_filename(video_path, title=title)
    
    def process_folder(self, folder_path: str, title: str = None) -> dict:
        """
        处理文件夹中的所有视频
        
        参数:
            folder_path: 文件夹路径
            title: 剧名（可选），如果提供则所有视频使用同一剧名，否则分别解析
            
        返回:
            字典，键为视频文件路径，值为格式化的字符串
        """
        # 支持的视频格式
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.m4v', 
                           '.mpg', '.mpeg', '.ts', '.rmvb', '.3gp'}
        
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            return {}
        
        results = {}
        video_files = [f for f in folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in video_extensions]
        
        for video_file in video_files:
            parsed_name = self.parse_filename(str(video_file), title=title)
            if parsed_name:
                results[str(video_file)] = parsed_name
        
        return results


def main():
    """测试用例"""
    parser = VideoNameParser()
    
    test_files = [
        # 标准格式
        "[Judas] Anne Shirley - S01E01v2.mkv",
        "MyShow S1E1.mp4",
        "Show Season 1 E01.mp4",
        "Show season_1 E_01.mp4",
        "Show - 05.mp4",
        "[Group] Anime - 12 [1080p].mkv",
        
        # OP/ED
        "MyAnime NCOP2.mkv",
        "Anime Opening Theme.mp4",
        "Show ED.mp4",
        
        # 特殊类型
        "Anime OVA.mp4",
        "Anime OVA2.mp4",
        "Show Special 3.mp4",
        "Movie Film.mp4",
        
        # PV/MV
        "Show PV2.mp4",
        "Anime MV.mp4",
        "Show Trailer.mp4",
        
        # 边界情况
        "Random File.mp4",
    ]
    
    print("=" * 60)
    print("视频文件名解析测试")
    print("=" * 60)
    
    for test_file in test_files:
        result = parser.parse_filename(test_file)
        print(f"{test_file}")
        print(f"  -> {result}")
        print()


if __name__ == '__main__':
    main()
