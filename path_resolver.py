# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
"""
统一路径解析器
整合多种路径获取方式，提供项目根目录和智能路径解析
"""

import os
from pathlib import Path
from typing import Union, Optional

# 尝试导入winshell（用于Windows快捷方式）
try:
    import winshell
    HAS_WINSHELL = True
except ImportError:
    HAS_WINSHELL = False


class PathResolver:
    """
    统一路径解析器
    
    功能：
    1. 获取项目根目录（当前脚本所在目录）
    2. 智能解析路径（支持：绝对路径、相对路径、用户目录~、Windows快捷方式.lnk）
    3. 向后兼容原有的resolve_path和_resolve_path逻辑
    """
    
    def __init__(self, base_file: Optional[str] = None):
        """
        初始化路径解析器
        
        Args:
            base_file: 基准文件路径（通常传入__file__），None则使用本文件路径
        """
        if base_file is None:
            base_file = __file__
        
        # 项目根目录（基准文件所在目录的绝对路径）
        self.project_root = Path(base_file).parent.absolute()
    
    def get_project_root(self) -> Path:
        """
        获取项目根目录（绝对路径）
        
        Returns:
            Path对象，项目根目录的绝对路径
        """
        return self.project_root
    
    def get_project_root_str(self) -> str:
        """
        获取项目根目录（字符串格式）
        
        Returns:
            字符串，项目根目录的绝对路径
        """
        return str(self.project_root)
    
    def resolve(
        self, 
        path: Union[str, Path], 
        base_dir: Optional[Union[str, Path]] = None,
        expand_user: bool = True,
        resolve_shortcut: bool = True
    ) -> Path:
        """
        智能路径解析（返回Path对象）
        
        功能：
        1. 去除引号
        2. 处理Windows快捷方式（.lnk）
        3. 展开用户目录（~）
        4. 绝对路径 → 直接使用
        5. 相对路径 → 基于base_dir拼接（默认使用project_root）
        
        Args:
            path: 待解析的路径
            base_dir: 基准目录（相对路径的基准），None则使用project_root
            expand_user: 是否展开用户目录（~）
            resolve_shortcut: 是否解析Windows快捷方式（.lnk）
        
        Returns:
            Path对象，解析后的绝对路径
        """
        if base_dir is None:
            base_dir = self.project_root
        else:
            base_dir = Path(base_dir)
        
        # 1. 转换为字符串并去除引号
        path_str = str(path).strip().strip('"').strip("'")
        
        # 2. 处理Windows快捷方式
        if resolve_shortcut and HAS_WINSHELL and path_str.lower().endswith('.lnk'):
            try:
                shortcut = winshell.shortcut(path_str)
                path_str = shortcut.path
            except (OSError, IOError, Exception):
                pass  # 解析失败，使用原路径
        
        # 3. 转换为Path对象
        path_obj = Path(path_str)
        
        # 4. 展开用户目录
        if expand_user:
            path_obj = path_obj.expanduser()
        
        # 5. 判断绝对/相对路径
        if path_obj.is_absolute():
            return path_obj.resolve()
        else:
            return (base_dir / path_obj).resolve()
    
    def resolve_str(
        self, 
        path: Union[str, Path], 
        base_dir: Optional[Union[str, Path]] = None,
        expand_user: bool = True,
        resolve_shortcut: bool = True
    ) -> str:
        """
        智能路径解析（返回字符串）
        
        参数同resolve()方法
        
        Returns:
            字符串，解析后的绝对路径
        """
        resolved = self.resolve(path, base_dir, expand_user, resolve_shortcut)
        return str(resolved)
    
    def resolve_normpath(
        self, 
        path: Union[str, Path], 
        base_dir: Optional[Union[str, Path]] = None,
        expand_user: bool = True,
        resolve_shortcut: bool = True
    ) -> str:
        """
        智能路径解析（返回规范化的字符串路径，兼容os.path风格）
        
        参数同resolve()方法
        
        Returns:
            字符串，规范化的绝对路径（使用os.path.normpath）
        """
        resolved = self.resolve(path, base_dir, expand_user, resolve_shortcut)
        return os.path.normpath(str(resolved))
    
    def join(self, *paths: Union[str, Path]) -> Path:
        """
        路径拼接（基于项目根目录）
        
        Args:
            *paths: 要拼接的路径片段
        
        Returns:
            Path对象，拼接后的路径
        """
        result = self.project_root
        for p in paths:
            result = result / p
        return result
    
    def join_str(self, *paths: Union[str, Path]) -> str:
        """
        路径拼接（返回字符串）
        
        Args:
            *paths: 要拼接的路径片段
        
        Returns:
            字符串，拼接后的路径
        """
        return str(self.join(*paths))


# ============================================================
# 便捷函数（全局单例）
# ============================================================

# 全局单例（基于当前文件）
_global_resolver = PathResolver()


def get_project_root() -> Path:
    """
    便捷函数：获取项目根目录（Path对象）
    
    Returns:
        Path对象，项目根目录
    """
    return _global_resolver.get_project_root()


def get_project_root_str() -> str:
    """
    便捷函数：获取项目根目录（字符串）
    
    Returns:
        字符串，项目根目录
    """
    return _global_resolver.get_project_root_str()


def resolve_path(
    path: Union[str, Path], 
    base_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    便捷函数：解析路径（返回字符串，兼容旧版video_utils.resolve_path）
    
    Args:
        path: 待解析路径
        base_dir: 基准目录（None则使用项目根目录）
    
    Returns:
        字符串，解析后的绝对路径
    """
    return _global_resolver.resolve_normpath(path, base_dir)


def resolve_path_obj(
    path: Union[str, Path], 
    base_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    便捷函数：解析路径（返回Path对象，兼容Video_Scene_Analyzer._resolve_path）
    
    Args:
        path: 待解析路径
        base_dir: 基准目录（None则使用项目根目录）
    
    Returns:
        Path对象，解析后的绝对路径
    """
    return _global_resolver.resolve(path, base_dir)


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 示例1: 创建解析器
    resolver = PathResolver(__file__)
    
    print("=" * 60)
    print("项目根目录:")
    print(f"  Path对象: {resolver.get_project_root()}")
    print(f"  字符串:   {resolver.get_project_root_str()}")
    
    print("\n" + "=" * 60)
    print("路径解析示例:")
    
    # 示例2: 相对路径
    print(f"\n相对路径 'config.json':")
    print(f"  -> {resolver.resolve('config.json')}")
    
    # 示例3: 绝对路径
    print(f"\n绝对路径 'C:/temp/test.txt':")
    print(f"  -> {resolver.resolve('C:/temp/test.txt')}")
    
    # 示例4: 用户目录
    print(f"\n用户目录 '~/Desktop/test.txt':")
    print(f"  -> {resolver.resolve('~/Desktop/test.txt')}")
    
    # 示例5: 带引号的路径
    quoted_path = '"models/test"'
    print(f"\n带引号 '{quoted_path}':")
    print(f"  -> {resolver.resolve(quoted_path)}")
    
    # 示例6: 路径拼接
    print(f"\n路径拼接 join('models', 'test', 'model.bin'):")
    print(f"  -> {resolver.join('models', 'test', 'model.bin')}")
    
    print("\n" + "=" * 60)
    print("便捷函数示例:")
    
    # 示例7: 使用便捷函数
    print(f"\nget_project_root():")
    print(f"  -> {get_project_root()}")
    
    print(f"\nresolve_path('config.json'):")
    print(f"  -> {resolve_path('config.json')}")
    
    print(f"\nresolve_path_obj('~/Desktop/test.txt'):")
    print(f"  -> {resolve_path_obj('~/Desktop/test.txt')}")
    
    print("\n" + "=" * 60)