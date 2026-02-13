# pipeline_app.py
# 视频处理流水线 - Flask Web界面版
# 支持通过网页可视化配置参数并运行流水线
# 顺序执行: indexer_app (视频索引) -> prompt_output_app (场景搜索)

import os
import time
import json
import multiprocessing as mp
import queue
import re
from urllib.parse import unquote

from flask import Flask, render_template, request, jsonify, Response

# 导入路径解析器
from path_resolver import PathResolver, resolve_path

# 创建路径解析器实例
_path_resolver = PathResolver(__file__)

# 获取当前文件所在目录
_SCRIPT_DIR = _path_resolver.get_project_root_str()

# ============================================================================
# Flask 应用初始化
# ============================================================================
app = Flask(__name__, template_folder='templates', static_folder='static')

# 模型目录路径
MODELS_DIR = _path_resolver.join_str('models')
CONFIG_PATH = _path_resolver.join_str('config.json')
PRESET_DIR = _path_resolver.join_str('preset')
PRESET_META_PATH = _path_resolver.join_str('preset', '_meta.json')
_PRESET_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# 全局运行状态
_running = False
_stop_requested = mp.Value('b', False)  # 跨进程共享的停止标志
_log_queue = queue.Queue()
_pipeline_process = None

# ============================================================================
# 导入索引模块
# ============================================================================
try:
    from A_coreUtils.embedding.embedding_model import create_video_index, batch_create_video_index
    _INDEXER_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] 无法导入 embedding_model 模块 - {e}")
    _INDEXER_AVAILABLE = False

# ============================================================================
# 导入搜索模块
# ============================================================================
try:
    from A_coreUtils.search.auto_scene_search import run_interactive_search
    _PROMPT_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] 无法导入 auto_scene_search 模块 - {e}")
    _PROMPT_SEARCH_AVAILABLE = False

try:
    from A_coreUtils.search.label_traverse_search import run_label_traverse_search
    _LABEL_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] 无法导入 label_traverse_search 模块 - {e}")
    _LABEL_SEARCH_AVAILABLE = False

try:
    from A_coreUtils.search.cloze_fill_search import run_cloze_fill_search
    _CLOZE_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] 无法导入 cloze_fill_search 模块 - {e}")
    _CLOZE_SEARCH_AVAILABLE = False


# ============================================================================
# 日志辅助函数
# ============================================================================
def log_message(message, level='info'):
    """添加日志消息到队列"""
    _log_queue.put({
        'type': 'log',
        'message': message,
        'level': level
    })
    print(f"[{level.upper()}] {message}")


def log_progress(percent):
    """更新进度"""
    _log_queue.put({
        'type': 'progress',
        'percent': percent
    })


def log_status(message):
    """更新状态"""
    _log_queue.put({
        'type': 'status',
        'message': message
    })


def log_complete(success, message):
    """标记完成"""
    _log_queue.put({
        'type': 'complete',
        'success': success,
        'message': message
    })


def _cleanup_pipeline_process():
    """清理已退出的子进程句柄"""
    global _pipeline_process, _running
    if _pipeline_process is not None and not _pipeline_process.is_alive():
        try:
            _pipeline_process.join(timeout=0.1)
        except Exception:
            pass
        _pipeline_process = None
        _running = False


def is_pipeline_running() -> bool:
    """是否有流水线子进程正在运行"""
    _cleanup_pipeline_process()
    return _pipeline_process is not None and _pipeline_process.is_alive()


def _run_pipeline_in_process(config, log_queue, stop_flag):
    """子进程入口：绑定日志队列和停止标志并执行流水线"""
    global _log_queue, _stop_requested
    _log_queue = log_queue
    _stop_requested = stop_flag
    run_pipeline_thread(config)


# ============================================================================
# 模型检测
# ============================================================================
def detect_model_type(model_path: str) -> str:
    """根据模型文件夹内容自动检测模型类型"""
    if not os.path.isdir(model_path):
        return 'auto'
    
    model_name_lower = os.path.basename(model_path).lower()
    
    # Reranker 仅用于重排，不应出现在视频向量化模型列表中
    if 'reranker' in model_name_lower:
        return 'reranker'
    
    # 检查 FG-CLIP2
    if 'fg-clip2' in model_name_lower or 'fg_clip2' in model_name_lower or 'fgclip2' in model_name_lower:
        return 'fgclip2'
    
    # 检查 Qwen
    if 'qwen' in model_name_lower:
        return 'qwen_embed'
    
    # 检查配置文件 (open_clip 格式)
    config_path = os.path.join(model_path, 'open_clip_config.json')
    if os.path.exists(config_path):
        return 'clip'
    
    return 'auto'


def _load_project_config():
    """读取项目 config.json"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_project_config(config):
    """保存项目 config.json"""
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def _resolve_diskcache_dir(custom_dir: str = None) -> str:
    """统一解析搜索结果 LMDB 缓存目录；未配置时返回 None 以使用各模块默认目录。"""
    if custom_dir is None:
        return None
    custom_dir = str(custom_dir).strip()
    if not custom_dir:
        return None
    cache_dir = custom_dir
    cache_dir = resolve_path(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _ensure_preset_dir():
    os.makedirs(PRESET_DIR, exist_ok=True)


def _sanitize_preset_name(name: str) -> str:
    if not isinstance(name, str):
        return ''
    preset_name = _PRESET_INVALID_CHARS.sub('_', name.strip())
    preset_name = preset_name.rstrip('. ')
    return preset_name[:120]


def _get_preset_path(name: str):
    preset_name = _sanitize_preset_name(name)
    if not preset_name:
        raise ValueError("预设名称不能为空")
    return preset_name, os.path.join(PRESET_DIR, f"{preset_name}.json")


def _load_preset_meta() -> dict:
    _ensure_preset_dir()
    if not os.path.exists(PRESET_META_PATH):
        return {}
    try:
        with open(PRESET_META_PATH, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _save_preset_meta(meta: dict):
    _ensure_preset_dir()
    safe_meta = meta if isinstance(meta, dict) else {}
    with open(PRESET_META_PATH, 'w', encoding='utf-8') as f:
        json.dump(safe_meta, f, ensure_ascii=False, indent=4)


def _list_preset_names() -> list:
    _ensure_preset_dir()
    names = []
    for file_name in os.listdir(PRESET_DIR):
        if not file_name.lower().endswith('.json'):
            continue
        if file_name == '_meta.json':
            continue
        names.append(os.path.splitext(file_name)[0])
    names.sort(key=str.lower)
    return names


# ============================================================================
# 流水线核心函数
# ============================================================================
def run_indexer_with_config(config):
    """
    运行视频索引，使用传入的配置
    """
    
    if not _INDEXER_AVAILABLE:
        log_message("索引模块不可用，请检查 embedding_model 是否正确安装", 'error')
        return False
    
    log_message("=" * 50, 'info')
    log_message("📹 步骤1: 视频索引", 'info')
    log_message("=" * 50, 'info')
    
    idx_config = config.get('indexer', {})
    
    # 解析路径
    input_dir = resolve_path(idx_config.get('input_directory', ''))
    output_dir = resolve_path(idx_config.get('output_directory', _path_resolver.join_str('indexes')))
    
    # 构建参数
    parameters = {
        'model_name': idx_config.get('model_name', 'qihoo360_fg-clip2-base'),
        'model_type': idx_config.get('model_type', 'fgclip2'),
        'truncate_dim': int(idx_config['truncate_dim']) if idx_config.get('truncate_dim') else None,
        'output_resolution': idx_config.get('output_resolution', '256'),
        'batch_size': idx_config.get('batch_size', 128),
        'workers': idx_config.get('workers', None),
        'io_workers': idx_config.get('io_workers', 8),
        'cosine_similarity_threshold': idx_config.get('cosine_similarity_threshold'),
        'brightness_threshold': idx_config.get('brightness_threshold', 32),
        'black_pixel_ratio': idx_config.get('black_pixel_ratio', 98),
        'sample_interval': idx_config.get('sample_interval', 5),
        'min_scene_length': idx_config.get('min_scene_length', 7),
        'localmax_order': idx_config.get('localmax_order', 2),
        'resume_processing': idx_config.get('resume_processing', True),
        'use_fp16': idx_config.get('use_fp16', True),
    }
    
    log_message(f"输入目录: {input_dir}", 'info')
    log_message(f"输出目录: {output_dir}", 'info')
    log_message(f"模型: {parameters['model_name']}", 'info')
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        log_message(f"错误: 输入目录不存在 - {input_dir}", 'error')
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查停止请求
    if _stop_requested.value:
        log_message("收到停止请求，终止索引", 'warning')
        return False
    
    time_start = time.time()
    
    # 判断是单文件夹还是批量处理
    supported_ext = ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.flv')
    has_videos = any(f.lower().endswith(supported_ext) for f in os.listdir(input_dir) 
                     if os.path.isfile(os.path.join(input_dir, f)))
    has_subfolders = any(os.path.isdir(os.path.join(input_dir, f)) for f in os.listdir(input_dir))
    
    try:
        if has_videos:
            log_message("📂 检测到视频文件，直接处理当前文件夹", 'info')
            log_progress(10)
            result = create_video_index(
                [input_dir],
                output_dir,
                **parameters
            )
            success = result is not None
        elif has_subfolders:
            log_message("📁 检测到子文件夹，批量处理模式", 'info')
            log_progress(10)
            success = batch_create_video_index(
                input_dir,
                output_dir,
                **parameters
            )
        else:
            log_message("错误: 输入目录中没有视频文件或子文件夹", 'error')
            return False
        
        time_end = time.time()
        log_message(f"索引耗时: {time_end - time_start:.2f} 秒", 'success')
        log_progress(50)
        
        return success
        
    except Exception as e:
        log_message(f"索引过程出错: {str(e)}", 'error')
        import traceback
        log_message(traceback.format_exc(), 'error')
        return False


def run_prompt_search_with_config(config):
    """
    运行 Prompt 组合搜索，使用传入的配置
    """
    
    if not _PROMPT_SEARCH_AVAILABLE:
        log_message("Prompt搜索模块不可用，请检查 auto_scene_search 是否正确安装", 'error')
        return False
    
    log_message("=" * 50, 'info')
    log_message("🔍 步骤2: Prompt 组合搜索", 'info')
    log_message("=" * 50, 'info')
    
    ps_config = config.get('prompt_search', {})
    
    # 检查停止请求
    if _stop_requested.value:
        log_message("收到停止请求，终止搜索", 'warning')
        return False
    
    time_start = time.time()
    log_progress(55)

    prompt_search_mode = int(ps_config.get('search_mode', 0))
    pkl_batch_size = ps_config.get('pkl_batch_size', 5)
    diskcache_dir = _resolve_diskcache_dir(ps_config.get('diskcache_dir'))
    
    try:
        search_results = run_interactive_search(
            # 路径配置
            index_directory=resolve_path(ps_config.get('index_directory', _path_resolver.join_str('indexes'))),
            output_directory=resolve_path(ps_config.get('output_directory', _path_resolver.join_str('output'))),
            # 基础配置
            use_fp16=ps_config.get('use_fp16', True),
            # Reranker 配置
            use_reranker=ps_config.get('use_reranker', False),
            rerank_top_k=ps_config.get('rerank_top_k', 50),
            rerank_batch_size=ps_config.get('rerank_batch_size', 4),
            reranker_output_resolution=str(ps_config.get('reranker_output_resolution', '384')),
            candidate_batch_size=ps_config.get('candidate_batch_size', 1000),
            # 组合限制
            # 保存配置
            # 视频导出配置
            video_output_directory=ps_config.get('video_output_directory'),
            video_copy_mode=ps_config.get('video_copy_mode', True),
            start_frame_offset=int(ps_config['start_frame_offset']) if ps_config.get('start_frame_offset') else None,
            end_frame_offset=int(ps_config['end_frame_offset']) if ps_config.get('end_frame_offset') else None,
            # 优化模式配置
            prompt_search_batch_size=ps_config.get('prompt_search_batch_size', 1024),
            feature_fp16=ps_config.get('feature_fp16', False),
            pkl_batch_size=pkl_batch_size,
            # 磁盘缓存配置
            use_diskcache=ps_config.get('use_diskcache', True),
            diskcache_dir=diskcache_dir,
            # Prompt模板配置
            prompt_template=ps_config.get('prompt_template'),
            video_name_format=ps_config.get('video_name_format'),
            # 调试配置
            debug_similarity=ps_config.get('debug_similarity', False),
            # 搜索模式配置
            search_mode=prompt_search_mode,
            top_k=ps_config.get('top_k', 50),
            # Prompt向量缓存配置
            prompt_cache_batch_size=ps_config.get('prompt_cache_batch_size', 512),
            # 中文模式配置
            use_chinese=ps_config.get('use_chinese', False),
            # 线程配置
            pkl_load_workers=ps_config.get('pkl_load_workers', 4),
            lmdb_write_batch_size=ps_config.get('lmdb_write_batch_size', 1000),
            # 向量去重配置
            vector_dedup_threshold=ps_config.get('vector_dedup_threshold'),
            # 相邻片段合并配置
            adjacent_merge_frames=ps_config.get('adjacent_merge_frames'),
        )
        
        if not isinstance(search_results, dict):
            log_message("Prompt搜索返回异常结果类型，判定为失败", 'error')
            return False

        time_end = time.time()
        log_message(f"搜索耗时: {time_end - time_start:.2f} 秒", 'success')
        log_progress(100)
        
        return True
        
    except Exception as e:
        log_message(f"搜索过程出错: {str(e)}", 'error')
        import traceback
        log_message(traceback.format_exc(), 'error')
        return False


def run_label_search_with_config(config):
    """
    运行遍历模式标签匹配搜索，使用传入的配置
    """
    
    if not _LABEL_SEARCH_AVAILABLE:
        log_message("标签搜索模块不可用，请检查 label_traverse_search 是否正确安装", 'error')
        return False
    
    log_message("=" * 50, 'info')
    log_message("🏷️ 步骤2: 遍历模式标签匹配搜索", 'info')
    log_message("=" * 50, 'info')
    
    ls_config = config.get('label_search', {})
    
    # 检查停止请求
    if _stop_requested.value:
        log_message("收到停止请求，终止搜索", 'warning')
        return False
    
    time_start = time.time()
    log_progress(55)
    diskcache_dir = _resolve_diskcache_dir(ls_config.get('diskcache_dir'))
    
    try:
        results = run_label_traverse_search(
            # 路径配置
            index_directory=resolve_path(ls_config.get('index_directory', _path_resolver.join_str('indexes'))),
            output_directory=resolve_path(ls_config.get('output_directory', _path_resolver.join_str('output'))),
            # 视频导出配置
            video_output_directory=ls_config.get('video_output_directory'),
            video_copy_mode=ls_config.get('video_copy_mode', False),
            video_name_format=ls_config.get('video_name_format'),
            debug_similarity=ls_config.get('debug_similarity', False),
            # 优化参数
            prompt_search_batch_size=ls_config.get('prompt_search_batch_size', 1024),
            pkl_batch_size=ls_config.get('pkl_batch_size'),
            # 搜索模式参数
            search_mode=ls_config.get('search_mode', 0),
            top_k=ls_config.get('top_k', 50),
            scene_chunk_size=ls_config.get('scene_chunk_size', 1000),
            # 视频帧偏移参数
            start_frame_offset=int(ls_config['start_frame_offset']) if ls_config.get('start_frame_offset') else None,
            end_frame_offset=int(ls_config['end_frame_offset']) if ls_config.get('end_frame_offset') else None,
            # LMDB 缓存参数
            use_diskcache=ls_config.get('use_diskcache', True),
            diskcache_dir=diskcache_dir,
            # 中文标签模式
            use_chinese_labels=ls_config.get('use_chinese_labels', False),
            # 线程配置参数
            pkl_load_workers=ls_config.get('pkl_load_workers', 4),
            lmdb_write_batch_size=ls_config.get('lmdb_write_batch_size', 1000),
            # 标签缓存批处理大小
            label_cache_batch_size=ls_config.get('label_cache_batch_size', 512),
            # 向量去重参数
            vector_dedup_threshold=ls_config.get('vector_dedup_threshold'),
            # 相邻片段合并参数
            adjacent_merge_frames=ls_config.get('adjacent_merge_frames'),
            # 计算与特征精度
            use_fp16=ls_config.get('use_fp16', True),
            feature_fp16=ls_config.get('feature_fp16', None),
        )
        
        time_end = time.time()
        log_message(f"遍历模式完成，共 {len(results) if results else 0} 个有效场景", 'success')
        log_message(f"搜索耗时: {time_end - time_start:.2f} 秒", 'success')
        log_progress(100)
        
        return True
        
    except Exception as e:
        log_message(f"搜索过程出错: {str(e)}", 'error')
        import traceback
        log_message(traceback.format_exc(), 'error')
        return False


def run_cloze_search_with_config(config):
    """
    运行选词填空模式搜索，使用传入的配置
    """
    
    if not _CLOZE_SEARCH_AVAILABLE:
        log_message("选词填空搜索模块不可用，请检查 cloze_fill_search 是否正确安装", 'error')
        return False
    
    log_message("=" * 50, 'info')
    log_message("📝 步骤2: 选词填空模式搜索", 'info')
    log_message("=" * 50, 'info')
    
    cs_config = config.get('cloze_search', {})
    
    # 检查停止请求
    if _stop_requested.value:
        log_message("收到停止请求，终止搜索", 'warning')
        return False
    
    time_start = time.time()
    log_progress(55)
    diskcache_dir = _resolve_diskcache_dir(cs_config.get('diskcache_dir'))
    
    try:
        results = run_cloze_fill_search(
            # 路径配置
            index_directory=resolve_path(cs_config.get('index_directory', _path_resolver.join_str('indexes'))),
            output_directory=resolve_path(cs_config.get('output_directory', _path_resolver.join_str('output'))),
            # 视频导出配置
            video_output_directory=cs_config.get('video_output_directory'),
            video_copy_mode=cs_config.get('video_copy_mode', False),
            video_name_format=cs_config.get('video_name_format'),
            debug_similarity=cs_config.get('debug_similarity', False),
            # 优化参数
            prompt_search_batch_size=cs_config.get('prompt_search_batch_size', 1024),
            pkl_batch_size=cs_config.get('pkl_batch_size'),
            # 搜索模式参数
            search_mode=cs_config.get('search_mode', 0),
            top_k=cs_config.get('top_k', 50),
            # 视频帧偏移参数
            start_frame_offset=int(cs_config['start_frame_offset']) if cs_config.get('start_frame_offset') else None,
            end_frame_offset=int(cs_config['end_frame_offset']) if cs_config.get('end_frame_offset') else None,
            # LMDB 缓存参数
            use_diskcache=cs_config.get('use_diskcache', True),
            diskcache_dir=diskcache_dir,
            # 中英文模式
            use_chinese=cs_config.get('use_chinese', False),
            # Reranker 参数
            use_reranker=cs_config.get('use_reranker', False),
            rerank_top_k=cs_config.get('rerank_top_k', 7),
            rerank_batch_size=cs_config.get('rerank_batch_size', 7),
            reranker_output_resolution=str(cs_config.get('reranker_output_resolution', '448')),
            candidate_batch_size=cs_config.get('candidate_batch_size', 1000),
            # 缓存批处理大小
            prompt_cache_batch_size=cs_config.get('prompt_cache_batch_size', 512),
            # 线程配置参数
            pkl_load_workers=cs_config.get('pkl_load_workers', 4),
            lmdb_write_batch_size=cs_config.get('lmdb_write_batch_size', 1000),
            # 向量去重参数
            vector_dedup_threshold=cs_config.get('vector_dedup_threshold'),
            # 相邻片段合并参数
            adjacent_merge_frames=cs_config.get('adjacent_merge_frames'),
            # 计算与特征精度
            use_fp16=cs_config.get('use_fp16', True),
            feature_fp16=cs_config.get('feature_fp16', None),
        )
        
        time_end = time.time()
        log_message(f"选词填空模式完成，共 {len(results) if results else 0} 个有效场景", 'success')
        log_message(f"搜索耗时: {time_end - time_start:.2f} 秒", 'success')
        log_progress(100)
        
        return True
        
    except Exception as e:
        log_message(f"搜索过程出错: {str(e)}", 'error')
        import traceback
        log_message(traceback.format_exc(), 'error')
        return False


def run_pipeline_thread(config):
    """
    在单独线程中运行流水线
    """
    global _running
    
    _running = True
    _stop_requested.value = False
    
    total_time_start = time.time()
    
    try:
        log_message("🎬 视频处理流水线启动", 'info')
        log_message(f"索引模块: {'✓ 可用' if _INDEXER_AVAILABLE else '✗ 不可用'}", 'info')
        log_message(f"Prompt搜索模块: {'✓ 可用' if _PROMPT_SEARCH_AVAILABLE else '✗ 不可用'}", 'info')
        log_message(f"标签搜索模块: {'✓ 可用' if _LABEL_SEARCH_AVAILABLE else '✗ 不可用'}", 'info')
        log_message(f"选词填空模块: {'✓ 可用' if _CLOZE_SEARCH_AVAILABLE else '✗ 不可用'}", 'info')
        
        run_indexer = config.get('run_indexer', True)
        run_search = config.get('run_search', True)
        search_entry_mode = config.get('search_entry_mode', 'prompt')
        
        log_progress(5)
        
        # 步骤1: 视频索引
        if run_indexer:
            log_status("正在运行索引...")
            success = run_indexer_with_config(config)
            if not success:
                log_complete(False, "索引步骤失败，流水线终止")
                return
        else:
            log_message("⏭️ 跳过索引步骤", 'info')
            log_progress(50)
        
        # 检查停止请求
        if _stop_requested.value:
            log_complete(False, "用户取消运行")
            return
        
        # 步骤2: 场景搜索
        if run_search:
            log_status("正在运行搜索...")
            if search_entry_mode == 'prompt':
                success = run_prompt_search_with_config(config)
            elif search_entry_mode == 'label':
                success = run_label_search_with_config(config)
            elif search_entry_mode == 'cloze':
                success = run_cloze_search_with_config(config)
            else:
                log_message(f"未知的搜索模式: {search_entry_mode}", 'error')
                success = False
            
            if not success:
                log_complete(False, "搜索步骤失败，流水线终止")
                return
        else:
            log_message("⏭️ 跳过搜索步骤", 'info')
            log_progress(100)
        
        total_time_end = time.time()
        log_message(f"总耗时: {total_time_end - total_time_start:.2f} 秒", 'success')
        log_complete(True, f"流水线执行完成！总耗时: {total_time_end - total_time_start:.2f} 秒")
        
    except Exception as e:
        log_message(f"流水线执行出错: {str(e)}", 'error')
        import traceback
        log_message(traceback.format_exc(), 'error')
        log_complete(False, f"流水线执行出错: {str(e)}")
    finally:
        _running = False


# ============================================================================
# Flask 路由
# ============================================================================
@app.route('/')
def index():
    """主页"""
    try:
        return render_template('Pipeline_control.html')
    except Exception as e:
        return f"<h1>错误: 无法加载模板</h1><p>{e}</p><p>请确保 templates/Pipeline_control.html 文件存在</p>"


@app.route('/browse')
def browse():
    """文件浏览器"""
    path = request.args.get('path', _SCRIPT_DIR)
    current_path = resolve_path(unquote(path))
    
    if os.path.isfile(current_path):
        current_path = os.path.dirname(current_path)
    if not os.path.exists(current_path) or not os.path.isdir(current_path):
        current_path = _SCRIPT_DIR
    
    try:
        dirs, files = [], []
        for item in os.listdir(current_path):
            try:
                if os.path.isdir(os.path.join(current_path, item)):
                    dirs.append(item)
                else:
                    files.append(item)
            except OSError:
                continue
        
        dirs.sort(key=str.lower)
        files.sort(key=str.lower)
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            parent_path = None
        
        return jsonify({
            "current_path": current_path,
            "parent_path": parent_path,
            "dirs": dirs,
            "files": files
        })
    except Exception as e:
        return jsonify({"error": str(e), "path": current_path}), 500


@app.route('/list_models')
def list_models():
    """列出 models 文件夹中的可用模型"""
    try:
        models = []
        if os.path.exists(MODELS_DIR):
            for item in os.listdir(MODELS_DIR):
                item_path = os.path.join(MODELS_DIR, item)
                if os.path.isdir(item_path):
                    # 检查是否是有效的模型文件夹
                    files = os.listdir(item_path)
                    is_valid = any(f.endswith(('.bin', '.pt', '.pth', '.safetensors', '.json')) for f in files)
                    if is_valid:
                        model_type = detect_model_type(item_path)
                        if model_type == 'reranker':
                            continue
                        models.append({
                            'name': item,
                            'path': item_path,
                            'type': model_type
                        })
        
        # 按名称排序
        models.sort(key=lambda x: x['name'].lower())
        
        return jsonify({
            "success": True,
            "models": models,
            "models_dir": MODELS_DIR
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/list_presets')
def list_presets():
    """列出所有参数预设"""
    try:
        presets = _list_preset_names()
        meta = _load_preset_meta()
        default_preset = meta.get('default_preset')
        if default_preset not in presets:
            default_preset = None
        return jsonify({
            "success": True,
            "presets": presets,
            "default_preset": default_preset
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"读取预设失败: {str(e)}"}), 500


@app.route('/load_preset')
def load_preset():
    """加载指定参数预设"""
    preset_name = request.args.get('name', '')
    try:
        safe_name, preset_path = _get_preset_path(preset_name)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400

    if not os.path.exists(preset_path):
        return jsonify({"success": False, "message": f"预设不存在: {safe_name}"}), 404

    try:
        with open(preset_path, 'r', encoding='utf-8') as f:
            preset_data = json.load(f)
        if isinstance(preset_data, dict) and 'data' in preset_data:
            payload = preset_data.get('data', {})
        else:
            payload = preset_data if isinstance(preset_data, dict) else {}
        return jsonify({
            "success": True,
            "name": safe_name,
            "data": payload
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"加载预设失败: {str(e)}"}), 500


@app.route('/save_preset', methods=['POST'])
def save_preset():
    """保存参数预设"""
    try:
        payload = request.get_json(silent=True) or {}
        preset_name = payload.get('name', '')
        preset_data = payload.get('data', {})
        set_as_default = bool(payload.get('set_as_default', False))

        if not isinstance(preset_data, dict):
            return jsonify({"success": False, "message": "预设数据格式错误"}), 400

        safe_name, preset_path = _get_preset_path(preset_name)
        _ensure_preset_dir()

        old_record = {}
        if os.path.exists(preset_path):
            try:
                with open(preset_path, 'r', encoding='utf-8') as f:
                    old_record = json.load(f)
            except Exception:
                old_record = {}

        now_ts = int(time.time())
        record = {
            "name": safe_name,
            "created_at": old_record.get('created_at', now_ts),
            "updated_at": now_ts,
            "data": preset_data
        }

        with open(preset_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=4)

        meta = _load_preset_meta()
        if set_as_default:
            meta['default_preset'] = safe_name
            _save_preset_meta(meta)

        return jsonify({
            "success": True,
            "message": f"预设已保存: {safe_name}",
            "name": safe_name,
            "default_preset": meta.get('default_preset')
        })
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"保存预设失败: {str(e)}"}), 500


@app.route('/set_default_preset', methods=['POST'])
def set_default_preset():
    """设置默认预设"""
    payload = request.get_json(silent=True) or {}
    preset_name = payload.get('name')

    try:
        meta = _load_preset_meta()
        if preset_name is None or str(preset_name).strip() == '':
            return jsonify({"success": False, "message": "请提供预设名称"}), 400

        safe_name, preset_path = _get_preset_path(str(preset_name))
        if not os.path.exists(preset_path):
            return jsonify({"success": False, "message": f"预设不存在: {safe_name}"}), 404

        meta['default_preset'] = safe_name
        _save_preset_meta(meta)
        return jsonify({
            "success": True,
            "message": f"默认预设已设置为: {safe_name}",
            "default_preset": safe_name
        })
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"设置默认预设失败: {str(e)}"}), 500


@app.route('/save_ffmpeg_config', methods=['POST'])
def save_ffmpeg_config():
    """保存 FFmpeg 参数到 config.json -> video_output"""
    try:
        payload = request.get_json(silent=True) or {}
        video_output_payload = payload.get('video_output', {})
        if not isinstance(video_output_payload, dict):
            return jsonify({"success": False, "message": "video_output 参数格式错误"}), 400

        config = _load_project_config()
        existing_video_output = config.get('video_output', {})
        if not isinstance(existing_video_output, dict):
            existing_video_output = {}

        allowed_fields = {
            'copy_mode', 'video_codec', 'hwaccel', 'gpu_device', 'crf',
            'preset', 'audio_codec', 'audio_bitrate', 'force_clean', 'pixel_format', 'movflags'
        }
        for key in allowed_fields:
            if key in video_output_payload:
                existing_video_output[key] = video_output_payload[key]

        nvenc_payload = video_output_payload.get('nvenc')
        if isinstance(nvenc_payload, dict):
            existing_nvenc = existing_video_output.get('nvenc', {})
            if not isinstance(existing_nvenc, dict):
                existing_nvenc = {}
            existing_nvenc.update(nvenc_payload)
            existing_video_output['nvenc'] = existing_nvenc

        config['video_output'] = existing_video_output
        _save_project_config(config)

        return jsonify({"success": True, "message": "FFmpeg 参数已写入 config.json"})
    except FileNotFoundError:
        return jsonify({"success": False, "message": f"找不到配置文件: {CONFIG_PATH}"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": f"写入配置失败: {str(e)}"}), 500


@app.route('/save_similarity_config', methods=['POST'])
def save_similarity_config():
    """保存相似度阈值到 config.json -> similarity_thresholds"""
    try:
        payload = request.get_json(silent=True) or {}
        thresholds_payload = payload.get('similarity_thresholds', {})
        if not isinstance(thresholds_payload, dict):
            return jsonify({"success": False, "message": "similarity_thresholds 参数格式错误"}), 400

        config = _load_project_config()
        existing_thresholds = config.get('similarity_thresholds', {})
        if not isinstance(existing_thresholds, dict):
            existing_thresholds = {}

        allowed_fields = {'clip_large', 'fgclip2', 'reranker', 'reranker_weight'}
        for key in allowed_fields:
            if key in thresholds_payload:
                existing_thresholds[key] = thresholds_payload[key]

        config['similarity_thresholds'] = existing_thresholds
        _save_project_config(config)

        return jsonify({"success": True, "message": "相似度阈值已写入 config.json"})
    except FileNotFoundError:
        return jsonify({"success": False, "message": f"找不到配置文件: {CONFIG_PATH}"}), 500
    except Exception as e:
        return jsonify({"success": False, "message": f"写入配置失败: {str(e)}"}), 500


@app.route('/run_pipeline_stream')
def run_pipeline_stream():
    """通过 SSE 流式返回运行日志，支持 EventSource 自动重连"""
    global _running, _log_queue, _pipeline_process
    
    # 如果 pipeline 已在运行，直接接入现有队列继续推送（支持断线自动重连）
    if is_pipeline_running() and _log_queue is not None:
        def generate_reconnect():
            completed_received = False
            while True:
                try:
                    msg = _log_queue.get(timeout=1)
                    yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                    if msg.get('type') == 'complete':
                        completed_received = True
                        break
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    if not is_pipeline_running():
                        if not completed_received:
                            fallback_msg = {
                                'type': 'complete',
                                'success': False,
                                'message': '用户强制停止运行' if _stop_requested.value else '流水线进程已退出'
                            }
                            yield f"data: {json.dumps(fallback_msg, ensure_ascii=False)}\n\n"
                        break
            _cleanup_pipeline_process()
        return Response(generate_reconnect(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        })
    
    config_str = request.args.get('config', '{}')
    try:
        config = json.loads(config_str)
    except json.JSONDecodeError:
        return jsonify({"error": "无效的配置JSON"}), 400
    
    _cleanup_pipeline_process()
    _log_queue = mp.Queue()
    _stop_requested.value = False
    _pipeline_process = mp.Process(target=_run_pipeline_in_process, args=(config, _log_queue, _stop_requested))
    _pipeline_process.daemon = True
    _pipeline_process.start()
    _running = True
    
    def generate():
        completed_received = False
        while True:
            try:
                # 等待日志消息，超时1秒
                msg = _log_queue.get(timeout=1)
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                
                # 如果是完成消息，结束流
                if msg.get('type') == 'complete':
                    completed_received = True
                    break
            except queue.Empty:
                # 发送心跳
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                
                # 检查子进程是否还在运行
                if not is_pipeline_running():
                    if not completed_received:
                        fallback_msg = {
                            'type': 'complete',
                            'success': False,
                            'message': '用户强制停止运行' if _stop_requested.value else '流水线进程已退出'
                        }
                        yield f"data: {json.dumps(fallback_msg, ensure_ascii=False)}\n\n"
                    break
        _cleanup_pipeline_process()
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
    })


@app.route('/stop_pipeline', methods=['POST'])
def stop_pipeline():
    """停止流水线"""
    global _pipeline_process
    
    if not is_pipeline_running():
        return jsonify({"success": False, "message": "流水线未在运行"})
    
    _stop_requested.value = True
    
    try:
        # 先等待子进程在检查点自行退出（优雅停止）
        _pipeline_process.join(timeout=5)
        # 超时仍在运行则强制终止
        if _pipeline_process.is_alive():
            _pipeline_process.terminate()
            _pipeline_process.join(timeout=3)
        if _pipeline_process.is_alive() and hasattr(_pipeline_process, 'kill'):
            _pipeline_process.kill()
            _pipeline_process.join(timeout=2)
    except Exception as e:
        return jsonify({"success": False, "message": f"停止失败: {e}"})
    finally:
        _cleanup_pipeline_process()
    
    try:
        _log_queue.put({'type': 'status', 'message': '⏹️ 已强制终止流水线进程'})
        _log_queue.put({'type': 'complete', 'success': False, 'message': '用户强制停止运行'})
    except Exception:
        pass
    
    return jsonify({"success": True, "message": "已强制停止流水线"})


@app.route('/status')
def get_status():
    """获取当前运行状态"""
    return jsonify({
        "running": is_pipeline_running(),
        "indexer_available": _INDEXER_AVAILABLE,
        "prompt_search_available": _PROMPT_SEARCH_AVAILABLE,
        "label_search_available": _LABEL_SEARCH_AVAILABLE,
        "cloze_search_available": _CLOZE_SEARCH_AVAILABLE
    })


def main():
    """
    主函数：启动 Web 控制面板
    
    用法:
        python pipeline_app.py
        python pipeline_app.py --port 5007
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='视频处理流水线（仅Web模式）')
    parser.add_argument('--port', type=int, default=5007, help='Web服务端口 (默认: 5007)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  🎬 视频处理流水线控制面板")
    print("  Flask Web界面版")
    print(f"  索引模块: {'✓ 可用' if _INDEXER_AVAILABLE else '✗ 不可用'}")
    print(f"  Prompt搜索模块: {'✓ 可用' if _PROMPT_SEARCH_AVAILABLE else '✗ 不可用'}")
    print(f"  标签搜索模块: {'✓ 可用' if _LABEL_SEARCH_AVAILABLE else '✗ 不可用'}")
    print(f"  选词填空模块: {'✓ 可用' if _CLOZE_SEARCH_AVAILABLE else '✗ 不可用'}")
    print(f"  访问: http://127.0.0.1:{args.port}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
