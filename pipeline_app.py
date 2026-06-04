# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
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
import threading
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
_stream_lock = threading.Lock()
_stream_active = False

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
    from A_coreUtils.search.auto_scene_search import run_interactive_search, cleanup_temp_after_export
    _PROMPT_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] 无法导入 auto_scene_search 模块 - {e}")
    _PROMPT_SEARCH_AVAILABLE = False


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


def _try_acquire_stream_slot() -> bool:
    """Queue-based SSE logs support one active consumer at a time."""
    global _stream_active
    with _stream_lock:
        if _stream_active:
            return False
        _stream_active = True
        return True


def _release_stream_slot():
    global _stream_active
    with _stream_lock:
        _stream_active = False


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


def _is_clip_model_name(model_name: str) -> bool:
    """仅允许名称中包含 clip 的模型。"""
    normalized = str(model_name).strip().lower()
    return bool(normalized and 'clip' in normalized)


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


def _normalize_top_k(value):
    """
    Normalize top_k from web payload:
    - None-like / non-positive -> None (unlimited)
    - positive integer -> int
    """
    top_k = _normalize_optional_int(value, field_name='top_k')
    if top_k is None:
        return None
    return top_k if top_k > 0 else None


def _normalize_optional_int(value, field_name: str = 'value'):
    """Normalize optional integer. None-like values map to None."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('', 'none', 'null', 'unlimited', 'default', 'auto', '默认', '不限'):
            return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name} value: {value!r}") from exc


def _normalize_optional_positive_int(value, field_name: str = 'value'):
    """Normalize optional positive integer. None-like/<=0 values map to None."""
    normalized = _normalize_optional_int(value, field_name=field_name)
    if normalized is None:
        return None
    return normalized if normalized > 0 else None


def _validate_lance_index_directory(index_directory: str):
    """Validate Lance-only index directory used by L/C modes."""
    try:
        if not os.path.exists(index_directory):
            return False, f"索引目录不存在: {index_directory}"
        if not os.path.isdir(index_directory):
            return False, f"索引路径不是目录: {index_directory}"

        has_lance = False
        first_pkl = None
        for file_name in os.listdir(index_directory):
            file_path = os.path.join(index_directory, file_name)
            if os.path.isdir(file_path) and file_name.endswith('.lance'):
                has_lance = True
            elif file_name.endswith('.pkl'):
                first_pkl = file_path

        if first_pkl is not None:
            return False, f"检测到不支持的 .pkl 索引，请先转换为 .lance: {first_pkl}"
        if not has_lance:
            return False, f"索引目录中没有 .lance 索引: {index_directory}"
        return True, ''
    except Exception as e:
        return False, f"检查索引目录失败: {e}"


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

    selected_model_name = str(idx_config.get('model_name', '')).strip() or 'qihoo360-fg-clip2-base'
    if not _is_clip_model_name(selected_model_name):
        log_message(f"错误: 仅允许使用名称包含 clip 的模型，当前模型为: {selected_model_name}", 'error')
        return False
    
    # 解析路径
    input_dir = resolve_path(idx_config.get('input_directory', ''))
    output_dir = resolve_path(idx_config.get('output_directory', _path_resolver.join_str('indexes')))
    
    # 构建参数 - 只传预设中明确存在的参数，未设置的让下游函数使用自己的默认值
    parameters = {
        'model_name': selected_model_name,
        'model_type': 'auto',
    }
    if idx_config.get('truncate_dim') is not None:
        parameters['truncate_dim'] = int(idx_config['truncate_dim'])
    # 直通参数：仅在预设中存在且非 None 时才传递，None 让下游函数使用自己的默认值
    for key in ('output_resolution', 'batch_size', 'workers', 'io_workers',
                'cosine_similarity_threshold', 'brightness_threshold', 'black_pixel_ratio',
                'white_threshold', 'white_pixel_ratio',
                'sample_interval', 'min_scene_length', 'localmax_order',
                'resume_processing', 'use_fp16'):
        if key in idx_config and idx_config[key] is not None:
            parameters[key] = idx_config[key]
    
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

    # 自动从 config.json 获取对应模型的 CLIP 相似度阈值
    if 'search_threshold' not in ps_config or ps_config.get('search_threshold') is None:
        project_cfg = _load_project_config()
        sim_thresholds = project_cfg.get('similarity_thresholds', {})
        idx_mname = str(config.get('indexer', {}).get('model_name', '')).lower()
        if 'fg-clip2' in idx_mname or 'fgclip2' in idx_mname:
            ps_config['search_threshold'] = float(sim_thresholds.get('fgclip2', 14))
        else:
            ps_config['search_threshold'] = float(sim_thresholds.get('clip_large', 21))
        log_message(f"自动设置搜索阈值(按模型): {ps_config['search_threshold']}", 'info')
    
    # 检查停止请求
    if _stop_requested.value:
        log_message("收到停止请求，终止搜索", 'warning')
        return False
    
    time_start = time.time()
    log_progress(55)

    diskcache_dir = _resolve_diskcache_dir(ps_config.get('diskcache_dir'))

    try:
        # 只传预设中明确存在的参数，未设置的让下游函数使用自己的默认值
        kwargs = {}
        # 路径配置（有默认回退）
        if 'index_directory' in ps_config:
            kwargs['index_directory'] = resolve_path(ps_config['index_directory'])
        if 'output_directory' in ps_config:
            kwargs['output_directory'] = resolve_path(ps_config['output_directory'])
        # 需要类型转换的参数
        if 'search_mode' in ps_config:
            normalized_search_mode = _normalize_optional_int(ps_config.get('search_mode'), field_name='prompt_search.search_mode')
            if normalized_search_mode is not None:
                kwargs['search_mode'] = normalized_search_mode
        if 'top_k' in ps_config:
            normalized_top_k = _normalize_top_k(ps_config.get('top_k'))
            kwargs['top_k'] = normalized_top_k
        normalized_start_offset = _normalize_optional_int(ps_config.get('start_frame_offset'), field_name='prompt_search.start_frame_offset')
        if normalized_start_offset is not None:
            kwargs['start_frame_offset'] = normalized_start_offset
        normalized_end_offset = _normalize_optional_int(ps_config.get('end_frame_offset'), field_name='prompt_search.end_frame_offset')
        if normalized_end_offset is not None:
            kwargs['end_frame_offset'] = normalized_end_offset
        if 'reranker_output_resolution' in ps_config:
            normalized_reranker_resolution = _normalize_optional_positive_int(
                ps_config.get('reranker_output_resolution'),
                field_name='prompt_search.reranker_output_resolution'
            )
            if normalized_reranker_resolution is not None:
                kwargs['reranker_output_resolution'] = str(normalized_reranker_resolution)
        # diskcache_dir 经过特殊解析
        if diskcache_dir is not None:
            kwargs['diskcache_dir'] = diskcache_dir
        # 直通参数：仅在预设中存在且非 None 时才传递
        for key in ('use_fp16', 'use_reranker',
                     'video_output_directory', 'video_copy_mode',
                     'feature_fp16',
                     'use_diskcache', 'debug_similarity',
                     'use_chinese',
                     'vector_dedup_threshold', 'adjacent_merge_frames',
                      'use_shot_analysis',
                      'use_mini_rerank', 'mini_rerank_min_matches',
                      'search_threshold'):
            if key in ps_config and ps_config[key] is not None:
                kwargs[key] = ps_config[key]
        # lance_indexes: 用户选中的索引文件名列表
        if 'lance_indexes' in ps_config and ps_config.get('lance_indexes'):
            kwargs['lance_indexes'] = ps_config['lance_indexes']
        for key in ('rerank_top_k', 'rerank_batch_size', 'candidate_batch_size',
                    'prompt_search_batch_size', 'lance_batch_size',
                    'prompt_cache_batch_size', 'lance_load_workers',
                    'lmdb_write_batch_size'):
            if key not in ps_config:
                continue
            normalized = _normalize_optional_positive_int(ps_config.get(key), field_name=f'prompt_search.{key}')
            # 字段存在即透传；显式 None/0 表示"不限制/自动"，不能回退为后端固定值
            kwargs[key] = normalized

        index_dir_for_check = kwargs.get('index_directory', _path_resolver.join_str('indexes'))
        valid_idx, idx_msg = _validate_lance_index_directory(index_dir_for_check)
        if not valid_idx:
            log_message(f"Prompt search failed: {idx_msg}", 'error')
            return False

        use_shot_analysis = kwargs.pop('use_shot_analysis', False)
        search_kwargs = kwargs
        search_kwargs['use_shot_analysis'] = use_shot_analysis
        search_results = run_interactive_search(**search_kwargs)
        if isinstance(search_results, dict) and search_results.get('export_succeeded'):
            cleanup_temp_after_export(_path_resolver)

        if not isinstance(search_results, dict):
            log_message("Prompt搜索返回异常结果类型，判定为失败", 'error')
            return False
        if not search_results.get('success', False):
            log_message(f"Prompt搜索失败: {search_results.get('error', 'unknown error')}", 'error')
            return False

        # logic_keywords.json 一致性校验警告
        for w in search_results.get('validation_warnings', []) or []:
            log_message(f"[logic_keywords] {w}", 'warning')

        result_count = int(search_results.get('result_count', 0) or 0)
        merged_count = int(search_results.get('merged_result_count', result_count) or 0)
        export_stats = search_results.get('export_stats') or {}
        log_message(f"Prompt搜索结果: 初始 {result_count}，导出前 {merged_count}", 'info')
        if export_stats:
            log_message(
                f"视频导出统计: 成功 {export_stats.get('success', 0)}，失败 {export_stats.get('failed', 0)}，跳过 {export_stats.get('skipped', 0)}",
                'info'
            )

        time_end = time.time()
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
        run_indexer = config.get('run_indexer', True)
        run_search = config.get('run_search', True)
        
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
        
        # 流水线串联：如果索引步骤运行了且搜索步骤未显式指定 index_directory，
        # 自动将索引器的 output_directory 传递给搜索步骤
        if run_indexer and run_search:
            idx_config = config.get('indexer', {})
            indexer_output = resolve_path(idx_config.get('output_directory', _path_resolver.join_str('indexes')))
            search_cfg = config.setdefault('prompt_search', {})
            if 'index_directory' not in search_cfg or not search_cfg['index_directory']:
                search_cfg['index_directory'] = indexer_output
                log_message(f"流水线串联: 搜索步骤自动使用索引输出目录 {indexer_output}", 'info')

        # 步骤2: 场景搜索
        if run_search:
            log_status("正在运行搜索...")
            success = run_prompt_search_with_config(config)
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


@app.route('/pick_directory', methods=['POST'])
def pick_directory():
    """打开系统原生目录选择对话框，返回选中路径"""
    try:
        data = request.get_json(silent=True) or {}
        initial_dir = data.get('initial_dir', _SCRIPT_DIR)
        if initial_dir and not os.path.isdir(initial_dir):
            initial_dir = _SCRIPT_DIR

        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        root.update()

        selected = filedialog.askdirectory(
            initialdir=initial_dir,
            title='选择目录'
        )

        root.destroy()
        root.quit()

        if selected:
            return jsonify({"success": True, "path": os.path.normpath(selected)})
        else:
            return jsonify({"success": False, "path": None, "message": "未选择目录"})
    except Exception as e:
        return jsonify({"success": False, "path": None, "error": str(e)}), 500


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
                        # clip-only model list for index page
                        if not _is_clip_model_name(item):
                            continue
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


@app.route('/list_lance_files')
def list_lance_files():
    """列出索引目录下的 .lance 索引子目录"""
    try:
        index_dir = request.args.get('dir', '')
        index_dir = resolve_path(index_dir) if index_dir else _path_resolver.join_str('indexes')
        lance_files = []
        if os.path.isdir(index_dir):
            for item in os.listdir(index_dir):
                item_path = os.path.join(index_dir, item)
                if os.path.isdir(item_path) and item.endswith('.lance'):
                    lance_files.append(item)
        lance_files.sort(key=str.lower)
        return jsonify({"success": True, "lance_files": lance_files, "index_dir": os.path.normpath(index_dir)})
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
    """Stream logs via SSE; supports EventSource reconnect."""
    global _running, _log_queue, _pipeline_process

    def _busy_stream():
        busy_msg = {
            'type': 'complete',
            'success': False,
            'message': 'Another client is already consuming logs; retry later'
        }
        yield f"data: {json.dumps(busy_msg, ensure_ascii=False)}\n\n"

    # If pipeline is running, allow reconnect to the same log stream.
    if is_pipeline_running() and _log_queue is not None:
        if not _try_acquire_stream_slot():
            return Response(_busy_stream(), mimetype='text/event-stream', headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            })

        def generate_reconnect():
            completed_received = False
            try:
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
                                    'message': 'Stopped by user' if _stop_requested.value else 'Pipeline process exited'
                                }
                                yield f"data: {json.dumps(fallback_msg, ensure_ascii=False)}\n\n"
                            break
            finally:
                _cleanup_pipeline_process()
                _release_stream_slot()

        return Response(generate_reconnect(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        })

    config_str = request.args.get('config', '{}')
    try:
        config = json.loads(config_str)
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid config JSON'}), 400

    if not _try_acquire_stream_slot():
        return Response(_busy_stream(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        })

    try:
        _cleanup_pipeline_process()
        _log_queue = mp.Queue()
        _stop_requested.value = False
        _pipeline_process = mp.Process(target=_run_pipeline_in_process, args=(config, _log_queue, _stop_requested))
        _pipeline_process.daemon = True
        _pipeline_process.start()
        _running = True
    except Exception:
        _release_stream_slot()
        raise

    def generate():
        completed_received = False
        try:
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
                                'message': 'Stopped by user' if _stop_requested.value else 'Pipeline process exited'
                            }
                            yield f"data: {json.dumps(fallback_msg, ensure_ascii=False)}\n\n"
                        break
        finally:
            _cleanup_pipeline_process()
            _release_stream_slot()

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
        "prompt_search_available": _PROMPT_SEARCH_AVAILABLE
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
    print(f"  访问: http://127.0.0.1:{args.port}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
