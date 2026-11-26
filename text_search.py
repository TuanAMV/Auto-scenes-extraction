# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
# text_search.py (v3.11 - CLIP/FG-CLIP2)
# 支持从models文件夹动态加载模型
# v3.11: 移除 Qwen3-VL-Embedding 支持，仅保留 CLIP/FG-CLIP2

import os
import atexit
import gc
from A_coreUtils.aftertreatment.jev_rerank import JevReranker, extracted_frames, score_prefix
from A_coreUtils.search.lmdb_store import LMDBCache, RankedResults, is_op_ed_video
from functools import lru_cache
from uuid import uuid4
from flask import abort
import json
import time
import re
from flask import Flask, render_template, request, jsonify, send_from_directory
from urllib.parse import unquote
import hashlib
import itertools

from A_coreUtils.video_processing.video_utils import VideoMetaHelper, cleanup_temp_folder, TEMP_DIR, extract_single_frame

# 导入视频切割工具
try:
    from A_coreUtils.video_processing.ffmpeg_precision_cutter import export_video_clip
    _FFMPEG_CUTTER_AVAILABLE = True
except ImportError:
    _FFMPEG_CUTTER_AVAILABLE = False

# 导入视频文件名解析器
try:
    from A_coreUtils.video_processing.video_name_parser import VideoNameParser
    _video_name_parser = VideoNameParser()
except ImportError:
    _video_name_parser = None

# 尝试导入不同的模型处理器
_EMBEDDING_MODEL_AVAILABLE = False

try:
    from A_coreUtils.embedding.embedding_model import EmbeddingModelProcessor, SemanticSearchEngine as EmbeddingSearchEngine
    _EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    pass

if not _EMBEDDING_MODEL_AVAILABLE:
    raise ImportError("需要安装 embedding_model 模块")

# 导入路径解析工具
from path_resolver import PathResolver, resolve_path

# 创建路径解析器实例（无参 → 以 path_resolver.py 所在目录为项目根）
_base_resolver = PathResolver()

app = Flask(__name__, template_folder='templates', static_folder='static')

# 使用PathResolver获取BASE_DIRECTORY
BASE_DIRECTORY = _base_resolver.get_project_root_str()
MODELS_DIR = _base_resolver.join_str('models')
PRESET_DIR = _base_resolver.join_str('preset', 'text_search')
PRESET_META_PATH = _base_resolver.join_str('preset', 'text_search', '_meta.json')
_PRESET_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
CONFIG_PATH = _base_resolver.join_str('config.json')
PREVIEW_FOLDER = _base_resolver.join_str('temp', 'previews')
RERANK_CACHE_DIR = _base_resolver.join_str('temp', 'rerank_cache')
FRAME_EXTRACTOR_CACHE_DIR = _base_resolver.join_str('temp', 'rerank_frames_single')
CLIPS_FOLDER = _base_resolver.join_str('temp', 'clips')  # 视频片段缓存目录
# 所有临时文件都放在 temp 下
PREVIEW_FOLDER = _base_resolver.join_str('temp', 'previews')
CACHE_DIR = _base_resolver.join_str('temp', 'search_cache')  # 重命名: faiss_cache -> search_cache

# 所有临时文件都放在 temp 下
PREVIEW_FOLDER = _base_resolver.join_str('temp', 'previews')
CACHE_DIR = _base_resolver.join_str('temp', 'search_cache')  # 重命名: faiss_cache -> search_cache

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PREVIEW_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RERANK_CACHE_DIR, exist_ok=True)
os.makedirs(CLIPS_FOLDER, exist_ok=True)
os.makedirs(PRESET_DIR, exist_ok=True)


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

_embedding_processor = None
_search_engine = None

_preview_fps = lru_cache(maxsize=128)(VideoMetaHelper.get_fps_cached)
_current_model_name = None


def release_clip_model():
    """释放 CLIP 模型，清理显存"""
    global _embedding_processor, _search_engine, _current_model_name
    if _embedding_processor is not None:
        print("[Info] 释放 CLIP 模型...")
        try:
            del _embedding_processor
            del _search_engine
        except Exception as e:
            print(f"[Warning] 释放 CLIP 模型时出错: {e}")
        _embedding_processor = None
        _search_engine = None
        _current_model_name = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        print("[Info] CLIP 模型已释放")


# ============================================================================
# Jev-rerank
# ============================================================================

def evaluate_text_matches(candidates, query_text, *, rerank, aesthetics, threshold, batch_size):
    """在检索选定的帧上核验；旧索引的中间帧由原始镜头边界计算。"""
    questions = {'query': f'这张图片是否符合以下描述？{query_text}'} if rerank else {}
    with JevReranker() as jev:
        iterator = iter(candidates)
        while True:
            rows = list(itertools.islice(iterator, batch_size))
            if not rows:
                break
            batch = [payload for _, payload in rows]
            targets = [(m['video_path'], m['matched_frame']) for m in batch]
            with extracted_frames(targets, RERANK_CACHE_DIR) as frames:
                for (key, _), match, target in zip(rows, batch, targets):
                    path = frames.get(target)
                    if path:
                        result = jev.evaluate(path, questions, aesthetics=aesthetics,
                                              min_matches=int(rerank), threshold=threshold)
                        if rerank:
                            score = result['scores']['query']
                            if score < threshold:
                                continue
                            match['jev_rerank_score'] = score
                            match['similarity'] += score * 50
                        if aesthetics:
                            match['aesthetic_score'] = result['aesthetic_score']
                    else:
                        if rerank:
                            print(f"[Jev-rerank] {target[0]} 命中帧 {target[1]} 读取失败，无法复核，淘汰")
                            continue
                        match['jev_note'] = '命中帧读取失败'
                        if aesthetics:
                            match['aesthetic_score'] = None
                    yield key, match


def detect_model_type(model_path: str) -> str:
    """根据模型文件夹内容自动检测模型类型"""
    if not os.path.isdir(model_path):
        return 'auto'
    
    model_name_lower = os.path.basename(model_path).lower()
    
    # [简化] 移除 Chinese CLIP 检测
    
    # 检查 FG-CLIP2
    if 'fg-clip2' in model_name_lower or 'fg_clip2' in model_name_lower or 'fgclip2' in model_name_lower:
        return 'fgclip2'

    # 检查配置文件 (open_clip 格式)
    config_path = os.path.join(model_path, 'open_clip_config.json')
    if os.path.exists(config_path):
        return 'clip'
    
    # 检查 config.json (transformers 格式)
    config_json_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_json_path):
        try:
            with open(config_json_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # [简化] 移除 Chinese CLIP 检测
            architectures = config.get('architectures', [])
            
            # 检测 FG-CLIP2 (通过 auto_map 或 architectures)
            if config.get('auto_map', {}).get('AutoModelForCausalLM'):
                return 'fgclip2'
            if any('fgclip' in arch.lower() for arch in architectures):
                return 'fgclip2'
        except:
            pass
    
    return 'auto'


def get_or_create_processor(model_name: str = None, truncate_dim: int = None, use_fp16: bool = True):
    """获取或创建全局模型处理器"""
    global _embedding_processor, _search_engine, _current_model_name
    
    # 设置默认模型
    if model_name is None:
        model_name = 'qihoo360-fg-clip2-base'
    
    # 构建模型路径
    if os.path.isabs(model_name) or os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = _base_resolver.join_str('models', model_name)
    
    # 自动检测模型类型
    model_type = detect_model_type(model_path)
    
    # 如果模型改变,重新创建处理器
    if _embedding_processor is None or _current_model_name != model_name:
        # [修复] 切换模型时清理旧模型缓存，释放显存/内存
        if _embedding_processor is not None and _current_model_name != model_name:
            print(f"[Info] 切换模型: {_current_model_name} -> {model_name}，清理旧模型缓存...")
            try:
                # 清理旧的处理器
                if hasattr(_embedding_processor, 'cleanup'):
                    _embedding_processor.cleanup()
                if hasattr(_embedding_processor, 'release_models'):
                    _embedding_processor.release_models()
                del _embedding_processor
                del _search_engine
                _embedding_processor = None
                _search_engine = None
                # 尝试释放GPU显存
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                print(f"[Info] 旧模型缓存已清理")
            except Exception as e:
                print(f"[Warning] 清理旧模型时出错: {e}")
        
        print(f"[Info] 加载模型: {model_name} (类型: {model_type}, FP16: {use_fp16})")
        
        # 根据模型类型选择处理器
        if not _EMBEDDING_MODEL_AVAILABLE:
            raise RuntimeError("CLIP 模型需要安装 embedding_model 模块")
        _embedding_processor = EmbeddingModelProcessor(
            model_name=model_name,
            model_type=model_type,
            batch_size=128,
            truncate_dim=truncate_dim,
            use_fp16=use_fp16
        )
        _search_engine = EmbeddingSearchEngine(_embedding_processor, CACHE_DIR)
        
        _current_model_name = model_name
    else:
        if hasattr(_embedding_processor, 'truncate_dim'):
            _embedding_processor.truncate_dim = truncate_dim
    
    return _embedding_processor, _search_engine


# text_search.py 自身使用的临时子目录（不包含搜索模式的 LMDB 缓存）
SEARCH_BASKET_DIR = os.path.join(os.path.dirname(CACHE_DIR), 'search_baskets')
_TEXT_SEARCH_TEMP_SUBDIRS = [PREVIEW_FOLDER, RERANK_CACHE_DIR, CLIPS_FOLDER]


def _clear_session_lmdbs(root):
    if not os.path.isdir(root):
        return
    for name in os.listdir(root):
        if not (name.startswith('result_') or name.startswith('basket_')):
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path):
            cleanup_temp_folder(path)
            try:
                os.rmdir(path)
            except OSError:
                pass


def cleanup_temp_folders():
    """Clear transient files and completed TextSearch session databases."""
    print("\n--- Clearing temporary files ---")
    for subdir in _TEXT_SEARCH_TEMP_SUBDIRS:
        if os.path.exists(subdir):
            cleanup_temp_folder(subdir)
            print(f"  Cleared {subdir}")
    _clear_session_lmdbs(CACHE_DIR)
    _clear_session_lmdbs(SEARCH_BASKET_DIR)
    if os.path.exists(TEMP_DIR):
        for item in os.listdir(TEMP_DIR):
            item_path = os.path.join(TEMP_DIR, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                try:
                    os.unlink(item_path)
                except (PermissionError, OSError):
                    pass
    print("Temporary cleanup complete")


def startup_cleanup():
    """Remove transient files and stale result/basket LMDBs from prior runs."""
    print("--- Startup cleanup ---")
    for subdir in _TEXT_SEARCH_TEMP_SUBDIRS:
        if os.path.exists(subdir) and os.listdir(subdir):
            print(f"Removing stale temporary files: {subdir}")
            cleanup_temp_folder(subdir)
    _clear_session_lmdbs(CACHE_DIR)
    _clear_session_lmdbs(SEARCH_BASKET_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PREVIEW_FOLDER, exist_ok=True)
    print("Startup cleanup complete")


@app.route('/')
def index():
    try:
        default_export_dir = EXPORTS_FOLDER
        default_export_hint = os.path.relpath(default_export_dir, BASE_DIRECTORY).replace('\\', '/')
        if default_export_hint in ('.', ''):
            default_export_hint = 'output'
        if not default_export_hint.endswith('/'):
            default_export_hint += '/'
        return render_template(
            'Frame_text_search.html',
            default_export_dir=default_export_dir,
            default_export_hint=default_export_hint
        )
    except Exception as e:
        return f"<h1>错误: 无法加载模板</h1><p>{e}</p>"


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """清理整个 temp 文件夹并释放模型"""
    try:
        cleared_items = []
        total_size = 0
        
        # 释放模型
        release_clip_model()
        cleared_items.append("已释放模型")
        
        # 计算并清理 text_search.py 自身的临时子目录（保留 cache 目录下的搜索结果 LMDB）
        for subdir in [*_TEXT_SEARCH_TEMP_SUBDIRS, CACHE_DIR, SEARCH_BASKET_DIR]:
            if os.path.exists(subdir):
                for root, dirs, files in os.walk(subdir):
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            total_size += os.path.getsize(fp)
                        except OSError:
                            pass
                cleanup_temp_folder(subdir)
                cleared_items.append(f"{os.path.basename(subdir)}/ ({subdir})")
        
        # 重新创建必要的子文件夹
        os.makedirs(TEMP_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(PREVIEW_FOLDER, exist_ok=True)
        os.makedirs(RERANK_CACHE_DIR, exist_ok=True)
        os.makedirs(CLIPS_FOLDER, exist_ok=True)
        
        # 显式清除持久化抽帧器缓存，避免正常结束前退出留下旧帧。
        if os.path.isdir(FRAME_EXTRACTOR_CACHE_DIR):
            for root, _, files in os.walk(FRAME_EXTRACTOR_CACHE_DIR):
                for name in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        pass
            from A_coreUtils.search.reranker_frame_extractor import RerankerFrameExtractor
            extractor = RerankerFrameExtractor(cache_dir=FRAME_EXTRACTOR_CACHE_DIR)
            extractor.clear_cache_dir()
            stats = extractor.get_cache_stats()
            cleared_items.append(f"帧提取缓存 ({stats['cache_dir']})")

        # 清理 FPS 缓存
        _preview_fps.cache_clear()
        cleared_items.append("FPS缓存")
        
        # 格式化大小
        if total_size >= 1024 * 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"
        elif total_size >= 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024):.2f} MB"
        elif total_size >= 1024:
            size_str = f"{total_size / 1024:.2f} KB"
        else:
            size_str = f"{total_size} B"
        
        return jsonify({
            "success": True,
            "message": f"已清理 {size_str} 缓存",
            "cleared": cleared_items
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/list_models')
def list_models():
    """列出models文件夹中的可用模型"""
    try:
        models = []
        if os.path.exists(MODELS_DIR):
            for item in os.listdir(MODELS_DIR):
                item_path = os.path.join(MODELS_DIR, item)
                if os.path.isdir(item_path):
                    # 检查是否是有效的模型文件夹
                    files = os.listdir(item_path)
                    is_valid = any(f.endswith(('.bin', '.pt', '.pth', '.safetensors', '.json')) for f in files)
                    if is_valid and 'clip' in item.lower():
                        model_type = detect_model_type(item_path)
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


@app.route('/get_model_info', methods=['POST'])
def get_model_info():
    """获取模型的详细信息（如 logit_scale）
    [修复] 不再加载模型，只返回基本信息。logit_scale 在搜索时才获取。
    """
    try:
        data = request.json
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({"success": False, "error": "未指定模型名称"}), 400
        
        # 构建模型路径
        if os.path.isabs(model_name) or os.path.exists(model_name):
            model_path = model_name
        else:
            model_path = _base_resolver.join_str('models', model_name)
        
        model_type = detect_model_type(model_path)
        
        # [修复] 不调用 get_or_create_processor，避免页面加载时就加载模型
        # 只返回模型类型信息，logit_scale 在搜索时才获取
        info = {
            "success": True,
            "model_name": model_name,
            "model_type": model_type,
            "logit_scale": None,  # 搜索时才获取
            "note": "模型将在搜索时加载"
        }
        
        return jsonify(info)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/list_presets')
def list_presets():
    """列出文本搜索预设"""
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
    """加载指定文本搜索预设"""
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
    """保存文本搜索预设"""
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
    """设置默认文本搜索预设"""
    payload = request.get_json(silent=True) or {}
    preset_name = payload.get('name')

    try:
        if preset_name is None or str(preset_name).strip() == '':
            return jsonify({"success": False, "message": "请提供预设名称"}), 400

        safe_name, preset_path = _get_preset_path(str(preset_name))
        if not os.path.exists(preset_path):
            return jsonify({"success": False, "message": f"预设不存在: {safe_name}"}), 404

        meta = _load_preset_meta()
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


def _strip_index_suffixes(index_name: str) -> str:
    """去掉索引文件名中的模式和维度后缀"""
    remaining = index_name
    mode_match = re.search(r'_(Single|Triplet|TripletAvg|SceneNative3f|SceneNative|SceneDetect|Video)$', remaining)
    if mode_match:
        remaining = remaining[:mode_match.start()]

    dim_match = re.search(r'_d\d+$', remaining)
    if dim_match:
        remaining = remaining[:dim_match.start()]

    return remaining


def _match_model_name_from_models_dir(index_name: str) -> str | None:
    """根据 models 目录中的模型名，从索引文件名后缀反推模型名"""
    try:
        model_names = []
        if os.path.isdir(MODELS_DIR):
            for item in os.listdir(MODELS_DIR):
                item_path = os.path.join(MODELS_DIR, item)
                if os.path.isdir(item_path):
                    model_names.append(item)

        # 优先匹配更长名称，避免短名称误命中
        model_names.sort(key=len, reverse=True)
        for model_name in model_names:
            if index_name == model_name or index_name.endswith(f"_{model_name}"):
                return model_name
    except Exception:
        pass
    return None


def extract_model_name_from_index(index_path: str) -> str:
    """从索引文件/目录名中提取模型名
    支持格式:
    1. {source}_{model_name}{_dXXX}_{mode}.lance
    2. {source}_{model_name}.lance
    """
    filename = os.path.basename(index_path)
    basename = filename.replace('.lance', '')

    remaining = _strip_index_suffixes(basename)

    # 优先用 models 目录反推，兼容 source 名称中含下划线的情况
    matched_model = _match_model_name_from_models_dir(remaining)
    if matched_model:
        return matched_model
    
    # 找到第一个下划线后的部分作为模型名
    first_underscore = remaining.find('_')
    if first_underscore == -1:
        return None
    
    return remaining[first_underscore + 1:]


# ============================================================
#  跨视频去重工具函数（复用P模式的去重逻辑）
# ============================================================

def _result_path(token, kind='result'):
    if not isinstance(token, str) or not re.fullmatch('[0-9a-f]{32}', token):
        abort(400)
    return os.path.join(CACHE_DIR, f'{kind}_{token}.lmdb')


def _clip_id(match):
    return f"{match['video_path']}|{match['start_frame']}|{match['end_frame']}"


def _result_page(store, sort, offset, limit):
    sort_key = sort if sort in {
        'time_asc', 'time_desc', 'similarity_desc', 'similarity_asc',
        'duration_desc', 'duration_asc',
    } else 'similarity_desc'
    limit, offset = max(1, min(200, int(limit))), max(0, int(offset))
    return {
        'matches': store.page(sort_key, offset, limit),
        'total': store.count(),
    }


def _basket_key(clip):
    return 'basket:' + str(clip)


def _tag_prefix(clip):
    return 'tag:' + str(clip) + ':'


def _find_match_by_clip(store, clip):
    for identifier, match, _, _ in store.iter_inserted():
        if _clip_id(match) == clip:
            return identifier, match
    return None, None


def _basket_page(cache, export, offset, limit=50):
    import heapq

    offset = max(0, int(offset))
    limit = max(1, min(200, int(limit)))
    def ordered_entries():
        for key, record in cache.iter_items('basket:'):
            clip = key[len('basket:'):]
            order = (int(record.get('added', 0)), clip) if export else (-int(record.get('updated', 0)), clip)
            yield order, clip, record

    selected = heapq.nsmallest(offset + limit, ordered_entries(), key=lambda entry: entry[0])
    selected = selected[offset:]
    return [{'clip_id': clip, 'match': record['data']} for _, clip, record in selected]


@app.route('/search_results/<token>', methods=['GET', 'DELETE'])
def search_results(token):
    path = _result_path(token)
    if not os.path.isdir(path):
        abort(404)
    if request.method == 'DELETE':
        RankedResults(path).destroy()
        return jsonify(success=True)
    store = RankedResults(path)
    try:
        return jsonify(_result_page(store, request.args.get('sort'), request.args.get('offset', 0), request.args.get('limit', 50)))
    finally:
        store.close()


@app.post('/candidates/<token>')
def candidates(token):
    data = request.get_json() or {}
    root = os.path.join(os.path.dirname(CACHE_DIR), 'search_baskets')
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, os.path.basename(_result_path(token, 'basket')))
    cache = LMDBCache(path)
    active_store = None
    try:
        ids = data.get('visible', [])
        if not isinstance(ids, list) or len(ids) > 200:
            abort(400)
        active = data.get('search_id')
        if active:
            result = _result_path(active)
            if not os.path.isdir(result):
                abort(404)
            active_store = RankedResults(result)
        action = data.get('action', 'status')
        if action in ('all', 'add', 'none') and active_store is None:
            abort(400)
        tag = str(data.get('prompt', '').strip() or '未命名')
        now = time.time_ns()

        def add_match(match):
            clip = _clip_id(match)
            key = _basket_key(clip)
            old = cache.get(key)
            if old is None or float(match.get('similarity', 0) or 0) >= float(old.get('score', 0) or 0):
                record = {
                    'data': match,
                    'score': float(match.get('similarity', 0) or 0),
                    'added': int(old.get('added', now) if old else now),
                    'updated': now,
                }
                cache.put(key, record)
            else:
                old['updated'] = now
                cache.put(key, old)
            cache.put(_tag_prefix(clip) + tag, True)

        if action == 'all':
            for _, match in active_store.iter_records('similarity_desc'):
                add_match(match)
        elif action == 'add':
            _, match = _find_match_by_clip(active_store, str(data.get('clip', '')))
            if match is not None:
                add_match(match)
        elif action == 'remove':
            clip = str(data.get('clip', ''))
            cache.delete(_basket_key(clip))
            cache.clear_prefix(_tag_prefix(clip))
        elif action == 'clear':
            cache.clear_prefix('basket:')
            cache.clear_prefix('tag:')
        elif action == 'none':
            for _, match in active_store.iter_records('similarity_desc'):
                clip = _clip_id(match)
                cache.delete(_basket_key(clip))
                cache.clear_prefix(_tag_prefix(clip))
        elif action != 'status':
            abort(400)

        total = cache.count_prefix('basket:')
        selected = 0
        if active_store is not None:
            for _, match in active_store.iter_records('similarity_desc'):
                if cache.get(_basket_key(_clip_id(match))) is not None:
                    selected += 1
        offset = max(0, int(data.get('offset', 0)))
        if not data.get('export'):
            offset = min(offset, max(0, ((total - 1) // 50) * 50))
        page = _basket_page(cache, bool(data.get('export')), offset)
        visible = set(str(value) for value in ids)
        visible.update(item['clip_id'] for item in page)
        tags = {}
        for clip in visible:
            values = [key[len(_tag_prefix(clip)):]
                      for key, _ in cache.iter_raw(_tag_prefix(clip))]
            if values:
                tags[clip] = values
        return jsonify(total=total, selected=selected, page=page, tags=tags, offset=offset)
    finally:
        if active_store is not None:
            active_store.close()
        cache.close()


@app.route('/search', methods=['POST'])
def search_similar_scenes():
    try:
        data = request.json or {}
        text_query = data.get('query')
        if not text_query:
            return jsonify({'success': False, 'error': '搜索描述不能为空'}), 400
        index_paths_raw = data.get('indices', '')
        index_paths = [resolve_path(line) for line in index_paths_raw.splitlines() if line.strip()]
        if not index_paths:
            return jsonify({'success': False, 'error': '请选择至少一个 Lance 索引'}), 400
        threshold = float(data.get('threshold', 20.0))
        truncate_dim = data.get('truncate_dim')
        if truncate_dim is not None:
            truncate_dim = int(truncate_dim)
        use_fp16 = data.get('use_fp16', False)
        use_jev_rerank = bool(data.get('use_jev_rerank', False))
        use_aesthetic_score = data.get('use_aesthetic_score') is True
        try:
            jev_top_k = max(1, int(data.get('jev_rerank_top_k', 10)))
        except (TypeError, ValueError):
            jev_top_k = 10
        try:
            jev_threshold = float(data.get('jev_rerank_threshold', 0.6))
        except (TypeError, ValueError):
            jev_threshold = 0.6
        use_dedup = bool(data.get('use_dedup', False))
        try:
            dedup_threshold = float(data.get('dedup_threshold', 0.95))
        except Exception:
            dedup_threshold = 0.95

        model_names = {extract_model_name_from_index(path) or 'unknown' for path in index_paths}
        if len(model_names) > 1:
            return jsonify({'success': False, 'error': f'所选索引来自不同向量模型: {", ".join(model_names)}'}), 400
        model_name = next(iter(model_names), None)
        if not model_name or model_name == 'unknown':
            return jsonify({'success': False, 'error': '无法从索引名称识别向量模型'}), 400
        print(f'[Info] 搜索 {len(index_paths)} 个索引，模型为 {model_name}')
        processor, search_engine = get_or_create_processor(model_name, truncate_dim, use_fp16)
        batch_size = max(1, min(int(data.get('scene_batch_size', 128)), 1024))
        token = uuid4().hex
        path = _result_path(token)
        hits = RankedResults(path)
        rerank_info = None
        rerank_used = False
        try:
            try:
                start_offset = int(data.get('start_frame_offset', -2))
                end_offset = int(data.get('end_frame_offset', 2))
            except (ValueError, TypeError):
                start_offset, end_offset = -2, 2

            pending = []
            for match, vectors in search_engine.iter_text_matches(text_query, index_paths, threshold, batch_size):
                if start_offset or end_offset:
                    sf, ef = int(match.get('start_frame', 0) or 0), int(match.get('end_frame', 0) or 0)
                    match = dict(match)
                    match.update(original_start_frame=sf, original_end_frame=ef)
                    match['start_frame'] = max(0, sf + start_offset)
                    match['end_frame'] = max(match['start_frame'] + 1, ef + end_offset)
                pending.append((match, match.get('source_lance', ''), is_op_ed_video(match.get('video_name') or match.get('video_path')), vectors if use_dedup else None))
                if len(pending) >= 256:
                    hits.add_many(pending)
                    pending.clear()
            if pending:
                hits.add_many(pending)

            if (use_jev_rerank or use_aesthetic_score) and hits.count():
                del processor, search_engine
                release_clip_model()
                shortlist = hits.shortlist(jev_top_k) if use_jev_rerank else hits.iter_records('similarity_desc')
                verified = []
                for identifier, match in evaluate_text_matches(
                    shortlist,
                    text_query,
                    rerank=use_jev_rerank,
                    aesthetics=use_aesthetic_score,
                    threshold=jev_threshold,
                    batch_size=min(batch_size, 256),
                ):
                    hits.update(identifier, match)
                    if use_jev_rerank:
                        verified.append(identifier)
                if use_jev_rerank:
                    hits.delete_except(verified, batch_size=batch_size)
                    rerank_used = True
                    rerank_info = {'type': 'Jev-rerank', 'kept': len(verified), 'verified': len(verified)}

            below = (
                identifier for identifier, match in hits.iter_records('__inserted__')
                if float(match.get('similarity', 0) or 0) < threshold
            )
            hits.delete_ids(below, batch_size=batch_size)
            dedup_info = None
            if use_dedup:
                before = hits.count()
                hits.deduplicate(dedup_threshold, batch_size)
                after = hits.count()
                dedup_info = {'before': before, 'after': after, 'removed': before - after}
            page = _result_page(hits, 'similarity_desc', 0, data.get('page_size', 50))
            hits.close()
            return jsonify(success=True, search_id=token, **page, rerank_used=rerank_used,
                           rerank_info=rerank_info, dedup_info=dedup_info, model_used=model_name)
        except BaseException:
            hits.destroy()
            raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get_preview')
def get_preview():
    video_path = request.args.get('video_path')
    frame = request.args.get('frame', type=int)
    
    if frame is None:
        return "Missing frame parameter", 400
    
    matched = request.args.get('matched') == '1'
    os.makedirs(PREVIEW_FOLDER, exist_ok=True)
    unique_id = f"{video_path}_{frame}_{matched}".encode('utf-8')
    filename = hashlib.md5(unique_id).hexdigest() + ".jpg"
    output_path = os.path.join(PREVIEW_FOLDER, filename)
    
    if not os.path.exists(output_path):
        if matched:
            with extracted_frames([(video_path, frame)], PREVIEW_FOLDER) as paths:
                image_path = paths.get((video_path, frame))
                if not image_path:
                    return "Failed to extract matched frame", 500
                os.replace(image_path, output_path)
        else:
            fps = _preview_fps(video_path)
            if not extract_single_frame(video_path, frame, fps, output_path):
                return "Failed to extract frame", 500
    
    return send_from_directory(PREVIEW_FOLDER, filename)


@app.route('/get_clip')
def get_clip():
    """
    切割视频片段并返回（使用 copy 模式快速切割）
    参数:
        video_path: 源视频路径
        start_frame: 起始帧
        end_frame: 结束帧
    返回:
        视频文件流
    """
    if not _FFMPEG_CUTTER_AVAILABLE:
        return jsonify({"success": False, "error": "FFmpeg cutter not available"}), 500
    
    video_path = request.args.get('video_path')
    start_frame = request.args.get('start_frame', type=int)
    end_frame = request.args.get('end_frame', type=int)
    
    if not video_path or start_frame is None or end_frame is None:
        return "Missing parameters (video_path, start_frame, end_frame)", 400
    
    if not os.path.exists(video_path):
        return f"Video file not found: {video_path}", 404
    
    # 生成唯一的缓存文件名
    unique_id = f"{video_path}_{start_frame}_{end_frame}".encode('utf-8')
    clip_hash = hashlib.md5(unique_id).hexdigest()
    clip_filename = f"{clip_hash}.mp4"
    clip_path = os.path.join(CLIPS_FOLDER, clip_filename)
    
    # 如果缓存存在，直接返回
    if os.path.exists(clip_path):
        return send_from_directory(CLIPS_FOLDER, clip_filename, mimetype='video/mp4')
    
    try:
        # 使用 export_video_clip 导出视频片段（copy模式，尾帧+2）
        success = export_video_clip(
            input_video=video_path,
            output_file=clip_path,
            start_frame=start_frame,
            end_frame=end_frame,
            copy_mode=True,
            end_frame_offset=2  # 尾帧+2
        )
        
        if success and os.path.exists(clip_path):
            return send_from_directory(CLIPS_FOLDER, clip_filename, mimetype='video/mp4')
        else:
            return "Failed to cut video clip", 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error cutting video: {str(e)}", 500


# 导出目录
EXPORTS_FOLDER = _base_resolver.join_str('output')
os.makedirs(EXPORTS_FOLDER, exist_ok=True)


@app.route('/export_clips', methods=['POST'])
def export_clips():
    """
    导出选中的视频片段
    参数:
        clips: 片段列表，每个包含 video_path, start_frame, end_frame
        mode: 'copy' 或 'precision'
        output_dir: 输出目录（可选）
        start_frame_offset: 起始帧偏移量（可选，默认-2）
        end_frame_offset: 结束帧偏移量（可选，默认2）
    返回:
        success: 是否成功
        exported_count: 导出数量
        output_dir: 输出目录
    """
    if not _FFMPEG_CUTTER_AVAILABLE:
        return jsonify({"success": False, "error": "FFmpeg cutter not available"}), 500
    
    try:
        data = request.json
        clips = data.get('clips', [])
        mode = data.get('mode', 'copy')  # 'copy' 或 'precision'
        output_dir = data.get('output_dir', '').strip()
        prompt = data.get('prompt', '').strip()
        
        # 获取帧偏移参数（从前端传入，有默认值）
        start_frame_offset = data.get('start_frame_offset', -2)
        end_frame_offset = data.get('end_frame_offset', 2)
        
        # 确保是整数
        try:
            start_frame_offset = int(start_frame_offset)
            end_frame_offset = int(end_frame_offset)
        except (ValueError, TypeError):
            start_frame_offset = -2
            end_frame_offset = 2
        
        if not clips:
            return jsonify({"success": False, "error": "没有选中的片段"}), 400
        
        # 确定输出目录
        if output_dir:
            output_dir = resolve_path(output_dir)
            if not os.path.isabs(output_dir):
                output_dir = _base_resolver.join_str(output_dir)
        else:
            # 使用 prompt 作为子文件夹名，无 prompt 则用时间戳
            if prompt:
                # 清理 prompt 中不合法的文件名字符，截断过长名称
                safe_prompt = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', prompt)[:80].strip('_. ')
                folder_name = safe_prompt if safe_prompt else 'export'
            else:
                import datetime
                folder_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(EXPORTS_FOLDER, folder_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 根据模式设置切割参数
        copy_mode = (mode == 'copy')
        
        exported_count = 0
        errors = []
        
        for i, clip in enumerate(clips):
            video_path = clip.get('video_path')
            start_frame = clip.get('start_frame')
            end_frame = clip.get('end_frame')
            
            if not video_path or start_frame is None or end_frame is None:
                errors.append(f"片段 {i+1}: 参数不完整")
                continue
            
            if not os.path.exists(video_path):
                errors.append(f"片段 {i+1}: 视频文件不存在")
                continue
            
            # 生成输出文件名: startFrame_剧名_剧集_.ext
            ext = os.path.splitext(os.path.basename(video_path))[1] or '.mp4'
            if _video_name_parser:
                try:
                    parsed_name = _video_name_parser.parse_filename(video_path)  # 剧名_剧集
                except Exception:
                    parsed_name = os.path.splitext(os.path.basename(video_path))[0]
            else:
                parsed_name = os.path.splitext(os.path.basename(video_path))[0]
            scores = score_prefix(float(clip['similarity']), clip) if 'aesthetic_score' in clip else ''
            output_filename = f"{scores}{start_frame}_{parsed_name}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            # 避免文件名冲突
            counter = 1
            while os.path.exists(output_path):
                output_filename = f"{scores}{start_frame}_{parsed_name}_{counter}{ext}"
                output_path = os.path.join(output_dir, output_filename)
                counter += 1
            
            try:
                # 使用 export_video_clip 导出视频片段
                # 使用前端传入的帧偏移参数
                if copy_mode:
                    # Copy模式：只使用结束帧偏移
                    success = export_video_clip(
                        input_video=video_path,
                        output_file=output_path,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        copy_mode=True,
                        end_frame_offset=end_frame_offset
                    )
                else:
                    # 精确模式：使用起始帧和结束帧偏移
                    success = export_video_clip(
                        input_video=video_path,
                        output_file=output_path,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        copy_mode=False,
                        start_frame_offset=start_frame_offset,
                        end_frame_offset=end_frame_offset
                    )
                
                if success and os.path.exists(output_path):
                    exported_count += 1
                    print(f"[Export] ✅ {output_filename}")
                else:
                    errors.append(f"片段 {i+1}: 切割失败")
            except Exception as e:
                errors.append(f"片段 {i+1}: {str(e)}")
        
        result = {
            "success": True,
            "exported_count": exported_count,
            "total_count": len(clips),
            "output_dir": output_dir,
            "mode": mode,
            "start_frame_offset": start_frame_offset,
            "end_frame_offset": end_frame_offset
        }
        
        if errors:
            result["errors"] = errors
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def _load_project_config():
    """读取项目 config.json"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_project_config(config):
    """保存项目 config.json"""
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


@app.route('/load_ffmpeg_config')
def load_ffmpeg_config():
    """从 config.json 读取 video_output 配置"""
    try:
        config = _load_project_config()
        video_output = config.get('video_output', {})
        return jsonify({"success": True, "video_output": video_output})
    except FileNotFoundError:
        return jsonify({"success": True, "video_output": {}})
    except Exception as e:
        return jsonify({"success": False, "message": f"读取配置失败: {str(e)}"}), 500


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


@app.route('/browse')
def browse():
    path = request.args.get('path', BASE_DIRECTORY)
    current_path = resolve_path(unquote(path))
    
    if os.path.isfile(current_path):
        current_path = os.path.dirname(current_path)
    if not os.path.exists(current_path) or not os.path.isdir(current_path):
        current_path = BASE_DIRECTORY
    
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
        initial_dir = data.get('initial_dir', BASE_DIRECTORY)
        if initial_dir and not os.path.isdir(initial_dir):
            initial_dir = BASE_DIRECTORY

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


if __name__ == '__main__':
    # 启动时清理残留文件
    startup_cleanup()
    
    # 注册退出时清理
    atexit.register(cleanup_temp_folders)
    
    print("=" * 60)
    print("  视频语义搜索工具 v3.9 (CLIP/FG-CLIP2)")
    print("  使用模型原生 logit_scale 计算相似度")
    print("  v3.9: 搜索和重排序分步执行，避免同时加载两个模型")
    print(f"  模型目录: {MODELS_DIR}")
    print(f"  临时目录: {TEMP_DIR}")
    print(f"    ├── previews/      (搜索预览图)")
    print(f"    └── search_cache/  (特征缓存)")
    print("  访问: http://127.0.0.1:5006")
    print("=" * 60)
    
    app.run(host=os.environ.get('ASE_WEB_HOST') or '127.0.0.1', port=5006, debug=False)
