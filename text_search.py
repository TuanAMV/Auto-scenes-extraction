# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
# text_search.py (v3.11 - CLIP/FG-CLIP2)
# 支持从models文件夹动态加载模型
# v3.11: 移除 Qwen3-VL-Embedding 支持，仅保留 CLIP/FG-CLIP2

import os
import atexit
import subprocess
import shutil
import gc
from A_coreUtils.lance_index_io import read_lance_index_for_reranker
import json
import time
import re
import numpy as np
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, send_from_directory
from urllib.parse import unquote
import hashlib

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
_RERANKER_AVAILABLE = False

try:
    from A_coreUtils.embedding.embedding_model import EmbeddingModelProcessor, SemanticSearchEngine as EmbeddingSearchEngine
    _EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    pass

try:
    from A_coreUtils.qwen_models.qwen3_vl_reranker import Qwen3VLReranker
    from A_coreUtils.search.reranker_frame_extractor import RerankerFrameExtractor
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False

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
DEFAULT_RERANKER_PATH = _base_resolver.join_str('models', 'Qwen3-VL-Reranker-2B')
PREVIEW_FOLDER = _base_resolver.join_str('temp', 'previews')
RERANK_CACHE_DIR = _base_resolver.join_str('temp', 'rerank_cache')
CLIPS_FOLDER = _base_resolver.join_str('temp', 'clips')  # 视频片段缓存目录
# 所有临时文件都放在 temp 下
PREVIEW_FOLDER = _base_resolver.join_str('temp', 'previews')
CACHE_DIR = _base_resolver.join_str('temp', 'search_cache')  # 重命名: faiss_cache -> search_cache

# 默认 output_resolution（从前端用户输入获取，此处仅作为默认值）
DEFAULT_OUTPUT_RESOLUTION = '384'

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
_fps_cache = {}
_current_model_name = None
_reranker_engine = None
_current_reranker_path = None
_current_rerank_resolution = None
_mini_reranker_analyzer = None


def release_clip_model():
    """释放 CLIP 模型，清理显存"""
    global _embedding_processor, _search_engine, _current_model_name
    if _embedding_processor is not None:
        print("[Info] 释放 CLIP 模型...")
        try:
            if hasattr(_embedding_processor, 'cleanup'):
                _embedding_processor.cleanup()
            if hasattr(_embedding_processor, 'release_models'):
                _embedding_processor.release_models()
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


def release_reranker_model():
    """释放 Reranker 模型，清理显存"""
    global _reranker_engine, _current_reranker_path, _current_rerank_resolution
    if _reranker_engine is not None:
        print("[Info] 释放 Reranker 模型...")
        try:
            # 调用 Qwen3VLReranker 的 cleanup 方法
            if hasattr(_reranker_engine, 'cleanup'):
                _reranker_engine.cleanup()
            del _reranker_engine
        except Exception as e:
            print(f"[Warning] 释放 Reranker 模型时出错: {e}")
        _reranker_engine = None
        _current_reranker_path = None
        _current_rerank_resolution = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        print("[Info] Reranker 模型已释放")


# ============================================================================
# MiniCPM Mini-Reranker (Qwen Reranker 平替)
# ============================================================================

def get_or_create_mini_reranker():
    """Get or create the MiniCPM mini-reranker (ShotAnalyzer 单例)."""
    global _mini_reranker_analyzer
    if _mini_reranker_analyzer is None:
        from A_coreUtils.aftertreatment.shot_analyzer import ShotAnalyzer
        _mini_reranker_analyzer = ShotAnalyzer()
        _mini_reranker_analyzer.load_model()
        print("[Info] MiniCPM mini-reranker 模型已加载")
    return _mini_reranker_analyzer


def release_mini_reranker_model():
    """释放 MiniCPM 模型显存"""
    global _mini_reranker_analyzer
    if _mini_reranker_analyzer is not None:
        _mini_reranker_analyzer.unload_model()
        _mini_reranker_analyzer = None
        gc.collect()
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except ImportError:
            pass
        print("[Info] MiniCPM mini-reranker 模型已释放")


def mini_rerank_text_results(matches: list, query_text: str, threshold: float = 0.6, top_k: int = 50) -> list:
    """
    MiniCPM Mini-Rerank: 用 MiniCPM-V-4-6 对 CLIP 搜索结果逐帧判断与查询描述的相似度。
    
    Args:
        matches: CLIP 搜索结果列表
        query_text: 用户输入的查询文本
        threshold: 相似度阈值 (0.0-1.0)，低于此值的结果将被移除
        top_k: 需要验证的 Top-K 结果数量
    
    Returns:
        过滤后的结果列表
    """
    import cv2
    import torch as _t
    
    if not matches:
        return matches
    
    matches_to_check = matches[:min(top_k, len(matches))]
    
    analyzer = get_or_create_mini_reranker()
    
    from A_coreUtils.search.reranker_frame_extractor import RerankerFrameExtractor
    frame_extractor = RerankerFrameExtractor(
        output_resolution='384', cache_dir=RERANK_CACHE_DIR,
    )
    
    # 构建提取场景列表
    scenes = []
    for match in matches_to_check:
        video_path = match.get('video_path', '')
        start_frame = match.get('start_frame', 0)
        scenes.append({
            'video_path': video_path,
            'start_frame': start_frame,
            'end_frame': start_frame,
            'fps': VideoMetaHelper.get_fps_cached(video_path, _fps_cache) if video_path else 25.0,
        })
    
    try:
        frame_paths = frame_extractor.extract_batch(scenes)
    except Exception as e:
        print(f"[MiniRerank] 帧提取异常: {e}")
        frame_paths = {}
    
    # MiniCPM 提示词 — JSON 格式输出
    prompt = (
        f'判断这张图片与以下描述的画面相似度，只输出一行JSON不要解释:\n'
        f'{{"相似度": <一个0到1之间的小数, 例如0.85>}}\n'
        f'描述: {query_text}'
    )
    
    kept = []
    removed_count = 0
    total = len(matches_to_check)
    
    for idx, match in enumerate(matches_to_check):
        video_path = match.get('video_path', '')
        start_frame = match.get('start_frame', 0)
        video_name = os.path.basename(video_path)
        frame_key = f"{start_frame}_1_{video_name}"
        frame_path = frame_paths.get(frame_key)
        
        if not frame_path or not os.path.exists(frame_path):
            kept.append(match)
            continue
        
        frame = cv2.imread(frame_path)
        if frame is None:
            kept.append(match)
            continue
        
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_img = Image.fromarray(img_rgb)
            
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ]}]
            
            inputs = analyzer._processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(analyzer.device)
            
            with _t.no_grad():
                generated_ids = analyzer._model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    eos_token_id=analyzer._processor.tokenizer.eos_token_id,
                )
            response_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
            text = analyzer._processor.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            
            # 从 JSON 解析相似度分数
            score = 0.0
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                try:
                    import json as _json
                    parsed = _json.loads(text[start:end + 1])
                    raw_val = parsed.get("相似度", 0)
                    score = float(raw_val)
                except Exception:
                    pass
            
            # 归一化：如果模型输出了百分比（如85），转为小数（0.85）
            raw_score = score
            if score > 1.0:
                score = score / 100.0
            
            if score >= threshold:
                match['mini_rerank_score'] = score
                boost = score * 50
                match['similarity'] = match.get('similarity', 0) + boost
                kept.append(match)
            else:
                removed_count += 1
        except Exception as e:
            print(f"  [MiniRerank] 处理异常: {e}")
            kept.append(match)
    
    # 清理帧提取缓存
    try:
        frame_extractor.cleanup(remove_files=True)
    except Exception:
        pass
    
    print(f"[MiniRerank] 完成: {total}条检查 → 保留 {len(kept)}条 (阈值={threshold})")
    return kept


def get_or_create_reranker(model_path: str, output_resolution: str = '384'):
    """Get or create the Qwen3-VL reranker engine."""
    global _reranker_engine, _current_reranker_path, _current_rerank_resolution
    if not _RERANKER_AVAILABLE:
        raise RuntimeError("Qwen3-VL Reranker module is not available")

    if not model_path:
        model_path = DEFAULT_RERANKER_PATH

    model_path = resolve_path(model_path)
    if not os.path.isabs(model_path):
        model_path = _base_resolver.join_str(model_path)

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Reranker model path not found: {model_path}")

    # 如果模型路径或分辨率变化，重新创建引擎
    if _reranker_engine is None or _current_reranker_path != model_path or _current_rerank_resolution != output_resolution:
        if _reranker_engine is not None:
            try:
                del _reranker_engine
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception:
                pass

        _reranker_engine = Qwen3VLReranker(
            model_name_or_path=model_path,
            cache_dir=RERANK_CACHE_DIR,
            torch_dtype="auto"
        )
        _current_reranker_path = model_path
        _current_rerank_resolution = output_resolution

    return _reranker_engine


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
_TEXT_SEARCH_TEMP_SUBDIRS = [PREVIEW_FOLDER, RERANK_CACHE_DIR, CLIPS_FOLDER, CACHE_DIR]


def cleanup_temp_folders():
    """清理 text_search.py 自身的临时文件夹（保留 cache 目录下的搜索结果 LMDB）"""
    print("\n--- 清理临时文件夹... ---")
    for subdir in _TEXT_SEARCH_TEMP_SUBDIRS:
        if os.path.exists(subdir):
            cleanup_temp_folder(subdir)
            print(f"  ✔ 已清理: {subdir}")
    # 清理 temp 根目录下的散落文件（不递归进子目录）
    if os.path.exists(TEMP_DIR):
        for item in os.listdir(TEMP_DIR):
            item_path = os.path.join(TEMP_DIR, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                try:
                    os.unlink(item_path)
                except (PermissionError, OSError):
                    pass
    print("清理完成")


def startup_cleanup():
    """启动时清理 text_search.py 自身的临时文件（保留 cache 目录下的搜索结果 LMDB）"""
    print("--- 启动时清理临时文件... ---")
    
    for subdir in _TEXT_SEARCH_TEMP_SUBDIRS:
        if os.path.exists(subdir):
            items = os.listdir(subdir)
            if items:
                print(f"  发现 {len(items)} 个残留项目在 {os.path.basename(subdir)}/，正在清理...")
                cleanup_temp_folder(subdir)
                print(f"  ✔ 已清理: {subdir}")
    
    # 确保文件夹存在
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PREVIEW_FOLDER, exist_ok=True)
    
    print("启动清理完成")


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
        release_reranker_model()
        release_mini_reranker_model()
        cleared_items.append("已释放模型")
        
        # 计算并清理 text_search.py 自身的临时子目录（保留 cache 目录下的搜索结果 LMDB）
        for subdir in _TEXT_SEARCH_TEMP_SUBDIRS:
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
        
        # 清理 FPS 缓存
        global _fps_cache
        _fps_cache.clear()
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
                    if is_valid:
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

def is_op_ed_video(video_name: str) -> bool:
    """
    检查视频名是否包含 OP 或 ED 字段（不分大小写）
    
    支持的格式：
    - 标准格式: op, ed, _op_, _ed_, op_, ed_, _op, _ed
    - 带编号: op01, ed01, op1, ed1, op_01, ed_01
    - NC版本: ncop, nced, ncop01, nced01
    - 混合格式: video_op01_xxx, video_ncop_xxx
    """
    import re
    if not video_name:
        return False
    
    name_lower = video_name.lower()
    
    patterns = [
        # NC版本（无字幕版）: ncop, nced, ncop01, nced01, ncop_01
        r'(?:^|_)nc(?:op|ed)(?:\d+|_\d+)?(?:_|$)',
        # 带编号的 OP/ED: op01, ed01, op1, ed1, op_01, ed_01
        r'(?:^|_)(?:op|ed)(?:\d+|_\d+)(?:_|$)',
        # 标准 OP/ED: _op_, _ed_, op_, ed_, _op, _ed
        r'(?:^|_)(?:op|ed)(?:_|$)',
    ]
    
    for pattern in patterns:
        if re.search(pattern, name_lower):
            return True
    return False


def _deduplicate_vectors_matrix(scene_vectors_list: list, threshold: float) -> list:
    """
    使用矩阵运算进行余弦相似度去重（支持多帧向量）
    
    比较两个场景时，计算所有帧向量之间的最大相似度。
    例如：场景A有3帧 [a1,a2,a3]，场景B有3帧 [b1,b2,b3]
    计算 3×3=9 个相似度，取最大值作为两个场景的相似度。
    
    Args:
        scene_vectors_list: 场景向量列表，每个元素是 [k, D] 的矩阵（k帧向量）
        threshold: 余弦相似度阈值
    
    Returns:
        保留的索引列表
    """
    n = len(scene_vectors_list)
    if n <= 1:
        return list(range(n))
    
    # 获取每个场景的帧数
    frame_counts = [v.shape[0] for v in scene_vectors_list]
    total_frames = sum(frame_counts)
    
    # 展平所有向量为 [total_frames, D]
    all_vectors = np.vstack(scene_vectors_list).astype(np.float16)
    
    # 向量已经归一化，直接计算相似度矩阵 [total_frames, total_frames]
    similarity_matrix = all_vectors @ all_vectors.T
    
    # 构建场景边界索引
    scene_boundaries = [0]
    for count in frame_counts:
        scene_boundaries.append(scene_boundaries[-1] + count)
    
    # 计算场景间最大相似度矩阵 [N, N]
    scene_max_sim = np.zeros((n, n), dtype=np.float16)
    for i in range(n):
        i_start, i_end = scene_boundaries[i], scene_boundaries[i+1]
        for j in range(i, n):
            j_start, j_end = scene_boundaries[j], scene_boundaries[j+1]
            # 取场景 i 和场景 j 之间所有帧向量的最大相似度
            block = similarity_matrix[i_start:i_end, j_start:j_end]
            max_sim = block.max()
            scene_max_sim[i, j] = max_sim
            scene_max_sim[j, i] = max_sim
    
    # 贪心去重：遍历上三角
    keep_mask = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not keep_mask[i]:
            continue
        # 标记与 i 相似度超过阈值的后续场景为重复
        for j in range(i + 1, n):
            if keep_mask[j] and scene_max_sim[i, j] > threshold:
                keep_mask[j] = False
    
    return np.where(keep_mask)[0].tolist()


def _deduplicate_vectors_matrix_cross_video(
    scene_vectors_list: list,
    scene_video_ids: list,
    threshold: float
) -> list:
    """
    跨视频去重版本：仅比较不同视频之间的场景相似度，同视频场景互不去重。
    """
    n = len(scene_vectors_list)
    if n <= 1:
        return list(range(n))

    frame_counts = [vectors.shape[0] for vectors in scene_vectors_list]
    all_vectors = np.vstack(scene_vectors_list).astype(np.float16)
    similarity_matrix = all_vectors @ all_vectors.T

    scene_boundaries = [0]
    for count in frame_counts:
        scene_boundaries.append(scene_boundaries[-1] + count)

    scene_max_sim = np.zeros((n, n), dtype=np.float16)
    for i in range(n):
        i_start, i_end = scene_boundaries[i], scene_boundaries[i + 1]
        for j in range(i, n):
            j_start, j_end = scene_boundaries[j], scene_boundaries[j + 1]
            block = similarity_matrix[i_start:i_end, j_start:j_end]
            max_sim = block.max()
            scene_max_sim[i, j] = max_sim
            scene_max_sim[j, i] = max_sim

    keep_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, n):
            if not keep_mask[j]:
                continue
            if scene_video_ids[i] == scene_video_ids[j]:
                continue
            if scene_max_sim[i, j] > threshold:
                keep_mask[j] = False

    return np.where(keep_mask)[0].tolist()


def deduplicate_text_search_results(
    all_matches: list,
    scene_features: dict,
    similarity_threshold: float = 0.95
) -> list:
    """
    text_search 结果去重（跨视频去重）
    
    优先规则：
    - 如果同Lance内有 OP/ED 视频，只保留 OP/ED 视频
    - 否则进行向量相似度去重
    
    去重范围：
    - 按 source_lance 分组，只在同一Lance内进行去重
    - 同一Lance内仅跨视频去重，同视频内不去重
    
    Args:
        all_matches: 搜索结果列表
        scene_features: 场景特征向量 {scene_key: np.ndarray}
        similarity_threshold: 余弦相似度阈值（0-1），超过则去重
    
    Returns:
        去重后的结果列表
    """
    if not all_matches or similarity_threshold is None:
        return all_matches
    
    if not scene_features:
        print("[去重] 无场景特征向量，跳过去重")
        return all_matches
    
    # Step 1: 按 source_lance 分组
    groups = defaultdict(list)
    for idx, match in enumerate(all_matches):
        lance_path = match.get('source_lance', 'unknown')
        groups[lance_path].append((idx, match))
    
    # Step 2: 每组内去重
    keep_indices = set()
    op_ed_priority_count = 0
    vector_dedup_count = 0
    
    for lance_path, group_items in groups.items():
        if len(group_items) == 1:
            # 只有一个，直接保留
            keep_indices.add(group_items[0][0])
            continue
        
        # ========== OP/ED 优先规则 ==========
        op_ed_items = []
        non_op_ed_items = []
        
        for idx, match in group_items:
            video_name = match.get('video_name', '')
            if is_op_ed_video(video_name):
                op_ed_items.append((idx, match))
            else:
                non_op_ed_items.append((idx, match))
        
        # 如果有 OP/ED 视频，只保留 OP/ED 视频
        if op_ed_items:
            for idx, _ in op_ed_items:
                keep_indices.add(idx)
            op_ed_priority_count += len(non_op_ed_items)
            continue
        
        # ========== 向量相似度去重（场景级 + 跨视频） ==========
        scene_vectors_list = []
        valid_items = []
        valid_video_ids = []

        for idx, match in group_items:
            scene_key = f"{match.get('video_path', '')}_{match.get('start_frame', 0)}_{match.get('end_frame', 0)}"
            vectors = scene_features.get(scene_key)
            if vectors is None:
                # 无向量无法判重，直接保留
                keep_indices.add(idx)
                continue
            scene_vectors_list.append(vectors)
            valid_items.append((idx, match))
            valid_video_ids.append(match.get('video_path', '') or '__unknown_video__')

        if len(scene_vectors_list) <= 1:
            for idx, _ in valid_items:
                keep_indices.add(idx)
            continue

        # 矩阵运算去重（仅跨视频比较）
        before_count = len(valid_items)
        keep_local_indices = _deduplicate_vectors_matrix_cross_video(
            scene_vectors_list,
            valid_video_ids,
            similarity_threshold
        )
        after_count = len(keep_local_indices)
        vector_dedup_count += (before_count - after_count)
        
        # 映射回原始索引
        for local_idx in keep_local_indices:
            keep_indices.add(valid_items[local_idx][0])
    
    # 打印统计信息
    total_removed = len(all_matches) - len(keep_indices)
    print(f"[去重] OP/ED优先过滤: {op_ed_priority_count} 个, 向量相似度去重: {vector_dedup_count} 个")
    print(f"[去重] 总计去除: {total_removed} 个, 保留: {len(keep_indices)} 个")
    
    # Step 3: 返回去重后的结果（保持原顺序）
    return [all_matches[i] for i in sorted(keep_indices)]


def load_scene_features_from_lance(lance_path: str, matches: list) -> dict:
    """
    从 Lance 索引加载场景特征向量（用于去重）
    
    Args:
        lance_path: Lance 索引目录路径
        matches: 需要加载特征的匹配结果列表
    
    Returns:
        {scene_key: np.ndarray} 场景特征向量字典
    """
    scene_features = {}
    
    if not os.path.exists(lance_path):
        return scene_features
    
    try:
        # 构建需要的场景键集合
        needed_keys = set()
        for match in matches:
            if match.get('source_lance') == lance_path:
                scene_key = f"{match.get('video_path', '')}_{match.get('start_frame', 0)}_{match.get('end_frame', 0)}"
                needed_keys.add(scene_key)
        
        if not needed_keys:
            return scene_features
        
        # 从 Lance 索引按需读取
        scene_features = read_lance_index_for_reranker(lance_path, needed_keys)
    except Exception as e:
        print(f"[Warning] 加载 Lance 特征失败 {lance_path}: {e}")
    
    return scene_features




@app.route('/search', methods=['POST'])
def search_similar_scenes():
    try:
        data = request.json
        text_query = data.get('query')
        if not text_query:
            return jsonify({"success": False, "error": "搜索描述不能为空"}), 400
        
        index_paths_raw = data.get('indices', '')
        index_paths = [resolve_path(line) for line in index_paths_raw.splitlines() if line.strip()]
        
        if not index_paths:
            return jsonify({"success": False, "error": "请提供索引库路径"}), 400
        
        # ✅ 修改：从前端获取 threshold，传递给 similarity_threshold
        threshold = float(data.get('threshold', 20.0))
        
        truncate_dim = data.get('truncate_dim')
        if truncate_dim is not None:
            truncate_dim = int(truncate_dim)
        
        # 获取 FP16 设置（默认 False）
        use_fp16 = data.get('use_fp16', False)

        # Reranker params
        use_reranker = bool(data.get('use_reranker', False))
        rerank_top_k = data.get('rerank_top_k', 50)
        rerank_batch_size = data.get('rerank_batch_size', 4)
        rerank_weight = data.get('rerank_weight', 0.6)
        # 使用固定默认路径，前端不再允许用户手动输入
        rerank_model_path = DEFAULT_RERANKER_PATH
        reranker_output_resolution = data.get('reranker_output_resolution', DEFAULT_OUTPUT_RESOLUTION)

        # MiniRerank params (MiniCPM-V-4-6)
        use_mini_rerank = bool(data.get('use_mini_rerank', False))
        mini_rerank_threshold = rerank_weight  # reuse rerank_weight slider as threshold

        try:
            rerank_top_k = int(rerank_top_k)
        except Exception:
            rerank_top_k = 50

        try:
            rerank_batch_size = int(rerank_batch_size)
        except Exception:
            rerank_batch_size = 4

        try:
            rerank_weight = float(rerank_weight)
        except Exception:
            rerank_weight = 0.6
        
        # 确保 reranker_output_resolution 是字符串
        reranker_output_resolution = str(reranker_output_resolution)
        
        # ✅ v3.10: 去重参数
        use_dedup = bool(data.get('use_dedup', False))
        dedup_threshold = data.get('dedup_threshold', 0.95)
        try:
            dedup_threshold = float(dedup_threshold)
        except Exception:
            dedup_threshold = 0.95
        
        # ✅ v3.11: 检查多索引是否来自相同模型
        model_names = set()
        for index_path in index_paths:
            model_name = extract_model_name_from_index(index_path)
            if model_name:
                model_names.add(model_name)
            else:
                model_names.add("unknown")
        
        # 如果有多个不同模型，返回错误
        if len(model_names) > 1:
            return jsonify({
                "success": False,
                "error": f"多索引搜索要求所有索引来自相同模型，但检测到 {len(model_names)} 个不同模型: {', '.join(model_names)}"
            }), 400
        
        # 使用检测到的模型名
        model_name = list(model_names)[0] if model_names else None
        if not model_name or model_name == "unknown":
            return jsonify({"success": False, "error": "无法从索引文件名中提取模型名"}), 400
        
        print(f"[Info] 多索引搜索: {len(index_paths)} 个索引，模型: {model_name}")
        
        processor, search_engine = get_or_create_processor(model_name, truncate_dim, use_fp16)
        
        # 检测模型类型
        model_path = _base_resolver.join_str('models', model_name) if not os.path.isabs(model_name) else model_name
        model_type = detect_model_type(model_path)
        
        all_matches = []
        
        for index_path in index_paths:
            print(f"  - 搜索索引: {os.path.basename(index_path)}")
            
            # CLIP 系模型搜索
            initial_threshold = threshold
            if use_reranker and rerank_weight > 0 and _RERANKER_AVAILABLE:
                initial_threshold = max(5, threshold * 0.5)
            
            matches = search_engine.search_by_text(
                text_query,
                [index_path],
                similarity_threshold=initial_threshold
            )
            # ✅ v3.10: 为每个 match 添加 source_lance 字段
            for match in matches:
                match['source_lance'] = index_path
            all_matches.extend(matches)
        
        # 按相似度排序（基础召回）
        all_matches.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        # ✅ 按需释放 CLIP 模型，避免与 MiniRerank 同时占用显存
        need_release_clip = use_mini_rerank and all_matches
        if need_release_clip:
            release_clip_model()

        rerank_used = False
        rerank_info = None

        # ========================================================================
        # MiniCPM Mini-Rerank（优先，与 Qwen Reranker 互斥）
        # ========================================================================
        if use_mini_rerank and all_matches:
            try:
                all_matches = mini_rerank_text_results(
                    all_matches,
                    query_text=text_query,
                    threshold=mini_rerank_threshold,
                    top_k=rerank_top_k,
                )
                all_matches.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                rerank_used = True
                rerank_info = {"type": "mini_rerank", "kept": len(all_matches)}
            except Exception as e:
                print(f"[Warning] MiniRerank failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                release_mini_reranker_model()



        # ========================================================================
        # 帧偏移：直接调整搜索结果帧范围（保存原始值，展示偏移后的值）
        # ========================================================================
        start_frame_offset = data.get('start_frame_offset', -2)
        end_frame_offset = data.get('end_frame_offset', 2)
        try:
            start_frame_offset = int(start_frame_offset)
            end_frame_offset = int(end_frame_offset)
        except Exception:
            start_frame_offset = -2
            end_frame_offset = 2

        if start_frame_offset != 0 or end_frame_offset != 0:
            offset_info = {"start_offset": start_frame_offset, "end_offset": end_frame_offset}
            for m in all_matches:
                sf = int(m.get('start_frame', 0) or 0)
                ef = int(m.get('end_frame', sf + 1) or sf + 1)
                m['original_start_frame'] = sf
                m['original_end_frame'] = ef
                m['start_frame'] = max(0, sf + start_frame_offset)
                m['end_frame'] = max(m['start_frame'] + 1, ef + end_frame_offset)
        else:
            offset_info = None

        # ✅ v3.10: 最终阈值过滤 - 应用于 rerank 后的最终相似度
        all_matches = [m for m in all_matches if m.get('similarity', 0) >= threshold]

        # ✅ v3.10: 跨视频去重（如果启用）
        dedup_info = None
        if use_dedup and all_matches:
            print(f"[去重] 开始跨视频去重，阈值={dedup_threshold}")
            before_count = len(all_matches)
            
            # 加载场景特征向量
            scene_features = {}
            unique_lances = set(m.get('source_lance') for m in all_matches if m.get('source_lance'))
            for lance_path in unique_lances:
                lance_feats = load_scene_features_from_lance(lance_path, all_matches)
                scene_features.update(lance_feats)
            
            # 执行去重
            all_matches = deduplicate_text_search_results(
                all_matches,
                scene_features,
                similarity_threshold=dedup_threshold
            )
            
            after_count = len(all_matches)
            dedup_info = {"before": before_count, "after": after_count, "removed": before_count - after_count}
            print(f"[去重] 完成: {before_count} -> {after_count} (去除 {before_count - after_count} 个)")

        return jsonify({
            "success": True,
            "matches": all_matches,
            "rerank_used": rerank_used,
            "rerank_info": rerank_info,
            "dedup_info": dedup_info,
            "model_used": model_name
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/get_preview')
def get_preview():
    video_path = request.args.get('video_path')
    frame = request.args.get('frame', type=int)
    
    if frame is None:
        return "Missing frame parameter", 400
    
    fps = VideoMetaHelper.get_fps_cached(video_path, _fps_cache)
    
    os.makedirs(PREVIEW_FOLDER, exist_ok=True)
    unique_id = f"{video_path}_{frame}".encode('utf-8')
    filename = hashlib.md5(unique_id).hexdigest() + ".jpg"
    output_path = os.path.join(PREVIEW_FOLDER, filename)
    
    if not os.path.exists(output_path):
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
    
    # 获取视频 fps
    fps = VideoMetaHelper.get_fps_cached(video_path, _fps_cache)
    
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
            output_filename = f"{start_frame}_{parsed_name}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            # 避免文件名冲突
            counter = 1
            while os.path.exists(output_path):
                output_filename = f"{start_frame}_{parsed_name}_{counter}{ext}"
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
    
    app.run(host='0.0.0.0', port=5006, debug=False)
