# text_search 执行逻辑（基于 `text_search.py` 当前实现）

更新时间：2026-02-16
代码依据：`Cut_DetectScene/text_search.py`

## 1. 入口与调度链路

1. 前端访问 `GET /`，后端返回 `Frame_text_search.html` 页面（`text_search.py:394-409`）。
2. 用户点击搜索后，前端调用 `POST /search`，提交检索参数（JSON）。
3. 后端执行 `search_similar_scenes()`（`text_search.py:1009-1245`）：
   - 参数校验
   - 多索引模型一致性检查（从 Lance 索引目录名提取模型名）
   - 文本召回（CLIP / FG-CLIP2）
   - 可选 Reranker 重排（Qwen3-VL-Reranker-2B）
   - 可选跨视频去重
4. 后端返回 `matches` 结果给前端展示；前端可继续调用：
   - `GET /get_preview` 获取缩略帧
   - `GET /get_clip` 获取临时预览片段
   - `POST /export_clips` 批量导出片段

## 2. 搜索执行步骤

`search_similar_scenes()`（`text_search.py:1009-1245`）的实际流程如下：

1. 读取 `request.json`，校验 `query` 和 `indices`。
2. 解析阈值和运行参数（`threshold`、`truncate_dim`、`use_fp16` 等）。
3. 解析重排参数（`use_reranker`、`rerank_top_k`、`rerank_batch_size`、`rerank_weight`、`reranker_output_resolution`）。Reranker 模型路径固定为 `DEFAULT_RERANKER_PATH`（`text_search.py:73`）。
4. 解析去重参数（`use_dedup`、`dedup_threshold`）。
5. 从每个 Lance 索引目录名提取模型名（`extract_model_name_from_index()`，`text_search.py:698-719`），要求多索引必须来自同一模型，否则直接返回 400。模型名提取支持两种策略：优先从 `models/` 目录反推匹配（`_match_model_name_from_models_dir()`），回退到按下划线分割。
6. 调用 `get_or_create_processor(model_name, truncate_dim, use_fp16)`（`text_search.py:284-348`）加载或复用嵌入模型与搜索引擎。模型切换时自动清理旧模型缓存并释放显存。
7. 遍历每个索引调用 `search_engine.search_by_text(...)` 做初筛召回，并附加 `source_lance` 字段。若启用 Reranker，初筛阈值自动降低为 `max(5, threshold * 0.5)` 以召回更多候选。
8. 合并所有召回结果，按 `similarity` 降序排序。
9. 若启用重排：先释放 CLIP 显存（`release_clip_model()`），再调用 Reranker 对前 `rerank_top_k` 结果重打分并融合分数。重排完成后释放 Reranker 模型（`release_reranker_model()`）。
10. 应用最终阈值过滤：只保留 `similarity >= threshold` 的结果。
11. 若启用去重：按 `source_lance` 分组加载命中场景向量（`load_scene_features_from_lance()`，`text_search.py:972-1004`），调用 `deduplicate_text_search_results(...)` 做跨视频去重。
12. 返回统一 JSON 结构（含 `matches`、`rerank_info`、`dedup_info`、`model_used`）。

## 3. 参数映射（请求 JSON -> `search_similar_scenes`）

### 3.1 检索输入

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `query` | 无（必填） | 文本检索描述，缺失直接返回 400 |
| `indices` | `''` | 索引路径字符串，按换行拆分为多个 Lance 索引路径，每行经 `resolve_path()` 解析为绝对路径 |
| `threshold` | `20.0` | 最终相似度阈值，重排后仍以该阈值做最终过滤 |
| `truncate_dim` | `None` | 向量截断维度，传入时会转换为 `int` |
| `use_fp16` | `False` | 嵌入模型推理时是否启用 FP16 |

### 3.2 重排参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `use_reranker` | `False` | 是否启用二阶段重排 |
| `rerank_top_k` | `50` | 参与重排的候选数（只处理前 K 条） |
| `rerank_batch_size` | `4` | Reranker 批大小 |
| `rerank_weight` | `0.6` | 融合权重：`final = clip*(1-w) + rerank*100*w` |
| `reranker_output_resolution` | `DEFAULT_OUTPUT_RESOLUTION`（默认 `'384'`） | 重排抽帧分辨率，强制转字符串 |
| `rerank_model_path` | 固定 `DEFAULT_RERANKER_PATH` | 后端固定路径，不接受前端自定义模型路径 |

### 3.3 去重参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `use_dedup` | `False` | 是否启用跨视频去重 |
| `dedup_threshold` | `0.95` | 场景向量去重阈值（余弦相似度） |

## 4. 关键子逻辑

### 4.1 模型检测与复用

| 函数 | 位置 | 作用 |
| --- | --- | --- |
| `detect_model_type(...)` | `text_search.py:245-281` | 自动识别 `clip` / `fgclip2` 等模型类型。检查顺序：文件夹名 → `open_clip_config.json` → `config.json` 中的 `architectures` / `auto_map` |
| `extract_model_name_from_index(...)` | `text_search.py:698-719` | 从 Lance 索引目录名提取模型名。先用 `_match_model_name_from_models_dir()` 反推，回退到按下划线分割 |
| `get_or_create_processor(...)` | `text_search.py:284-348` | 创建或复用嵌入模型（`EmbeddingModelProcessor`）与搜索引擎（`SemanticSearchEngine`）；模型切换时清理旧缓存并释放显存 |
| `release_clip_model()` | `text_search.py:151-175` | 释放 CLIP 模型和相关缓存，调用 `torch.cuda.empty_cache()` |
| `release_reranker_model()` | `text_search.py:178-200` | 释放 Reranker 模型和相关缓存 |
| `get_or_create_reranker(...)` | `text_search.py:203-242` | 获取或创建 Qwen3-VL Reranker 引擎，模型路径或分辨率变化时自动重建 |

### 4.2 重排流程（`text_search.py:1128-1205`）

1. 取初筛结果前 `rerank_top_k` 条。
2. 使用 `RerankerFrameExtractor`（`reranker_frame_extractor.py`）批量抽取每条结果对应帧图，缓存到 `RERANK_CACHE_DIR`。
3. 收集有效帧路径，组装 `query(text) + documents(images)` 调用 `reranker.process(inputs, batch_size=rerank_batch_size)`。
4. 融合分数：`final = clip_sim * (1 - rerank_weight) + rerank_score * 100.0 * rerank_weight`。
5. 重排序后只保留 `matches_to_rerank` 部分（未重排的尾部结果被丢弃）。
6. 重排完成后立即释放 Reranker 模型。

### 4.3 去重流程（`text_search.py:1210-1232`）

1. 按命中结果中的 `source_lance` 收集唯一 Lance 路径，调用 `load_scene_features_from_lance()` 按需加载场景特征向量（使用 `read_lance_index_for_reranker()`）。
2. `deduplicate_text_search_results()`（`text_search.py:862-969`）按 `source_lance` 分组，每组内：
   - 先执行 OP/ED 优先规则（`is_op_ed_video()`）：若组内有 OP/ED 视频，只保留 OP/ED 视频。
   - 否则执行跨视频矩阵相似度去重（`_deduplicate_vectors_matrix_cross_video()`，`text_search.py:817-859`）：仅比较不同视频之间的场景，同视频内不去重。
3. 返回去重统计：`before`、`after`、`removed`。

## 5. 返回结构与错误处理

### 5.1 成功返回（200）

| 字段 | 说明 |
| --- | --- |
| `success` | 固定 `True` |
| `matches` | 结果列表（每项含视频路径、帧区间、相似度等） |
| `rerank_used` | 是否实际执行了重排 |
| `rerank_info` | 重排统计（`reranked`、`total`） |
| `dedup_info` | 去重统计（`before`、`after`、`removed`） |
| `model_used` | 实际使用的模型名（从索引文件名推断） |

### 5.2 失败返回

| 场景 | 状态码 | 说明 |
| --- | --- | --- |
| 参数缺失（如 `query`、`indices`） | `400` | 直接返回错误描述 |
| 多索引模型不一致 | `400` | 禁止混用不同模型索引 |
| 运行期异常 | `500` | 返回异常信息字符串 |

## 6. 相关接口（同模块）

| 接口 | 作用 |
| --- | --- |
| `POST /clear_cache` | 清理 `text_search.py` 自身的临时子目录（`previews/`、`rerank_cache/`、`clips/`、`search_cache/`）并释放 CLIP 和 Reranker 模型 |
| `GET /list_models` | 列出 `models/` 目录中的可用模型（含 Reranker），返回自动识别类型 |
| `POST /get_model_info` | 返回模型基础信息（不主动加载大模型，`logit_scale` 在搜索时才获取） |
| `GET /get_preview` | 提取/缓存指定帧图片（MD5 哈希命名，缓存到 `PREVIEW_FOLDER`） |
| `GET /get_clip` | 临时切出短视频片段供前端播放（缓存到 `CLIPS_FOLDER`） |
| `POST /export_clips` | copy / precision 两种模式批量导出，支持自定义输出目录和文件名格式 |
| `GET /load_ffmpeg_config` | 读取 `config.json` 中 `video_output` 段 |
| `POST /save_ffmpeg_config` | 保存 FFmpeg 参数到 `config.json` |
| `GET /browse` | 文件/目录浏览接口（前端路径选择器使用） |
| `GET /list_presets` | 列出文本搜索预设（存储在 `preset/text_search/` 目录） |
| `GET /load_preset` | 加载指定文本搜索预设 |
| `POST /save_preset` | 保存文本搜索预设 |
| `POST /set_default_preset` | 设置默认文本搜索预设 |
| `POST /delete_preset` | 删除指定文本搜索预设 |
