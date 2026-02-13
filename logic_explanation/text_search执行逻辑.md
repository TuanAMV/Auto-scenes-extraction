# text_search 执行逻辑（基于 `text_search.py` 当前实现）

更新时间：2026-02-13  
代码依据：`Cut_DetectScene/text_search.py:288`、`Cut_DetectScene/text_search.py:1005`、`Cut_DetectScene/text_search.py:1329`

## 1. 入口与调度链路

1. 前端访问 `GET /`，后端返回 `Frame_text_search.html` 页面。
2. 用户点击搜索后，前端调用 `POST /search`，提交检索参数（JSON）。
3. 后端执行 `search_similar_scenes()`：
   - 参数校验
   - 多索引模型一致性检查
   - 文本召回（CLIP / FG-CLIP2）
   - 可选 Reranker 重排
   - 可选跨视频去重
4. 后端返回 `matches` 结果给前端展示；前端可继续调用：
   - `GET /get_preview` 获取缩略帧
   - `GET /get_clip` 获取临时预览片段
   - `POST /export_clips` 批量导出片段

## 2. 搜索执行步骤

`search_similar_scenes()` 的实际流程如下：

1. 读取 `request.json`，校验 `query` 和 `indices`。
2. 解析阈值和运行参数（`threshold`、`truncate_dim`、`use_fp16` 等）。
3. 解析重排参数（`use_reranker`、`rerank_top_k`、`rerank_batch_size`、`rerank_weight`、`reranker_output_resolution`）。
4. 解析去重参数（`use_dedup`、`dedup_threshold`）。
5. 从每个 PKL 文件名提取模型名，要求多索引必须来自同一模型，否则直接返回 400。
6. 调用 `get_or_create_processor(model_name, truncate_dim, use_fp16)` 加载或复用嵌入模型与搜索引擎。
7. 遍历每个索引调用 `search_engine.search_by_text(...)` 做初筛召回，并附加 `source_pkl` 字段。
8. 合并所有召回结果，按 `similarity` 降序排序。
9. 若启用重排：先释放 CLIP 显存，再调用 Reranker 对前 `rerank_top_k` 结果重打分并融合分数。
10. 应用最终阈值过滤：只保留 `similarity >= threshold` 的结果。
11. 若启用去重：加载命中场景向量并调用 `deduplicate_text_search_results(...)` 做跨视频去重。
12. 返回统一 JSON 结构（含 `matches`、`rerank_info`、`dedup_info`、`model_used`）。

## 3. 参数映射（请求 JSON -> `search_similar_scenes`）

### 3.1 检索输入

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `query` | 无（必填） | 文本检索描述，缺失直接返回 400 |
| `indices` | `''` | 索引路径字符串，按换行拆分为多个 PKL 路径 |
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

| 函数 | 作用 |
| --- | --- |
| `detect_model_type(...)` | 自动识别 `clip` / `fgclip2` 等模型类型 |
| `get_or_create_processor(...)` | 创建或复用嵌入模型与搜索引擎；模型切换时清理旧缓存并释放显存 |
| `release_clip_model()` | 释放 CLIP 模型和相关缓存 |
| `release_reranker_model()` | 释放 Reranker 模型和相关缓存 |

### 4.2 重排流程

1. 取初筛结果前 `rerank_top_k` 条。
2. 使用 `RerankerFrameExtractor` 批量抽取每条结果对应帧图。
3. 组装 `query + documents(images)` 调用 `reranker.process(...)`。
4. 将重排分数融合回 `similarity` 并重排序。

### 4.3 去重流程

1. 按命中结果中的 `source_pkl` 加载对应场景特征（仅加载命中场景）。
2. `deduplicate_text_search_results(...)` 先执行 OP/ED 优先规则。
3. 对剩余结果执行跨视频矩阵相似度去重（仅跨视频比较）。
4. 返回去重统计：`before`、`after`、`removed`。

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
| `POST /clear_cache` | 清理临时目录并释放模型 |
| `GET /list_models` | 列出可用模型目录并返回自动识别类型 |
| `POST /get_model_info` | 返回模型基础信息（不主动加载大模型） |
| `GET /get_preview` | 提取/缓存指定帧图片 |
| `GET /get_clip` | 临时切出短视频片段供前端播放 |
| `POST /export_clips` | copy / precision 两种模式批量导出 |
| `GET /load_ffmpeg_config` | 读取 `config.json` 中 `video_output` |
| `POST /save_ffmpeg_config` | 保存 FFmpeg 参数到 `config.json` |
| `GET /browse` | 文件/目录浏览接口（前端路径选择器使用） |
