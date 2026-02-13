# P模式执行逻辑（基于 `pipeline_app.py` 当前实现）

更新时间：2026-02-13  
代码依据：`Cut_DetectScene/pipeline_app.py:341`、`Cut_DetectScene/pipeline_app.py:592`、`Cut_DetectScene/pipeline_app.py:941`

## 1. 入口与调度链路

1. 前端调用 `GET /run_pipeline_stream`，把完整 `config` 作为 query 参数传入（SSE 流式返回日志）。
2. 后端创建子进程执行 `_run_pipeline_in_process -> run_pipeline_thread(config)`。
3. `run_pipeline_thread` 根据三个总开关决定是否进入 P 模式：
   - `run_indexer`（默认 `True`）
   - `run_search`（默认 `True`）
   - `search_entry_mode`（默认 `'prompt'`）
4. 当 `search_entry_mode == 'prompt'` 时，进入 `run_prompt_search_with_config(config)`。

## 2. P模式执行步骤

`run_prompt_search_with_config` 的实际流程如下：

1. 检查模块可用性：`_PROMPT_SEARCH_AVAILABLE` 为 `False` 时直接失败。
2. 读取配置段：`ps_config = config.get('prompt_search', {})`。
3. 检查停止标志：`_stop_requested.value` 为 `True` 时直接退出。
4. 预处理关键参数：
   - `prompt_search_mode = int(ps_config.get('search_mode', 0))`
   - `pkl_batch_size = ps_config.get('pkl_batch_size', 5)`
   - `diskcache_dir = _resolve_prompt_cache_dir(...)`
5. 调用 `run_interactive_search(...)`（实际检索主逻辑在 `auto_scene_search`）。
6. 返回值校验：返回值必须是 `dict`，否则判定为失败。
7. 成功后记录耗时、推送进度到 `100%`。

## 3. 参数映射（`prompt_search` -> `run_interactive_search`）

### 3.1 路径配置

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `index_directory` | `indexes`（再做 `resolve_path`） | 索引文件目录，包含 `.pkl` 特征文件。模型名称从 PKL 文件名自动提取（格式 `VideoName_modelname.pkl`） |
| `output_directory` | `output`（再做 `resolve_path`） | 搜索结果输出目录，保存匹配结果的 JSON 文件 |

### 3.2 模型精度

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `use_fp16` | `True` | CLIP 模型推理精度。开启后模型以 FP16 运行，节省约 50% 显存 |
| `feature_fp16` | `False` | PKL 特征向量在 GPU 上的存储精度。`False` = FP32 存储（精度高），`True` = FP16 存储（显存减半）。底层逻辑：若传 `None` 则回退为 `use_fp16` 的值 |

### 3.3 Reranker 配置

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `use_reranker` | `False` | 是否启用二阶段重排序。开启后先用 CLIP 粗筛，再用 Reranker 模型对 top-K 候选精排 |
| `rerank_top_k` | `50` | 从初始召回结果中选取前 K 个进入 Reranker 精排，推荐 30-100 |
| `rerank_batch_size` | `4` | Reranker 每批处理的图像数量，推荐 2-8。越大越快但显存占用越高 |
| `reranker_output_resolution` | `'384'`（字符串） | Reranker 读取视频帧时的短边分辨率（像素），推荐 `'384'` 或 `'512'` |
| `candidate_batch_size` | `1000` | Reranker 候选批次大小，控制一次送入 Reranker 的候选场景数量上限 |

### 3.4 视频导出

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `video_output_directory` | `None` | 视频切割输出目录，`None` 时使用 `output/videos` |
| `video_copy_mode` | `True` | 视频切割方式。`True` = copy 模式（ffmpeg stream copy，快速但不精确到帧），`False` = 精确切割模式（重编码，慢但帧级精确） |
| `start_frame_offset` | `None`（有值时转 `int`） | 起始帧偏移量。负数向前扩展、正数向后收缩。`None` 时使用默认值（copy 模式=0，精确切割=-2） |
| `end_frame_offset` | `None`（有值时转 `int`） | 结束帧偏移量。负数向前收缩、正数向后扩展。`None` 时使用默认值（copy 模式=2，精确切割=2） |
| `video_name_format` | `None` | 导出视频文件名模板。支持占位符 `{镜头}`, `{情绪}`, `{场景}`, `{主体}`, `{动作}`, `{起始帧}`, `{视频解析名}` 及 JSON 中定义的扩展大类。`None` 时自动生成默认格式 |

### 3.5 搜索模式与结果控制

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `search_mode` | `0`（入口强制 `int(...)` 转换） | 搜索粒度模式：`-1` = 按视频独立搜索（每个视频返回 top_k 个结果）；`0` = 按 PKL 独立搜索（每个 PKL 返回 top_k 个结果）；`1` = 跨 PKL 全局搜索（返回全局 top_k 个结果） |
| `top_k` | `50` | 每组返回的最大结果数。`None` 则不限制数量 |
| `debug_similarity` | `False` | 调试模式，开启后在导出文件名前添加相似度分数 |

### 3.6 Prompt 模板

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `prompt_template` | `None` | 自定义 prompt 组合模板。支持占位符 `{mood}`, `{lens}`, `{subject}`, `{action}`, `{scene}` 及 JSON 扩展大类。`None` 时使用默认模板 `"A {mood} {lens} of a {subject} {action} in {scene}"` |
| `use_chinese` | `False` | 中文模式开关。`False` = 用英文标签值生成 prompt（适合英文 CLIP），`True` = 用中文标签键名生成 prompt（适合中文 CLIP 如 FG-CLIP2） |

### 3.7 PKL 特征加载策略

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `pkl_batch_size` | `5` | 每批加载的 PKL 文件数量。`None` 或 `>= PKL 总数` = 一次性全部加载到 GPU（显存占用高但搜索快）；`< PKL 总数` = 分批加载用完释放（显存友好但稍慢） |
| `pkl_load_workers` | `4` | PKL 文件加载的并行线程数，全量预加载时使用。推荐 4-8 |

### 3.8 Prompt 向量缓存

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `prompt_cache_batch_size` | `512` | 生成 prompt 向量缓存时的批处理大小。控制一次编码多少条 prompt 文本为向量，推荐 256-1024。缓存通过哈希自动验证（prompt_template + keywords + use_chinese 变化时自动重新生成） |
| `prompt_search_batch_size` | `1024` | 搜索时每批从缓存加载的 prompt 向量数量，用于 GPU 矩阵运算。推荐 1024-4096，越大搜索越快但显存占用越高 |

### 3.9 LMDB 磁盘缓存（搜索结果）

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `use_diskcache` | `True` | 是否使用 LMDB 存储搜索结果到磁盘。开启后避免大量搜索结果撑爆内存 |
| `diskcache_dir` | `templates/prompt_cache`（可自定义） | LMDB 缓存目录。通过 `_resolve_prompt_cache_dir()` 解析，未配置时使用默认路径 |
| `lmdb_write_batch_size` | `1000` | LMDB 单事务写入的记录数。分批加载模式（`pkl_batch_size < PKL 总数`）时使用，控制每次事务提交的数据量 |

### 3.10 后处理

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `vector_dedup_threshold` | `None` | 向量去重余弦相似度阈值。`None` = 不去重；`0.90~0.98` = 超过此阈值的同标签视频只保留一个（优先保留 OP/ED 视频） |
| `adjacent_merge_frames` | `None` | 相邻片段合并帧阈值。`None` = 不合并；`N`（正整数）= 当片段 A 的 endframe 与片段 B 的 startframe 差值 ≤ N 时合并为一个片段（同类标签用 `_` 连接并去重） |


