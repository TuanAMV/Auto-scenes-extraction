# Auto-scenes-extraction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Search](https://img.shields.io/badge/Scene%20Search-CLIP%20%2B%20MiniCPM-orange)
![FP](https://img.shields.io/badge/Precision-FP16%20Full--Link-blueviolet)

## Runtime Versions

- Python: `3.11.2`
- torch: `2.9.1+cu128` (embedded Python)
- transformers: `5.7.0` (embedded Python)

---

## 中文版

### 项目简介

`Auto-scenes-extraction` 是一个本地视频场景检索工具：先把视频做成场景向量索引，再通过自然语言搜索，最后可导出命中的视频片段。

### 核心能力

- **视频向量化**：按场景抽帧并生成 Lance 向量索引（全链路 FP16）
- **文本检索**：单句自然语言查询，CLIP/FG-CLIP2 双模型支持
- **新标签模式（自动提纯）**：从 `logic_keywords.json` 自动生成 ~12,500 条 prompt，4 大类交叉搜索，交集过滤确保质量
- **MiniCPM 标签验证与剥离**：用 MiniCPM-V-4-6 验证 CLIP 初筛标签，剔除假阳性，提升准确度
- **视频导出**：ffmpeg precision cut / copy 双模式导出命中片段

### 整体流程

```
视频输入 → 场景向量索引 (Lance, FP16)
    │
    ├→ text_search: 单句查询 → CLIP 召回 → (可选重排/去重) → 预览/导出
    │
    └→ 新标签模式: Prompt 缓存 → 4 类交叉搜索 → 交集过滤
                    → MiniCPM 标签验证/剥离 → (景别分析) → 视频导出
```

### 入口文件

| 入口 | 用途 | 启动 |
|------|------|------|
| `pipeline_app.py` | 新标签模式 Web 界面（建索引 + 自动提纯搜索） | `启动自动提纯.bat` |
| `text_search.py` | 语义检索 Web 服务（交互式单句文搜） | `启动语义搜视频.bat` |

### 逻辑文档 (HTML, 可直接浏览器打开)

| 文档 | 说明 |
|------|------|
| `logic_explanation/new/新标签模式执行流程图.html` | 新标签模式完整执行流程图 |
| `logic_explanation/new/自动提纯逻辑.html` | 自动提纯详细逻辑 |
| `logic_explanation/new/视频向量化逻辑.html` | 视频向量化全流程 |
| `logic_explanation/new/text_search执行逻辑.html` | text_search 执行逻辑 |
| `logic_explanation/new/logic_keywords.html` | logic_keywords.json 设计原理与结构 |

### `logic_keywords.json` 的作用

- **三级标签树**：大类（主体/动作/场景/情绪）→ 子类（生物/生物行为/物体/物理运动...）→ 孙类（男人/running/城市/horror...）
- **中英映射**：每个标签 `"中文": "english"`，自动生成中/英文 prompt
- **自动提纯模板**：每个大类有专用 prompt 模板，占位符从标签树动态解析
- **输出命名格式**：`{情绪}_{场景}_{主体}_{动作}_{起始帧}_{视频解析名}`
- **必有标签**：`["主体", "动作", "场景", "情绪"]` — 场景必须同时命中这 4 类才输出
- **MiniCPM 验证 prompt**：每个大类有独立的中/英文验证句式
- **缓存失效**：文件变更自动触发 prompt 向量缓存重建

### 两大子系统

| 子系统 | 入口 | 前端页面 | 用途 |
|------|------|------|------|
| 新标签模式流水线 | `pipeline_app.py` | `Pipeline_control.html` | 视频建索引 + 批量自动搜索导出 |
| 语义搜视频 | `text_search.py` | `Frame_text_search.html` | 交互式单句文搜视频，支持预览/导出 |

两者共享 `A_coreUtils/` 下的核心模块（嵌入模型、Lance 索引读写、视频处理、MiniCPM 验证等），但运行时互相独立。

### 视频向量化流程

1. FFmpeg 按 `sample_interval` 间隔抽帧，**短边缩放**到 `output_resolution`（默认 256，1920×1080 → 456×256）
2. 双缓冲并行架构：I/O 线程池加载 + GPU 批量推理（`batch_size=512`）并行执行
3. **全链路 FP16**：模型推理 → Lance 存储 → 搜索加载，全程半精度，磁盘/显存减半
4. 计算相邻帧向量的余弦相似度，低于 `cosine_similarity_threshold` 处切分场景边界
5. 短场景（< `min_scene_length` 帧）合并到相邻最相似场景
6. 每个场景提取首/中/尾三帧特征，L2 归一化后写入 Lance 索引
7. 支持增量处理（`resume_processing`）：跳过已索引视频

支持的嵌入模型：CLIP（`openai-clip-vit-large-patch14`）、FG-CLIP2（`qihoo360-fg-clip2-base`），自动检测模型类型。

| 模型 | 短边缩放后 | 预处理 | 模型看到 |
|------|-----------|--------|---------|
| CLIP | 456×256 | Resize(224) → CenterCrop(224) | 224×224 中心区域 |
| FG-CLIP2 | 456×256 | Siglip2ImageProcessorFast 动态分块 | 456×256 完整画面 (28×16 patch) |

### 新标签模式（自动提纯）

**核心流程**：

1. **Prompt 生成**：从 `logic_keywords.json` 读取 4 大类模板，每类独立生成 prompt（约 12,500 条），非笛卡尔积
2. **Prompt 向量缓存**：CLIP/FG-CLIP2 编码 → FP16 向量 → Lance 缓存，MD5 哈希校验自动失效重建
3. **GPU 矩阵搜索**：`logit_scale(100) × prompts @ scenes.T` → 余弦相似度矩阵（12,500 × scenes）
4. **每类取最高分**：按 category 分组，scene 的每类保留相似度最高的 prompt 及其标签
5. **交集过滤**：场景必须同时命中 4 个必有标签（主体/动作/场景/情绪），缺少任一类则丢弃
6. **MiniCPM 标签验证与剥离**：
   - 抽帧 → MiniCPM-V-4-6 逐类判定标签 True/False
   - 匹配数 ≥ `min_matches` → 保留场景，**剥除判定为 False 的标签**（删除 `xxx_cn/_en`，重建 result_name，写回 LMDB）
   - 不足 → 删除场景
   - 解析失败 → ValueError 冒泡终止管道
7. **后处理**：向量去重（`vector_dedup_threshold`）→ 相邻合并（`adjacent_merge_frames`）→ 景别分析（`use_shot_analysis`）
8. **视频导出**：ffmpeg precision cut / copy 模式，命名格式从 `输出视频命名格式` 读取

**性能参数**：

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `fgclip2` 阈值 | 10~14 | FG-CLIP2 阈值需低于 CLIP（余弦相似度天然偏低） |
| `mini_rerank_min_matches` | 3 | MiniCPM 最低匹配类别数 |
| `use_chinese` | False | 仅 FG-CLIP2 可用 |

### text_search 语义检索

独立的交互式文搜视频服务：

- 单句自然语言查询，多索引联合检索（要求同一嵌入模型）
- 可选二阶段重排（Qwen3-VL Reranker）：初筛阈值自动降低以召回更多候选
- 可选跨视频去重：OP/ED 优先保留规则 + 矩阵余弦相似度去重
- 帧预览（`/get_preview`）、片段预览（`/get_clip`）、批量导出（`/export_clips`）
- 模型热切换：切换模型时自动释放旧模型显存
- `start_frame_offset` / `end_frame_offset` 控制输出片段起止帧偏移

### 快速开始

1. 将所需模型放到 `models/` 目录下：
   - 嵌入模型（至少一个）：`openai-clip-vit-large-patch14` 或 `qihoo360-fg-clip2-base`
   - MiniCPM 验证模型：`MiniCPM-V-4-6`
   - 景别分析模型：`aslakey_shot_scale`
   - FFmpeg：`models/ffmpeg/bin/ffmpeg.exe`
2. 将待处理视频放入 `videos/` 目录（或在 Web UI 中指定路径）
3. 运行：
   - `启动自动提纯.bat` → 打开流水线界面，先建索引再执行新标签模式搜索
   - `启动语义搜视频.bat` → 打开语义检索界面，输入描述语句交互式搜索
4. 在 Web UI 中配置参数并执行任务
5. 导出结果默认保存到 `output/` 目录

> 两个 bat 脚本使用项目内嵌入式 Python（`python/python.exe`），无需额外安装 Python 环境。

---

## English

### Overview

`Auto-scenes-extraction` is a local video scene retrieval tool. It builds scene-level vector indexes from videos, supports natural-language search, and exports matched clips.

### Key Features

- Scene vector indexing from videos (full-link FP16)
- Text-to-scene retrieval (CLIP / FG-CLIP2)
- New Label Mode (auto-purify): ~12,500 auto-generated prompts, 4-category cross-search with intersection filtering
- MiniCPM label verification & stripping: VLM validates CLIP labels, removes false positives
- Clip export with frame-range control

### Main Entry Points

| Entry | Purpose | Launch |
|------|------|------|
| `pipeline_app.py` | New Label Mode pipeline (indexing + auto search & export) | `启动自动提纯.bat` |
| `text_search.py` | Semantic search service (interactive text-to-video) | `启动语义搜视频.bat` |

### Deep-Dive Docs (HTML, open in browser)

| Document | Description |
|------|------|
| `logic_explanation/new/新标签模式执行流程图.html` | New Label Mode full flowchart |
| `logic_explanation/new/自动提纯逻辑.html` | Auto-purify detailed logic |
| `logic_explanation/new/视频向量化逻辑.html` | Video vectorization pipeline |
| `logic_explanation/new/text_search执行逻辑.html` | text_search execution logic |
| `logic_explanation/new/logic_keywords.html` | logic_keywords.json design |

---

## Acknowledgements

Thanks to all contributors to this project.

Core contributors:

- @彩绘的图案丶 [Bilibili](https://space.bilibili.com/4854520) | [YouTube](https://www.youtube.com/@TuanAMV)

Special thanks:

- @QQ346713889

Primary upstream projects used:

- CLIP: https://github.com/openai/CLIP
- MiniCPM-V: https://github.com/OpenBMB/MiniCPM-o
- FG-CLIP: https://github.com/360CVGroup/FG-CLIP
- Qwen3-VL-Embedding / Reranker: https://github.com/QwenLM/Qwen3-VL-Embedding
- FFmpeg: https://github.com/FFmpeg/FFmpeg
- aslakey/shot-scale: https://huggingface.co/aslakey/shot-scale

## License

Project source code is licensed under Apache License 2.0. See `licenses/LICENSE`.

## Third-Party Components

This repository includes third-party software, model files, and binaries with their own licenses and terms.

See:

- `licenses/THIRD_PARTY_LICENSES.md`
- `licenses/NOTICE`

Important:

- `models/ffmpeg` is a GPLv3 build. If you redistribute this project with that binary, GPL obligations apply to the redistributed package.
- Model directories under `models/` may have separate terms from the project source code license. Review each model directory before redistribution.
