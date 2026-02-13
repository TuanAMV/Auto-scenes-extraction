# Auto-scenes-extraction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Search](https://img.shields.io/badge/Scene%20Search-CLIP%20%2B%20Reranker-orange)

## Runtime Versions

- Python: `3.11.2`
- torch: `2.9.1+cu130` (embedded Python)
- transformers: `4.57.1` (embedded Python)

🌐 **Language / 语言**: 中文 | English

---

## 中文版

### 项目简介 🎬

`Auto-scenes-extraction` 是一个本地视频场景检索工具：  
先把视频做成场景向量索引，再通过自然语言搜索，最后可导出命中的视频片段。

### 核心能力 ✨

- 🧩 视频向量化：按场景抽帧并生成向量索引
- 🔎 文本检索：输入描述语句召回相关场景
- 🧠 二阶段重排：CLIP 召回 + Qwen3-VL Reranker 精排
- ✂️ 结果导出：按帧区间切片导出命中片段

### 主体逻辑（端到端）🛠️

```mermaid
flowchart LR
  A[🎞️ 视频输入] --> B[🧮 场景向量化/建索引]
  B --> C{🔍 搜索模式}
  C --> C1[Prompt 模式]
  C --> C2[Label 遍历模式]
  C --> C3[Cloze 选词填空模式]
  C1 --> D[📌 候选场景]
  C2 --> D
  C3 --> D
  D --> E{🧠 use_reranker?}
  E -->|Yes| F[Qwen3-VL 重排]
  E -->|No| G[直接排序]
  F --> H[🧹 去重/合并后处理]
  G --> H
  H --> I[✂️ 导出片段]
```

### 入口文件 📍

- `pipeline_app.py`：流水线主入口（建索引 + 多种搜索模式）
- `text_search.py`：语义检索 Web 服务入口
- `启动自动提纯.bat`：启动流水线界面
- `启动语义搜索.bat`：启动语义检索界面

### Logic Explanation 📚

- 主流程图：`logic_explanation/P模式主体逻辑.png`
- Prompt 模式执行逻辑：`logic_explanation/P模式执行逻辑.md`
- Prompt 模式流程图（Mermaid）：`logic_explanation/P模式流程图.md`
- text_search 执行逻辑：`logic_explanation/text_search执行逻辑.md`
- 视频向量化逻辑：`logic_explanation/视频向量化逻辑.md`

![P模式主体逻辑](logic_explanation/P模式主体逻辑.png)

### 快速开始 🚀

1. 将模型和工具放到 `models/`。
2. 运行：
   - `启动自动提纯.bat`（流水线）
   - `启动语义搜索.bat`（检索）
3. 在 Web UI 中配置参数并执行任务。

### 目录结构 🗂️

- `A_coreUtils/`：检索、向量化、视频处理核心模块
- `models/`：本地模型与二进制工具
- `preset/`：参数预设
- `logic_explanation/`：流程与实现说明
- `temp/`, `output/`, `indexes/`：运行时文件

---

## English

### Overview 🎬

`Auto-scenes-extraction` is a local video scene retrieval tool.  
It builds scene-level vector indexes from videos, supports natural-language
search, and exports matched clips.

### Key Features ✨

- Scene vector indexing from videos
- Text-to-scene retrieval
- Two-stage ranking (embedding recall + reranker refinement)
- Clip export with frame-range control

### Main Entry Points 📍

- `pipeline_app.py`: end-to-end pipeline entry
- `text_search.py`: semantic search service entry
- `启动自动提纯.bat`: launch pipeline UI
- `启动语义搜索.bat`: launch text search UI

### Deep-Dive Docs 📚

- `logic_explanation/P模式执行逻辑.md`
- `logic_explanation/P模式流程图.md`
- `logic_explanation/text_search执行逻辑.md`
- `logic_explanation/视频向量化逻辑.md`

---

## Acknowledgements 🙌

Thanks to all contributors to this project.

Core contributors:

- @TuanAMV
- @彩绘的图案丶

Primary upstream projects used:

- CLIP: https://github.com/openai/CLIP
- Qwen3-VL-Embedding / Reranker: https://github.com/QwenLM/Qwen3-VL-Embedding
- FG-CLIP: https://github.com/360CVGroup/FG-CLIP
- FFmpeg: https://github.com/FFmpeg/FFmpeg

## License ⚖️

Project source code is licensed under Apache License 2.0.  
See `LICENSE.txt`.

## Third-Party Components 📦

This repository includes third-party software, model files, and binaries with
their own licenses and terms.

See:

- `THIRD_PARTY_LICENSES.md`
- `NOTICE`

Important:

- `models/ffmpeg` is a GPLv3 build. If you redistribute this project with that
  binary, GPL obligations apply to the redistributed package.
- Model directories under `models/` may have separate terms from the project
  source code license. Review each model directory before redistribution.
