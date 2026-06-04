# -*- coding: utf-8 -*-
# 本文件使用 UTF-8 编码，请勿使用 GBK 或其他编码打开/保存
# prompt_vector_cache.py
#
# v7.0 (Lance):
# - Persistent cache is stored as a single Lance dataset directory:
#     prompt_cache_{model_name}_{lang}.lance
# - No legacy memmap / pkl / lmdb cache formats are supported.
#
# The public API is kept compatible for callers:
# - PromptVectorCache.cache_exists()
# - PromptVectorCache.generate_cache()
# - PromptVectorCache.load_cache_batched() -> iterator yielding:
#     (batch_vectors: torch.Tensor [B, dim] on DEVICE,
#      batch_prompts: List[str],
#      batch_metadata: List[Dict],
#      batch_info: Dict)

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch

# ============================================================
# Path bootstrap (keep behavior compatible with original file)
# ============================================================
_current_file = os.path.abspath(__file__)
_prompt_dir = os.path.dirname(_current_file)
_a_core_utils_dir = os.path.dirname(_prompt_dir)
_cut_detect_scene_dir = os.path.dirname(_a_core_utils_dir)
if _cut_detect_scene_dir not in sys.path:
    sys.path.insert(0, _cut_detect_scene_dir)

from path_resolver import PathResolver  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PromptVectorCache:
    """
    Prompt vector cache backed by Lance.

    Cache file:
      {cache_dir}/prompt_cache_{model_name}_{lang}.lance
    """

    def __init__(
        self,
        processor=None,
        keywords_path: str = None,
        cache_dir: str = None,
        batch_size: int = 512,
        use_chinese: bool = False,
    ):
        self.resolver = PathResolver()
        self.use_chinese = use_chinese

        if keywords_path is None:
            keywords_path = str(self.resolver.project_root / "logic_keywords.json")
        self.keywords_path = keywords_path

        if cache_dir is None:
            cache_dir = str(self.resolver.project_root / "templates" / "prompt_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.processor = processor
        self.batch_size = int(batch_size)

        with open(self.keywords_path, "r", encoding="utf-8") as f:
            self.keywords_data = json.load(f)

        self._cache_hash = self._compute_cache_hash()
        self._model_name = None
        if processor is not None:
            self._model_name = self._get_model_name()

    def _get_model_name(self) -> str:
        if self.processor is None:
            return "unknown"
        if hasattr(self.processor, "model_name"):
            return str(self.processor.model_name)
        if hasattr(self.processor, "_model_name"):
            return str(self.processor._model_name)
        return "unknown"

    def _compute_cache_hash(self) -> str:
        with open(self.keywords_path, "r", encoding="utf-8") as f:
            keywords_content = f.read()
        lang_mode = "chinese" if self.use_chinese else "english"
        combined = f"prompt_vector_cache_v7\n{keywords_content}\n{lang_mode}"
        return hashlib.md5(combined.encode("utf-8")).hexdigest()[:8]

    def get_cache_path(self, model_name: str = None) -> str:
        if model_name is None:
            model_name = self._model_name or "unknown"

        safe_model_name = str(model_name).replace("/", "_").replace("\\", "_").replace(":", "_")
        lang_suffix = "cn" if self.use_chinese else "en"
        base_name = f"prompt_cache_{safe_model_name}_{lang_suffix}"
        return os.path.join(self.cache_dir, base_name)

    def get_lance_path(self, model_name: str = None) -> str:
        return self.get_cache_path(model_name) + ".lance"

    def get_category_idx_map(self, model_name: str = None) -> Dict[str, List[int]]:
        """从 Lance 缓存中读取每个类别的 prompt 索引列表。
        Returns: {"主体": [0,1,2,...], "动作": [...], "场景": [...], "情绪": [...]}
        """
        lance_path = self.get_lance_path(model_name)
        if not os.path.exists(lance_path):
            return {}
        import lance, pickle
        ds = lance.dataset(lance_path)
        cat_map = {}
        for i, row in enumerate(ds.to_table(columns=["metadata"]).column(0)):
            meta = pickle.loads(row.as_py()) if hasattr(row, 'as_py') else pickle.loads(row)
            cat = meta.get("category", "?")
            cat_map.setdefault(cat, []).append(i)
        return cat_map

    def cache_exists(self, model_name: str = None) -> bool:
        lance_path = self.get_lance_path(model_name)
        if not os.path.exists(lance_path):
            return False

        try:
            import lance

            ds = lance.dataset(lance_path)
            md = ds.metadata or {}
            if md.get("format") != "prompt_vector_cache_lance_v7":
                return False

            cached_hash = md.get("cache_hash", "")
            if cached_hash != self._cache_hash:
                return False

            expected_total = int(md.get("total_prompts", "0") or 0)
            actual_total = int(ds.count_rows())
            if expected_total and expected_total != actual_total:
                return False

            return True
        except Exception:
            return False

    def _generate_all_prompts(self) -> Generator[Tuple[str, Dict], None, None]:
        from A_coreUtils.search.auto_scene_search import PromptGenerator

        generator = PromptGenerator(
            keywords_path=self.keywords_path,
            use_chinese=self.use_chinese,
        )

        for combo in generator.iterate_all_labels_flat():
            yield combo["prompt"], combo

    def generate_cache(self, progress_callback=None) -> str:
        if self.processor is None:
            raise RuntimeError("processor is required to generate cache")

        import lance
        import pyarrow as pa

        self._model_name = self._get_model_name()
        cache_base_path = self.get_cache_path()
        lance_path = self.get_lance_path()

        if self.cache_exists():
            print(f"[缓存有效] {cache_base_path}")
            return cache_base_path

        if os.path.exists(lance_path):
            shutil.rmtree(lance_path, ignore_errors=True)

        print("=" * 70)
        print("📝 Prompt向量缓存生成器 (v7.0 Lance)")
        print("=" * 70)
        print(f"  模型: {self._model_name}")
        print(f"  缓存目录: {self.cache_dir}")
        print(f"  写入目标: {os.path.basename(lance_path)}")
        print(f"  编码 batch_size: {self.batch_size}")

        print("\n[步骤1] 统计 prompt 总数...")
        from A_coreUtils.search.auto_scene_search import PromptGenerator

        temp_generator = PromptGenerator(
            keywords_path=self.keywords_path,
            use_chinese=self.use_chinese,
        )
        total_prompts = sum(1 for _ in temp_generator.iterate_all_labels_flat())
        del temp_generator
        if total_prompts <= 0:
            raise RuntimeError("total_prompts == 0, check logic_keywords.json / template")
        print(f"  总计: {total_prompts:,} 个 prompt")

        first_prompt = next(self._generate_all_prompts())[0]
        sample_vector = self.processor.encode_text([first_prompt])
        vector_dim = int(sample_vector.shape[1])
        print(f"  向量维度: {vector_dim}")

        schema = pa.schema(
            [
                pa.field("prompt", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), vector_dim)),
                pa.field("metadata", pa.binary()),  # pickle bytes
            ]
        )

        def _vectors_to_fixed_size_list(vectors: np.ndarray) -> pa.FixedSizeListArray:
            vec = np.asarray(vectors, dtype=np.float16, order="C")
            values = pa.array(vec.reshape(-1), type=pa.float16())
            return pa.FixedSizeListArray.from_arrays(values, vector_dim)

        def record_batches():
            current_idx = 0
            batch_prompts: List[str] = []
            batch_metadata: List[Dict] = []

            start_time = time.time()
            last_log = 0

            for prompt_text, prompt_meta in self._generate_all_prompts():
                batch_prompts.append(prompt_text)
                batch_metadata.append(prompt_meta)

                if len(batch_prompts) < self.batch_size:
                    continue

                vectors = self.processor.encode_text(batch_prompts)
                meta_bytes = [pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL) for m in batch_metadata]

                yield pa.record_batch(
                    [
                        pa.array(batch_prompts, type=pa.string()),
                        _vectors_to_fixed_size_list(vectors),
                        pa.array(meta_bytes, type=pa.binary()),
                    ],
                    names=["prompt", "vector", "metadata"],
                )

                current_idx += len(batch_prompts)
                batch_prompts.clear()
                batch_metadata.clear()

                # progress
                if progress_callback:
                    progress_callback(current_idx, total_prompts, f"处理 {current_idx}/{total_prompts}")
                else:
                    if current_idx - last_log >= max(1, self.batch_size * 20):
                        elapsed = time.time() - start_time
                        speed = current_idx / elapsed if elapsed > 0 else 0
                        print(f"  [{current_idx:,}/{total_prompts:,}] {speed:.1f} prompts/s")
                        last_log = current_idx

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            if batch_prompts:
                vectors = self.processor.encode_text(batch_prompts)
                meta_bytes = [pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL) for m in batch_metadata]
                yield pa.record_batch(
                    [
                        pa.array(batch_prompts, type=pa.string()),
                        _vectors_to_fixed_size_list(vectors),
                        pa.array(meta_bytes, type=pa.binary()),
                    ],
                    names=["prompt", "vector", "metadata"],
                )
                current_idx += len(batch_prompts)

            if current_idx != total_prompts:
                raise RuntimeError(f"数量不一致: 预期 {total_prompts}, 实际 {current_idx}")

        print("\n[步骤2] 写入 Lance 数据集...")
        t0 = time.time()
        ds = lance.write_dataset(record_batches(), lance_path, schema=schema, mode="create")

        ds.update_metadata(
            {
                "format": "prompt_vector_cache_lance_v7",
                "model_name": str(self._model_name),
                "cache_hash": str(self._cache_hash),
                "total_prompts": str(total_prompts),
                "vector_dim": str(vector_dim),
                "keywords_path": str(self.keywords_path),
                "use_chinese": "1" if self.use_chinese else "0",
                "created_at": datetime.now().isoformat(),
                "normalized": "1",
            }
        )

        elapsed = time.time() - t0
        print(f"  写入完成: {int(ds.count_rows()):,} 行, 耗时 {elapsed:.2f}s")

        # quick size
        size_mb = 0.0
        for root, _, files in os.walk(lance_path):
            for fn in files:
                fp = os.path.join(root, fn)
                if os.path.isfile(fp):
                    size_mb += os.path.getsize(fp) / (1024 * 1024)
        print(f"  数据集大小: {size_mb:.2f} MB")

        print("\n" + "=" * 70)
        print("✅ 缓存生成完成")
        print("=" * 70)
        return cache_base_path

    def load_cache_batched(self, model_name: str = None, batch_size: int = 10000, use_fp16: bool = True) -> "PromptVectorBatchIterator":
        lance_path = self.get_lance_path(model_name)
        if not os.path.exists(lance_path):
            raise FileNotFoundError(f"Lance 数据集不存在: {lance_path}")
        return PromptVectorBatchIterator(lance_path=lance_path, batch_size=int(batch_size), use_fp16=use_fp16)

    def load_meta_lookup(self, model_name: str = None) -> "PromptMetaLookup":
        lance_path = self.get_lance_path(model_name)
        if not os.path.exists(lance_path):
            raise FileNotFoundError(f"Lance 数据集不存在: {lance_path}")
        return PromptMetaLookup(lance_path=lance_path)

    def get_cache_info(self, model_name: str = None) -> Dict:
        import lance

        lance_path = self.get_lance_path(model_name)
        if not os.path.exists(lance_path):
            raise FileNotFoundError(f"Lance 数据集不存在: {lance_path}")
        ds = lance.dataset(lance_path)
        md = ds.metadata or {}
        total_prompts = int(ds.count_rows())

        size_mb = 0.0
        for root, _, files in os.walk(lance_path):
            for fn in files:
                fp = os.path.join(root, fn)
                if os.path.isfile(fp):
                    size_mb += os.path.getsize(fp) / (1024 * 1024)

        return {
            "total_prompts": total_prompts,
            "vector_dim": int(md.get("vector_dim", "0") or 0),
            "metadata": md,
            "lance_path": lance_path,
            "file_size_mb": size_mb,
        }


class PromptVectorBatchIterator:
    """
    Batch iterator for prompt vectors stored in Lance.

    Yields fixed-size batches (except the last batch) to keep batch_idx semantics
    stable for checkpointing in BatchTextSearchEngine.
    """

    def __init__(self, lance_path: str, batch_size: int = 10000, use_fp16: bool = True):
        import lance

        self.lance_path = lance_path
        self.batch_size = int(batch_size)
        self.use_fp16 = bool(use_fp16)

        self._ds = lance.dataset(self.lance_path)
        self._ds_meta = self._ds.metadata or {}

        self.total_prompts = int(self._ds.count_rows())
        self.num_batches = (self.total_prompts + self.batch_size - 1) // self.batch_size
        self._current_batch = 0

        self._vector_dim = int(self._ds_meta.get("vector_dim", "0") or 0)
        if self._vector_dim <= 0:
            try:
                vec_field = self._ds.schema.field("vector")
                self._vector_dim = int(getattr(vec_field.type, "list_size", 0) or 0)
            except Exception:
                self._vector_dim = 0
        if self._vector_dim <= 0:
            raise RuntimeError("无法解析 vector_dim，请删除缓存并重新生成")

        self._init_scanner()

    def _init_scanner(self):
        # Use an internal scanner batch size that is >= desired batch_size.
        # We'll re-batch to fixed-size output regardless of fragment boundaries.
        self._scanner = self._ds.scanner(
            columns=["vector", "prompt", "metadata"],
            scan_in_order=True,
            batch_size=max(1024, self.batch_size),
        )
        self._rb_iter = iter(self._scanner.to_batches())

        self._rb = None
        self._rb_pos = 0
        self._rb_vectors = None
        self._rb_prompts = None
        self._rb_meta_bytes = None

    def _load_next_rb(self):
        rb = next(self._rb_iter)  # may raise StopIteration

        vec_col = rb.column(0)  # fixed_size_list<float32>[dim]
        flat = vec_col.values.to_numpy(zero_copy_only=False)
        vecs = flat.reshape(len(vec_col), self._vector_dim)

        self._rb = rb
        self._rb_pos = 0
        self._rb_vectors = vecs
        self._rb_prompts = rb.column(1).to_pylist()
        self._rb_meta_bytes = rb.column(2).to_pylist()

    def __iter__(self):
        self.reset()
        return self

    def __next__(self) -> Tuple[torch.Tensor, List[str], List[Dict], Dict]:
        if self._current_batch >= self.num_batches:
            raise StopIteration

        start_idx = self._current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_prompts)
        desired = end_idx - start_idx

        out_vec = np.empty((desired, self._vector_dim), dtype=np.float16)
        out_prompts: List[str] = ["" for _ in range(desired)]
        out_meta: List[Dict] = [{} for _ in range(desired)]

        filled = 0
        while filled < desired:
            if self._rb is None or self._rb_pos >= len(self._rb_prompts):
                self._load_next_rb()

            rb_remain = len(self._rb_prompts) - self._rb_pos
            take = min(rb_remain, desired - filled)

            out_vec[filled : filled + take] = self._rb_vectors[self._rb_pos : self._rb_pos + take]
            out_prompts[filled : filled + take] = self._rb_prompts[self._rb_pos : self._rb_pos + take]

            mb = self._rb_meta_bytes[self._rb_pos : self._rb_pos + take]
            for i, b in enumerate(mb):
                try:
                    out_meta[filled + i] = pickle.loads(b) if b is not None else {}
                except Exception:
                    out_meta[filled + i] = {}

            self._rb_pos += take
            filled += take

        batch_vectors = torch.from_numpy(out_vec).to(DEVICE)
        if self.use_fp16 and DEVICE == "cuda":
            batch_vectors = batch_vectors.half()

        batch_info = {
            "batch_idx": self._current_batch,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "total_prompts": self.total_prompts,
            "num_batches": self.num_batches,
        }

        self._current_batch += 1
        return batch_vectors, out_prompts, out_meta, batch_info

    def __len__(self) -> int:
        return self.num_batches

    @property
    def metadata(self) -> Dict:
        return self._ds_meta

    def reset(self):
        self._current_batch = 0
        self._init_scanner()

    def close(self):
        self._rb = None
        self._rb_vectors = None
        self._rb_prompts = None
        self._rb_meta_bytes = None
        self._rb_iter = None
        self._scanner = None
        self._ds = None

    def __del__(self):
        self.close()


class PromptMetaLookup:
    """
    按 prompt_idx 批量回查 metadata，避免候选里冗余携带 meta 副本。
    """

    def __init__(self, lance_path: str):
        import lance

        self.lance_path = lance_path
        self._ds = lance.dataset(self.lance_path)

    def get_many(self, prompt_indices: List[int]) -> Dict[int, Dict]:
        uniq = sorted({int(i) for i in prompt_indices if i is not None and int(i) >= 0})
        if not uniq:
            return {}

        try:
            table = self._ds.take(uniq, columns=["metadata"])
        except Exception as e:
            raise RuntimeError(f"PromptMetaLookup.take() 失败: {e}")

        meta_col = table.column("metadata").to_pylist()
        out: Dict[int, Dict] = {}
        for idx, raw in zip(uniq, meta_col):
            if raw is None:
                out[idx] = {}
                continue
            try:
                data = pickle.loads(raw)
                out[idx] = data if isinstance(data, dict) else {}
            except Exception:
                out[idx] = {}
        return out

    def get_metadata_by_indices(self, prompt_indices: List[int]) -> Dict[int, Dict]:
        return self.get_many(prompt_indices)
