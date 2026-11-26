"""Bounded Lance search and LMDB result views for automatic purification."""

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import lance
import numpy as np
import torch

from A_coreUtils.lance_index_io import iter_scene_batches, iter_vector_tiles


from .lmdb_store import (
    LMDBCache,
    LMDBResultView,
    RankedResults,
    _group_id,
    _sortable_float,
    is_op_ed_video,
)


def extract_label_key_from_result_name(
    result_name: str, video_name_format: str, exclude_fields: list = None
) -> str:
    """
    从 `result_name` 中提取标签组合键（用于分组）。
    Args:
        result_name: 例如 `城市_奔跑_男人_恐怖_1234_视频名`
        video_name_format: 例如 `{场景}_{动作}_{主体}_{情绪}_{起始帧}_{视频解析名}`
        exclude_fields: 排除字段，默认 `['起始帧', '视频解析名']`
    Returns:
        标签组合键，例如 `城市_奔跑_男人_恐怖`
    """
    import re

    if exclude_fields is None:
        exclude_fields = ["起始帧", "视频解析名"]
    # 解析格式字符串中的全部占位符
    placeholders = re.findall(r"\{([^}]+)\}", video_name_format)
    # 将格式模板转换为正则表达式
    pattern = video_name_format
    for ph in placeholders:
        pattern = pattern.replace(f"{{{ph}}}", "(.+?)")
    pattern = pattern + "$"
    # 匹配 result_name
    match = re.match(pattern, result_name)
    if not match:
        return result_name  # 无法解析时返回原值

    # 提取标签值（排除指定字段）
    label_values = []
    for i, ph in enumerate(placeholders):
        if ph not in exclude_fields:
            label_values.append(match.group(i + 1))
    return "_".join(label_values)


def extract_video_parsed_name(result_name: str, video_name_format: str) -> str:
    """
    从 `result_name` 中提取视频解析名。
    Args:
        result_name: 例如 `城市_奔跑_男人_恐怖_1234_视频名`
        video_name_format: 例如 `{场景}_{动作}_{主体}_{情绪}_{起始帧}_{视频解析名}`
    Returns:
        视频解析名，例如 `视频名`
    """
    import re

    placeholders = re.findall(r"\{([^}]+)\}", video_name_format)
    pattern = video_name_format
    for ph in placeholders:
        pattern = pattern.replace(f"{{{ph}}}", "(.+?)")
    pattern = pattern + "$"
    match = re.match(pattern, result_name)
    if not match:
        return ""

    # 找到"视频解析名"占位符的位置
    for i, ph in enumerate(placeholders):
        if ph == "视频解析名":
            return match.group(i + 1)

    return ""


class BatchTextSearchEngine:
    def __init__(self, row_batch, fp16, truncate_dim):
        self.row_batch = row_batch
        self.dtype = torch.float16 if fp16 else torch.float32
        self.truncate_dim = truncate_dim

    def _winners(self, scenes, prompts, threshold, categories):
        if not scenes:
            return
        best = [{} for _ in scenes]
        for vectors, _, metadata, info in prompts:
            category_slots = defaultdict(list)
            for slot, meta in enumerate(metadata):
                category_slots[meta["category"]].append(slot)
            categories.update(category_slots)
            device = vectors.device
            queries = vectors.to(self.dtype) * 100.0
            maxima = torch.full(
                (len(vectors), len(scenes)), -torch.inf, device=device, dtype=self.dtype
            )
            for values, owner in iter_vector_tiles(scenes, self.row_batch):
                if self.truncate_dim and values.shape[1] > self.truncate_dim:
                    values = values[:, : self.truncate_dim]
                    values = values / (
                        np.linalg.norm(values, axis=1, keepdims=True) + 1e-8
                    )
                values = torch.as_tensor(values, device=device, dtype=self.dtype)
                scores = queries @ values.T
                indices = torch.as_tensor(owner, device=device).expand(len(vectors), -1)
                maxima.scatter_reduce_(1, indices, scores, reduce="amax")
            scores = maxima.float().cpu().numpy().T
            for i, by_category in enumerate(best):
                for cat, slots in category_slots.items():
                    # The best of each prompt batch is sufficient for the final maximum.
                    slot = slots[int(scores[i, slots].argmax())]
                    value = float(scores[i, slot])
                    if value >= threshold:
                        previous = by_category.get(cat)
                        if previous is None or value > previous[0]:
                            by_category[cat] = (
                                value,
                                info["start_idx"] + slot,
                                metadata[slot],
                            )
        for by_category in best:
            yield by_category.values()

    def _scan(self, scans, parallel):
        # A worker holds at most one Arrow batch. Stable consumption order also
        # makes equal-score ties and checkpoint identities repeatable.
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            for offset in range(0, len(scans), parallel):
                active = []
                for path, version, row_offset in scans[offset : offset + parallel]:
                    batches = iter_scene_batches(
                        path, self.row_batch, offset=row_offset, version=version
                    )
                    active.append((path, batches, pool.submit(next, batches, None)))
                while active:
                    again = []
                    for path, batches, future in active:
                        block = future.result()
                        if block is not None:
                            row_end, rows = block
                            yield path, row_end, rows
                            again.append(
                                (path, batches, pool.submit(next, batches, None))
                            )
                    active = again

    @staticmethod
    def _scene_key(source, scene):
        payload = json.dumps(
            [source, scene.get("video_path", ""), scene.get("start_frame", 0), scene.get("end_frame", 0)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _done_key(path):
        return "done:" + hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()

    def _record_batch(self, store, path, row_end, rows, prompts, threshold, store_vectors):
        updates = {}
        vector_updates = {}
        for (scene, values), winners in zip(
            rows, self._winners(rows, prompts, threshold, set())
        ):
            if not winners:
                continue
            scene_key = self._scene_key(path, scene)
            store_key = "scene:" + scene_key
            record = store.get(store_key) or {
                "video_path": scene["video_path"],
                "start_frame": int(scene["start_frame"]),
                "end_frame": int(scene["end_frame"]),
                "fps": float(scene.get("fps") or 0),
                "source_lance": path,
                "dim": 0,
                "has_vectors": False,
                "winners": {},
            }
            for score, prompt_idx, metadata in winners:
                category = metadata["category"]
                previous = record["winners"].get(category)
                if previous is None or float(score) > float(previous["score"]):
                    record["winners"][category] = {
                        "score": float(score),
                        "idx": int(prompt_idx),
                        "meta": dict(metadata),
                    }
            if store_vectors and not record.get("has_vectors"):
                vector_values = values
                if self.truncate_dim and vector_values.shape[1] > self.truncate_dim:
                    vector_values = vector_values[:, : self.truncate_dim]
                    vector_values = vector_values / (
                        np.linalg.norm(vector_values, axis=1, keepdims=True) + 1e-8
                    )
                vector_values = np.ascontiguousarray(vector_values, dtype=np.float16)
                record["dim"] = int(vector_values.shape[1])
                record["has_vectors"] = True
                vector_updates["vector:" + scene_key] = vector_values.tobytes()
            updates[store_key] = record
            if len(updates) >= 256:
                if vector_updates:
                    store.put_raw_many(vector_updates)
                    vector_updates.clear()
                store.put_many(updates)
                updates.clear()
        if vector_updates:
            store.put_raw_many(vector_updates)
        if updates:
            store.put_many(updates)
        store.put(self._done_key(path), {"path": path, "row_end": int(row_end)})

    @staticmethod
    def _iter_scene_records(store):
        yield from store.iter_items("scene:")

    @staticmethod
    def _selected_scenes(store, generator):
        from .auto_scene_search import _generate_merged_result_name

        categories = set()
        for _, record in BatchTextSearchEngine._iter_scene_records(store):
            categories.update(record.get("winners", {}).keys())
        required = set(generator._required_categories or categories)
        for scene_key, record in BatchTextSearchEngine._iter_scene_records(store):
            winners = record.get("winners", {})
            present = set(winners)
            labels, fields = {}, {}
            if not required <= present:
                continue
            per_category = {}
            for category, winner in winners.items():
                metadata = dict(winner.get("meta") or {})
                per_category[category] = (winner, metadata)
                labels.update(metadata.get("labels") or {})
                for key, value in metadata.items():
                    if key.endswith(("_cn", "_en")):
                        fields.setdefault(key, value)
            if not set(generator._required_subcategories) <= labels.keys():
                continue
            if not set(generator._required_labels).issubset(labels.values()):
                continue
            best = max(per_category.values(), key=lambda entry: float(entry[0]["score"]))
            best_winner, best_meta = best
            item = {
                **fields,
                **best_meta,
                "video_path": record["video_path"],
                "start_frame": record["start_frame"],
                "end_frame": record["end_frame"],
                "fps": record.get("fps") or 25.0,
                "source_lance": record["source_lance"],
                "similarity": float(best_winner["score"]),
                "prompt_idx": int(best_winner["idx"]),
                "all_labels": labels,
            }
            item["result_name"] = _generate_merged_result_name(item, generator.name_template)
            key = hashlib.sha256(
                json.dumps(
                    [record["video_path"], record["start_frame"], record["end_frame"]],
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            # _iter_scene_records returns the storage key (``scene:<sha256>``),
            # while vector sidecars use only the digest. Keep the two key
            # namespaces aligned when the caller loads the raw float16 vector.
            yield key, scene_key.removeprefix("scene:"), item, record

    def search(
        self,
        paths,
        prompts,
        generator,
        threshold,
        mode,
        top_k,
        dedup,
        root,
        resume,
        parallel,
        write_batch,
    ):
        os.makedirs(root, exist_ok=True)

        def index_revision(path):
            dataset = lance.dataset(path)
            return (
                os.path.abspath(path),
                dataset.version,
                str(dataset.get_version()["timestamp"]),
            )

        revisions = [index_revision(p) for p in paths]
        signature = [
            8,
            revisions,
            prompts.revision,
            prompts.batch_size,
            threshold,
            self.row_batch,
            str(self.dtype),
            self.truncate_dim,
            dedup is not None,
            generator._full_data,
        ]
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        state_path = os.path.join(root, digest + ".lmdb") if resume else tempfile.mkdtemp(prefix="search_state_", dir=root)
        state = LMDBCache(state_path)
        output = tempfile.mkdtemp(prefix="results_", dir=root)
        result_cache = LMDBCache(output)
        ranked = None
        completed = False
        try:
            scans = []
            for path, version, _ in revisions:
                done = state.get(self._done_key(path), {}) or {}
                scans.append((path, version, int(done.get("row_end", 0))))
            for path, row_end, rows in self._scan(scans, parallel):
                done = state.get(self._done_key(path), {}) or {}
                if int(done.get("row_end", 0)) >= int(row_end):
                    continue
                self._record_batch(state, path, row_end, rows, prompts, threshold, dedup is not None)

            if dedup is not None:
                ranked_path = tempfile.mkdtemp(prefix="ranked_", dir=root)
                ranked = RankedResults(ranked_path)
            buffer = {}
            candidate_buffer = {}
            rank_buffer = {}

            def flush_candidates():
                if candidate_buffer:
                    result_cache.put_many(candidate_buffer)
                    candidate_buffer.clear()
                if rank_buffer:
                    result_cache.put_raw_many(rank_buffer)
                    rank_buffer.clear()

            def write_candidate(key, item):
                if top_k is None:
                    buffer[key] = item
                    if len(buffer) >= write_batch:
                        result_cache.put_results_many(buffer)
                        buffer.clear()
                    return
                group = {
                    -1: item.get("video_path", ""),
                    0: item.get("source_lance", ""),
                    1: "__global__",
                }[mode]
                group_id = _group_id(group).encode("ascii")
                candidate_key = "candidate:" + key
                rank_key = (
                    b"rank:" + group_id + b":"
                    + _sortable_float(item.get("similarity", 0.0), descending=True)
                    + b":" + key.encode("ascii")
                )
                candidate_buffer[candidate_key] = item
                rank_buffer[rank_key] = key.encode("ascii")
                if len(candidate_buffer) >= write_batch:
                    flush_candidates()

            ranked_entries = []
            for key, _scene_key, item, record in self._selected_scenes(state, generator):
                if ranked is None:
                    write_candidate(key, item)
                else:
                    vectors = state.get_raw("vector:" + _scene_key)
                    dim = int(record.get("dim", 0) or 0)
                    vector_array = None
                    if vectors and dim:
                        vector_array = np.frombuffer(vectors, np.float16).reshape(-1, dim)
                    label = extract_label_key_from_result_name(item["result_name"], generator.name_template)
                    parsed = extract_video_parsed_name(item["result_name"], generator.name_template)
                    item["_scene_key"] = key
                    ranked_entries.append((
                        item,
                        json.dumps([item["source_lance"], label], ensure_ascii=False),
                        is_op_ed_video(parsed),
                        vector_array,
                    ))
                    if len(ranked_entries) >= write_batch:
                        ranked.add_many(ranked_entries)
                        ranked_entries.clear()
            if ranked_entries:
                ranked.add_many(ranked_entries)
                ranked_entries.clear()
            if ranked is not None:
                ranked.deduplicate(dedup, self.row_batch, by_score=False)
                for _, item, _, _ in ranked.iter_inserted():
                    item = dict(item)
                    key = item.pop("_scene_key")
                    write_candidate(key, item)
            if top_k is not None:
                flush_candidates()
                current_group = None
                group_count = 0
                cleanup = []
                for rank_key, value in result_cache.iter_raw(b"rank:"):
                    group_id = rank_key[5:69]
                    if group_id != current_group:
                        current_group = group_id
                        group_count = 0
                    scene_key = bytes(value).decode("ascii")
                    candidate_key = "candidate:" + scene_key
                    if group_count < int(top_k):
                        item = result_cache.get(candidate_key)
                        if item is not None:
                            buffer[scene_key] = item
                            group_count += 1
                            if len(buffer) >= write_batch:
                                result_cache.put_results_many(buffer)
                                buffer.clear()
                    cleanup.append(rank_key)
                    cleanup.append(candidate_key)
                    if len(cleanup) >= 512:
                        result_cache.delete_many(cleanup)
                        cleanup.clear()
                if cleanup:
                    result_cache.delete_many(cleanup)
            result_cache.put_results_many(buffer)
            result_cache.close()
            result_cache = None
            completed = True
            return LMDBResultView(output)
        except BaseException:
            if result_cache is not None:
                result_cache.destroy()
                result_cache = None
            raise
        finally:
            if ranked is not None:
                ranked.destroy()
            if completed or not resume:
                state.destroy()
            else:
                state.close()
