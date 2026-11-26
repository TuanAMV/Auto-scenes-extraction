"""LMDB-backed stores used by the bounded search paths.

The search pipeline only needs an append/update store and ordered cursors.  A
relational query engine is unnecessary here: LMDB keeps the values on disk and
its lexicographic key order gives us streaming ranking and time indexes.
"""

from __future__ import annotations

import math
import pickle
import re
import shutil
import struct
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import lmdb
import numpy as np


_U64_MASK = (1 << 64) - 1


def _sortable_float(value: float, *, descending: bool = False) -> bytes:
    """Return an IEEE-754 key that sorts numerically in LMDB byte order."""
    value = float(value)
    if not math.isfinite(value):
        value = -math.inf if descending else math.inf
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    ordered = bits ^ (_U64_MASK if bits >> 63 else (1 << 63))
    if descending:
        ordered = _U64_MASK - ordered
    return ordered.to_bytes(8, "big")


def _sortable_int(value: int, *, descending: bool = False) -> bytes:
    value = int(value)
    ordered = (value + (1 << 63)) & _U64_MASK
    if descending:
        ordered = _U64_MASK - ordered
    return ordered.to_bytes(8, "big")


def _id_bytes(identifier: int) -> bytes:
    return f"{int(identifier):016x}".encode("ascii")


def _group_id(group: str) -> str:
    """Return a filesystem/LMDB-safe stable group identifier."""
    import hashlib

    return hashlib.sha256(str(group).encode("utf-8", "surrogatepass")).hexdigest()


def is_op_ed_video(name: str) -> bool:
    return bool(name and re.search(r"(?:^|_)(?:nc)?(?:op|ed)(?:_?\d+)?(?:_|$)", name, re.I))


class LMDBCache:
    """Small typed wrapper around one LMDB environment.

    Values written through ``put``/``put_many`` are pickle-serialized because
    the existing result records are nested Python dictionaries.  Raw methods
    are used for indexes and packed feature vectors, so those hot paths do not
    add another serialization layer.
    """

    RESULT_PREFIX = "result:"

    def __init__(self, path: str, map_size: int = 16 * 1024 * 1024):
        self.path = str(path)
        self._closed = True
        self._destroyed = False
        import os

        os.makedirs(self.path, exist_ok=True)
        self.env = lmdb.open(
            self.path,
            # Python already created the Unicode directory. Skip LMDB's ANSI mkdir on Windows.
            create=False,
            map_size=map_size,
            max_dbs=1,
            subdir=True,
            lock=True,
            readahead=False,
        )
        self._closed = False

    @staticmethod
    def _key(key) -> bytes:
        return key if isinstance(key, bytes) else str(key).encode("utf-8", "surrogatepass")

    def _write(self, operation):
        while True:
            try:
                with self.env.begin(write=True) as txn:
                    return operation(txn)
            except lmdb.MapFullError:
                self.env.set_mapsize(max(2 * self.env.info()["map_size"], 64 * 1024 * 1024))

    def get(self, key, default=None):
        raw = self.get_raw(key)
        return default if raw is None else pickle.loads(raw)

    def get_raw(self, key) -> Optional[bytes]:
        with self.env.begin(buffers=True) as txn:
            value = txn.get(self._key(key))
            return None if value is None else bytes(value)

    def put(self, key, value):
        self.put_many({key: value})

    def put_raw(self, key, value: bytes):
        self.put_raw_many({key: value})

    def put_many(self, items: Dict):
        if not items:
            return
        raw = {self._key(key): pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
               for key, value in items.items()}
        self.put_raw_many(raw)

    def put_raw_many(self, items: Dict):
        if not items:
            return

        def insert(txn):
            for key, value in items.items():
                txn.put(self._key(key), bytes(value))

        self._write(insert)

    def delete(self, key) -> bool:
        return bool(self.delete_many([key]))

    def delete_many(self, keys: Iterable) -> int:
        keys = [self._key(key) for key in keys]
        if not keys:
            return 0

        def remove(txn):
            return sum(int(txn.delete(key)) for key in keys)

        return self._write(remove)

    def count_prefix(self, prefix: str | bytes) -> int:
        prefix = self._key(prefix)
        count = 0
        with self.env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix):
                for key, _ in cursor:
                    if not bytes(key).startswith(prefix):
                        break
                    count += 1
        return count

    def iter_raw(self, prefix: str | bytes = b"") -> Iterator[Tuple[bytes, bytes]]:
        prefix = self._key(prefix)
        with self.env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            if prefix and not cursor.set_range(prefix):
                return
            if not prefix and not cursor.first():
                return
            for key, value in cursor:
                key_bytes = bytes(key)
                if prefix and not key_bytes.startswith(prefix):
                    break
                yield key_bytes, bytes(value)

    def iter_items(self, prefix: str | bytes = b"") -> Iterator[Tuple[str, object]]:
        for key, value in self.iter_raw(prefix):
            yield key.decode("utf-8", "surrogatepass"), pickle.loads(value)

    def clear_prefix(self, prefix: str | bytes):
        prefix = self._key(prefix)
        pending = []
        for key, _ in self.iter_raw(prefix):
            pending.append(key)
            if len(pending) >= 512:
                self.delete_many(pending)
                pending.clear()
        if pending:
            self.delete_many(pending)

    def put_results_many(self, results: Dict[str, dict]):
        if not results:
            return
        self.put_many({self.RESULT_PREFIX + str(key): value for key, value in results.items()})

    def delete_results_many(self, scene_keys: Iterable[str]) -> int:
        return self.delete_many(self.RESULT_PREFIX + key for key in scene_keys)

    def count_results(self) -> int:
        return self.count_prefix(self.RESULT_PREFIX)

    def iter_results_batches(self, batch_size: int = 1000):
        """Yield logical scene keys; release the read transaction before callers write."""
        prefix = self._key(self.RESULT_PREFIX)
        seek_key = prefix
        batch_size = max(1, int(batch_size))
        while True:
            batch = []
            with self.env.begin() as txn:
                cursor = txn.cursor()
                if cursor.set_range(seek_key):
                    for key, value in cursor:
                        if not key.startswith(prefix):
                            break
                        scene_key = key[len(prefix):].decode("utf-8", "surrogatepass")
                        batch.append((scene_key, pickle.loads(value)))
                        if len(batch) == batch_size:
                            # First possible key strictly after this one, even if it is deleted.
                            seek_key = key + b"\0"
                            break
            if batch:
                yield batch
            if len(batch) < batch_size:
                return

    def close(self):
        if not self._closed:
            self.env.close()
            self._closed = True

    def destroy(self):
        if self._destroyed:
            return
        self.close()
        self._destroyed = True
        try:
            shutil.rmtree(self.path, ignore_errors=True)
        except OSError:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class LMDBResultView:
    """Streaming view over ``result:*`` records."""

    def __init__(self, cache_dir: str):
        self.cache_dir = str(cache_dir)
        self._cache = LMDBCache(self.cache_dir)
        self._closed = False

    def iter_batches(self, batch_size: int = 1000):
        yield from self._cache.iter_results_batches(batch_size=batch_size)

    def __iter__(self):
        for batch in self.iter_batches(batch_size=1000):
            yield from batch

    def count(self) -> int:
        return self._cache.count_results()

    def close(self):
        if not self._closed:
            self._cache.close()
            self._closed = True

    def destroy(self):
        self._cache.destroy()
        self._closed = True

    def __len__(self):
        return self.count()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RankedResults:
    """Disk-backed hit store with ordered LMDB cursors.

    It replaces the old SQLite table.  Metadata is stored once per hit and
    score/time indexes are fixed-width keys, so page reads never materialize
    the complete result set in Python.
    """

    HIT_PREFIX = b"hit:"
    VECTOR_PREFIX = b"vec:"
    NEXT_KEY = b"meta:next"

    def __init__(self, path: str, map_size: int = 16 * 1024 * 1024):
        self.cache = LMDBCache(path, map_size=map_size)
        self.path = str(path)
        self._next = int(self.cache.get(self.NEXT_KEY, 0) or 0)

    @staticmethod
    def _hit_key(identifier: int) -> bytes:
        return RankedResults.HIT_PREFIX + _id_bytes(identifier)

    @staticmethod
    def _vec_key(identifier: int) -> bytes:
        return RankedResults.VECTOR_PREFIX + _id_bytes(identifier)

    @staticmethod
    def _score_key(identifier: int, score: float, descending: bool) -> bytes:
        prefix = b"idx:score:d:" if descending else b"idx:score:a:"
        return prefix + _sortable_float(score, descending=descending) + b":" + _id_bytes(identifier)

    @staticmethod
    def _duration_key(identifier: int, duration: float, descending: bool) -> bytes:
        prefix = b"idx:duration:d:" if descending else b"idx:duration:a:"
        return prefix + _sortable_float(duration, descending=descending) + b":" + _id_bytes(identifier)

    @staticmethod
    def _time_key(identifier: int, video: str, frame: int, descending: bool) -> bytes:
        raw_video = str(video or "").encode("utf-8", "surrogatepass")
        if descending:
            raw_video = bytes(255 - value for value in raw_video)
            frame_key = _sortable_int(frame, descending=True)
            prefix = b"idx:time:d:"
        else:
            frame_key = _sortable_int(frame)
            prefix = b"idx:time:a:"
        return prefix + raw_video + b"\0" + frame_key + b":" + _id_bytes(identifier)

    def _index_keys(self, identifier: int, record: dict) -> List[bytes]:
        return [
            self._score_key(identifier, record["score"], True),
            self._score_key(identifier, record["score"], False),
            self._duration_key(identifier, record["duration"], True),
            self._duration_key(identifier, record["duration"], False),
            self._time_key(identifier, record["video"], record["frame"], False),
            self._time_key(identifier, record["video"], record["frame"], True),
        ]

    def _record(self, match, group, op, vector):
        score = float(match.get("similarity", 0.0) or 0.0)
        record = {
            "match": dict(match),
            "score": score,
            "group": str(group or ""),
            "op": bool(op),
            "video": str(match.get("video_path", "") or ""),
            "frame": int(match.get("start_frame", 0) or 0),
            "duration": float(match.get("duration_seconds", 0.0) or 0.0),
        }
        if vector is not None:
            arr = np.ascontiguousarray(vector, dtype=np.float16)
            record["dim"] = int(arr.shape[1])
            vector = arr.tobytes()
        return record, vector

    def add_many(self, entries: Sequence[Tuple[dict, str, bool, Optional[np.ndarray]]]) -> List[int]:
        if not entries:
            return []
        raw = {}
        ids = []
        for match, group, op, vector in entries:
            identifier = self._next
            self._next += 1
            record, vector_bytes = self._record(match, group, op, vector)
            hit_key = self._hit_key(identifier)
            raw[hit_key] = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
            if vector_bytes is not None:
                raw[self._vec_key(identifier)] = vector_bytes
            for index_key in self._index_keys(identifier, record):
                raw[index_key] = _id_bytes(identifier)
            ids.append(identifier)
        raw[self.NEXT_KEY] = pickle.dumps(self._next, protocol=pickle.HIGHEST_PROTOCOL)
        self.cache.put_raw_many(raw)
        return ids

    def _load(self, identifier: int):
        raw = self.cache.get_raw(self._hit_key(identifier))
        if raw is None:
            return None
        record = pickle.loads(raw)
        vector = self.cache.get_raw(self._vec_key(identifier))
        return record, vector

    def iter_records(self, sort: str = "similarity_desc") -> Iterator[Tuple[int, dict]]:
        if sort == "__inserted__":
            with self.cache.env.begin(buffers=True) as txn:
                cursor = txn.cursor()
                prefix = self.HIT_PREFIX
                if cursor.set_range(prefix):
                    for key, value in cursor:
                        key_bytes = bytes(key)
                        if not key_bytes.startswith(prefix):
                            break
                        try:
                            identifier = int(key_bytes[len(prefix):].decode("ascii"), 16)
                        except (TypeError, ValueError):
                            continue
                        yield identifier, pickle.loads(value)["match"]
            return
        index_prefix = {
            "similarity_desc": b"idx:score:d:",
            "similarity_asc": b"idx:score:a:",
            "duration_desc": b"idx:duration:d:",
            "duration_asc": b"idx:duration:a:",
            "time_asc": b"idx:time:a:",
            "time_desc": b"idx:time:d:",
        }.get(sort, b"idx:score:d:")
        with self.cache.env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            if cursor.set_range(index_prefix):
                for key, value in cursor:
                    if not bytes(key).startswith(index_prefix):
                        break
                    try:
                        identifier = int(bytes(value).decode("ascii"), 16)
                    except (TypeError, ValueError):
                        continue
                    raw = txn.get(self._hit_key(identifier))
                    if raw is not None:
                        yield identifier, pickle.loads(raw)["match"]

    def iter_inserted(self) -> Iterator[Tuple[int, dict, dict, Optional[bytes]]]:
        with self.cache.env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            if cursor.set_range(self.HIT_PREFIX):
                for key, value in cursor:
                    key_bytes = bytes(key)
                    if not key_bytes.startswith(self.HIT_PREFIX):
                        break
                    try:
                        identifier = int(key_bytes[len(self.HIT_PREFIX):].decode("ascii"), 16)
                    except (TypeError, ValueError):
                        continue
                    record = pickle.loads(value)
                    vector = txn.get(self._vec_key(identifier))
                    yield identifier, record["match"], record, None if vector is None else bytes(vector)

    def count(self) -> int:
        return self.cache.count_prefix(self.HIT_PREFIX)

    def page(self, sort: str, offset: int, limit: int) -> List[dict]:
        offset = max(0, int(offset))
        limit = max(1, min(200, int(limit)))
        output = []
        for position, (_, match) in enumerate(self.iter_records(sort)):
            if position < offset:
                continue
            output.append(match)
            if len(output) >= limit:
                break
        return output

    def shortlist(self, limit: int) -> List[Tuple[int, dict]]:
        output = []
        for identifier, match in self.iter_records("similarity_desc"):
            output.append((identifier, match))
            if len(output) >= max(0, int(limit)):
                break
        return output

    def update(self, identifier: int, match: dict):
        loaded = self._load(identifier)
        if loaded is None:
            return
        old_record, vector = loaded
        record, _ = self._record(match, old_record["group"], old_record["op"], None)
        if "dim" in old_record:
            record["dim"] = old_record["dim"]
        raw = {
            self._hit_key(identifier): pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL),
        }
        for key in self._index_keys(identifier, old_record):
            raw[key] = None
        for key in self._index_keys(identifier, record):
            raw[key] = _id_bytes(identifier)
        self.cache._write(lambda txn: self._replace_raw(txn, raw))

    @staticmethod
    def _replace_raw(txn, items):
        for key, value in items.items():
            if value is None:
                txn.delete(key)
            else:
                txn.put(key, value)

    def delete_except(self, identifiers: Iterable[int], batch_size: int = 512):
        """Delete all hits except a streamed, bounded identifier set.

        The identifier set is normally the small shortlist produced by Jev.  It
        is intentionally materialized only once; physical keys are deleted in
        batches so a large result set never becomes one giant delete list.
        """
        keep = {int(identifier) for identifier in identifiers}
        remove = []
        for identifier, _, record, _ in self.iter_inserted():
            if identifier in keep:
                continue
            remove.extend([
                self._hit_key(identifier), self._vec_key(identifier),
                *self._index_keys(identifier, record),
            ])
            if len(remove) >= max(1, int(batch_size)) * 8:
                self.cache.delete_many(remove)
                remove.clear()
        if remove:
            self.cache.delete_many(remove)

    def delete_ids(self, identifiers: Iterable[int], batch_size: int = 512):
        remove = []
        for identifier in identifiers:
            loaded = self._load(int(identifier))
            if loaded is None:
                continue
            record, _ = loaded
            remove.extend([
                self._hit_key(int(identifier)),
                self._vec_key(int(identifier)),
                *self._index_keys(int(identifier), record),
            ])
            if len(remove) >= max(1, int(batch_size)) * 8:
                self.cache.delete_many(remove)
                remove.clear()
        if remove:
            self.cache.delete_many(remove)

    def deduplicate(self, threshold: float, tile_size: int, *, by_score: bool = True):
        """Keep OP/ED records and remove cross-video vector duplicates.

        Deduplication is deliberately two-pass and disk-backed.  The first
        pass records groups containing an OP/ED hit; the second pass compares
        one candidate at a time with the already-kept identifiers for that
        group.  No full ``records`` dictionary or full delete list is built.
        """
        tile_size = max(1, int(tile_size))
        order = "similarity_desc" if by_score else "__inserted__"
        op_prefix = b"dedup:op:"
        kept_prefix = b"dedup:kept:"
        self.cache.clear_prefix(op_prefix)
        self.cache.clear_prefix(kept_prefix)
        try:
            source = self.iter_records(order) if order != "__inserted__" else (
                (identifier, match)
                for identifier, match, _, _ in self.iter_inserted()
            )
            op_updates = {}
            for identifier, _ in source:
                loaded = self._load(identifier)
                if loaded is None:
                    continue
                record, _ = loaded
                if record.get("op"):
                    op_updates[op_prefix + _group_id(record["group"]).encode("ascii")] = b"1"
                    if len(op_updates) >= 512:
                        self.cache.put_raw_many(op_updates)
                        op_updates.clear()
            if op_updates:
                self.cache.put_raw_many(op_updates)

            source = self.iter_records(order) if order != "__inserted__" else (
                (identifier, match)
                for identifier, match, _, _ in self.iter_inserted()
            )
            remove = []
            for identifier, _ in source:
                loaded = self._load(identifier)
                if loaded is None:
                    continue
                record, vector_blob = loaded
                group_id = _group_id(record["group"])
                keep = bool(record.get("op"))
                if not keep and self.cache.get_raw(op_prefix + group_id.encode("ascii")) is None:
                    keep = True
                    dim = int(record.get("dim", 0) or 0)
                    if vector_blob is not None and dim:
                        scene = np.frombuffer(vector_blob, np.float16).reshape(-1, dim)
                        prior_prefix = kept_prefix + group_id.encode("ascii") + b":"
                        for _, prior_blob in self.cache.iter_raw(prior_prefix):
                            prior = pickle.loads(prior_blob)
                            if prior["video"] == record["video"] or int(prior.get("dim", 0) or 0) != dim:
                                continue
                            other_blob = self.cache.get_raw(self._vec_key(prior["id"]))
                            if other_blob is None:
                                continue
                            other = np.frombuffer(other_blob, np.float16).reshape(-1, dim)
                            duplicate = any(
                                (scene[i:i + tile_size] @ other[j:j + tile_size].T).max() > threshold
                                for i in range(0, len(scene), tile_size)
                                for j in range(0, len(other), tile_size)
                            )
                            if duplicate:
                                keep = False
                                break
                if keep:
                    kept_key = kept_prefix + group_id.encode("ascii") + b":" + _id_bytes(identifier)
                    self.cache.put_raw(kept_key, pickle.dumps({
                        "id": identifier,
                        "video": record["video"],
                        "dim": int(record.get("dim", 0) or 0),
                    }, protocol=pickle.HIGHEST_PROTOCOL))
                else:
                    remove.extend([
                        self._hit_key(identifier), self._vec_key(identifier),
                        *self._index_keys(identifier, record),
                    ])
                    if len(remove) >= tile_size * 8:
                        self.cache.delete_many(remove)
                        remove.clear()
            if remove:
                self.cache.delete_many(remove)
        finally:
            self.cache.clear_prefix(op_prefix)
            self.cache.clear_prefix(kept_prefix)

    def close(self):
        self.cache.close()

    def destroy(self):
        self.cache.destroy()
