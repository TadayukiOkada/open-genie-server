"""On-disk KV-cache snapshots for system-prompt prefixes.

MISS: reset -> query(prefix, SENTENCE_BEGIN, noop) -> save
           -> query(remaining, SENTENCE_END, cb)
HIT:  reset -> restore -> query(remaining, SENTENCE_END, cb)
"""

import contextlib
import hashlib
import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# What key() produces. Anything else is refused before it reaches a path.
KEY_RE = re.compile(r"[0-9a-f]{16}")


class PrefixCache:
    def __init__(self, cache_dir: str) -> None:
        self.last_restore_ms: float | None = None
        self.last_save_ms: float | None = None
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Keys a save or restore is working on right now, with a count. A
        # prune skips them: deleting a directory GenieDialog_save is still
        # writing would leave a half-written entry that lists as reachable.
        # _lock is held across each prune deletion too, so a save or restore
        # of that key waits for it rather than starting halfway through.
        self._lock = threading.Lock()
        self._in_use: dict[str, int] = {}
        logger.info(f"PrefixCache: {self._dir}")

    def key(self, text: str, namespace: str = "") -> str:
        """namespace must encode whatever changes the KV-cache's meaning for
        a given prompt text — slot + active model id + LoRA adapter, at
        minimum. Without this, restoring a cache saved under one slot/model/
        LoRA into a dialog now running a different one would silently feed
        garbage KV state into generation. Since the key is content-addressed,
        switching slot/model/LoRA just changes the hash — old entries are
        never touched, they simply become unreachable (and can still be
        listed/deleted via the /v1/prefix/cache endpoints, which are
        namespace-agnostic)."""
        return hashlib.sha256(f"{namespace}\x1f{text}".encode("utf-8")).hexdigest()[:16]

    def _path(self, key: str) -> str:
        if not KEY_RE.fullmatch(key):
            raise ValueError(f"not a prefix cache key: {key!r} "
                             "(16 lowercase hex digits)")
        return str(self._dir / f"prefix_{key}.geniestate")

    def _meta_path(self, key: str) -> Path:
        """A small sidecar naming the namespace the entry was saved under.
        The key is a hash, so without it nobody can tell which entries a
        model or LoRA change has made unreachable."""
        return Path(self._path(key)).with_suffix(".json")

    def namespace_of(self, key: str) -> str | None:
        """The namespace recorded at save time; None for an entry saved
        before this was recorded, or whose record is unreadable."""
        try:
            ns = json.loads(self._meta_path(key).read_text()).get("namespace")
        except (OSError, ValueError, AttributeError):
            return None
        return ns if isinstance(ns, str) else None

    @contextlib.contextmanager
    def _using(self, key: str):
        with self._lock:
            self._in_use[key] = self._in_use.get(key, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                if self._in_use[key] > 1:
                    self._in_use[key] -= 1
                else:
                    del self._in_use[key]

    def _record_namespace(self, key: str, namespace: str) -> None:
        try:
            self._meta_path(key).write_text(json.dumps(
                {"namespace": namespace, "saved": int(time.time())}))
        except OSError as e:
            # The entry itself is fine; it just lists as unknown.
            logger.warning(f"Prefix cache: could not record the "
                           f"namespace of key={key}: {e}")

    def note_namespace(self, key: str, namespace: str) -> None:
        """Records the namespace of an entry saved before namespaces were,
        once a slot reaches it. The key hashes the namespace, so a key a slot
        computed from its current namespace and found on disk can only
        belong to that namespace. Without this an upgraded cache would list
        every old entry as unknown forever, and only scope=all could clear
        them."""
        if self.exists(key) and self.namespace_of(key) is None:
            self._record_namespace(key, namespace)

    def exists(self, key: str) -> bool:
        return Path(self._path(key)).exists()

    def save(self, lib, handle, key: str, namespace: str | None = None) -> bool:
        t0 = time.perf_counter()
        with self._using(key):
            ret = lib.save_state(handle, self._path(key))
            if ret == 0 and namespace is not None:
                self._record_namespace(key, namespace)
        self.last_save_ms = (time.perf_counter() - t0) * 1000
        if ret == 0:
            p = Path(self._path(key))
            size = (sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    if p.is_dir() else p.stat().st_size if p.exists() else -1)
            logger.info(f"Prefix cache SAVED  key={key}  size={size} B")
            return True
        logger.warning(f"GenieDialog_save failed: {ret}  key={key}")
        return False

    def restore(self, lib, handle, key: str) -> bool:
        """GenieDialog_restore is a blocking call that reads the saved KV
        state itself, and the SDK does not profile it — there is no
        DIALOG_RESTORE event type and `applyEngineState` covers a different
        path (qualla/dialog.cpp:2653) — so time it here. `last_restore_ms`
        is what /v1/server/profile reports as host-measured."""
        t0 = time.perf_counter()
        with self._using(key):
            ret = lib.restore_state(handle, self._path(key))
        self.last_restore_ms = (time.perf_counter() - t0) * 1000
        if ret == 0:
            logger.info(f"Prefix cache HIT    key={key}  "
                        f"restore={self.last_restore_ms:.1f}ms")
            return True
        logger.warning(f"GenieDialog_restore failed: {ret}  key={key}")
        return False

    def delete(self, key: str) -> bool:
        p = Path(self._path(key))
        self._meta_path(key).unlink(missing_ok=True)
        if p.is_dir():
            try:
                shutil.rmtree(p)
            except FileNotFoundError:
                pass  # a concurrent delete got there first; it finishes
            logger.info(f"Prefix cache DELETED key={key} (dir)")
            return True
        if p.is_file():
            p.unlink(missing_ok=True)
            logger.info(f"Prefix cache DELETED key={key} (file)")
            return True
        return False

    def list_entries(self, current_namespaces: set[str] | None = None) -> list:
        """Every entry, with the namespace it was saved under and, given the
        slots' current namespaces, whether any slot can still reach it
        (None when the namespace was not recorded)."""
        entries = []
        for p in sorted(self._dir.glob("prefix_*.geniestate")):
            key = p.name.removeprefix("prefix_").removesuffix(".geniestate")
            if not KEY_RE.fullmatch(key):
                continue
            try:
                st = p.stat()
                size = (sum(f.stat().st_size for f in p.rglob("*")
                            if f.is_file())
                        if p.is_dir() else st.st_size)
            except FileNotFoundError:
                continue  # deleted by a concurrent call while we listed
            ns = self.namespace_of(key)
            entries.append({
                "key": key,
                "path": str(p),
                "kind": "directory" if p.is_dir() else "file",
                "size_bytes": size,
                "mtime": int(st.st_mtime),
                "namespace": ns,
                "reachable": (None if ns is None or current_namespaces is None
                              else ns in current_namespaces),
            })
        return entries

    def prune(self, current_namespaces: set[str], *,
              include_unknown: bool = False) -> dict:
        """Deletes the entries no slot can reach: saved under a namespace no
        slot has now. An entry whose namespace was not recorded is kept
        unless include_unknown, since it may well be reachable. Nothing is
        ever deleted on its own -- the cache fills only on an explicit
        warmup, and it empties only on an explicit call like this one, so a
        TTFT measurement never changes behind the caller's back.

        An entry a save or restore is working on right now is kept and
        listed in kept_in_use. A namespace record left without its entry
        (the entry removed by hand) is swept too."""
        deleted: list[str] = []
        unknown: list[str] = []
        in_use: list[str] = []
        freed = 0
        for e in self.list_entries(current_namespaces):
            if e["reachable"] is None and not include_unknown:
                unknown.append(e["key"])
            elif not e["reachable"]:
                with self._lock:
                    if e["key"] in self._in_use:
                        in_use.append(e["key"])
                    elif self.delete(e["key"]):
                        deleted.append(e["key"])
                        freed += max(e["size_bytes"], 0)
        for meta in self._dir.glob("prefix_*.json"):
            key = meta.name.removeprefix("prefix_").removesuffix(".json")
            if not KEY_RE.fullmatch(key):
                continue
            with self._lock:
                if key not in self._in_use and not self.exists(key):
                    meta.unlink(missing_ok=True)
                    logger.info(f"Prefix cache: removed the namespace record "
                                f"of missing entry key={key}")
        return {"deleted": deleted, "freed_bytes": freed,
                "kept_unknown": unknown, "kept_in_use": in_use}
