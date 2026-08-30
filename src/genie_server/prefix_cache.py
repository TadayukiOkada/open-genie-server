"""On-disk KV-cache snapshots for system-prompt prefixes.

MISS: reset -> query(prefix, SENTENCE_BEGIN, noop) -> save
           -> query(remaining, SENTENCE_END, cb)
HIT:  reset -> restore -> query(remaining, SENTENCE_END, cb)
"""

import hashlib
import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class PrefixCache:
    def __init__(self, cache_dir: str) -> None:
        self.last_restore_ms: float | None = None
        self.last_save_ms: float | None = None
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
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
        return str(self._dir / f"prefix_{key}.geniestate")

    def exists(self, key: str) -> bool:
        return Path(self._path(key)).exists()

    def save(self, lib, handle, key: str) -> bool:
        t0 = time.perf_counter()
        ret = lib.save_state(handle, self._path(key))
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
        if p.is_dir():
            shutil.rmtree(p)
            logger.info(f"Prefix cache DELETED key={key} (dir)")
            return True
        if p.is_file():
            p.unlink()
            logger.info(f"Prefix cache DELETED key={key} (file)")
            return True
        return False

    def list_entries(self) -> list:
        entries = []
        for p in sorted(self._dir.glob("prefix_*")):
            st = p.stat()
            size = (sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    if p.is_dir() else st.st_size)
            entries.append({
                "key": p.name.removeprefix("prefix_").removesuffix(".geniestate"),
                "path": str(p),
                "kind": "directory" if p.is_dir() else "file",
                "size_bytes": size,
                "mtime": int(st.st_mtime),
            })
        return entries
