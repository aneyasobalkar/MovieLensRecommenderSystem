import time
from typing import Any, Optional


class TTLCache:
    """Thread-unsafe in-memory TTL cache — fine for single-process dev server."""

    def __init__(self, default_ttl: int = 300):
        self._store: dict[str, tuple[Any, float]] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._store[key] = (value, time.monotonic() + (ttl or self.default_ttl))

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def delete_prefix(self, prefix: str) -> None:
        for k in [k for k in self._store if k.startswith(prefix)]:
            del self._store[k]

    @property
    def size(self) -> int:
        return len(self._store)


# Module-level singleton used by the API
rec_cache = TTLCache(default_ttl=300)
