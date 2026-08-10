import json
import hashlib


def _key(community_id: str, seeds: list[str], alpha: float, engine: str, **kwargs) -> str:
    seed_hash = hashlib.md5(json.dumps(sorted(seeds)).encode(), usedforsecurity=False).hexdigest()
    prior_hash = kwargs.get('prior_hash', '')
    return f"{community_id}:{engine}:{alpha:.4f}:{seed_hash}:{prior_hash}"


class PPRCache:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._cache: dict[str, object] = {}

    def get(self, key: str):
        return self._cache.get(key)

    def put(self, key: str, value: object):
        if len(self._cache) >= self.capacity:
            # Simple FIFO eviction
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value


