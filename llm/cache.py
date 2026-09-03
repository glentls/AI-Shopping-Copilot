"""On-disk cache for LLM rerank calls, keyed by the exact request. This is what
guarantees determinism (CLAUDE.md: "Two consecutive evaluator runs must produce
identical scores") -- not a temperature parameter, since claude-opus-5 and the rest of
the current model family removed temperature/top_p/top_k control entirely (sending them
is a 400, not a no-op). Same model + same query + same candidate set always resolves to
the same cached ranking after the first call, regardless of any underlying model
stochasticity on that first call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_CACHE_DIR = "data/llm_cache"


def cache_key(model: str, query: str, candidate_ids: list[str]) -> str:
    payload = json.dumps({"model": model, "query": query, "candidates": candidate_ids}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)

    def get(self, key: str) -> dict | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
