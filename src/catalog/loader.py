"""Shared, process-cached catalog row loader.

``Catalog._build_index``, ``popularity_top10``, and ``load_catalog_vocab``
each used to open and ``json.loads`` every line of ``data/catalog.jsonl``
independently -- three redundant full-file scans. ``load_catalog_rows`` does
the parse once per distinct path and caches the result; whichever consumer
runs first pays for the parse, the other two reuse it.

Returned rows are read-only: callers must not mutate the dicts in place, or
the mutation would leak into every other consumer sharing the cached tuple.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def load_catalog_rows(catalog_path: str) -> tuple[dict, ...]:
    """Parse ``catalog_path`` (JSONL) once and cache the result by path."""
    with Path(catalog_path).open(encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle if line.strip())
