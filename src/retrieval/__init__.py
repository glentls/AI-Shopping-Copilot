"""Retrieval leaf component. Never imports ranking/, dialog/, or memory/.

The Phase-2 walking skeleton's retrieval IS the shipped weak baseline (SQLite FTS5 BM25, ported
from starter/agent.py) -- per the Phase-2 instructions, retrieval's permanent fallback is the
existing baseline, not an empty result set, since an empty candidate list can never score.
"""

from .bm25 import BM25Index, build_index, search

__all__ = ["BM25Index", "build_index", "search"]
