"""Retrieval leaf component. Never imports ranking/, dialog/, or memory/.

The Phase-2 walking skeleton's retrieval IS the shipped weak baseline (SQLite FTS5 BM25, ported
from starter/agent.py) -- per the Phase-2 instructions, retrieval's permanent fallback is the
existing baseline, not an empty result set, since an empty candidate list can never score.
"""

from src.contracts import RetrievalResult

from .bm25 import BM25Index, load_products
from .bm25 import build_index as build_bm25_index
from .bm25 import search as bm25_search
from .dense import DenseIndex, build_dense_index, dense_search, dense_search_batch
from .fusion import fuse_results, rrf, score_fusion
from .multiturn import (
    SessionMemory,
    accumulate_query,
    blend_profile,
    build_effective_query,
    rocchio_terms,
)
from .postprocess import (
    CategoryIndex,
    apply_popularity_prior,
    apply_soft_prefs,
    filter_candidates,
    postprocess,
)
from .retriever import HybridIndex, build_index, search

__all__ = [
    # public entry point (fusion-aware; BM25-only unless config.retrieval.fusion.enabled)
    "build_index", "search", "HybridIndex", "RetrievalResult",
    # primitives
    "BM25Index", "build_bm25_index", "bm25_search", "load_products",
    "DenseIndex", "build_dense_index", "dense_search", "dense_search_batch",
    "fuse_results", "rrf", "score_fusion",
    "CategoryIndex", "postprocess", "filter_candidates", "apply_popularity_prior", "apply_soft_prefs",
    "SessionMemory", "build_effective_query", "accumulate_query", "blend_profile", "rocchio_terms",
]
