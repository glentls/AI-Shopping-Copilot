"""The retrieval component's public entry point: `build_index()` + `search()`.

`search()` is BM25 by default. When `config.retrieval.fusion.enabled` AND a dense index is
available, it fuses BM25 + dense (src/retrieval/fusion.py). Anything that goes wrong on the
dense/fusion path -- disabled, cache missing, encoder import fails, a runtime error -- falls
back to the plain BM25 result and logs. There is no configuration in which `search()` returns
fewer candidates than BM25 alone would have.

agent.py imports `build_index` and `search` from `src.retrieval` and is unchanged: `build_index`
still returns something whose `.products` / `.fallback_pool` the agent's fallback path reads,
and `search(index, request, config)` still returns a `RetrievalResult`.
"""

from __future__ import annotations

import dataclasses
import traceback
from dataclasses import dataclass
from pathlib import Path

from src.config import load_config
from src.contracts import RetrievalRequest, RetrievalResult

from .bm25 import BM25Index
from .bm25 import build_index as build_bm25_index
from .bm25 import search as bm25_search
from .dense import DenseIndex, build_dense_index
from .dense import dense_search
from .fusion import fuse_results
from .multiturn import SessionMemory, build_effective_query
from .postprocess import CategoryIndex, postprocess


@dataclass
class HybridIndex:
    """Carries the BM25 index and, optionally, the dense index. Delegates the attributes
    agent.py's fallback path expects (`products`, `fallback_pool`) to the BM25 index so the
    orchestrator needs no changes."""

    bm25: BM25Index
    dense: DenseIndex | None = None
    cat_index: CategoryIndex | None = None
    session_memory: SessionMemory | None = None

    @property
    def products(self):
        return self.bm25.products

    @property
    def fallback_pool(self):
        return self.bm25.fallback_pool

    @property
    def connection(self):
        return self.bm25.connection

    @property
    def stopwords(self):
        return self.bm25.stopwords


def build_index(catalog_path: str | Path, config: dict | None = None) -> HybridIndex:
    config = config or load_config()
    bm25 = build_bm25_index(catalog_path, config)
    cat_index = CategoryIndex.build(bm25.products)

    fusion_cfg = config["retrieval"].get("fusion", {})
    dense: DenseIndex | None = None
    if fusion_cfg.get("enabled", False):
        dense = build_dense_index(catalog_path, bm25.products, config)
        if dense is not None:
            try:
                # Load the encoder now, on the constructing thread. agent.py calls search()
                # from a 1-worker ThreadPoolExecutor with a short timeout; a ~70s lazy model
                # load in there would time out every component call until it finished.
                dense.encode_queries(["warmup"])
            except Exception:
                print("[retriever] dense encoder warmup failed; disabling dense route:\n"
                      + traceback.format_exc())
                dense = None
    return HybridIndex(bm25=bm25, dense=dense, cat_index=cat_index, session_memory=SessionMemory())


def search(index: HybridIndex, request: RetrievalRequest, config: dict | None = None) -> RetrievalResult:
    config = config or load_config()
    retrieval_cfg = config["retrieval"]
    bm25_index = index.bm25 if isinstance(index, HybridIndex) else index
    dense_index = index.dense if isinstance(index, HybridIndex) else None
    cat_index = index.cat_index if isinstance(index, HybridIndex) else None

    # Phase 7: rebuild the query from the whole session (+ profile + feedback) before anything
    # else touches it. Inert unless config.retrieval.multiturn.enabled AND the request carries
    # a turn/session_id -- under today's wiring it returns request.canonical_query unchanged.
    session_memory = index.session_memory if isinstance(index, HybridIndex) else None
    if session_memory is not None:
        try:
            effective_query = build_effective_query(
                request, session_memory, config, bm25_index.stopwords, bm25_index.products
            )
            if effective_query != request.canonical_query:
                request = dataclasses.replace(request, canonical_query=effective_query)
        except Exception:
            print("[retriever] multi-turn query build failed; using the raw query:\n"
                  + traceback.format_exc())

    fusion_cfg = retrieval_cfg.get("fusion", {})
    fusion_on = dense_index is not None and fusion_cfg.get("enabled", False)

    # When an enforceable hard filter is active, read further down each list so post-filtering
    # doesn't starve the pool; then trim back to a normal ranking pool. Inert today (NullDialog
    # sends hard_filters={}), so `base_depth` collapses to exactly the pre-Phase-6 value.
    has_filter = bool(request.hard_filters) and any(
        k == "category" for k in request.hard_filters
    )
    fusion_depth = int(fusion_cfg.get("depth", 200))
    normal_pool = max(request.top_k, int(retrieval_cfg["candidate_pool_size"]))
    if fusion_on:
        base_depth = int(retrieval_cfg.get("filters", {}).get("prefilter_pool_size", 800)) if has_filter else fusion_depth
    else:
        base_depth = int(retrieval_cfg.get("filters", {}).get("prefilter_pool_size", 800)) if has_filter else request.top_k

    widened = request if request.top_k >= base_depth else dataclasses.replace(request, top_k=base_depth)
    bm25_result = bm25_search(bm25_index, widened, config)

    result = bm25_result
    if fusion_on:
        try:
            dense_result = dense_search(dense_index, request.canonical_query, base_depth, config=config)
            result = fuse_results({"bm25": bm25_result, "dense": dense_result}, fusion_cfg)
        except Exception:
            print("[retriever] fusion failed; returning BM25 only:\n" + traceback.format_exc())
            result = bm25_result

    try:
        result = postprocess(result, request.hard_filters, request.soft_prefs, cat_index, config)
    except Exception:
        print("[retriever] postprocess failed; returning pre-postprocess result:\n" + traceback.format_exc())

    if has_filter and len(result) > normal_pool:
        result = RetrievalResult(
            list(result)[:normal_pool],
            pool_size=result.pool_size,
            dropped_constraints=result.dropped_constraints,
        )
    return result
