"""Rank/score fusion of the BM25 and dense candidate lists.

Two methods behind one config flag (`config.retrieval.fusion.method`):

  rrf     Reciprocal Rank Fusion: score(d) = Σ_i w_i / (k + rank_i(d)). Robust, ignores score
          magnitude entirely -- only position matters.
  minmax  per-list min-max normalise the raw scores to [0, 1], then weighted sum.
  zscore  per-list z-score normalise, then weighted sum.

RRF is the default. minmax/zscore keep score magnitude, which can help on a homogeneous
catalog where rank gaps understate how much better the top hit is -- Phase 5 A/Bs all three
(see docs/r1_log.md) rather than picking on principle.

Fusion is done over the top ~`depth` of each list, not the top 10 -- a gold ranked 150 by BM25
but 30 by dense is only recoverable if both lists are read that deep.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from src.contracts import Candidate, ProductMeta, RetrievalResult


def rrf(ranked_lists: dict[str, list[str]], k: float, weights: dict[str, float]) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for name, ids in ranked_lists.items():
        weight = weights.get(name, 1.0)
        for rank, parent_asin in enumerate(ids, start=1):
            scores[parent_asin] += weight / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def _normalise(scored: list[tuple[str, float]], method: str) -> dict[str, float]:
    if not scored:
        return {}
    values = [score for _, score in scored]
    if method == "minmax":
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        return {asin: (score - low) / span for asin, score in scored}
    if method == "zscore":
        mean = statistics.fmean(values)
        sd = statistics.pstdev(values) or 1.0
        return {asin: (score - mean) / sd for asin, score in scored}
    raise ValueError(f"unknown normalisation method: {method!r}")


def score_fusion(
    scored_lists: dict[str, list[tuple[str, float]]], weights: dict[str, float], method: str
) -> list[tuple[str, float]]:
    """Weighted sum of per-list normalised scores. An item absent from a list contributes 0
    (i.e. the min of a min-max list, ~the mean of a z-score list) -- a deliberate, mild penalty
    for not appearing at all."""
    totals: dict[str, float] = defaultdict(float)
    for name, scored in scored_lists.items():
        weight = weights.get(name, 1.0)
        for asin, norm in _normalise(scored, method).items():
            totals[asin] += weight * norm
    return sorted(totals.items(), key=lambda pair: pair[1], reverse=True)


def fuse_results(
    results: dict[str, RetrievalResult | list[Candidate]], config: dict
) -> RetrievalResult:
    """Fuse named candidate lists into one RetrievalResult (route='fused'). `config` is the
    `retrieval.fusion` block."""
    method = config.get("method", "rrf")
    depth = int(config.get("depth", 200))
    weights = {name: float(w) for name, w in config.get("weights", {}).items()}

    meta: dict[str, ProductMeta] = {}
    for candidates in results.values():
        for candidate in candidates:
            meta.setdefault(candidate.parent_asin, candidate.meta)

    if method == "rrf":
        ranked_lists = {
            name: [c.parent_asin for c in list(candidates)[:depth]]
            for name, candidates in results.items()
        }
        fused = rrf(ranked_lists, float(config.get("rrf_k", 60)), weights)
    else:
        scored_lists = {
            name: [(c.parent_asin, c.score) for c in list(candidates)[:depth]]
            for name, candidates in results.items()
        }
        fused = score_fusion(scored_lists, weights, method)

    candidates = [
        Candidate(parent_asin=asin, score=float(score), route="fused", meta=meta[asin])
        for asin, score in fused
        if asin in meta
    ]
    primary = results.get("bm25")
    pool_size = getattr(primary, "pool_size", len(candidates))
    dropped = list(getattr(primary, "dropped_constraints", []))
    return RetrievalResult(candidates, pool_size=pool_size or len(candidates), dropped_constraints=dropped)
