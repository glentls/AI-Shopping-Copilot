"""Reciprocal Rank Fusion over an arbitrary number of ranked lists, with config-driven
per-route weights (CLAUDE.md Phase 2: "Make the fusion weights config-driven so we can
ablate them rather than argue about them")."""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    weights: list[float],
    rrf_k: int = 60,
) -> list[str]:
    if len(ranked_lists) != len(weights):
        raise ValueError("ranked_lists and weights must be the same length")
    scores: dict[str, float] = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        if weight <= 0:
            continue
        for rank, asin in enumerate(ranked_list, start=1):
            scores[asin] = scores.get(asin, 0.0) + weight / (rrf_k + rank)
    return sorted(scores, key=lambda asin: scores[asin], reverse=True)
