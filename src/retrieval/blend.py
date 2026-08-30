"""Rank fusion, soft slot reranking, and recommendation explanations."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence

from src.contracts import Candidate, ConversationState

from .bm25 import BM25Hit
from .dense import DenseHit


RRF_K = 60


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def reciprocal_rank_fusion(
    bm25_hits: Sequence[BM25Hit],
    stemmed_hits: Sequence[BM25Hit],
    dense_hits: Sequence[DenseHit],
    exact_hits: Sequence[BM25Hit],
    profile_ranks: Mapping[str, int],
    metadata: Mapping[str, tuple[float | None, int]],
    fallback: Sequence[str],
    top_n: int,
    dense_weight: float = 0.10,
    exact_phrase_mode: bool = False,
    profile_weight: float = 0.0,
) -> list[Candidate]:
    """Fuse incomparable scorer outputs by rank, never by raw magnitude."""
    weights = {
        "bm25": _env_float("TJ_BM25_WEIGHT", 1.0),
        "stemmed": _env_float(
            "TJ_STEM_EXACT_WEIGHT" if exact_phrase_mode else "TJ_STEM_WEIGHT",
            0.20 if exact_phrase_mode else 0.40,
        ),
        "dense": _env_float("TJ_DENSE_WEIGHT", dense_weight),
        "exact": _env_float("TJ_EXACT_WEIGHT", 0.275),
    }
    candidates: dict[str, Candidate] = {}
    routes = (
        ("bm25", bm25_hits),
        ("stemmed", stemmed_hits),
        ("dense", dense_hits),
        ("exact", exact_hits),
    )
    for component, hits in routes:
        weight = weights[component]
        if weight <= 0.0:
            continue
        for rank, hit in enumerate(hits, 1):
            candidate = candidates.setdefault(hit.parent_asin, Candidate(hit.parent_asin, 0.0))
            contribution = weight / (RRF_K + rank)
            candidate.components[component] = contribution
            candidate.components[f"{component}_raw"] = float(hit.score)
            candidate.score += contribution

    # Profile tags are a personalized context route, but never a catalog-wide
    # recall route: they may reorder candidates already retrieved for the live
    # request, not inject unrelated products or override explicit constraints.
    effective_profile_weight = _env_float("TJ_PROFILE_WEIGHT", profile_weight)
    if effective_profile_weight > 0.0:
        for asin, rank in profile_ranks.items():
            candidate = candidates.get(asin)
            if candidate is None or rank <= 0:
                continue
            contribution = effective_profile_weight / (RRF_K + rank)
            candidate.components["profile_rrf"] = contribution
            candidate.score += contribution

    # No route may produce an empty/short recommendation list. Popular items
    # are weak lottery tickets, but still better than unused slots.
    for asin in fallback:
        if len(candidates) >= max(10, top_n):
            break
        candidates.setdefault(asin, Candidate(asin, 0.0, {"fallback": 1.0}))

    for candidate in candidates.values():
        profile_rank = profile_ranks.get(candidate.parent_asin)
        candidate.components["profile"] = 0.0 if profile_rank is None else 1.0 / profile_rank
        rating_number = metadata.get(candidate.parent_asin, (None, 0))[1]
        candidate.components["popularity"] = math.log1p(max(0, rating_number))

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.score,
            candidate.components["profile"],
            candidate.components["popularity"],
            candidate.parent_asin,
        ),
        reverse=True,
    )
    return ranked[:top_n]


def lock_hard_constraints(
    candidates: list[Candidate],
    constraints: Sequence[tuple[str, Sequence[str]]],
    table,
    min_results: int = 10,
) -> list[Candidate]:
    """Put catalog-backed Buying matches first, with a recall-safe backoff.

    A constraint is OR within one slot and AND across slots.  The lock is only
    activated when enough complete matches exist to fill the evaluator's top
    K; sparse or noisy catalog metadata therefore cannot empty the result set.
    """
    normalized = [
        (slot, {value for value in values if value})
        for slot, values in constraints
        if values
    ]
    if not normalized or not candidates:
        return candidates

    complete: list[Candidate] = []
    remainder: list[Candidate] = []
    for candidate in candidates:
        matched = 0
        for slot, wanted in normalized:
            held = set(table.values(candidate.parent_asin, slot))
            if held & wanted:
                matched += 1
        candidate.components["hard_match_ratio"] = matched / len(normalized)
        (complete if matched == len(normalized) else remainder).append(candidate)

    required = min(max(1, min_results), len(candidates))
    if len(complete) < required:
        for candidate in candidates:
            candidate.components["hard_filter_backoff"] = 1.0
        return candidates

    for candidate in complete:
        candidate.components["hard_filter"] = 1.0
    return complete + remainder


def rerank_candidates(
    candidates: list[Candidate],
    state: ConversationState,
    table,
    metadata: Mapping[str, tuple[float | None, int]],
) -> list[Candidate]:
    """Move matches a few rank places and apply soft-only constraint penalties."""
    slot_weight = _env_float("TJ_SLOT_WEIGHT", 2.0)
    budget_weight = _env_float("TJ_BUDGET_WEIGHT", 0.5)
    catalog_confidence_power = max(
        0.0, _env_float("TJ_CATALOG_CONFIDENCE_POWER", 1.0)
    )
    tie_data: dict[str, tuple[float, float]] = {}
    catalog_confidence_for = getattr(table, "confidence", None)

    for base_rank, candidate in enumerate(candidates):
        slot_signal = 0.0
        best_match: tuple[float, str, str] | None = None
        for slot, requested in state.slots.items():
            if slot == "budget":
                continue
            held = set(table.values(candidate.parent_asin, slot))
            if not held:
                continue
            for value in requested:
                if value.value not in held:
                    continue
                customer_confidence = max(0.0, min(1.0, float(value.confidence)))
                reported_catalog_confidence = (
                    catalog_confidence_for(candidate.parent_asin, slot, value.value)
                    if callable(catalog_confidence_for)
                    else None
                )
                # A table built without provenance still has valid values.
                # Treat its missing/zero metadata as unknown, not as evidence
                # that an otherwise confirmed catalog match should score zero.
                catalog_confidence = (
                    float(reported_catalog_confidence)
                    if reported_catalog_confidence
                    else 1.0
                )
                catalog_confidence = max(0.0, min(1.0, float(catalog_confidence)))
                provenance_weight = (
                    catalog_confidence ** catalog_confidence_power
                    if catalog_confidence_power > 0.0
                    else 1.0
                )
                confidence = customer_confidence * provenance_weight
                if value.polarity:
                    slot_signal += confidence
                    if best_match is None or confidence > best_match[0]:
                        best_match = confidence, slot, value.value
                else:
                    slot_signal -= 1.5 * confidence

        slot_contribution = slot_signal * slot_weight
        budget_contribution = 0.0
        price = metadata.get(candidate.parent_asin, (None, 0))[0]
        if (
            state.budget_max
            and state.budget_max > 0
            and price is not None
            and price > state.budget_max
        ):
            over_ratio = price / state.budget_max
            budget_contribution = -budget_weight * min(2.0, math.log2(over_ratio))

        candidate.components["slot"] = slot_contribution
        candidate.components["budget"] = budget_contribution
        # Base ordering is RRF. Slot and price contributions are expressed in
        # rank places, so the environment weights stay independent of raw
        # BM25 and cosine score scales.
        candidate.score = -float(base_rank) + slot_contribution + budget_contribution

        if candidate.components.get("exact", 0.0) > 0:
            candidate.why = "it matches a specific requirement you mentioned"
        elif best_match is not None:
            candidate.why = f"it matches your {best_match[2]} preference"
        elif candidate.components.get("dense", 0.0) > candidate.components.get(
            "bm25", 0.0
        ):
            candidate.why = "it semantically matches your request"
        elif max(
            candidate.components.get("bm25", 0.0),
            candidate.components.get("stemmed", 0.0),
        ) > 0:
            candidate.why = "it is a strong keyword match for your request"
        else:
            candidate.why = "it is a popular option while preferences are broad"

        tie_data[candidate.parent_asin] = (
            candidate.components.get("profile", 0.0),
            candidate.components.get("popularity", 0.0),
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.score,
            tie_data[candidate.parent_asin][0],
            tie_data[candidate.parent_asin][1],
            candidate.parent_asin,
        ),
        reverse=True,
    )
    return candidates
