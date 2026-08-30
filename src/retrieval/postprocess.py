"""Phase 6 -- candidate-pool post-processing applied after BM25 / fusion, before the result
leaves `retriever.search()`.

Three passes, each independently config-gated and each a no-op on an empty input:

  1. category hard filter + relaxation ladder  (driven by `request.hard_filters`)
  2. popularity prior                           (`config.retrieval.popularity`)
  3. soft preferences                           (driven by `request.soft_prefs`)

`request.hard_filters` / `request.soft_prefs` are empty under NullDialog / NullMemory, so with
today's wiring passes 1 and 3 are inert and `search()` is byte-for-byte its pre-Phase-6 self.
They exist so that when R3/R5 populate those fields the machinery is already there, measured,
and behind flags. Pass 2 (popularity) applies to every query and is the one live lever -- see
docs/r1_log.md for the A/B.

Reordering passes preserve each Candidate's original `score` and only change list order: a
downstream reranker keys off `route` + position + `meta`, and rewriting `score` with a blended
[0,1] value would mislead it about the lexical/vector signal strength.

Nothing here imports another component.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from src.contracts import Candidate, ProductMeta, RetrievalResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Breadcrumb scaffolding that carries no filtering signal -- every clothing row has these.
_CATEGORY_STOPWORDS = frozenset({
    "clothing", "shoes", "jewelry", "men", "women", "womens", "mens", "girls", "boys",
    "kids", "baby", "unisex", "adult", "novelty", "more", "the", "and", "of", "s",
})


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1}


@dataclass
class CategoryIndex:
    """`parent_asin -> set of category breadcrumb tokens`, for the `category` hard filter.
    Built once from the same ProductMeta map BM25 uses; no FTS, no catalog re-read."""

    tokens_of: dict[str, frozenset[str]]

    @classmethod
    def build(cls, products: dict[str, ProductMeta]) -> "CategoryIndex":
        tokens_of: dict[str, frozenset[str]] = {}
        for asin, meta in products.items():
            toks: set[str] = set()
            for crumb in meta.categories:
                toks |= _tokens(crumb)
            tokens_of[asin] = frozenset(toks - _CATEGORY_STOPWORDS)
        return cls(tokens_of=tokens_of)

    def matches(self, asin: str, want: frozenset[str]) -> bool:
        """AND semantics: every meaningful term in `want` must appear in the row's breadcrumb.
        An empty `want` (all terms were scaffolding) matches everything -- nothing to filter on."""
        if not want:
            return True
        return want <= self.tokens_of.get(asin, frozenset())


# --------------------------------------------------------------------------- #
# Pass 1: category hard filter + relaxation ladder
# --------------------------------------------------------------------------- #

_FILTERABLE = ("category",)  # the only hard_filters key R1 can enforce today


def filter_candidates(
    candidates: list[Candidate],
    hard_filters: dict,
    cat_index: CategoryIndex | None,
    config: dict,
) -> tuple[list[Candidate], list[str]]:
    """Apply the enforceable hard filters, relaxing the lowest-priority one whenever the
    surviving pool would fall below `min_pool_size`. Returns `(kept, dropped_keys)`.

    `dropped_keys` lists every requested filter not reflected in `kept`: ones the ladder
    relaxed *and* ones R1 cannot enforce (e.g. `color`) -- so a downstream caller knows which
    constraints to re-ask rather than assume were honoured.
    """
    if not hard_filters:
        return candidates, []

    rel_cfg = config["retrieval"].get("relaxation", {})
    min_pool = int(rel_cfg.get("min_pool_size", 10))
    priority = list(rel_cfg.get("priority") or _FILTERABLE)

    unenforceable = [k for k in hard_filters if k not in _FILTERABLE]
    active = [k for k in priority if k in hard_filters and k in _FILTERABLE]
    active += [k for k in hard_filters if k in _FILTERABLE and k not in active]

    dropped: list[str] = list(unenforceable)
    if cat_index is None:
        return candidates, dropped + [k for k in active]

    want = {k: _tokens(str(hard_filters[k])) - _CATEGORY_STOPWORDS for k in active}

    while True:
        kept = [
            c for c in candidates
            if all(cat_index.matches(c.parent_asin, frozenset(want[k])) for k in active if k == "category")
        ]
        if len(kept) >= min_pool or not active:
            return kept if active else candidates, dropped
        dropped.append(active.pop())  # relax the lowest-priority active filter, retry


# --------------------------------------------------------------------------- #
# Pass 2: popularity prior
# --------------------------------------------------------------------------- #


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    return [(v - low) / span for v in values]


def apply_popularity_prior(candidates: list[Candidate], config: dict) -> list[Candidate]:
    """Reorder the pool by `(1 - w) * lexical_norm + w * popularity_norm`, both min-max
    normalised across the *current pool*. `w = config.retrieval.popularity.weight`.

    Rationale (Phase 0): the gold target sits at catalog rating-count percentile ~99.5, so a
    mild lean toward well-reviewed items rescues golds that BM25 buried at rank 30-90 without
    displacing a strong lexical match. Scores are preserved; only order changes.
    """
    pop_cfg = config["retrieval"].get("popularity", {})
    if not pop_cfg.get("enabled", False) or len(candidates) < 2:
        return candidates
    weight = float(pop_cfg.get("weight", 0.0))
    if weight <= 0.0:
        return candidates

    lex_norm = _minmax([c.score for c in candidates])
    pop_norm = _minmax([math.log1p(max(0, c.meta.rating_number)) for c in candidates])
    blended = sorted(
        range(len(candidates)),
        key=lambda i: (1.0 - weight) * lex_norm[i] + weight * pop_norm[i],
        reverse=True,
    )
    return [candidates[i] for i in blended]


# --------------------------------------------------------------------------- #
# Pass 3: soft preferences
# --------------------------------------------------------------------------- #


def apply_soft_prefs(candidates: list[Candidate], soft_prefs: dict, config: dict) -> list[Candidate]:
    """Small additive nudges from `request.soft_prefs` (a dict R5/memory populates; empty
    today). A missing attribute on a candidate is neutral (0), never a penalty -- `price` is
    null in ~79% of the catalog, so penalising unknown price would punish most of it.

    Supported keys: `store` (str -- substring match, small boost), `price_max` (float --
    boost items at or under it, neutral for null price). Order changes only; scores preserved.
    """
    soft_cfg = config["retrieval"].get("soft_prefs", {})
    if not soft_prefs or not soft_cfg.get("enabled", False) or len(candidates) < 2:
        return candidates

    store_boost = float(soft_cfg.get("store_boost", 0.0))
    price_boost = float(soft_cfg.get("price_boost", 0.0))
    want_store = str(soft_prefs.get("store") or "").lower().strip()
    price_max = soft_prefs.get("price_max")
    price_max = float(price_max) if isinstance(price_max, (int, float)) else None
    if not ((want_store and store_boost) or (price_max is not None and price_boost)):
        return candidates

    def bonus(meta: ProductMeta) -> float:
        b = 0.0
        if want_store and meta.store and want_store in meta.store.lower():
            b += store_boost
        if price_max is not None and meta.price is not None and meta.price <= price_max:
            b += price_boost
        return b

    lex_norm = _minmax([c.score for c in candidates])
    order = sorted(
        range(len(candidates)),
        key=lambda i: lex_norm[i] + bonus(candidates[i].meta),
        reverse=True,
    )
    return [candidates[i] for i in order]


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def postprocess(
    result: RetrievalResult,
    hard_filters: dict,
    soft_prefs: dict,
    cat_index: CategoryIndex | None,
    config: dict,
) -> RetrievalResult:
    """filter -> popularity -> soft-prefs, order only. Does not truncate -- the caller
    (`retriever.search`) owns the pool depth it hands to ranking. `pool_size` reports the
    post-filter pool depth; `dropped_constraints` lists relaxed / unenforceable filter keys."""
    candidates = list(result)
    kept, dropped = filter_candidates(candidates, hard_filters, cat_index, config)
    pool_size = len(kept)

    kept = apply_popularity_prior(kept, config)
    kept = apply_soft_prefs(kept, soft_prefs, config)

    existing_dropped = list(getattr(result, "dropped_constraints", []))
    return RetrievalResult(
        kept,
        pool_size=pool_size,
        dropped_constraints=existing_dropped + dropped,
    )
