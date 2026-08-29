from __future__ import annotations

from starter.retrieval.catalog_store import CatalogStore, ProductMeta
from starter.retrieval.filters import constraint_phrases

EXACT_PHRASE_SCORE = 10
TERM_MATCH_SCORE = 3
PRICE_OUTLIER_PENALTY = 5


def rerank_candidates(
    candidates: list[str],
    filters: dict,
    query_text: str,
    slot_values: list[str],
    store: CatalogStore,
) -> list[str]:
    if not candidates:
        return []

    phrases = constraint_phrases(query_text, slot_values)
    terms = _unique_terms(phrases)
    max_price = _parse_max_price(filters)

    scored: list[tuple[float, int, str]] = []
    for index, asin in enumerate(candidates):
        product = store.get(asin)
        if product is None:
            continue
        score = _score_product(product, phrases, terms, max_price, filters.get("profile"))
        scored.append((score, -index, asin))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [asin for _, _, asin in scored]


def _score_product(
    product: ProductMeta,
    phrases: list[str],
    terms: list[str],
    max_price: float | None,
    profile: object = None,
) -> float:
    corpus = product.searchable_text.lower()
    score = 0.0

    for phrase in phrases:
        normalized = phrase.lower().strip()
        if normalized and normalized in corpus:
            score += EXACT_PHRASE_SCORE

    for term in terms:
        if term in corpus:
            score += TERM_MATCH_SCORE

    if max_price is not None and product.price is not None and product.price > max_price * 1.2:
        score -= PRICE_OUTLIER_PENALTY

    if profile is not None:
        from starter.personalization.rerank_boost import compute_profile_boost

        score += compute_profile_boost(product, profile)

    return score


def _unique_terms(phrases: list[str]) -> list[str]:
    terms: list[str] = []
    for phrase in phrases:
        for token in phrase.lower().split():
            if len(token) > 2 and token not in terms:
                terms.append(token)
    return terms


def _parse_max_price(filters: dict) -> float | None:
    raw = filters.get("max_price")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
