from __future__ import annotations

import math
import re

from starter.retrieval.catalog_store import CatalogStore, ProductMeta
from starter.retrieval.filters import constraint_phrases
from starter.retrieval.query_builder import STOPWORDS

EXACT_PHRASE_SCORE = 10.0
TERM_MATCH_SCORE = 1.5
CATEGORY_MATCH_SCORE = 6.0
PRICE_EXACT_SCORE = 14.0
PRICE_NEAR_SCORE = 3.0
PRICE_OUTLIER_PENALTY = 5.0
POPULARITY_WEIGHT = 2.0

CATEGORY_RE = re.compile(r"i'?m looking for\s+([^.\n]+)", re.I)
CATEGORY_NOISE = {"clothing", "item", "shoes", "jewelry"}
PRICE_RE = re.compile(r"(?:budget around|under|<=|about)?\s*\$\s*([\d]+(?:\.[\d]+)?)", re.I)
PRICE_EXACT_TOLERANCE = 0.01
PRICE_NEAR_RATIO = 0.10
PRICE_OUTLIER_RATIO = 1.25


def rerank_candidates(
    candidates: list[str],
    filters: dict,
    query_text: str,
    slot_values: list[str],
    store: CatalogStore,
) -> list[str]:
    if not candidates:
        return []

    phrases = [_normalize(phrase) for phrase in constraint_phrases(query_text, slot_values)]
    phrases = [phrase for phrase in phrases if phrase]
    terms = _unique_terms(phrases)
    category_tokens = _category_tokens(query_text)
    target_price = _target_price(filters, query_text)
    profile = filters.get("profile")

    scored: list[tuple[float, int, str]] = []
    for index, asin in enumerate(candidates):
        product = store.get(asin)
        if product is None:
            continue
        score = _score_product(product, phrases, terms, category_tokens, target_price, profile)
        scored.append((score, -index, asin))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [asin for _, _, asin in scored]


def _score_product(
    product: ProductMeta,
    phrases: list[str],
    terms: list[str],
    category_tokens: list[str],
    target_price: float | None,
    profile: object = None,
) -> float:
    corpus = _normalize(product.searchable_text)
    score = 0.0

    for phrase in phrases:
        if phrase in corpus:
            score += EXACT_PHRASE_SCORE

    for term in terms:
        if term in corpus:
            score += TERM_MATCH_SCORE

    score += _category_score(product, category_tokens)
    score += _price_score(product, target_price)
    score += _popularity_score(product)

    if profile is not None:
        from starter.personalization.rerank_boost import compute_profile_boost

        score += compute_profile_boost(product, profile)

    return score


def _category_score(product: ProductMeta, category_tokens: list[str]) -> float:
    """The stated product type comes from the target's own category path, so it always holds."""
    if not category_tokens:
        return 0.0
    categories = product.categories.lower()
    matched = sum(1 for token in category_tokens if token in categories)
    return CATEGORY_MATCH_SCORE * matched / len(category_tokens)


def _price_score(product: ProductMeta, target_price: float | None) -> float:
    if target_price is None or product.price is None:
        return 0.0
    if abs(product.price - target_price) <= PRICE_EXACT_TOLERANCE:
        return PRICE_EXACT_SCORE
    if abs(product.price - target_price) <= target_price * PRICE_NEAR_RATIO:
        return PRICE_NEAR_SCORE
    if product.price > target_price * PRICE_OUTLIER_RATIO:
        return -PRICE_OUTLIER_PENALTY
    return 0.0


def _popularity_score(product: ProductMeta) -> float:
    """Purchased items skew popular, so review volume is a useful prior."""
    if not product.rating_number:
        return 0.0
    return POPULARITY_WEIGHT * math.log10(1 + product.rating_number)


def _category_tokens(query_text: str) -> list[str]:
    match = CATEGORY_RE.search(query_text)
    if not match:
        return []
    tokens = _content_tokens(_normalize(match.group(1)))
    return [token for token in tokens if token not in CATEGORY_NOISE]


def _target_price(filters: dict, query_text: str) -> float | None:
    raw = filters.get("max_price")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    match = PRICE_RE.search(query_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace(":", " ").replace("%", " % ")).strip().lower()


def _content_tokens(phrase: str) -> list[str]:
    return [
        token
        for token in dict.fromkeys(phrase.split())
        if len(token) > 2 and token not in STOPWORDS
    ]


def _unique_terms(phrases: list[str]) -> list[str]:
    terms: list[str] = []
    for phrase in phrases:
        for token in _content_tokens(phrase):
            if token not in terms:
                terms.append(token)
    return terms
