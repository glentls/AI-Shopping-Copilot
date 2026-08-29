from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from starter.retrieval.catalog_store import CatalogStore
from starter.retrieval.feedback import build_retrieval_feedback
from starter.retrieval.filters import apply_metadata_filters
from starter.retrieval.query_builder import build_fts_expression, strip_boilerplate, tokenize_terms
from starter.retrieval.reranker import rerank_candidates

CANDIDATE_POOL_SIZE = 60
FALLBACK_BROAD_EXPRESSION = '"clothing" OR "shoes" OR "apparel"'


@dataclass(frozen=True)
class SearchResult:
    asins: list[str]
    feedback: dict[str, object] = field(default_factory=dict)


class HybridSearcher:
    """Pillar I entry point: query text and filters in, ranked ASINs and feedback out."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self._store = CatalogStore(catalog_path)

    def search(
        self,
        query_text: str,
        mode: str = "browsing",
        filters: dict | None = None,
        top_k: int = 10,
    ) -> SearchResult:
        filters = filters or {}
        normalized_mode = "buying" if mode == "buying" else "browsing"
        slot_values = _slot_values_from_filters(filters, normalized_mode)
        pool_size = max(top_k, CANDIDATE_POOL_SIZE)
        relaxed_search = False

        # Buying starts from a precision-first AND expression; when nothing satisfies every
        # disclosed requirement at once, the recall-first route below takes over.
        expression = build_fts_expression(query_text, normalized_mode, slot_values)
        if not expression:
            expression = _fallback_expression(query_text)

        candidate_count = self._store.count_matches(expression)
        asins = self._store.bm25_search(expression, limit=pool_size)

        if not asins and normalized_mode == "buying":
            relaxed_expression = build_fts_expression(query_text, "browsing", slot_values)
            if relaxed_expression:
                relaxed_search = True
                expression = relaxed_expression
                candidate_count = self._store.count_matches(expression)
                asins = self._store.bm25_search(expression, limit=pool_size)

        if not asins:
            relaxed_search = True
            expression = _fallback_expression(query_text)
            candidate_count = self._store.count_matches(expression)
            asins = self._store.bm25_search(expression, limit=max(top_k, CANDIDATE_POOL_SIZE))

        if not asins:
            relaxed_search = True
            asins = self._store.all_asins()[: max(top_k, CANDIDATE_POOL_SIZE)]
            candidate_count = len(self._store.all_asins())

        asins = apply_metadata_filters(asins, filters, self._store, normalized_mode)
        if not asins:
            relaxed_search = True
            asins = self._store.bm25_search(expression, limit=max(top_k, CANDIDATE_POOL_SIZE))

        asins = rerank_candidates(asins, filters, query_text, slot_values, self._store)

        feedback = build_retrieval_feedback(
            candidate_count,
            filters,
            relaxed_search=relaxed_search,
        )
        return SearchResult(asins=asins[:top_k], feedback=feedback)

def _slot_values_from_filters(filters: dict, mode: str = "browsing") -> list[str]:
    values: list[str] = []
    for key in (
        "material",
        "color",
        "colour",
        "size",
        "style",
        "brand",
        "feature",
        "use_case",
        "category",
        "category_tokens",
        "budget",
    ):
        raw = filters.get(key)
        if isinstance(raw, list):
            values.extend(
                str(item).strip() for item in raw if _is_concise_slot_value(str(item).strip())
            )
        elif raw not in (None, ""):
            value = str(raw).strip()
            if _is_concise_slot_value(value):
                values.append(value)
    if filters.get("max_price") is not None:
        values.append(f"budget around ${filters['max_price']}")
    return list(dict.fromkeys(value for value in values if value))


def _is_concise_slot_value(value: str) -> bool:
    """Whole utterances crowd out real constraints in the phrase budget, so drop them."""
    lowered = value.lower()
    utterance_markers = (
        "i'm looking for", "i am looking for", "key requirement is",
        "what matters is", "what i need is", "those options",
    )
    return bool(value) and len(value) <= 80 and not any(marker in lowered for marker in utterance_markers)


def _fallback_expression(query_text: str) -> str:
    terms = tokenize_terms(strip_boilerplate(query_text))
    if terms:
        return " OR ".join(f'"{term}"' for term in terms[:10])
    return FALLBACK_BROAD_EXPRESSION
