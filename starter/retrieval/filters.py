from __future__ import annotations

import re

from starter.retrieval.catalog_store import CatalogStore, ProductMeta
from starter.retrieval.query_builder import extract_constraint_phrases

PRICE_TOLERANCE = 0.20


def apply_metadata_filters(
    candidates: list[str],
    filters: dict,
    store: CatalogStore,
    mode: str = "browsing",
) -> list[str]:
    if not candidates or not _has_active_constraints(filters):
        return candidates

    filtered = [asin for asin in candidates if _matches_filters(store.get(asin), filters)]
    if not filtered:
        return candidates if mode == "browsing" else filtered
    return filtered


def _has_active_constraints(filters: dict) -> bool:
    constraint_keys = (
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
        "max_price",
        "budget",
    )
    return any(filters.get(key) not in (None, "", []) for key in constraint_keys)


def _matches_filters(product: ProductMeta | None, filters: dict) -> bool:
    if product is None:
        return False
    if not _price_matches(product, filters):
        return False
    if not _text_constraints_match(product, "material", _values(filters, "material")):
        return False
    if not _text_constraints_match(product, "color", _values(filters, "color", "colour")):
        return False
    if not _text_constraints_match(product, "size", _values(filters, "size")):
        return False
    if not _text_constraints_match(product, "style", _values(filters, "style")):
        return False
    if not _text_constraints_match(product, "feature", _values(filters, "feature")):
        return False
    if not _text_constraints_match(product, "use_case", _values(filters, "use_case")):
        return False
    if not _brand_matches(product, _values(filters, "brand")):
        return False
    if not _category_matches(product, _values(filters, "category", "category_tokens")):
        return False
    return True


def _values(filters: dict, *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        raw = filters.get(key)
        if isinstance(raw, list):
            values.extend(str(item).strip() for item in raw if str(item).strip())
        elif raw not in (None, ""):
            values.append(str(raw).strip())
    return list(dict.fromkeys(value for value in values if value))


def _price_matches(product: ProductMeta, filters: dict) -> bool:
    max_price = filters.get("max_price")
    if max_price is None:
        return True
    if product.price is None:
        return True
    try:
        limit = float(max_price)
    except (TypeError, ValueError):
        return True
    return product.price <= limit * (1.0 + PRICE_TOLERANCE)


def _text_constraints_match(product: ProductMeta, _field: str, values: list[str]) -> bool:
    if not values:
        return True
    corpus = product.searchable_text.lower()
    return any(value.lower() in corpus for value in values)


def _brand_matches(product: ProductMeta, values: list[str]) -> bool:
    if not values:
        return True
    corpus = f"{product.store} {product.title}".lower()
    return any(value.lower() in corpus for value in values)


def _category_matches(product: ProductMeta, values: list[str]) -> bool:
    if not values:
        return True
    corpus = product.categories.lower()
    return any(value.lower() in corpus for value in values)


def constraint_phrases(query_text: str, slot_values: list[str]) -> list[str]:
    phrases = extract_constraint_phrases(query_text)
    for value in slot_values:
        if value and value not in phrases:
            phrases.append(value)
    return list(dict.fromkeys(phrase for phrase in phrases if phrase))
