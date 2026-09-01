"""Loads the read-only catalog once and builds the shared text/field views every
retrieval route needs, so each route doesn't re-derive them from raw records.

Schema actually observed on the 50k-row catalog (see docs/ablations.md Phase 2 notes):
- `price` is null on 79% of rows (39,473/50,000) -- never hard-filter on it.
- `details.Brand`/`details["Brand Name"]` covers only ~4.7% of rows; `store` covers
  99.4% (only 314 nulls) and is the reliable brand-like signal.
- `categories` is always a list, 3-8 entries, rooted at "Clothing, Shoes & Jewelry".
"""

from __future__ import annotations

import json
from pathlib import Path


def _join(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def load_catalog(catalog_path: str | Path) -> dict[str, dict]:
    """parent_asin -> raw catalog record, insertion order preserved (dict is ordered)."""
    products: dict[str, dict] = {}
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            products[str(record["parent_asin"])] = record
    return products


def category_path(product: dict) -> str:
    categories = product.get("categories") or []
    return " ".join(str(c) for c in categories)


def lexical_text(product: dict) -> str:
    """title + description + category path, per CLAUDE.md Phase 2 spec for the lexical route."""
    return " ".join(
        part
        for part in (str(product.get("title") or ""), _join(product.get("description")), category_path(product))
        if part
    )


DENSE_TEXT_CHAR_CAP = 500


def dense_text(product: dict) -> str:
    """Richer text for the semantic route: adds features, which the lexical route omits.

    Capped at DENSE_TEXT_CHAR_CAP characters. This isn't just a size nicety: on this
    project's dev hardware, uncapped catalog text (avg 1,242 chars, max 5,836) made a
    single embedding pass over the 50k catalog project to ~22 hours with BAAI/bge-small
    -en-v1.5, vs. ~37 minutes for a title+lead-description+features window at this cap
    with all-MiniLM-L6-v2 (see retrieval/dense.py for the model choice writeup). Longer
    catalog text is dominated by boilerplate (shipping/care instructions, brand history)
    well past the point of adding retrieval-relevant signal, so the truncation is a
    reasonable trade, not just a speed hack.
    """
    text = " ".join(
        part
        for part in (
            str(product.get("title") or ""),
            _join(product.get("description")),
            _join(product.get("features")),
            category_path(product),
        )
        if part
    )
    return text[:DENSE_TEXT_CHAR_CAP]


def price_of(product: dict) -> float | None:
    price = product.get("price")
    return float(price) if isinstance(price, (int, float)) else None


def store_of(product: dict) -> str:
    store = product.get("store")
    return str(store) if store else ""
