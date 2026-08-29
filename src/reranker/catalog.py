"""Catalog loading for the reranker.

Loads the frozen catalog once and precomputes, per product:
    - lowercased searchable text (mirrors the evaluator's field flattening)
    - price (float or None)
    - rating_number, average_rating (for popularity tiebreaks / fallback)

The searchable-text construction intentionally matches the evaluator's
``searchable_text`` (local_evaluator.py) so that coverage matching aligns with
how the simulator discloses constraints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")


def _flatten(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key} {item}" for key, item in value.items()]
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is not None:
        return [str(value)]
    return []


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        parts.extend(_flatten(product.get(field)))
    return " ".join(parts).strip()


def _price(product: dict) -> float | None:
    value = product.get("price")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Product:
    parent_asin: str
    text: str          # lowercased searchable text
    price: float | None
    rating_number: int
    average_rating: float


class Catalog:
    """In-memory catalog: id -> Product, plus field rows for FTS indexing."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.products: dict[str, Product] = {}
        self._rows: list[tuple] = []
        self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                pid = str(raw["parent_asin"])
                self.products[pid] = Product(
                    parent_asin=pid,
                    text=searchable_text(raw).lower(),
                    price=_price(raw),
                    rating_number=int(raw.get("rating_number") or 0),
                    average_rating=float(raw.get("average_rating") or 0.0),
                )
                self._rows.append(
                    (
                        pid,
                        " ".join(_flatten(raw.get("title"))),
                        " ".join(_flatten(raw.get("categories"))),
                        " ".join(_flatten(raw.get("features"))),
                        " ".join(_flatten(raw.get("details"))),
                        " ".join(_flatten(raw.get("store"))),
                        " ".join(_flatten(raw.get("description"))),
                    )
                )

    def fts_rows(self) -> list[tuple]:
        return self._rows

    def popularity_top(self, k: int) -> list[str]:
        ordered = sorted(
            self.products.values(),
            key=lambda p: (p.rating_number, p.average_rating),
            reverse=True,
        )
        return [p.parent_asin for p in ordered[:k]]

    def __len__(self) -> int:
        return len(self.products)
