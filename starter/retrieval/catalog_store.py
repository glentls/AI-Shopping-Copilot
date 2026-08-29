from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

# BM25 column weights: parent_asin, title, categories, features, details, store, description
BM25_WEIGHTS = (0.0, 6.0, 4.0, 4.0, 4.0, 1.5, 1.0)
BM25_ORDER_BY = f"bm25(products, {', '.join(str(weight) for weight in BM25_WEIGHTS)})"


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _parse_price(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ProductMeta:
    parent_asin: str
    title: str
    categories: str
    features: str
    details: str
    store: str
    description: str
    price: float | None
    average_rating: float | None
    rating_number: int | None
    searchable_text: str


class CatalogStore:
    """In-memory FTS5 index and structured metadata cache for the product catalog."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self._connection = sqlite3.connect(":memory:")
        self._products: dict[str, ProductMeta] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                features = _text(product.get("features"))
                details = _text(product.get("details"))
                store = _text(product.get("store"))
                description = _text(product.get("description"))
                searchable_text = " ".join(
                    part for part in (title, categories, features, details, store, description) if part
                )
                self._products[parent_asin] = ProductMeta(
                    parent_asin=parent_asin,
                    title=title,
                    categories=categories,
                    features=features,
                    details=details,
                    store=store,
                    description=description,
                    price=_parse_price(product.get("price")),
                    average_rating=_parse_price(product.get("average_rating")),
                    rating_number=_parse_int(product.get("rating_number")),
                    searchable_text=searchable_text,
                )
                batch.append((parent_asin, title, categories, features, details, store, description))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self._connection.commit()

    def get(self, parent_asin: str) -> ProductMeta | None:
        return self._products.get(parent_asin)

    def all_asins(self) -> list[str]:
        return list(self._products.keys())

    def bm25_search(self, fts_expression: str, limit: int) -> list[str]:
        if not fts_expression.strip() or limit <= 0:
            return []
        rows = self._connection.execute(
            f"SELECT parent_asin FROM products WHERE products MATCH ? ORDER BY {BM25_ORDER_BY} LIMIT ?",
            (fts_expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def count_matches(self, fts_expression: str) -> int:
        if not fts_expression.strip():
            return 0
        row = self._connection.execute(
            "SELECT COUNT(*) FROM products WHERE products MATCH ?",
            (fts_expression,),
        ).fetchone()
        return int(row[0]) if row else 0

    @property
    def size(self) -> int:
        return len(self._products)
