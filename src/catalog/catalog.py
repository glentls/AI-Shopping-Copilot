"""In-memory catalog index.

Loads ``data/catalog.jsonl`` once into a single in-memory sqlite3 FTS5
virtual table. One :class:`Catalog` instance owns the connection and is shared
(read-only) by downstream components such as the retriever, so the 50k-row
index is built exactly once at startup.

Table schema (``products``):

* ``parent_asin``     UNINDEXED  -- returned, never matched
* ``title``, ``categories``, ``features``, ``details``, ``store``,
  ``description``     -- BM25 text columns
* ``price``, ``average_rating`` UNINDEXED  -- numeric filter columns
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


# The BM25 text columns, in declared / insert order.
TEXT_COLUMNS: tuple[str, ...] = (
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
)

TABLE_NAME = "products"


def _text(value: object) -> str:
    """Flatten a catalog value into a single searchable string."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


class Catalog:
    """Owns the shared in-memory FTS5 index over the product catalog."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._build_index()

    @property
    def text_columns(self) -> tuple[str, ...]:
        return TEXT_COLUMNS

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            f"CREATE VIRTUAL TABLE {TABLE_NAME} USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "price UNINDEXED, average_rating UNINDEXED, rating_number UNINDEXED, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        insert = f"INSERT INTO {TABLE_NAME} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                        product.get("price"),
                        product.get("average_rating"),
                        product.get("rating_number"),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany(insert, batch)
                    batch.clear()
        if batch:
            cursor.executemany(insert, batch)
        self.connection.commit()

    def execute(self, sql: str, params: tuple | list = ()) -> list:
        """Run a read query against the shared index and return all rows."""
        return self.connection.execute(sql, params).fetchall()
