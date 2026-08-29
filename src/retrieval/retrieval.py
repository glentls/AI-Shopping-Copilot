"""BM25 retrieval over the product catalog.

Backed by sqlite3 FTS5 (same engine used by the weak baseline in
``src/agent.py``), so it adds no new dependencies and scales to the full
50k-row catalog. A single ``Retriever`` instance builds one in-memory index
and is safe to reuse across sessions (reads only).

The public entrypoint is :meth:`Retriever.retrieve_bm25`, which consumes the
same ``dict[str, list]`` "search key" shape the ledger stores, e.g.::

    {"type": ["jacket"], "price": [{"lte": 30.0}]}

Field values are auto-classified by shape:

* list of **strings**  -> soft, weighted BM25 *text* terms (e.g. ``type``,
  ``color``, ``material``, ``brand``, ``style``, ``keywords``).
* list of **dicts** with ``gte``/``lte``/``gt``/``lt``/``eq`` keys -> a
  *numeric range filter* (e.g. ``price``, ``average_rating``).
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# The FTS columns, in the order they are declared / inserted. ``parent_asin``
# is UNINDEXED (returned, never matched) and the two trailing numeric columns
# are UNINDEXED filter columns.
_TEXT_COLUMNS: tuple[str, ...] = (
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
)

# Default per-column BM25 weights, mirroring src/agent.py's bm25() call
# (parent_asin column is weight 0.0 -- never contributes to the score).
_DEFAULT_WEIGHTS: dict[str, float] = {
    "title": 6.0,
    "categories": 4.0,
    "features": 2.5,
    "details": 2.5,
    "store": 1.5,
    "description": 1.0,
}

# Maps a search-key *text field* to the FTS columns it should search. A field
# not listed here falls back to every text column.
_DEFAULT_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "type": ("title", "categories"),
    "category": ("title", "categories"),
    "color": ("title", "features", "description"),
    "material": ("features", "details"),
    "brand": ("store", "title"),
    "store": ("store", "title"),
    "style": ("title", "features", "description"),
    "use_case": ("title", "features", "description"),
    "feature": _TEXT_COLUMNS,
    "keywords": _TEXT_COLUMNS,
}

# Search-key operator -> SQL comparison operator.
_OP_TO_SQL: dict[str, str] = {
    "gte": ">=",
    "lte": "<=",
    "gt": ">",
    "lt": "<",
    "eq": "=",
}

# Search-key field -> numeric catalog column it filters on.
_NUMERIC_FIELD_TO_COLUMN: dict[str, str] = {
    "price": "price",
    "average_rating": "average_rating",
    "rating": "average_rating",
}


def _text(value: object) -> str:
    """Flatten a catalog value into a single searchable string."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    """Lowercase content tokens, dropping stopwords and 1-char noise."""
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _is_numeric_filter(value: object) -> bool:
    """A value is a numeric range filter iff it is a list of dicts whose keys
    are all recognized operators (``gte``/``lte``/``gt``/``lt``/``eq``)."""
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict) or not item:
            return False
        if any(key not in _OP_TO_SQL for key in item):
            return False
    return True


class Retriever:
    """Stateless (read-only) BM25 retriever over the product catalog."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        weights: dict[str, float] | None = None,
        field_map: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.weights = {**_DEFAULT_WEIGHTS, **(weights or {})}
        self.field_map = {**_DEFAULT_FIELD_MAP, **(field_map or {})}
        self.connection = sqlite3.connect(":memory:")
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "price UNINDEXED, average_rating UNINDEXED, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
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
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
                    )
                    batch.clear()
        if batch:
            cursor.executemany(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
            )
        self.connection.commit()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve_bm25(self, search_key: dict[str, list], top_k: int = 10) -> list[str]:
        """Return up to ``top_k`` ``parent_asin`` strings ranked by weighted
        BM25, filtered by any numeric range constraints in ``search_key``.

        Text fields (list-of-strings) drive the weighted BM25 score. Numeric
        fields (list-of-``{op: value}``) are hard range filters, except that a
        product with a NULL value for that column always passes (missing price
        should not exclude a product)."""
        search_key = search_key or {}

        match_expression = self._build_match_expression(search_key)
        where_clause, where_params = self._build_numeric_filter(search_key)

        bm25_args = ", ".join(
            str(self.weights.get(col, 0.0)) for col in _TEXT_COLUMNS
        )
        # parent_asin (col 0) is weight 0.0; trailing UNINDEXED numeric columns
        # are omitted -- bm25() only weights the text columns.
        rank = f"bm25(products, 0.0, {bm25_args})"

        if match_expression:
            sql = (
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                f"{where_clause} ORDER BY {rank} LIMIT ?"
            )
            params = [match_expression, *where_params, top_k]
        elif where_clause:
            # No text terms: numeric-filter-only query, best-rated first.
            sql = (
                "SELECT parent_asin FROM products "
                f"{where_clause.replace('AND', 'WHERE', 1)} "
                "ORDER BY average_rating DESC LIMIT ?"
            )
            params = [*where_params, top_k]
        else:
            return []

        rows = self.connection.execute(sql, params).fetchall()
        return [str(row[0]) for row in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_match_expression(self, search_key: dict[str, list]) -> str:
        """Build the FTS5 MATCH expression from the text fields, expanding
        each field's terms into its mapped columns via column-filter syntax."""
        clauses: list[str] = []
        for field, value in search_key.items():
            if field in _NUMERIC_FIELD_TO_COLUMN or _is_numeric_filter(value):
                continue
            if not isinstance(value, list):
                continue
            columns = self.field_map.get(field, _TEXT_COLUMNS)
            column_filter = "{" + " ".join(columns) + "}"
            for raw in value:
                for term in dict.fromkeys(_terms(str(raw))):
                    clauses.append(f'{column_filter}: "{term}"')
        # OR-combine every term/column clause, mirroring the baseline's loose
        # recall-oriented matching.
        return " OR ".join(dict.fromkeys(clauses))

    def _build_numeric_filter(self, search_key: dict[str, list]) -> tuple[str, list]:
        """Build the SQL WHERE fragment (leading ``AND``) and bound params for
        numeric range filters. NULL column values always pass."""
        conditions: list[str] = []
        params: list = []
        for field, value in search_key.items():
            if not _is_numeric_filter(value):
                continue
            column = _NUMERIC_FIELD_TO_COLUMN.get(field)
            if column is None:
                continue
            for item in value:
                for op, bound in item.items():
                    sql_op = _OP_TO_SQL[op]
                    conditions.append(f"({column} IS NULL OR {column} {sql_op} ?)")
                    params.append(bound)
        if not conditions:
            return "", []
        return "AND " + " AND ".join(conditions), params
