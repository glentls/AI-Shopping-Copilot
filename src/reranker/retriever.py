"""Candidate retrieval: SQLite FTS5 BM25 over the catalog.

Reuses the proven starter configuration. Returns a ranked ``list[str]`` of
``parent_asin`` (the retrieval -> reranker boundary). Kept separable so a
different retrieval implementation can replace it without touching the reranker.
"""

from __future__ import annotations

import re
import sqlite3

from src.reranker.catalog import Catalog

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# BM25 column weights: title, categories, features, details, store, description.
_BM25_WEIGHTS = "bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)"


def terms(text: str) -> list[str]:
    return [
        tok.lower()
        for tok in TOKEN_RE.findall(text)
        if len(tok) > 1 and tok.lower() not in STOPWORDS
    ]


class Retriever:
    def __init__(self, catalog: Catalog) -> None:
        self.connection = sqlite3.connect(":memory:")
        self._build_index(catalog)

    def _build_index(self, catalog: Catalog) -> None:
        cur = self.connection.cursor()
        cur.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        cur.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", catalog.fts_rows()
        )
        self.connection.commit()

    def search(self, query: str, limit: int = 200) -> list[str]:
        unique = list(dict.fromkeys(terms(query)))[:40]
        if not unique:
            return []
        expression = " OR ".join(f'"{t}"' for t in unique)
        rows = self.connection.execute(
            f"SELECT parent_asin FROM products WHERE products MATCH ? "
            f"ORDER BY {_BM25_WEIGHTS} LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(r[0]) for r in rows]
