"""Persistent SQLite FTS5 keyword retrieval for the product catalog."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ARTIFACT_NAME = "bm25.sqlite3"
SCHEMA_VERSION = "2"
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# These words describe the conversation rather than the desired product. FTS
# sees every repeated word as evidence, so removing simulator scaffolding is
# materially more important than a long general-purpose stop-word list.
STOPWORDS = {
    "a", "additional", "an", "and", "are", "as", "ask", "at", "attribute",
    "be", "but", "by", "could", "do", "exploring", "for", "from", "have", "i",
    "in", "is", "it", "judgment", "key", "looking", "matters", "me", "my",
    "need", "no", "not", "of", "on", "options", "or", "please", "preference",
    "quite", "requirement", "right", "some", "specific", "still", "that",
    "the", "this", "those", "to", "use", "want", "what", "with", "would",
    "well", "yet", "you", "your",
}

# FTS5's bm25() expects one weight per column, including the unindexed ASIN.
# Product-name and taxonomy matches are the most discriminative; descriptions
# are verbose and therefore deliberately light.
FIELD_WEIGHTS = (0.0, 6.0, 2.5, 1.0, 4.0, 1.5, 2.5)


@dataclass(frozen=True)
class BM25Hit:
    parent_asin: str
    score: float


def text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(text: str) -> list[str]:
    """Return unique, FTS-safe terms while preserving their first order."""
    result: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(text or ""):
        token = token.lower()
        if len(token) <= 1 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _insert_batches(cursor: sqlite3.Cursor, rows: Iterable[tuple], sql: str) -> None:
    batch: list[tuple] = []
    for row in rows:
        batch.append(row)
        if len(batch) == 1000:
            cursor.executemany(sql, batch)
            batch.clear()
    if batch:
        cursor.executemany(sql, batch)


def build_bm25_index(catalog_path: str | Path, artifacts_dir: str | Path) -> Path:
    """Build the disk-backed FTS index and compact ranking metadata."""
    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ARTIFACT_NAME
    temporary = directory / f"{ARTIFACT_NAME}.tmp"
    if temporary.exists():
        temporary.unlink()

    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            "PRAGMA journal_mode=OFF;"
            "PRAGMA synchronous=OFF;"
            "PRAGMA temp_store=MEMORY;"
            "CREATE TABLE artifact_info(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, features, description, categories, store, details, "
            "tokenize='unicode61 remove_diacritics 2');"
            "CREATE VIRTUAL TABLE products_stemmed USING fts5("
            "parent_asin UNINDEXED, title, features, description, categories, store, details, "
            "tokenize='porter unicode61 remove_diacritics 2');"
            "CREATE TABLE product_meta("
            "parent_asin TEXT PRIMARY KEY, price REAL, rating_number INTEGER NOT NULL);"
        )
        connection.execute(
            "INSERT INTO artifact_info VALUES('schema_version', ?)", (SCHEMA_VERSION,)
        )

        def product_rows() -> Iterable[tuple]:
            with Path(catalog_path).open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    product = json.loads(line)
                    yield (
                        str(product["parent_asin"]),
                        text_value(product.get("title")),
                        text_value(product.get("features")),
                        text_value(product.get("description")),
                        text_value(product.get("categories")),
                        text_value(product.get("store")),
                        text_value(product.get("details")),
                    )

        _insert_batches(
            connection.cursor(), product_rows(), "INSERT INTO products VALUES(?,?,?,?,?,?,?)"
        )
        _insert_batches(
            connection.cursor(),
            product_rows(),
            "INSERT INTO products_stemmed VALUES(?,?,?,?,?,?,?)",
        )

        def metadata_rows() -> Iterable[tuple]:
            with Path(catalog_path).open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    product = json.loads(line)
                    raw_price = product.get("price")
                    price = float(raw_price) if isinstance(raw_price, (int, float)) else None
                    raw_count = product.get("rating_number") or 0
                    count = int(raw_count) if isinstance(raw_count, (int, float)) else 0
                    yield str(product["parent_asin"]), price, count

        _insert_batches(
            connection.cursor(), metadata_rows(), "INSERT INTO product_meta VALUES(?,?,?)"
        )
        connection.execute("INSERT INTO products(products) VALUES('optimize')")
        connection.execute(
            "INSERT INTO products_stemmed(products_stemmed) VALUES('optimize')"
        )
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, target)
    return target


class BM25Index:
    """Read-only query wrapper around the built FTS5 artifact."""

    def __init__(self, artifact_path: str | Path) -> None:
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 artifact missing: {path}")
        self.connection = sqlite3.connect(
            f"file:{path.resolve()}?mode=ro", uri=True, check_same_thread=False
        )
        version = self.connection.execute(
            "SELECT value FROM artifact_info WHERE key='schema_version'"
        ).fetchone()
        if not version or version[0] != SCHEMA_VERSION:
            self.connection.close()
            raise ValueError(f"unsupported BM25 artifact schema in {path}")

    @staticmethod
    def _weights_sql() -> str:
        return ", ".join(str(weight) for weight in FIELD_WEIGHTS)

    def _execute(
        self,
        expression: str,
        limit: int,
        table: str = "products",
    ) -> list[BM25Hit]:
        if not expression or limit <= 0:
            return []
        if table not in {"products", "products_stemmed"}:
            raise ValueError(f"unsupported FTS table: {table}")
        rows = self.connection.execute(
            f"SELECT parent_asin, bm25({table}, " + self._weights_sql() + ") AS rank "
            f"FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?",
            (expression, int(limit)),
        ).fetchall()
        return [BM25Hit(str(asin), -float(rank)) for asin, rank in rows]

    def search(self, query: str, limit: int) -> list[BM25Hit]:
        query_terms = terms(query)[:48]
        expression = " OR ".join(f'"{term}"' for term in query_terms)
        return self._execute(expression, limit)

    def stemmed_search(self, query: str, limit: int) -> list[BM25Hit]:
        """Search grammatical variants using FTS5's built-in Porter stemmer."""
        query_terms = terms(query)[:48]
        expression = " OR ".join(f'"{term}"' for term in query_terms)
        return self._execute(expression, limit, table="products_stemmed")

    def exact_search(self, phrases: Sequence[str], limit: int) -> list[BM25Hit]:
        """Search exact token sequences copied from customer constraint clauses."""
        expressions: list[str] = []
        seen: set[tuple[str, ...]] = set()
        for phrase in phrases:
            # Keep stopwords here: FTS phrase queries require consecutive
            # tokens, so turning "made in USA" into "made USA" cannot match.
            phrase_terms = tuple(token.lower() for token in TOKEN_RE.findall(phrase)[:32])
            # A one-word phrase such as "cotton" is a broad keyword route, not
            # evidence that the simulator copied a catalog phrase.
            if len(phrase_terms) < 2 or phrase_terms in seen:
                continue
            seen.add(phrase_terms)
            expressions.append('"' + " ".join(phrase_terms) + '"')
        return self._execute(" OR ".join(expressions), limit)

    def profile_ranks(self, tags: Sequence[str], limit: int = 2000) -> dict[str, int]:
        query = " ".join(str(tag) for tag in tags if str(tag).strip())
        return {hit.parent_asin: rank for rank, hit in enumerate(self.search(query, limit), 1)}

    def metadata(self) -> tuple[dict[str, tuple[float | None, int]], list[str]]:
        rows = self.connection.execute(
            "SELECT parent_asin, price, rating_number FROM product_meta"
        ).fetchall()
        values = {
            str(asin): (None if price is None else float(price), int(rating_number))
            for asin, price, rating_number in rows
        }
        popular = [asin for asin, _ in sorted(
            ((str(asin), int(count)) for asin, _, count in rows),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )]
        return values, popular
