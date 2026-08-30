"""BM25 lexical retrieval over the frozen catalog, via SQLite FTS5.

This is a straight port of starter/agent.py's weak-baseline logic (docs/plan/RECON.md V4), made
config-driven per config.yaml's `retrieval` block instead of hard-coded constants, and wrapped to
return src.contracts.Candidate objects with real ProductMeta instead of bare dicts. It is the
permanent Null/fallback retrieval path -- there is no "empty" retrieval fallback, because an
empty candidate list can never score a hit.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.config import load_config
from src.contracts import Candidate, ProductMeta, RetrievalRequest, RetrievalResult

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
_WEIGHT_COLUMN_ORDER = (
    "parent_asin", "title", "categories", "features", "details", "store", "description",
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _details_brand(details: object) -> str | None:
    if not isinstance(details, dict):
        return None
    for key in ("Brand", "Brand Name"):
        value = details.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _product_meta(product: dict) -> ProductMeta:
    price = product.get("price")
    return ProductMeta(
        title=str(product.get("title") or ""),
        price=float(price) if isinstance(price, (int, float)) else None,
        categories=[str(item) for item in (product.get("categories") or [])],
        features=[str(item) for item in (product.get("features") or [])],
        description=[str(item) for item in (product.get("description") or [])],
        store=str(product["store"]) if product.get("store") not in (None, "") else None,
        details_brand=_details_brand(product.get("details")),
        average_rating=float(product.get("average_rating") or 0.0),
        rating_number=int(product.get("rating_number") or 0),
    )


@dataclass
class BM25Index:
    connection: sqlite3.Connection
    products: dict[str, ProductMeta]
    fallback_pool: list[str]  # catalog IDs sorted by rating_number desc, for deterministic padding


def build_index(catalog_path: str | Path, config: dict | None = None) -> BM25Index:
    config = config or load_config()
    # check_same_thread=False: agent.py calls search() from its ThreadPoolExecutor timeout-guard
    # worker thread, not the thread that built this connection. The connection is only ever used
    # from that one worker thread after construction, so this is safe, not a concurrency hazard.
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    cursor = connection.cursor()
    cursor.execute(
        "CREATE VIRTUAL TABLE products USING fts5("
        "parent_asin UNINDEXED, title, categories, features, details, store, description, "
        "tokenize='unicode61 remove_diacritics 2')"
    )
    batch_size = config["retrieval"]["index_batch_size"]
    products: dict[str, ProductMeta] = {}
    ratings: list[tuple[int, str]] = []
    batch: list[tuple[str, str, str, str, str, str, str]] = []
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            products[parent_asin] = _product_meta(product)
            ratings.append((int(product.get("rating_number") or 0), parent_asin))
            batch.append((
                parent_asin,
                _text(product.get("title")),
                _text(product.get("categories")),
                _text(product.get("features")),
                _text(product.get("details")),
                _text(product.get("store")),
                _text(product.get("description")),
            ))
            if len(batch) >= batch_size:
                cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
    if batch:
        cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
    connection.commit()

    pad_pool_size = config["fallback"]["pad_pool_size"]
    ratings.sort(key=lambda pair: (-pair[0], pair[1]))  # deterministic: rating desc, then asin
    fallback_pool = [asin for _, asin in ratings[:pad_pool_size]]

    return BM25Index(connection=connection, products=products, fallback_pool=fallback_pool)


def search(index: BM25Index, request: RetrievalRequest, config: dict | None = None) -> RetrievalResult:
    config = config or load_config()
    retrieval_cfg = config["retrieval"]
    unique_terms = list(dict.fromkeys(_terms(request.canonical_query)))[: retrieval_cfg["max_query_terms"]]
    expression = " OR ".join(f'"{term}"' for term in unique_terms)
    if not expression:
        return RetrievalResult([], pool_size=0, dropped_constraints=[])

    weights = retrieval_cfg["bm25_weights"]
    weight_args = [weights[column] for column in _WEIGHT_COLUMN_ORDER]
    pool_size = max(request.top_k, retrieval_cfg["candidate_pool_size"])

    rows = index.connection.execute(
        "SELECT parent_asin, bm25(products, ?, ?, ?, ?, ?, ?, ?) AS rank_val "
        "FROM products WHERE products MATCH ? ORDER BY rank_val ASC LIMIT ?",
        (*weight_args, expression, pool_size),
    ).fetchall()

    candidates: list[Candidate] = []
    for parent_asin, rank_val in rows:
        parent_asin = str(parent_asin)
        meta = index.products.get(parent_asin)
        if meta is None:
            continue
        candidates.append(Candidate(parent_asin=parent_asin, score=-float(rank_val), route="bm25", meta=meta))
    # BM25 applies no hard category filter and runs no relaxation ladder, so the surviving
    # pool is just what FTS5 returned and nothing was dropped. Phases 6+ populate these for real.
    return RetrievalResult(candidates, pool_size=len(candidates), dropped_constraints=[])
