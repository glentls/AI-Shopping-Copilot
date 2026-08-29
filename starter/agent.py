from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Union

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# Numeric fields stored outside FTS5 and the operators supported for them.
# A numeric filter value is a dict like {"lte": 50.0} or {"gte": 4.0, "lte": 5.0}.
NUMERIC_FIELDS = {"price", "average_rating", "rating_number"}
_OP_MAP = {"lt": "__lt__", "lte": "__le__", "gt": "__gt__", "gte": "__ge__", "eq": "__eq__"}

NumericFilter = dict[str, float]   # e.g. {"lte": 50.0}
SearchValue = Union[str, NumericFilter]


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


class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: set[str] = set()
        # Numeric metadata keyed by parent_asin — used for post-filtering.
        # Structure: {asin: {"price": float|None, "average_rating": float|None, "rating_number": int|None}}
        self._numeric: dict[str, dict[str, float | int | None]] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                batch.append(
                    (
                        asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                self._numeric[asin] = {
                    "price": product.get("price"),
                    "average_rating": product.get("average_rating"),
                    "rating_number": product.get("rating_number"),
                }
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def retrieve(
        self,
        search_key: dict[str, list[SearchValue]],
        top_k: int = 10,
    ) -> list[str]:
        """Return a ranked list of parent_asin strings matching *search_key*.

        Parameters
        ----------
        search_key:
            Mapping of attribute name -> list of values.

            Text attributes (any key not in NUMERIC_FIELDS):
                Values are strings. Within one attribute they are OR-ed;
                across attributes everything is AND-ed.
                Example: ``{"color": ["red", "crimson"], "style": ["casual"]}``

            Numeric attributes (``price``, ``average_rating``, ``rating_number``):
                Values are operator dicts applied as post-filters after the
                FTS5 candidate retrieval.
                Supported operators: ``lt``, ``lte``, ``gt``, ``gte``, ``eq``.
                Multiple dicts for the same key are AND-ed (range syntax).
                Example: ``{"price": [{"lte": 50}], "average_rating": [{"gte": 4.0}]}``

            Text and numeric keys can be combined freely:
                ``{"color": ["red"], "price": [{"lte": 50}]}``

        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[str]
            Ranked ``parent_asin`` values, best match first.
        """
        # Split keys into text vs numeric
        text_key: dict[str, list[str]] = {}
        numeric_filters: dict[str, list[NumericFilter]] = {}

        for attr, values in search_key.items():
            if attr in NUMERIC_FIELDS:
                numeric_filters[attr] = [v for v in values if isinstance(v, dict)]
            else:
                text_key[attr] = [v for v in values if isinstance(v, str)]

        # ── Step 1: FTS5 text search ─────────────────────────────────────────
        # Fetch a larger candidate pool when numeric post-filtering is needed
        # so we still return top_k results after filtering.
        candidate_k = top_k * 20 if numeric_filters else top_k

        attribute_exprs: list[str] = []
        for values in text_key.values():
            tokens: list[str] = []
            for value in values:
                tokens.extend(_terms(value))
            if not tokens:
                continue
            or_clause = " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))
            attribute_exprs.append(f"({or_clause})")

        if attribute_exprs:
            expression = " AND ".join(attribute_exprs)
            candidates: list[str] = [
                str(row[0])
                for row in self.connection.execute(
                    "SELECT parent_asin FROM products WHERE products MATCH ? "
                    "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                    (expression, candidate_k),
                ).fetchall()
            ]
        elif numeric_filters:
            # No text query — start from all products, numeric filter will narrow down
            candidates = [
                str(row[0])
                for row in self.connection.execute(
                    "SELECT parent_asin FROM products LIMIT ?",
                    (candidate_k,),
                ).fetchall()
            ]
        else:
            return []

        # ── Step 2: numeric post-filter ──────────────────────────────────────
        if not numeric_filters:
            return candidates[:top_k]

        results: list[str] = []
        for asin in candidates:
            meta = self._numeric.get(asin, {})
            passed = True
            for field, filter_dicts in numeric_filters.items():
                raw = meta.get(field)
                if raw is None:
                    passed = False
                    break
                val = float(raw)
                for fdict in filter_dicts:
                    for op, threshold in fdict.items():
                        method = _OP_MAP.get(op)
                        if method and not getattr(val, method)(float(threshold)):
                            passed = False
                            break
                    if not passed:
                        break
            if passed:
                results.append(asin)
            if len(results) == top_k:
                break

        return results

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions.add(session_id)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        unique_terms = list(dict.fromkeys(_terms(user_message)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            recommendations: list[dict] = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
