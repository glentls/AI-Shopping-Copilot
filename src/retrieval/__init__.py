"""Conversation state -> ranked candidates.

LANE B OWNS THIS PACKAGE. This is a working BM25-only placeholder lifted from
the starter so the skeleton runs end to end and reproduces the published
baseline. Lane B replaces it with BM25 + ONNX dense retrieval + slot matching,
fused by reciprocal rank fusion.

Two rules that hold for every implementation in here:
  1. Never hard-filter. Score and rank only. 15% of sessions retract a
     constraint on turn 3 or 4, and a filter built on a since-retracted
     constraint has already deleted the right answer permanently.
  2. Always return a full top_k. An empty slot is a wasted free chance at a hit.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from src.attributes import AttributeTable
from src.contracts import SLOTS, Candidate, ConversationState

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "still", "exploring", "need", "key", "requirement", "have", "preference",
}


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


class Retriever:
    def __init__(
        self,
        catalog_path: str | Path,
        artifacts_dir: str | Path,
        table: AttributeTable,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.table = table
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, ...]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                batch.append((
                    str(product["parent_asin"]),
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    _text(product.get("features")),
                    _text(product.get("details")),
                    _text(product.get("store")),
                    _text(product.get("description")),
                ))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
        self.connection.commit()

    def _query_text(self, state: ConversationState) -> str:
        """Everything the customer has said, plus their live slot values."""
        said = " ".join(text for role, text in state.history if role == "customer")
        values = " ".join(v for slot in SLOTS for v in state.active(slot))
        return f"{said} {values}"

    def search(self, state: ConversationState, top_n: int = 300) -> list[Candidate]:
        terms = list(dict.fromkeys(_terms(self._query_text(state))))[:40]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS rank "
            "FROM products WHERE products MATCH ? ORDER BY rank LIMIT ?",
            (expression, top_n),
        ).fetchall()
        return [
            Candidate(parent_asin=str(asin), score=-float(rank), components={"bm25": -float(rank)})
            for asin, rank in rows
        ]

    def rerank(self, cands: list[Candidate], state: ConversationState) -> list[Candidate]:
        """Soft slot boost on top of BM25 order. Never removes a candidate."""
        for position, candidate in enumerate(cands):
            bonus = 0.0
            for slot in SLOTS:
                held = set(self.table.values(candidate.parent_asin, slot))
                if not held:
                    continue
                for value in state.active(slot):
                    if value in held:
                        bonus += 1.0
                for value in state.excluded(slot):
                    if value in held:
                        bonus -= 1.5
            candidate.components["slot"] = bonus
            candidate.score = bonus - position * 0.01
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands
