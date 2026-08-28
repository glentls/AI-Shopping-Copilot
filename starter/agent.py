from __future__ import annotations

import json //read product data
import re //patterns for finding words in a sentence
import sqlite3 //a tiny database that lives in memory
from pathlib import Path

//finds chunks of letters and numbers (basically words) and removes the unn
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]: //"I want black shoes" -> ["black", "shoes"]
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path) //get the data on this agent so that other methods can use later
        self.connection = sqlite3.connect(":memory:") //a databse js for the session
        self._sessions: set[str] = set() //each session is "a user having one convo with the agent"
        self._build_index() //immediately loads every product into the database

    def _build_index(self) -> None: //loading the catlog into a search table
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
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None: //start new session
        # The profile is anonymized and may be used for personalization.
        self._sessions.add(session_id)

    def respond( //ans one turn of the conversation
        self,
        session_id: str,
        user_message: str, //what the shopper says
        turn: int, //1 through 10, imp: unused in this starter (a smarter agent might ask questions early and recommend later)
        //Early turns (like turn 1-2): the agent probably doesn't have enough information yet — maybe it should ask a clarifying question instead of guessing blindly ("What's your budget?" "What color are you looking for?")
        //Later turns (like turn 5+): by now, hopefully enough information has been gathered through the conversation, so the agent should stop asking and start recommending actual products — because turns are limited (remember, max 10, and running out = zero score), so at some point it needs to commit to an answer rather than keep asking more questions forever

        top_k: int, //how many products to return
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
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?", //bm25 ranking
                (expression, top_k),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
