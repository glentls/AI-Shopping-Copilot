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

ATTRIBUTES = ("category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case")
OVERRIDE_MARKERS = ("actually", "instead", "never mind", "nevermind", "forget that", "change of plans", "i meant", "rather")
ATTRIBUTE_TERMS = {
    "category": ("looking for", "need", "want", "shoes", "dress", "shirt", "bag", "jewelry", "boots"),
    "material": ("leather", "cotton", "wool", "linen", "suede", "silk", "denim", "material"),
    "color": ("black", "white", "blue", "red", "green", "brown", "pink", "grey", "gray", "color"),
    "size": ("size", "small", "medium", "large", " xs ", " s ", " m ", " l ", " xl "),
    "style": ("style", "casual", "formal", "vintage", "minimalist", "classic", "sporty"),
    "brand": ("brand",),
    "budget": ("$", "budget", "under", "less than", "cheap", "affordable", "price"),
    "feature": ("feature", "waterproof", "comfortable", "durable", "pockets", "slip resistant"),
    "use_case": ("for work", "for running", "for hiking", "for a wedding", "for travel", "gift", "occasion"),
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


class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, dict] = {}
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

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "profile": user_profile or {},
            "slots": {attribute: None for attribute in ATTRIBUTES},
            "slot_status": {attribute: "unknown" for attribute in ATTRIBUTES},
            "unconstrained": set(),
            "asked": [],
            "history": [],
            "override_pending": False,
            "retrieval_feedback": {},
        }

    def update_retrieval_feedback(self, session_id: str, feedback: dict) -> None:
        """Optional Pillar 1 -> Pillar 2 handoff for proactive guidance."""
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before updating feedback")
        self._sessions[session_id]["retrieval_feedback"] = dict(feedback or {})

    def _extract_updates(self, message: str) -> dict[str, str]:
        """Lightweight slot extraction; deliberately dependency-free."""
        text = " " + message.lower() + " "
        updates: dict[str, str] = {}
        for attribute, terms in ATTRIBUTE_TERMS.items():
            found = next((term.strip() for term in terms if term in text), None)
            if found:
                # Preserve the full utterance because the retrieval component
                # still uses the original user message for lexical context.
                updates[attribute] = message.strip()
        return updates

    def _apply_message(self, state: dict, message: str) -> None:
        lowered = message.lower()
        override = any(marker in lowered for marker in OVERRIDE_MARKERS)
        no_preference = any(phrase in lowered for phrase in (
            "no preference", "anything is fine", "you decide", "doesn't matter",
            "does not matter", "i'm flexible", "im flexible",
        ))
        updates = self._extract_updates(message)
        if override:
            # Overrides invalidate old conversational constraints. The new message is
            # applied below, so explicit values in the override are retained.
            for attribute in state["slots"]:
                state["slots"][attribute] = None
            state["unconstrained"].clear()
            state["asked"].clear()
            state["override_pending"] = True
        for attribute, value in updates.items():
            if attribute in state["slots"]:
                state["slots"][attribute] = value
                state["slot_status"][attribute] = "confirmed"
                state["unconstrained"].discard(attribute)
        if no_preference and state["asked"]:
            attribute = state["asked"][-1]
            if attribute in state["slots"]:
                state["slots"][attribute] = None
                state["slot_status"][attribute] = "unconstrained"
                state["unconstrained"].add(attribute)
        state["history"].append(message)

    def _choose_question(self, state: dict, turn: int, candidate_count: int) -> str | None:
        feedback = state.get("retrieval_feedback", {})
        overloaded = bool(feedback.get("overloaded")) or int(feedback.get("candidate_count", 0) or 0) > 100
        if turn >= 8:
            return None
        if state.get("override_pending"):
            # The evaluator's override response may contain a new requirement
            # without naming its field. `other` intentionally asks for that
            # requirement directly and prevents stale-slot questioning.
            state["asked"].append("other")
            state["override_pending"] = False
            return "What is the most important requirement for this new request?"
        slots = state["slots"]
        missing = feedback.get("missing_attributes")
        priority = tuple(attribute for attribute in missing if attribute in ATTRIBUTES) if isinstance(missing, list) else ()
        priority += ("category", "use_case", "budget", "size", "color", "material", "style", "brand", "feature")
        for attribute in priority:
            if slots[attribute] is None and attribute not in state["unconstrained"] and attribute not in state["asked"]:
                if attribute == "category":
                    question = "What type of item are you looking for?"
                elif attribute == "use_case":
                    question = "What will you mainly use it for?"
                else:
                    question = f"Do you have a preference for {attribute}?"
                state["asked"].append(attribute)
                return question
        if overloaded:
            return None
        return None

    def _is_overgeneral(self, state: dict, message: str) -> bool:
        """Conversation-only overload signal, used when Pillar 1 gives no metadata."""
        vague = (
            "something", "anything", "some options", "show me options",
            "surprise me", "not sure", "whatever", "just browsing",
        )
        known_slots = sum(value is not None for value in state["slots"].values())
        return known_slots == 0 and (len(_terms(message)) <= 4 or any(phrase in message.lower() for phrase in vague))

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        self._apply_message(state, user_message)
        feedback = state.get("retrieval_feedback", {})
        overloaded = bool(feedback.get("overloaded"))
        state["overgeneral"] = self._is_overgeneral(state, user_message)
        context = " ".join(value for value in state["slots"].values() if value)
        unique_terms = list(dict.fromkeys(_terms(user_message + " " + context)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression or (overloaded and turn < 8):
            recommendations: list[dict] = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
        question = self._choose_question(state, turn, len(recommendations))
        return {
            "message": question or "Here are the closest matches I found.",
            "ask_attribute": state["asked"][-1] if question else None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
