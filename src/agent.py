from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.intent_router import detect_scenario, extract_attributes
# from src.reranker import Reranker
# from src.decision_engine import DecisionEngine
# from src.output_formatter import OutputFormatter
from starter.agent import Agent as _BM25Agent
from starter.ledger import LedgerService


@dataclass
class PriceConstraint:
    operator: str  # "<" | "<=" | ">" | ">=" | "~"
    amount: float


_PRICE_RE = re.compile(
    r"(?:"
    r"(?P<op1>under|less\s+than|below|cheaper\s+than|max|maximum|no\s+more\s+than|at\s+most)\s*\$?(?P<amt1>[\d,]+(?:\.\d+)?)"
    r"|(?P<op2>over|more\s+than|above|at\s+least|minimum|min)\s*\$?(?P<amt2>[\d,]+(?:\.\d+)?)"
    r"|(?P<op3>around|about|approximately|budget\s+(?:is|of)?|~)\s*\$?(?P<amt3>[\d,]+(?:\.\d+)?)"
    r"|\$?(?P<amt4>[\d,]+(?:\.\d+)?)\s*(?P<op4>or\s+less|or\s+under|and\s+under|and\s+below|-)"
    r"|\$(?P<amt5>[\d,]+(?:\.\d+)?)"
    r")",
    re.IGNORECASE,
)


def _parse_price_constraint(text: str) -> PriceConstraint | None:
    m = _PRICE_RE.search(text)
    if not m:
        return None

    def _clean(s: str | None) -> float | None:
        return float(s.replace(",", "")) if s else None

    if m.group("op1"):
        return PriceConstraint("<", _clean(m.group("amt1")))
    if m.group("op2"):
        op_word = re.sub(r"\s+", " ", m.group("op2").lower().strip())
        op = ">=" if op_word in ("at least", "minimum", "min") else ">"
        return PriceConstraint(op, _clean(m.group("amt2")))
    if m.group("op3"):
        return PriceConstraint("~", _clean(m.group("amt3")))
    if m.group("amt4"):
        return PriceConstraint("<=", _clean(m.group("amt4")))
    if m.group("amt5"):
        return PriceConstraint("~", _clean(m.group("amt5")))
    return None


class Agent:
    """Full pipeline agent: Intent → Ledger → BM25 → Reranker → Decision."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self._ledger = LedgerService()
        self._retriever = _BM25Agent(catalog_path)
        # self._reranker = Reranker()
        # self._decision = DecisionEngine()
        # self._formatter = OutputFormatter()


    def reset(self, session_id: str, user_profile: dict) -> None:
        self._ledger.create(session_id, user_profile or {})

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        self._ledger.increment_turn(session_id)

        # ── 1. Intent Router ──────────────────────────────────────────────
        session = self._ledger.read(session_id)
        scenario = detect_scenario(user_message, session.get("history", []))

        if scenario == "intent_override":
            self._ledger.clear_constraints(session_id)
            self._ledger.set_intent(session_id, "buying")
        elif scenario == "boundary":
            asked = self._ledger.read(session_id)["asked_attributes"]
            if asked:
                last_asked = asked[-1]
                # Remove the boundary attribute from constraints so it isn't searched
                with self._ledger.session(session_id) as s:
                    s["constraints"].pop(last_asked, None)
            self._ledger.set_intent(session_id, "boundary")
        else:
            self._ledger.set_intent(session_id, scenario)

        # ── 2. Attribute Extraction ───────────────────────────────────────
        new_attrs = extract_attributes(user_message)
        price = _parse_price_constraint(user_message)

        for attr, value in new_attrs.items():
            self._ledger.set_constraint(session_id, attr, value)

        if price:
            with self._ledger.session(session_id) as s:
                s["price_constraint"] = {"operator": price.operator, "amount": price.amount}

        # ── 3. Update history ─────────────────────────────────────────────
        with self._ledger.session(session_id) as s:
            s.setdefault("history", []).append({"turn": turn, "role": "user", "content": user_message})

        # ── 4. Build search key ───────────────────────────────────────────
        session = self._ledger.read(session_id)
        search_key: dict = {}
        for attr, values in session["constraints"].items():
            search_key[attr] = values
        price_c = session.get("price_constraint")
        if price_c:
            search_key["price"] = [{"lte": price_c["amount"]}] if price_c["operator"] in ("<", "<=", "~") else [{"gte": price_c["amount"]}]
        self._ledger.set_search_key(session_id, search_key)

        # # ── 5. BM25 Retrieval ─────────────────────────────────────────────
        # candidates = self._retriever.retrieve(search_key, top_k=100)

        # # ── 7. Reranker ───────────────────────────────────────────────────
        # ranked = self._reranker.rerank(candidates, session)

        # # ── 8. Decision Engine ────────────────────────────────────────────
        # confidence = self._reranker.get_confidence(ranked)
        # kiv_next = self._ledger.next_unasked_attribute(session_id)

        # if self._decision.should_recommend(confidence, len(ranked), turn, session):
        #     return self._formatter.format_recommend(ranked, top_k)

        # # ── 9. Ask path ───────────────────────────────────────────────────
        # attribute = self._decision.select_question_attribute(ranked, session, kiv_next)
        # self._ledger.mark_attribute_asked(session_id, attribute)
        # return self._formatter.format_ask(attribute, session)
