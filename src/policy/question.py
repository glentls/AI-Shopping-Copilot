"""Deciding what to ask next.

LANE C OWNS THIS FILE. Working placeholder using expected information gain.

BUNDLE YOUR QUESTIONS. The organizers' own worked example asks about use_case,
material and budget in one breath and the customer answers all three. The API
allows only one ask_attribute, so: write prose covering 2-3 topics, and set
ask_attribute to the highest-value one. You cannot lose by doing this, and
turn count is 20% of the score.

Measured value per topic on this catalog (halvings of the candidate pool):
  brand 12.8 | use_case 4.1 | style 3.9 | feature 3.6 | material 3.5
  color 3.2  | gender 3.1 (never ask -- the opener gives it free)
  budget 1.9 (nearly worthless alone -- only ever bundle it)
Going from 50,000 candidates to 10 needs about 12 halvings.
"""

from __future__ import annotations

import math
from typing import Optional

from src.attributes import AttributeTable
from src.contracts import Candidate, ConversationState

# Prior value per slot, from the measured halvings above. Used to break ties
# and to seed turn 1 when there are no candidates to compute entropy over.
PRIOR = {
    "brand": 1.00, "use_case": 0.72, "style": 0.68, "feature": 0.64,
    "material": 0.62, "color": 0.56, "size": 0.40, "budget": 0.18,
    "category": 0.50,
}
BUNDLE_SIZE = 3


def _entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def _askable(state: ConversationState, slot: str) -> bool:
    if slot in state.unanswerable or slot in state.asked:
        return False
    return not state.active(slot)


def score_slots(
    state: ConversationState,
    cands: list[Candidate],
    table: AttributeTable,
) -> list[tuple[str, float]]:
    """Every askable slot, best first, by expected information gain."""
    pool = {c.parent_asin for c in cands}
    scored: list[tuple[str, float]] = []
    for slot in PRIOR:
        if not _askable(state, slot):
            continue
        gain = _entropy(table.distribution(slot, pool)) if pool else 0.0
        # Discount by how often the catalog can even answer this slot.
        answerable = table.coverage(slot) or 0.5
        scored.append((slot, (gain * answerable) + PRIOR[slot]))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def choose_question(
    state: ConversationState,
    cands: list[Candidate],
    table: AttributeTable,
) -> tuple[Optional[str], list[str]]:
    """Returns (ask_attribute for the API, extra topics to bundle in prose)."""
    ranked = score_slots(state, cands, table)
    if not ranked:
        return "other", []
    primary = ranked[0][0]
    extras = [slot for slot, _ in ranked[1:BUNDLE_SIZE]]
    return primary, extras
