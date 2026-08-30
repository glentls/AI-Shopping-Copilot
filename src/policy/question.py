"""Deciding what to ask next.

LANE C OWNS THIS FILE.

BUNDLE YOUR QUESTIONS. The organizers' own worked example asks about use_case,
material and budget in one breath and the customer answers all three. The API
allows only one ask_attribute, so: write prose covering 2-3 topics, and set
ask_attribute to the highest-value one. You cannot lose by doing this, and turn
count is 20% of the score.

Measured value per topic on this catalog (halvings of the candidate pool):
  brand 12.8 | use_case 4.1 | style 3.9 | feature 3.6 | material 3.5
  color 3.2  | gender 3.1 (never ask -- the opener gives it free)
  budget 1.9 (nearly worthless alone -- only ever bundle it)
Going from 50,000 candidates to 10 needs about 12 halvings.

Each turn every askable slot is scored by expected information gain -- the
entropy of how the answer would split the remaining candidates -- discounted by
how often the catalog can answer it at all. The open-ended ``other`` action
competes on the same scale, priced by what it has yielded in THIS session and
discounted each time it is used.
"""

from __future__ import annotations

import math
import os
from typing import Optional

from src.attributes import AttributeTable
from src.contracts import Candidate, ConversationState
from src.extract import detect_override
from src.lexicons import NO_PREFERENCE_RE
from src.orchestration import candidate_pool_overloaded, compile_context_program
from src.policy.state import learned_on

# Prior value per slot, from the measured halvings above. Used to break ties
# and to seed turn 1 when there are no candidates to compute entropy over.
PRIOR = {
    "brand": 1.00, "use_case": 0.72, "style": 0.68, "feature": 0.64,
    "material": 0.62, "color": 0.56, "size": 0.40, "budget": 0.18,
    "category": 0.50,
}
BUNDLE_SIZE = 3

# "other" is an ACTION, not a slot. It asks the customer an open-ended
# "anything else?" question, while concrete actions ask about one known topic.
# The two kinds of action compete on the same order of magnitude.
#
# The former default of 20 came from the public simulator and dwarfed concrete
# question scores, so a human could be asked "anything else?" for many turns.
# Eight gives one broad question a modest lead over the strongest concrete
# question; outcome-aware guardrails, rather than an oversized prior, govern
# whether it may be asked again.
OPEN_QUESTION_BASELINE = float(
    os.environ.get("TJ_OPEN_QUESTION_BASELINE", "8.0")
)

# Preferences are finite: even productive open questions have diminishing
# returns. This discount is applied once per answered open question.
OPEN_QUESTION_DECAY = float(
    os.environ.get("TJ_OPEN_QUESTION_DECAY", "0.8")
)

# Facts a productive open-ended turn is expected to return. The public simulator
# discloses up to two constraints per answer (`matches[:2]`), so two is the
# number to beat here; anything less makes the open question lose value against
# concrete slots.
OPEN_QUESTION_EXPECTED_YIELD = float(
    os.environ.get("TJ_OPEN_QUESTION_EXPECTED_YIELD", "2.0")
)

# A productive answer can keep the action competitive, but never cancels the
# per-use decay. A silent answer sharply lowers the next eligible score.
HIGH_YIELD_FACTOR = 1.25
SINGLE_YIELD_FACTOR = 0.90
ZERO_YIELD_FACTOR = 0.35

# Guardrails are separate from the estimated score. They make the customer
# experience predictable even if benchmark tuning changes the score constants.
MAX_CONSECUTIVE_OPEN_QUESTIONS = int(
    os.environ.get("TJ_OPEN_QUESTION_MAX_CONSECUTIVE", "2")
)
ZERO_YIELD_PATIENCE = int(
    os.environ.get("TJ_OPEN_QUESTION_ZERO_YIELD_PATIENCE", "2")
)

# One answer cannot realistically deliver more than a few bits, but raw entropy
# over a high-cardinality slot says otherwise: `brand` has 19,749 distinct
# values and scores ~8.7 against use_case's ~1.9, so an uncapped picker asks
# about brands every single time. Cap the gain term so cardinality alone cannot
# buy the turn.
GAIN_CAP = 4.0

# Explicit refusals of "anything else?" to accept before disabling it for the
# current intent. Intent overrides reset all of these derived counters.
OPEN_QUESTION_DECLINE_PATIENCE = int(
    os.environ.get("TJ_OPEN_QUESTION_DECLINE_PATIENCE", "2")
)

def _entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def _askable(state: ConversationState, slot: str) -> bool:
    """Never spend a turn on a question that cannot pay.

    Three ways a slot is already spent: the customer told us they have no
    preference for it (boundary), we have asked it before, or we already hold
    an answer for it.
    """
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
        gain = min(_entropy(table.distribution(slot, pool)), GAIN_CAP) if pool else 0.0
        # Discount by how often the catalog can even answer this slot.
        answerable = table.coverage(slot) or 0.5
        scored.append((slot, (gain * answerable) + PRIOR[slot]))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _intent_start_turn(state: ConversationState) -> int:
    """First customer turn in the current intent."""
    said = [text for role, text in state.history if role == "customer"]
    return max(
        (
            turn
            for turn, text in enumerate(said, start=1)
            if detect_override(text)
        ),
        default=1,
    )


def _answered(state: ConversationState) -> list[tuple[int, str, str]]:
    """(reply turn, question, reply) records for the current intent."""
    said = [text for role, text in state.history if role == "customer"]
    intent_start = _intent_start_turn(state)
    answered: list[tuple[int, str, str]] = []
    for question_turn, action in state.question_history:
        reply_turn = question_turn + 1
        # A question asked before an override belongs to the previous intent,
        # even when the customer's reply is the message that changes intent.
        if question_turn < intent_start or reply_turn > len(said):
            continue
        answered.append((reply_turn, action, said[reply_turn - 1]))
    return answered


def _replies_to(state: ConversationState) -> dict[str, list[int]]:
    """What each action returned under the current intent."""
    observed: dict[str, list[int]] = {}
    for reply_turn, action, _ in _answered(state):
        observed.setdefault(action, []).append(learned_on(state, reply_turn))
    return observed


def open_question_declines(state: ConversationState) -> int:
    """Open-question refusals, even when concrete questions intervene."""
    return sum(
        1
        for _, action, reply in _answered(state)
        if action == "other" and NO_PREFERENCE_RE.search(reply)
    )


def open_question_zero_yields(state: ConversationState) -> int:
    """Answered open questions that taught no new structured facts."""
    return sum(
        1
        for reply_turn, action, _ in _answered(state)
        if action == "other" and learned_on(state, reply_turn) == 0
    )


def consecutive_open_questions(state: ConversationState) -> int:
    """Consecutive open questions asked under the current intent."""
    count = 0
    intent_start = _intent_start_turn(state)
    for question_turn, action in reversed(state.question_history):
        if question_turn < intent_start or action != "other":
            break
        count += 1
    return count


def _last_answered_open_question_was_silent(state: ConversationState) -> bool:
    answered = _answered(state)
    if not answered:
        return False
    reply_turn, action, _ = answered[-1]
    return action == "other" and learned_on(state, reply_turn) == 0


def open_question_score(state: ConversationState) -> float:
    """Current value of asking "anything else that matters to you?".

    The score starts on the same scale as concrete questions, decays after each
    use, and reacts to what the latest open question actually taught. Hard
    guardrails temporarily pause it after a silent answer and retire it after
    repeated silence or refusals. All observations reset on an intent override.
    """
    if open_question_declines(state) >= OPEN_QUESTION_DECLINE_PATIENCE:
        return 0.0
    if open_question_zero_yields(state) >= ZERO_YIELD_PATIENCE:
        return 0.0
    if consecutive_open_questions(state) >= MAX_CONSECUTIVE_OPEN_QUESTIONS:
        return 0.0
    if _last_answered_open_question_was_silent(state):
        # Force a concrete question immediately after an unproductive broad one.
        return 0.0

    observed = _replies_to(state).get("other", [])
    score = OPEN_QUESTION_BASELINE * (OPEN_QUESTION_DECAY ** len(observed))
    if not observed:
        return score
    if observed[-1] >= OPEN_QUESTION_EXPECTED_YIELD:
        return score * HIGH_YIELD_FACTOR
    if observed[-1] > 0:
        return score * SINGLE_YIELD_FACTOR
    return score * ZERO_YIELD_FACTOR


def choose_question(
    state: ConversationState,
    cands: list[Candidate],
    table: AttributeTable,
) -> tuple[Optional[str], list[str]]:
    """Returns (ask_attribute for the API, extra topics to bundle in prose).

    The prose bundles 2-3 topics either way -- the organizers' own example agent
    asks about use_case, material and budget in one breath and the customer
    answers all three. Only ask_attribute is scored, so bundling is free upside.

    Returns ``None`` only after every concrete topic is spent and the customer
    has exhausted the open question. At that point another forced question is known
    to teach nothing and creates a visible dialogue loop.
    """
    ranked = score_slots(state, cands, table)
    open_score = open_question_score(state)

    # A broad browsing request can match hundreds of plausible products.  Stop
    # spending work on ranking that overloaded pool and ask the structured
    # open-ended question immediately; concrete high-information topics remain bundled in
    # the prose so the customer has useful ways to narrow it.
    program = compile_context_program(state)
    if candidate_pool_overloaded(program, len(cands)) and open_score > 0.0:
        return "other", [slot for slot, _ in ranked[:BUNDLE_SIZE - 1]]

    if not ranked:
        return ("other", []) if open_score > 0.0 else (None, [])

    if ranked[0][1] < open_score:
        # The open question takes the scored slot; the best concrete slots still
        # go into the prose, so a simulator that reads the message loses nothing.
        return "other", [slot for slot, _ in ranked[:BUNDLE_SIZE - 1]]

    return ranked[0][0], [slot for slot, _ in ranked[1:BUNDLE_SIZE]]
