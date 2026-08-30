"""Turn-scoped intent routing and context programming.

The evaluator exposes one fixed ``respond`` method, but the work performed
inside it does not need to be fixed.  This module compiles the live dialogue
state into a small, inspectable program on every turn:

* Buying requests retain a wide candidate recall set and may lock catalog-
  backed, high-confidence constraints before the final top ten.
* Browsing requests lean on semantic and profile evidence.  When the request
  is still too broad, candidate processing is capped and the policy is told to
  clarify immediately.

Only aggregate profile tags enter the program.  Explicit session constraints
always take precedence, and a profile tag is never treated as a hard filter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.contracts import SLOTS, ConversationState
from src.extract import detect_override


IntentRoute = Literal["buying", "browsing"]

_BROWSING_RE = re.compile(
    r"\b(?:brows(?:e|ing)|explor(?:e|ing)|ideas?|inspiration|"
    r"not sure|open to|surprise me|recommend(?:ation)?s?|something nice)\b",
    re.IGNORECASE,
)
_BUYING_RE = re.compile(
    r"\b(?:key requirement|what i need|must have|need|"
    r"exactly|specific|under\s+[$€£]?\d|no more than)\b",
    re.IGNORECASE,
)

PROFILE_TERM_LIMIT = 8
BROAD_CANDIDATE_CUTOFF = 200
CANDIDATE_OVERLOAD_THRESHOLD = 100
HARD_CONFIDENCE = 0.94


@dataclass(frozen=True)
class ContextProgram:
    """The retrieval and dialogue strategy selected for one turn."""

    route: IntentRoute
    active_terms: tuple[str, ...]
    profile_terms: tuple[str, ...]
    hard_constraints: tuple[tuple[str, tuple[str, ...]], ...]
    lock_hard_constraints: bool
    over_general: bool
    candidate_cutoff: int
    dense_weight: float
    profile_weight: float


def _dedupe_terms(values: object, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple, set)):
        return ()
    terms: list[str] = []
    for value in values:
        normalized = " ".join(str(value).strip().lower().split())
        if normalized and normalized not in terms:
            terms.append(normalized)
        if len(terms) >= limit:
            break
    return tuple(terms)


def _current_intent_messages(state: ConversationState) -> list[str]:
    messages = [text for role, text in state.history if role == "customer"]
    start = 0
    for index, message in enumerate(messages):
        if detect_override(message):
            start = index
    return messages[start:]


def compile_context_program(state: ConversationState) -> ContextProgram:
    """Distill history, live slots, and the safe profile into a turn plan."""
    active: list[str] = []
    constraints: list[tuple[str, tuple[str, ...]]] = []
    specific_slots: set[str] = set()

    for slot in SLOTS:
        values = [
            value
            for value in state.slots.get(slot, [])
            if value.polarity
        ]
        live = tuple(dict.fromkeys(value.value for value in values if value.value))
        if live:
            active.extend(live)
            specific_slots.add(slot)
        hard = tuple(dict.fromkeys(
            value.value
            for value in values
            if value.value and value.confidence >= HARD_CONFIDENCE
        ))
        if hard and slot != "budget":
            constraints.append((slot, hard))

    messages = _current_intent_messages(state)
    current_text = " ".join(messages)
    detail_slots = specific_slots - {"category"}
    browsing_cue = bool(_BROWSING_RE.search(current_text))
    buying_cue = bool(_BUYING_RE.search(current_text))
    customer_turns = sum(1 for role, _ in state.history if role == "customer")
    intent_start_turn = max(1, customer_turns - len(messages) + 1)
    later_detail = any(
        slot != "category" and value.polarity and value.turn > intent_start_turn
        for slot, values in state.slots.items()
        for value in values
    )

    # An explicit browsing opener remains exploratory even when taxonomy words
    # happen to map to style/use-case slots.  A detail supplied on a later turn
    # switches the workflow to precision.  Strong buying language wins when
    # both cue families happen to occur in the same natural sentence.
    if buying_cue or later_detail:
        route: IntentRoute = "buying"
    elif browsing_cue:
        # Taxonomy words in an exploratory opener may map to style or use-case
        # slots.  They describe the browsing area; they do not turn the opener
        # into a targeted purchase until a later turn supplies a real detail.
        route = "browsing"
    elif detail_slots or state.budget_max is not None:
        route = "buying"
    else:
        route = "browsing"

    over_general = route == "browsing"
    profile = state.user_profile if isinstance(state.user_profile, dict) else {}
    profile_terms = _dedupe_terms(profile.get("preference_tags", ()), PROFILE_TERM_LIMIT)

    return ContextProgram(
        route=route,
        active_terms=tuple(dict.fromkeys(active)),
        profile_terms=profile_terms,
        hard_constraints=tuple(constraints) if route == "buying" else (),
        # Supplying a detail during exploratory browsing switches retrieval to
        # the precision-oriented route, but only explicit requirement language
        # authorizes a hard top-K lock.  This distinction prevents an inferred
        # catalog label from masquerading as a customer-mandated constraint.
        lock_hard_constraints=route == "buying" and buying_cue,
        over_general=over_general,
        candidate_cutoff=BROAD_CANDIDATE_CUTOFF if over_general else 300,
        # Exact requirement clauses override this in the retriever.  A broad
        # request uses the dense route as a diversity hedge; a precise natural
        # request gives semantic evidence equal footing with lexical evidence.
        dense_weight=0.10 if over_general else 1.0,
        # Personalization helps choose among relevant browsing candidates.  It
        # never competes with explicit Buying constraints.
        profile_weight=0.10 if route == "browsing" and profile_terms else 0.0,
    )


def candidate_pool_overloaded(program: ContextProgram, candidate_count: int) -> bool:
    """Whether retrieval should hand control to proactive clarification."""
    return (
        program.over_general
        and candidate_count >= CANDIDATE_OVERLOAD_THRESHOLD
    )


__all__ = [
    "BROAD_CANDIDATE_CUTOFF",
    "CANDIDATE_OVERLOAD_THRESHOLD",
    "ContextProgram",
    "candidate_pool_overloaded",
    "compile_context_program",
]
