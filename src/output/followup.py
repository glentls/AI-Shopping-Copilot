"""Situational follow-up message generation (hardcoded, first pass).

The confidence policy (``src/confidence/policy.py``) decides *whether* to ask
(``clarify``) and *which* attribute (``ask_attribute``). Two policies exist:
the shipped ``always_ask`` (fixed ``ask_attribute="other"``, a wildcard) and
the measured-alternative ``attribute_cycle`` (a specific, not-yet-asked
attribute each turn -- see ``decide_specific_attribute``'s docstring). This
module phrases the question either way: a specific attribute gets its own
question text; "other"/None falls back to situational phrasing (vague opener,
boundary brush-off, intent override, late turn).

Note: the evaluator's ``customer_reply()`` (``evaluator/local_evaluator.py``)
decides what to reveal next turn purely from ``ask_attribute`` -- it never
reads ``message`` text. This module therefore has zero effect on
HitRate@10/MRR/MTTC/TechnicalScore; it is a demo/UX quality improvement, not a
scoring lever. Kept deliberately separate from the confidence *decision*
(``policy.py``) so a change here can never regress either policy arm.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FollowUpContext:
    """Situational signals already computed every turn in ``src/agent.py``,
    threaded through so message selection doesn't need to recompute or
    guess at anything.

    Attributes:
        scenario: One of "buying" / "browsing" / "intent_override" /
            "boundary", from ``src.intent_router.detect_scenario``.
        n_constraints_known: ``SessionLedger.n_constraints_known`` -- how many
            distinct constraints the customer has disclosed so far.
        exhausted: ``SessionLedger.exhausted`` -- customer signaled no further
            preferences exist.
        turn: Current turn number (1-indexed).
        override_seen: ``SessionLedger.override_seen`` -- an intent override
            has occurred at some point this session (used to soften the
            "still nothing?" late-turn phrasing after a genuine pivot).
        ask_attribute: ``ConfidencePayload.ask_attribute`` for this turn.
            Under ``attribute_cycle`` this is a specific, not-yet-asked
            attribute (e.g. "color") and gets its own question text; under
            the shipped ``always_ask`` it is always "other", which falls
            through to the situational phrasing below.
    """

    scenario: str
    n_constraints_known: int
    exhausted: bool
    turn: int
    override_seen: bool = False
    ask_attribute: str | None = None


# Late-turn threshold: past this, phrasing shifts to acknowledge we're running
# low on turns rather than opening with a fresh discovery question.
_LATE_TURN = 6

# Specific-attribute question text, used when ask_attribute names a real
# attribute (attribute_cycle policy). Reachable dead weight under always_ask,
# where ask_attribute is always "other".
_ATTRIBUTE_QUESTIONS = {
    "category": "What type of item are you looking for?",
    "material": "Do you have a material preference?",
    "color": "Any color in mind?",
    "size": "What size do you need?",
    "style": "What style are you going for?",
    "brand": "Any brand you prefer?",
    "budget": "What's your budget?",
    "feature": "Are there any specific features that matter most to you?",
    "use_case": "What will you mainly use it for?",
}

# -- Situational fallback (used when ask_attribute is "other"/None) ----------

_ASK_INTENT_OVERRIDE = (
    "Got it, updating my search based on that! Is there anything else — "
    "like size, color, material, or budget — that would help me narrow it down?"
)
_ASK_BOUNDARY = (
    "No worries, I'll use my judgment there. Anything else you'd like me to "
    "keep in mind — size, color, material, brand, or budget?"
)
_ASK_ZERO_INFO = (
    "I'd love to help you find the right thing! Could you tell me a bit more "
    "about what you're looking for — the type of item, and anything like "
    "color, material, size, or budget that matters to you?"
)
_ASK_LATE_TURN = (
    "We're getting close! Anything specific — size, color, material, or "
    "price range — that would help me lock in the best match?"
)
_ASK_DEFAULT = (
    "Thanks! Anything else that would help me narrow this down — color, "
    "size, material, brand, or budget?"
)

# -- Recommend-only variants (used when payload.clarify is False) ------------

_RECOMMEND_EXHAUSTED = "Based on everything you've shared, here's what I found!"
_RECOMMEND_LATE_TURN = "Here's my best selection based on our conversation so far!"
_RECOMMEND_DEFAULT = "Here are the closest matches I found."


def build_ask_message(context: FollowUpContext | None) -> str:
    """Message text for a turn where the agent is asking a follow-up
    question. Falls back to the generic default with no context.

    A specific ``ask_attribute`` (attribute_cycle policy) takes priority and
    is layered with a situational lead-in for override/boundary turns;
    "other"/None (always_ask policy) falls through to purely situational
    phrasing.
    """
    if context is None:
        return _ASK_DEFAULT

    question = _ATTRIBUTE_QUESTIONS.get(context.ask_attribute or "")

    if context.scenario == "intent_override":
        if question:
            return f"Got it, updating my search based on that! {question}"
        return _ASK_INTENT_OVERRIDE
    if context.scenario == "boundary":
        if question:
            return f"No worries, I'll use my judgment there. {question}"
        return _ASK_BOUNDARY
    if question and context.n_constraints_known == 0:
        # First question, nothing known yet: the attribute question alone is
        # a natural opener -- no filler needed.
        return question
    if context.n_constraints_known == 0:
        return _ASK_ZERO_INFO
    if question and context.turn >= _LATE_TURN:
        return f"We're getting close! {question}"
    if question:
        return question
    if context.turn >= _LATE_TURN:
        return _ASK_LATE_TURN
    return _ASK_DEFAULT


def build_recommend_message(context: FollowUpContext | None) -> str:
    """Message text for a turn where the agent is recommending without
    asking (clarify=False). Falls back to the generic default with no
    context."""
    if context is None:
        return _RECOMMEND_DEFAULT
    if context.exhausted:
        return _RECOMMEND_EXHAUSTED
    if context.turn >= _LATE_TURN:
        return _RECOMMEND_LATE_TURN
    return _RECOMMEND_DEFAULT
