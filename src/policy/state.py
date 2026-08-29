"""Turn-by-turn conversation state.

LANE C OWNS THIS FILE.

The two behaviours that punish a naive design live here:
  Intent Override (15% of sessions) -- a preference is REPLACED on turn 3 or 4.
    State that merely accumulates ends up holding two contradictory constraints
    and ranks WORSE than before the customer spoke. Slots must overwrite.
  Boundary (5%) -- the customer has no preference for what we asked. Mark the
    slot unanswerable so we never spend another turn on it.

A third, quieter failure lives here too: repetition. A customer who says
"leather" on turn 2 and again on turn 4 is one constraint, not two, but a state
that appends both hands the reranker a doubled slot bonus for the same fact.
`_absorb` folds a repeat into the value already held.
"""

from __future__ import annotations

from src.contracts import ConversationState, SlotValue
from src.extract import detect_no_preference, detect_override, extract_slots, parse_budget


def _absorb(state: ConversationState, slot: str, incoming: SlotValue) -> None:
    """Merge one extracted value into the slot, folding repeats.

    Appending a duplicate is not free. The reranker scores a candidate by
    counting how many live slot values it carries, so holding "leather" twice
    makes a leather belt outrank a better overall match on the strength of one
    thing the customer said once. Fold instead: refresh polarity so a
    re-assertion can revive a value we had retracted, and keep the ORIGINAL
    turn, because SlotValue.turn is documented as the turn a fact was learned
    on and the question picker reads it to tell a productive turn from a
    silent one.
    """
    for held in state.slots.get(slot, []):
        if held.value == incoming.value:
            held.polarity = incoming.polarity
            # A repeat is emphasis: the customer has now said this twice. Raise
            # confidence rather than stacking a second copy. Lane B's reranker
            # currently scores every live value at a flat 1.0 and ignores this
            # field -- see docs/lane_c_notes.md, it is worth ~0.02 MRR on the
            # override sessions once rerank weights by it.
            held.confidence = min(1.0, max(held.confidence, incoming.confidence) + 0.05)
            return
    state.add(slot, incoming)


def _apply_boundary(state: ConversationState, user_message: str) -> None:
    """'No preference' -> never spend another turn on that slot.

    The customer may name the slot ("no preference on colour") or leave it
    implicit ("you decide"), in which case it refers to whatever we asked last.
    """
    for slot in detect_no_preference(user_message):
        resolved = state.last_asked if slot == "*" else slot
        if resolved:
            state.unanswerable.add(resolved)


def _apply_override(
    state: ConversationState,
    user_message: str,
    incoming: dict[str, list[SlotValue]],
) -> None:
    """Retract what the new message contradicts, before the new values land."""
    retracted = detect_override(user_message)
    if not retracted:
        return
    targets = set(incoming) if retracted == ["*"] else set(retracted)
    for slot in targets:
        # Only retract what the new message actually CONTRADICTS. "Actually,
        # what I need is leather" when we already had leather is a customer
        # restating their priority, not changing it -- retracting there throws
        # away a correct constraint and the ranking gets worse.
        asserted = {value.value for value in incoming.get(slot, []) if value.polarity}
        for held in state.slots.get(slot, []):
            if held.polarity and held.value not in asserted:
                held.polarity = False


def update(state: ConversationState, user_message: str, turn: int) -> ConversationState:
    state.turn = turn
    state.history.append(("customer", user_message))

    _apply_boundary(state, user_message)

    incoming = extract_slots(user_message, turn, state)
    _apply_override(state, user_message, incoming)

    for slot, values in incoming.items():
        for value in values:
            _absorb(state, slot, value)

    budget = parse_budget(user_message)
    if budget is not None:
        state.budget_max = budget

    return state


def learned_on(state: ConversationState, turn: int) -> int:
    """How many facts the message on `turn` taught us.

    Only genuinely new values count: `_absorb` leaves a repeat stamped with the
    turn it was first learned on, so a customer restating an old preference
    reads as the silent turn it really is.
    """
    return sum(
        1 for values in state.slots.values() for value in values if value.turn == turn
    )
