"""Turn-by-turn conversation state.

LANE C OWNS THIS FILE. Working placeholder.

The two behaviours that punish a naive design live here:
  Intent Override (15% of sessions) -- a preference is REPLACED on turn 3 or 4.
    State that merely accumulates ends up holding two contradictory constraints
    and ranks WORSE than before the customer spoke. Slots must overwrite.
  Boundary (5%) -- the customer has no preference for what we asked. Mark the
    slot unanswerable so we never spend another turn on it.
"""

from __future__ import annotations

from src.contracts import ConversationState
from src.extract import detect_no_preference, detect_override, extract_slots, parse_budget


def update(state: ConversationState, user_message: str, turn: int) -> ConversationState:
    state.turn = turn
    state.history.append(("customer", user_message))

    # Boundary: they have no preference for whatever we last asked about.
    for slot in detect_no_preference(user_message):
        resolved = state.last_asked if slot == "*" else slot
        if resolved:
            state.unanswerable.add(resolved)

    # Override: retract the old value before adding the new one.
    retracted = detect_override(user_message)
    incoming = extract_slots(user_message, turn, state)
    if retracted:
        targets = set(incoming) if retracted == ["*"] else set(retracted)
        for slot in targets:
            # Only retract what the new message actually CONTRADICTS. "Actually,
            # what I need is leather" when we already had leather is a customer
            # restating their priority, not changing it -- retracting there
            # throws away a correct constraint and the ranking gets worse.
            asserted = {v.value for v in incoming.get(slot, []) if v.polarity}
            for held in state.slots.get(slot, []):
                if held.polarity and held.value not in asserted:
                    held.polarity = False

    for slot, values in incoming.items():
        for value in values:
            state.add(slot, value)

    budget = parse_budget(user_message)
    if budget is not None:
        state.budget_max = budget

    return state
