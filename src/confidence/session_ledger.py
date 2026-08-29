"""Per-session conversation state consumed by the confidence policy.

The agent cannot see the simulator's hidden intent card, so ``exhausted`` is
inferred from the *user replies*. The brush-off phrasings mirror the
evaluator's own generated strings (see ``local_evaluator.py`` ``customer_reply``):

    - boundary:    "I don't have a preference for {attr}; ..."
    - no-more-info: "I don't have an additional preference for {attr}."
    - not-yet:     "Those options are not quite right yet. ..."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# True exhaustion: the customer has NO further constraint to give at all.
# Must require "additional" so it does not collide with the one-off boundary
# brush-off below (evaluator: "I don't have an additional preference for {attr}.").
_NO_MORE_INFO_RE = re.compile(r"don't have an additional preference", re.IGNORECASE)

# Boundary brush-off: the customer has no preference for THIS attribute, but may
# still have others (evaluator: "I don't have a preference for {attr}; please use
# your judgment."). This must NOT latch exhaustion.
_BOUNDARY_RE = re.compile(
    r"don't have a preference for .*please use your judgment", re.IGNORECASE
)

# Turn at/after which we stop asking regardless. Single source of truth lives in
# ``policy.TURN_CUTOFF``; imported lazily inside ``observe`` to avoid a cycle.


@dataclass
class SessionLedger:
    """Mutable state accumulated across a session.

    Attributes:
        session_id: Evaluator-supplied session identifier.
        turn: Current turn number (1-indexed).
        constraints_known: Ordered, de-duplicated constraint strings disclosed
            by the customer so far. ``len`` is the ``#constraints_known`` signal.
        asked_attributes: Attributes already asked about this session.
        exhausted: True once the customer signals no further constraints exist
            (or the hard turn cutoff is reached with no new info). Latches True
            until an override resets it.
        override_seen: True after an intent-override reply has been processed.
        no_progress_turns: Consecutive turns without a new constraint.
    """

    session_id: str
    turn: int = 0
    constraints_known: list[str] = field(default_factory=list)
    asked_attributes: set[str] = field(default_factory=set)
    exhausted: bool = False
    override_seen: bool = False
    no_progress_turns: int = 0

    @property
    def n_constraints_known(self) -> int:
        return len(self.constraints_known)

    def note_ask(self, attribute: str | None) -> None:
        """Record that the agent asked about ``attribute`` this turn."""
        if attribute:
            self.asked_attributes.add(attribute)

    def add_constraint(self, value: str) -> bool:
        """Add a disclosed constraint. Returns True if it was new."""
        cleaned = re.sub(r"\s+", " ", value).strip(" -;,.\t\n")
        if not cleaned or cleaned in self.constraints_known:
            return False
        self.constraints_known.append(cleaned)
        return True

    def observe(self, user_message: str, turn: int) -> None:
        """Update state from an incoming customer message.

        Detects intent override (resets exhaustion), brush-off / no-more-info
        replies (latches exhaustion), and tracks lack of progress.
        """
        from src.confidence.policy import TURN_CUTOFF  # lazy: avoids import cycle

        self.turn = turn
        lowered = user_message.lower()

        # Intent override: the customer replaces an earlier preference. Reset
        # the exhaustion latch so clarification can resume on the new intent.
        if re.search(r"\bactually\b|\binstead\b|ignore my earlier", lowered):
            self.override_seen = True
            self.exhausted = False
            self.no_progress_turns = 0
            return

        # Boundary brush-off: no preference for THIS attribute only. The customer
        # may still have other constraints, so do NOT latch exhaustion and do NOT
        # count it as lack of progress -- clarification should continue next turn.
        if _BOUNDARY_RE.search(user_message):
            return

        # Explicit "no further preference at all" -> exhausted (until override).
        if _NO_MORE_INFO_RE.search(user_message):
            self.exhausted = True
            self.no_progress_turns += 1
            return

        # Fell through: treat as no new structured info this turn.
        self.no_progress_turns += 1
        if turn >= TURN_CUTOFF and self.no_progress_turns >= 2:
            self.exhausted = True

    def reset_progress(self) -> None:
        """Call when a genuinely new constraint arrives."""
        self.no_progress_turns = 0
