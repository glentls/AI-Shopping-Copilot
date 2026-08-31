"""Query construction from accumulated dialog state, not the raw last message.

This is deliberately the minimal Phase 2 slice of "slot state": it accumulates every
user message seen so far and weights the most recent turn highest, so a session never
loses turn-1's rich disclosure to a later uninformative reply (see docs/ablations.md /
docs/evaluator_notes.md -- the baseline lost ~88% of its turns to an identical generic
reply for exactly this reason). It does NOT do structured slot extraction, contradiction
-based intent-override detection, or belief tracking -- that state machine is Phase 3's
job (CLAUDE.md Phase 3a/3d). Phase 2 only needs "what text do we search with," and that
only requires accumulation + recency weighting, not the full slot model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DialogState:
    messages: list[str] = field(default_factory=list)

    def add_turn(self, user_message: str) -> None:
        if user_message:
            self.messages.append(user_message)

    def build_query(self, user_profile: dict | None = None) -> str:
        if not self.messages:
            return ""
        latest = self.messages[-1]
        history = " ".join(self.messages[:-1])
        # Repeat the latest turn so it dominates BM25 term frequency and the mean-pooled
        # dense query embedding, without discarding earlier turns entirely.
        parts = [latest, latest, history]
        if user_profile:
            tags = user_profile.get("preference_tags")
            if tags:
                parts.append(" ".join(str(tag) for tag in tags))
        return " ".join(part for part in parts if part).strip()
