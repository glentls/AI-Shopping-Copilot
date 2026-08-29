"""NullMemory: the permanent memory fallback. Returns an empty profile -- no boosts, no summary.

Per docs/plan/FEASIBILITY.md: cross-session persistence has no data to attach to (there is no
user ID anywhere in the contract, only a fresh random session_id per session), so this component
is scoped to intra-session distillation only. NullMemory doing nothing is a legitimate permanent
fallback, not a placeholder for a missing cross-session store.
"""

from __future__ import annotations

from src.contracts import MemoryProfile, SessionState


def distill(state: SessionState, user_profile: dict) -> MemoryProfile:
    return MemoryProfile(boosts={}, summary="")
