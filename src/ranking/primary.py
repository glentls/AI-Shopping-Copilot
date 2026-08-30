"""Primary ranking selector for the deterministic local cross-encoder."""

from __future__ import annotations

from src.contracts import Candidate, SessionState

from .cross_encoder import rerank as cross_encoder_rerank


def rerank(state: SessionState, candidates: list[Candidate]) -> list[Candidate]:
    """Use the local cross-encoder behind NullReranker's exact public signature."""
    return cross_encoder_rerank(state, candidates)
