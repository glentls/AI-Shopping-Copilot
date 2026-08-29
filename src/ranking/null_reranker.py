"""NullReranker: the permanent ranking fallback. Returns candidates in input order, unchanged.

This is not scaffolding -- it is a real, permanent implementation. Every future reranker
(cross-encoder, LLM listwise) must degrade to exactly this behavior on failure or timeout.
"""

from __future__ import annotations

from src.contracts import Candidate, SessionState


def rerank(state: SessionState, candidates: list[Candidate]) -> list[Candidate]:
    return list(candidates)
