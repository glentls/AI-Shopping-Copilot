"""Ranking leaf component. Never imports retrieval/, dialog/, or memory/."""

from .null_reranker import rerank

__all__ = ["rerank"]
