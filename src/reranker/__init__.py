"""Reranker component.

Owned alongside the confidence component. Produces :class:`RankResult`, the
frozen contract consumed downstream by the confidence check.

Retrieval -> Reranker : ``list[str]`` of ``parent_asin`` (catalog IDs).
Reranker  -> Confidence: :class:`RankResult`.
"""

from src.reranker.rank import Reranker, build_reranker, default_query
from src.reranker.types import RankResult

__all__ = ["RankResult", "Reranker", "build_reranker", "default_query"]
