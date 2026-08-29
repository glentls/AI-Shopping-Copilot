"""Reranker core: retrieve -> coverage-rerank -> RankResult.

Public API:
    build_reranker(catalog_path) -> Reranker
    Reranker.rank(query, constraints, top_k) -> RankResult
    default_query(constraints) -> str        # helper to build a query from constraints

``rank`` reranks retrieved candidates by (coverage desc, retrieval rank asc,
rating desc) and assembles the internals the confidence check needs:
``max_coverage`` and ``top_tier_crowd``.
"""

from __future__ import annotations

from src.reranker.catalog import Catalog
from src.reranker.coverage import compile_constraints
from src.reranker.retriever import Retriever
from src.reranker.types import RankResult

DEFAULT_POOL = 200


def default_query(constraints: list[str], extra: str = "") -> str:
    """Build a retrieval query string from known constraints (+ optional text)."""
    return " ".join([*constraints, extra]).strip()


class Reranker:
    def __init__(self, catalog: Catalog, retriever: Retriever) -> None:
        self.catalog = catalog
        self.retriever = retriever

    def rank(
        self,
        query: str,
        constraints: list[str] | None = None,
        top_k: int = 10,
        pool_size: int = DEFAULT_POOL,
    ) -> RankResult:
        constraints = constraints or []
        candidate_ids = self.retriever.search(query, limit=pool_size)

        if not candidate_ids:
            return RankResult()

        # Compile each constraint once, then reuse across all candidates.
        matchers = compile_constraints(constraints)

        # Score each candidate: coverage, retrieval rank (lower=better), rating.
        # Track max coverage and its crowd in the same scan (no second pass).
        scored = []
        max_coverage = 0
        top_tier_crowd = 0
        for retrieval_rank, pid in enumerate(candidate_ids):
            product = self.catalog.products.get(pid)
            if product is None:
                continue
            cov = sum(1 for m in matchers if m.matches(product))
            scored.append((cov, retrieval_rank, product))
            if cov > max_coverage:
                max_coverage = cov
                top_tier_crowd = 1
            elif cov == max_coverage:
                top_tier_crowd += 1

        if not scored:
            return RankResult()

        # Rerank: coverage desc, retrieval rank asc, rating desc, id asc (stable).
        scored.sort(
            key=lambda s: (
                -s[0],
                s[1],
                -s[2].rating_number,
                -s[2].average_rating,
                s[2].parent_asin,
            )
        )

        ranked_ids = [s[2].parent_asin for s in scored[:top_k]]

        return RankResult(
            ranked=ranked_ids,
            pool_size=len(scored),
            max_coverage=max_coverage,
            top_tier_crowd=top_tier_crowd,
        )


def build_reranker(catalog_path: str) -> Reranker:
    catalog = Catalog(catalog_path)
    retriever = Retriever(catalog)
    return Reranker(catalog, retriever)
