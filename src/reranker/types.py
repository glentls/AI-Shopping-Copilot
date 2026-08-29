"""Reranker output contract.

``RankResult`` is the frozen interface between the reranker (this component) and
the confidence check. The confidence function reads only ``max_coverage``,
``top_tier_crowd`` and ``pool_size`` off this struct (plus the ledger's
``constraints_known`` count). Keep this frozen so the two components never form
a build/branch dependency.

The upstream interface (retrieval -> reranker) is simply ``list[str]`` of
``parent_asin`` values (catalog IDs, e.g. "B09PYB7B6Z").
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RankResult:
    """Reranker output plus the internals the confidence function needs.

    Attributes:
        ranked: Top-10 ``parent_asin`` values, best-to-worst. These are also
            the recommendations emitted every turn.
        pool_size: Number of candidates handed in by retrieval (pre-truncation).
        max_coverage: Highest number of known constraints matched by any single
            product in the pool.
        top_tier_crowd: Number of products sharing ``max_coverage``. A large
            crowd means the current constraints are ambiguous.
    """

    ranked: list[str] = field(default_factory=list)
    pool_size: int = 0
    max_coverage: int = 0
    top_tier_crowd: int = 0
