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

from src.catalog.catalog import Catalog
from src.reranker.coverage import Product, compile_constraints
from src.retrieval.retrieval import Retriever
from src.reranker.types import RankResult

DEFAULT_POOL = 200

_ROW_COLUMNS = (
    "parent_asin",
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
    "price",
    "average_rating",
    "rating_number",
)


def default_query(constraints: list[str], extra: str = "") -> str:
    """Build a retrieval query string from known constraints (+ optional text)."""
    return " ".join([*constraints, extra]).strip()


def _hydrate_products(
    catalog: Catalog,
    parent_asins: list[str],
    cache: dict[str, Product] | None = None,
) -> dict[str, Product]:
    """Batch-fetch catalog rows for ``parent_asins`` and build reranker ``Product``
    shims keyed by parent_asin.

    ``cache`` (if given) is a persistent, content-addressed store of
    previously hydrated products (keyed by ``parent_asin``, never mutated by
    the catalog during a run) -- only the ids missing from it are fetched,
    and the cache is updated in place with any newly fetched rows."""
    if not parent_asins:
        return {}
    if cache is None:
        missing = parent_asins
    else:
        missing = [pid for pid in parent_asins if pid not in cache]
    if missing:
        placeholders = ", ".join("?" for _ in missing)
        sql = (
            f"SELECT {', '.join(_ROW_COLUMNS)} FROM products "
            f"WHERE parent_asin IN ({placeholders})"
        )
        rows = catalog.execute(sql, missing)
        fetched = _rows_to_products(rows)
        if cache is not None:
            cache.update(fetched)
    else:
        fetched = {}
    if cache is None:
        return fetched
    return {pid: cache[pid] for pid in parent_asins if pid in cache}


def _rows_to_products(rows: list[tuple]) -> dict[str, Product]:
    products: dict[str, Product] = {}
    for row in rows:
        (
            parent_asin,
            title,
            categories,
            features,
            details,
            store,
            description,
            price,
            average_rating,
            rating_number,
        ) = row
        text = " ".join(
            str(part)
            for part in (title, categories, features, details, store, description)
            if part
        ).lower()
        products[str(parent_asin)] = Product(
            parent_asin=str(parent_asin),
            text=text,
            price=float(price) if price is not None else None,
            rating_number=int(rating_number) if rating_number is not None else 0,
            average_rating=float(average_rating) if average_rating is not None else 0.0,
        )
    return products


class Reranker:
    def __init__(self, catalog: Catalog, retriever: Retriever) -> None:
        self.catalog = catalog
        self.retriever = retriever
        # Process-lifetime, content-addressed caches: the catalog is
        # read-only for the duration of a run, so a cache hit is always
        # exactly the value the uncached path would have computed.
        self._bm25_cache: dict[tuple[str, int], list[str]] = {}
        self._product_cache: dict[str, Product] = {}

    def rank(
        self,
        query: str,
        constraints: list[str] | None = None,
        top_k: int = 10,
        pool_size: int = DEFAULT_POOL,
    ) -> RankResult:
        constraints = constraints or []
        cache_key = (query, pool_size)
        candidate_ids = self._bm25_cache.get(cache_key)
        if candidate_ids is None:
            candidate_ids = self.retriever.retrieve_bm25({"keywords": [query]}, top_k=pool_size)
            self._bm25_cache[cache_key] = candidate_ids

        if not candidate_ids:
            return RankResult()

        products = _hydrate_products(self.catalog, candidate_ids, cache=self._product_cache)

        # Compile each constraint once, then reuse across all candidates.
        matchers = compile_constraints(constraints)

        # Score each candidate: coverage, retrieval rank (lower=better), rating.
        # Track max coverage and its crowd in the same scan (no second pass).
        scored = []
        max_coverage = 0
        top_tier_crowd = 0
        for retrieval_rank, pid in enumerate(candidate_ids):
            product = products.get(pid)
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
