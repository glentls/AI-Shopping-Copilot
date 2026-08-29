"""Retrieval component: BM25 candidate retrieval over the frozen catalog.

Loads the catalog once (:class:`Catalog`) and serves ranked ``parent_asin``
candidates via SQLite FTS5 (:class:`Retriever`). This is the upstream boundary
feeding the reranker; kept separable so a different retrieval backend can
replace it without touching the reranker.

    Retrieval -> Reranker : ``list[str]`` of ``parent_asin`` (catalog IDs).
"""

from src.retrieval.catalog import Catalog, Product, searchable_text
from src.retrieval.retriever import Retriever, terms

__all__ = ["Catalog", "Product", "searchable_text", "Retriever", "terms"]
