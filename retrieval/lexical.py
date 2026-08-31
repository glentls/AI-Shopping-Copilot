"""BM25 lexical route over title + description + category path (retrieval.catalog.lexical_text),
via bm25s for speed. Index is built once in memory at Agent construction; no disk cache needed
since bm25s indexing the 50k-row catalog is fast (seconds, not the minutes an embedding
model needs) -- see retrieval/dense.py for why that route needs a precompute step instead.
"""

from __future__ import annotations

import bm25s


class LexicalRetriever:
    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens, show_progress=False)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not query.strip():
            return []
        query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        k = min(k, len(self.ids))
        indices, scores = self._bm25.retrieve(query_tokens, k=k, show_progress=False)
        return [(self.ids[idx], float(score)) for idx, score in zip(indices[0], scores[0]) if score > 0]
