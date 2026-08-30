"""Dense retrieval over the cached embedding matrix (src/retrieval/embed_index.py).

`dense_search()` returns the exact `list[Candidate]` shape `bm25.search()` returns, so it drops
into the same slot. The vector search itself is one matmul + one argpartition on a 50k x 384
float32 matrix -- a few ms, no FAISS (unjustified at this scale). The cost that is *not* free is
encoding the query through the transformer on CPU (~tens of ms, and a one-time ~70s model load);
callers that issue many queries (eval/recall_probe.py) should batch-encode via
`dense_search_batch()` instead of calling `dense_search()` in a loop.

Fail-soft: `build_dense_index()` returns None if the embedding feature is disabled, the cache is
missing and cannot be built, or torch/sentence-transformers will not import. Retrieval then just
runs BM25 -- exactly the Phase 0 contingency. Nothing here imports another component.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import load_config
from src.contracts import Candidate, ProductMeta, RetrievalResult

from .embed_index import EmbeddingIndex, Encoder, build_or_load, make_encoder


@dataclass
class DenseIndex:
    embedding: EmbeddingIndex                 # vectors (n, dim) float32 L2-normed + parent_asins
    products: dict[str, ProductMeta]          # shared ref (from BM25Index.products) for Candidate.meta
    query_prefix: str = ""                    # bge asymmetric-retrieval prefix; "" for most models
    _encoder: Encoder | None = None
    _row_of: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._row_of:
            self._row_of = {asin: i for i, asin in enumerate(self.embedding.parent_asins)}

    @property
    def row_of(self) -> dict[str, int]:
        return self._row_of

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """(len(queries), dim) float32, L2-normalised. Lazily loads the model on first call."""
        if self._encoder is None:
            raise RuntimeError("DenseIndex has no query encoder attached")
        prefixed = [self.query_prefix + q for q in queries]
        return self._encoder.encode(prefixed)


def build_dense_index(
    catalog_path: str | Path,
    products: dict[str, ProductMeta],
    config: dict | None = None,
) -> DenseIndex | None:
    config = config or load_config()
    emb_cfg = config["retrieval"].get("embedding", {})
    fusion_enabled = config["retrieval"].get("fusion", {}).get("enabled", False)
    if not emb_cfg.get("enabled", False) and not fusion_enabled:
        return None
    try:
        embedding = build_or_load(catalog_path, config, allow_rebuild=emb_cfg.get("allow_rebuild", True))
        encoder = make_encoder(
            embedding.model_name,
            batch_size=int(emb_cfg.get("batch_size", 128)),
            max_seq_length=int(emb_cfg.get("max_seq_length", 256)),
        )
        index = DenseIndex(
            embedding=embedding,
            products=products,
            query_prefix=str(emb_cfg.get("query_prefix", "")),
        )
        index._encoder = encoder
        return index
    except Exception:  # fail soft -- retrieval falls back to BM25-only
        print("[dense] build_dense_index failed; dense route disabled:\n" + traceback.format_exc())
        return None


def _rank_from_scores(
    index: DenseIndex, scores: np.ndarray, k: int, mask: np.ndarray | None
) -> list[Candidate]:
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    n = scores.shape[0]
    k = max(1, min(k, n))
    # argpartition for the top-k, then sort just those k by score desc
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part])]
    asins = index.embedding.parent_asins
    out: list[Candidate] = []
    for row in order:
        score = float(scores[row])
        if score == -np.inf:  # fully masked out -- stop, don't pad with junk
            break
        asin = asins[row]
        meta = index.products.get(asin)
        if meta is None:
            continue
        out.append(Candidate(parent_asin=asin, score=score, route="dense", meta=meta))
    return out


def dense_search_batch(
    index: DenseIndex, query_vectors: np.ndarray, k: int, masks: list[np.ndarray | None] | None = None
) -> list[list[Candidate]]:
    """Vectorised path for many queries at once (probe / offline eval). `query_vectors` is
    (nq, dim), already encoded + L2-normalised."""
    score_matrix = query_vectors @ index.embedding.vectors.T  # (nq, n)
    results: list[list[Candidate]] = []
    for i in range(score_matrix.shape[0]):
        mask = masks[i] if masks is not None else None
        results.append(_rank_from_scores(index, score_matrix[i], k, mask))
    return results


def dense_search(
    index: DenseIndex,
    query: str,
    k: int,
    mask: np.ndarray | None = None,
    config: dict | None = None,
) -> RetrievalResult:
    """Single-query dense retrieval. Encodes `query` (transformer forward pass on CPU -- the
    slow part), then one matmul + argpartition. Returns the same shape bm25.search() returns.

    `mask`: optional bool array aligned to `index.embedding.parent_asins`; False rows are scored
    -inf and never returned (Phase 6 hard-filter hook).
    """
    query_vector = index.encode_queries([query])[0]
    scores = index.embedding.vectors @ query_vector
    candidates = _rank_from_scores(index, scores, k, mask)
    return RetrievalResult(candidates, pool_size=len(candidates), dropped_constraints=[])
