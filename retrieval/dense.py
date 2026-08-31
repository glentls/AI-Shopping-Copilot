"""Dense semantic route: CPU-friendly small embedding model (BAAI/bge-small-en-v1.5,
ONNX-quantized via fastembed -- no torch dependency), catalog embeddings precomputed
once offline by scripts/build_dense_index.py and cached to disk, loaded here as a plain
in-memory numpy array. No vector DB.

DenseRetriever only *loads* the cache; it never re-embeds the catalog at runtime -- if
the cache is missing or doesn't match the current catalog, it fails loudly (this is a
correctness issue, not something to silently paper over with a stale or misaligned
index) and tells you which script to run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_CACHE_DIR = "data/model_cache"


class DenseIndexMissing(RuntimeError):
    pass


def load_cached_index(cache_dir: str | Path) -> tuple[list[str], np.ndarray]:
    cache_dir = Path(cache_dir)
    ids_path = cache_dir / "ids.json"
    embeddings_path = cache_dir / "embeddings.npy"
    if not ids_path.exists() or not embeddings_path.exists():
        raise DenseIndexMissing(
            f"Dense index not found at {cache_dir}. Run "
            f"`python -m scripts.build_dense_index` first (see README)."
        )
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path)
    if len(ids) != embeddings.shape[0]:
        raise DenseIndexMissing(
            f"Dense index at {cache_dir} is corrupt: {len(ids)} ids vs "
            f"{embeddings.shape[0]} embedding rows. Rebuild with `python -m scripts.build_dense_index`."
        )
    return ids, embeddings


class DenseRetriever:
    def __init__(
        self,
        ids: list[str],
        cache_dir: str | Path = "data/dense_index",
        model_name: str = DEFAULT_MODEL_NAME,
        model_cache_dir: str = DEFAULT_MODEL_CACHE_DIR,
    ):
        cached_ids, embeddings = load_cached_index(cache_dir)
        if cached_ids != ids:
            raise DenseIndexMissing(
                f"Dense index at {cache_dir} was built from a different catalog ordering/version. "
                f"Rebuild with `python -m scripts.build_dense_index`."
            )
        self.ids = cached_ids
        self.embeddings = embeddings  # (N, dim), L2-normalized rows

        from fastembed import TextEmbedding  # deferred: avoids paying ONNX runtime import cost when unused

        self._model = TextEmbedding(model_name=model_name, cache_dir=model_cache_dir)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not query.strip():
            return []
        query_embedding = next(self._model.embed([query])).astype(np.float32)
        # Rows are L2-normalized at build time, and fastembed normalizes query embeddings
        # too, so dot product == cosine similarity.
        scores = self.embeddings @ query_embedding
        k = min(k, len(self.ids))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.ids[i], float(scores[i])) for i in top_idx]
