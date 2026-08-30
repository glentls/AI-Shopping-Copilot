"""Unit tests for src/retrieval/dense.py -- fabricated 5-row index, fake encoder, no model,
no catalog, no torch."""

from __future__ import annotations

import unittest

import numpy as np

from src.contracts import ProductMeta
from src.retrieval.dense import DenseIndex, dense_search, dense_search_batch
from src.retrieval.embed_index import Encoder, EmbeddingIndex


def _meta(title: str) -> ProductMeta:
    return ProductMeta(title=title, price=None, categories=[], features=[], description=[],
                       store=None, details_brand=None, average_rating=0.0, rating_number=0)


class _FakeEncoder(Encoder):
    """Maps a query string to a 3-d unit vector by reading three ints out of it, e.g. "1,0,0"."""

    name = "fake"
    dim = 3

    def _load(self) -> None:
        self._loaded = True

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            nums = [float(x) for x in t.replace("PFX", "").strip().split(",")]
            rows.append(nums)
        return np.array(rows, dtype=np.float32)


def _index() -> DenseIndex:
    # 5 catalog rows, unit vectors pointing at axes / diagonals
    vecs = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]], dtype=np.float32
    )
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    asins = ["A", "B", "C", "D", "E"]
    emb = EmbeddingIndex(vectors=vecs, parent_asins=asins, model_name="fake",
                         template_version="t", catalog_sha="sha", dim=3)
    idx = DenseIndex(embedding=emb, products={a: _meta(a) for a in asins}, query_prefix="PFX")
    idx._encoder = _FakeEncoder()
    return idx


class DenseSearchTest(unittest.TestCase):
    def test_returns_candidates_sorted_by_cosine_desc(self) -> None:
        result = dense_search(_index(), "1,0,0", k=3)
        self.assertEqual([c.parent_asin for c in result], ["A", "D", "E"])  # A exact, D/E share x
        self.assertTrue(all(result[i].score >= result[i + 1].score for i in range(len(result) - 1)))
        self.assertTrue(all(c.route == "dense" for c in result))

    def test_k_caps_the_result_length(self) -> None:
        self.assertEqual(len(dense_search(_index(), "1,1,1", k=2)), 2)

    def test_mask_excludes_rows_and_they_never_appear(self) -> None:
        mask = np.array([False, True, True, False, False])  # only B, C eligible
        result = dense_search(_index(), "1,0,0", k=5, mask=mask)
        self.assertEqual(set(c.parent_asin for c in result), {"B", "C"})

    def test_result_is_a_list_and_carries_pool_size(self) -> None:
        result = dense_search(_index(), "0,1,0", k=3)
        self.assertIsInstance(result, list)
        self.assertEqual(result.pool_size, len(result))
        self.assertEqual(result.dropped_constraints, [])

    def test_batch_matches_single(self) -> None:
        idx = _index()
        qv = idx.encode_queries(["1,0,0", "0,0,1"])
        batch = dense_search_batch(idx, qv, k=3)
        self.assertEqual([c.parent_asin for c in batch[0]], ["A", "D", "E"])
        self.assertEqual(batch[1][0].parent_asin, "C")


if __name__ == "__main__":
    unittest.main()
