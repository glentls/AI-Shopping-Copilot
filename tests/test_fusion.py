"""Unit tests for src/retrieval/fusion.py -- fabricated ranked lists, no index."""

from __future__ import annotations

import unittest

from src.contracts import Candidate, ProductMeta, RetrievalResult
from src.retrieval.fusion import fuse_results, rrf, score_fusion


def _meta(t):
    return ProductMeta(title=t, price=None, categories=[], features=[], description=[],
                       store=None, details_brand=None, average_rating=0.0, rating_number=0)


class RRFTest(unittest.TestCase):
    def test_item_ranked_well_in_both_lists_wins(self) -> None:
        lists = {"a": ["X", "Y", "Z"], "b": ["Y", "X", "W"]}
        out = dict(rrf(lists, k=60, weights={"a": 1.0, "b": 1.0}))
        self.assertEqual(max(out, key=out.get), "X")  # 1/61+1/62 vs Y 1/62+1/61 -> tie broken by X's rank1... equal actually
        self.assertGreater(out["X"], out["Z"])
        self.assertGreater(out["Y"], out["W"])

    def test_weights_bias_the_winner(self) -> None:
        lists = {"a": ["A", "B"], "b": ["B", "A"]}
        heavy_a = dict(rrf(lists, 60, {"a": 5.0, "b": 1.0}))
        self.assertGreater(heavy_a["A"], heavy_a["B"])

    def test_k_flattens_rank_differences(self) -> None:
        lists = {"a": ["A", "B"]}
        small_k = dict(rrf(lists, 1, {"a": 1.0}))
        big_k = dict(rrf(lists, 1000, {"a": 1.0}))
        self.assertGreater(small_k["A"] - small_k["B"], big_k["A"] - big_k["B"])


class ScoreFusionTest(unittest.TestCase):
    def test_minmax_puts_each_list_on_0_1(self) -> None:
        lists = {"bm25": [("A", 10.0), ("B", 2.0)], "dense": [("B", 0.9), ("A", 0.1)]}
        out = dict(score_fusion(lists, {"bm25": 1.0, "dense": 1.0}, "minmax"))
        # A: 1.0 + 0.0 ; B: 0.0 + 1.0 -> tie
        self.assertAlmostEqual(out["A"], out["B"], places=6)

    def test_zscore_runs_and_orders(self) -> None:
        lists = {"bm25": [("A", 100.0), ("B", 1.0), ("C", 0.5)]}
        out = dict(score_fusion(lists, {"bm25": 1.0}, "zscore"))
        self.assertEqual(max(out, key=out.get), "A")


class FuseResultsTest(unittest.TestCase):
    def _res(self, ids, base=5.0):
        cs = [Candidate(parent_asin=i, score=base - n, route="bm25", meta=_meta(i)) for n, i in enumerate(ids)]
        return RetrievalResult(cs, pool_size=len(cs), dropped_constraints=["price"])

    def test_rrf_fusion_returns_candidates_with_fused_route(self) -> None:
        bm = self._res(["A", "B", "C", "D"])
        dn = self._res(["C", "A", "E", "F"])
        fused = fuse_results({"bm25": bm, "dense": dn},
                             {"method": "rrf", "depth": 200, "rrf_k": 60, "weights": {"bm25": 1, "dense": 1}})
        self.assertTrue(all(c.route == "fused" for c in fused))
        self.assertIn(fused[0].parent_asin, {"A", "C"})  # the two that appear in both
        self.assertEqual(set(c.parent_asin for c in fused), {"A", "B", "C", "D", "E", "F"})

    def test_depth_limits_what_is_fused(self) -> None:
        bm = self._res(["A", "B", "C"])
        dn = self._res(["X", "Y", "Z"])
        fused = fuse_results({"bm25": bm, "dense": dn},
                             {"method": "rrf", "depth": 1, "rrf_k": 60, "weights": {"bm25": 1, "dense": 1}})
        self.assertEqual(set(c.parent_asin for c in fused), {"A", "X"})

    def test_carries_pool_size_and_dropped_from_bm25(self) -> None:
        fused = fuse_results({"bm25": self._res(["A"]), "dense": self._res(["B"])},
                             {"method": "rrf", "weights": {}})
        self.assertEqual(fused.dropped_constraints, ["price"])


if __name__ == "__main__":
    unittest.main()
