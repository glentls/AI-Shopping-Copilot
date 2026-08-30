"""Unit tests for src/retrieval/postprocess.py -- fabricated Candidates, no catalog, no index."""

from __future__ import annotations

import unittest

from src.contracts import Candidate, ProductMeta, RetrievalResult
from src.retrieval.postprocess import (
    CategoryIndex,
    apply_popularity_prior,
    apply_soft_prefs,
    filter_candidates,
    postprocess,
)


def _meta(cats=(), rating_number=0, store=None, price=None) -> ProductMeta:
    return ProductMeta(title="t", price=price, categories=list(cats), features=[], description=[],
                       store=store, details_brand=None, average_rating=0.0, rating_number=rating_number)


def _cand(asin, score, **meta_kw) -> Candidate:
    return Candidate(parent_asin=asin, score=score, route="bm25", meta=_meta(**meta_kw))


def _cfg(**retrieval_over) -> dict:
    base = {
        "retrieval": {
            "candidate_pool_size": 50,
            "popularity": {"enabled": False, "weight": 0.15},
            "filters": {"prefilter_pool_size": 800},
            "relaxation": {"min_pool_size": 1, "priority": ["category"]},
            "soft_prefs": {"enabled": False, "store_boost": 0.1, "price_boost": 0.05},
        }
    }
    base["retrieval"].update(retrieval_over)
    return base


class CategoryIndexTest(unittest.TestCase):
    def test_strips_scaffolding_and_keeps_signal_tokens(self) -> None:
        idx = CategoryIndex.build({"A": _meta(cats=["Clothing, Shoes & Jewelry", "Women", "Running Shoes"])})
        self.assertIn("running", idx.tokens_of["A"])
        self.assertNotIn("women", idx.tokens_of["A"])
        self.assertNotIn("clothing", idx.tokens_of["A"])

    def test_matches_is_AND_over_terms(self) -> None:
        idx = CategoryIndex.build({"A": _meta(cats=["Athletic Running Shorts"])})
        self.assertTrue(idx.matches("A", frozenset({"running", "shorts"})))
        self.assertFalse(idx.matches("A", frozenset({"running", "jacket"})))
        self.assertTrue(idx.matches("A", frozenset()))  # nothing to filter on


class FilterTest(unittest.TestCase):
    def test_no_hard_filters_is_identity(self) -> None:
        cands = [_cand("A", 3.0), _cand("B", 2.0)]
        kept, dropped = filter_candidates(cands, {}, None, _cfg())
        self.assertEqual(kept, cands)
        self.assertEqual(dropped, [])

    def test_category_filter_keeps_only_matches(self) -> None:
        cands = [_cand("A", 3.0, cats=["Running Jacket"]), _cand("B", 2.0, cats=["Dress Shirt"])]
        idx = CategoryIndex.build({c.parent_asin: c.meta for c in cands})
        kept, dropped = filter_candidates(cands, {"category": "jacket"}, idx, _cfg())
        self.assertEqual([c.parent_asin for c in kept], ["A"])
        self.assertEqual(dropped, [])

    def test_relaxation_ladder_drops_filter_when_pool_too_small(self) -> None:
        cands = [_cand("A", 3.0, cats=["Dress Shirt"]), _cand("B", 2.0, cats=["Dress Shirt"])]
        idx = CategoryIndex.build({c.parent_asin: c.meta for c in cands})
        cfg = _cfg(relaxation={"min_pool_size": 5, "priority": ["category"]})
        kept, dropped = filter_candidates(cands, {"category": "jacket"}, idx, cfg)
        self.assertEqual(kept, cands)          # filter relaxed -> original pool back
        self.assertEqual(dropped, ["category"])

    def test_unenforceable_keys_are_reported_dropped(self) -> None:
        cands = [_cand("A", 3.0, cats=["Running Jacket"]), _cand("B", 2.0, cats=["Running Jacket"])]
        idx = CategoryIndex.build({c.parent_asin: c.meta for c in cands})
        kept, dropped = filter_candidates(cands, {"category": "jacket", "color": "blue"}, idx, _cfg())
        self.assertEqual([c.parent_asin for c in kept], ["A", "B"])
        self.assertIn("color", dropped)


class PopularityTest(unittest.TestCase):
    def test_disabled_is_identity(self) -> None:
        cands = [_cand("A", 3.0, rating_number=1), _cand("B", 2.0, rating_number=9999)]
        self.assertEqual(apply_popularity_prior(cands, _cfg()), cands)

    def test_enabled_lifts_a_popular_near_miss(self) -> None:
        cands = [_cand("A", 3.0, rating_number=1), _cand("B", 2.9, rating_number=50000)]
        cfg = _cfg(popularity={"enabled": True, "weight": 0.6})
        out = apply_popularity_prior(cands, cfg)
        self.assertEqual([c.parent_asin for c in out], ["B", "A"])

    def test_small_weight_keeps_strong_lexical_on_top(self) -> None:
        cands = [_cand("A", 10.0, rating_number=1), _cand("B", 1.0, rating_number=50000)]
        cfg = _cfg(popularity={"enabled": True, "weight": 0.15})
        out = apply_popularity_prior(cands, cfg)
        self.assertEqual(out[0].parent_asin, "A")

    def test_scores_are_preserved(self) -> None:
        cands = [_cand("A", 3.0, rating_number=1), _cand("B", 2.9, rating_number=50000)]
        out = apply_popularity_prior(cands, _cfg(popularity={"enabled": True, "weight": 0.6}))
        self.assertEqual({c.parent_asin: c.score for c in out}, {"A": 3.0, "B": 2.9})


class SoftPrefsTest(unittest.TestCase):
    def test_disabled_is_identity(self) -> None:
        cands = [_cand("A", 3.0), _cand("B", 2.0, store="Nike")]
        self.assertEqual(apply_soft_prefs(cands, {"store": "nike"}, _cfg()), cands)

    def test_store_match_boosts(self) -> None:
        cands = [_cand("A", 1.0, store="Adidas"), _cand("B", 0.9, store="Nike Official")]
        cfg = _cfg(soft_prefs={"enabled": True, "store_boost": 5.0, "price_boost": 0.0})
        out = apply_soft_prefs(cands, {"store": "nike"}, cfg)
        self.assertEqual(out[0].parent_asin, "B")

    def test_missing_price_is_neutral_not_penalised(self) -> None:
        cands = [_cand("A", 1.0, price=None), _cand("B", 0.5, price=20.0)]
        cfg = _cfg(soft_prefs={"enabled": True, "store_boost": 0.0, "price_boost": 5.0})
        out = apply_soft_prefs(cands, {"price_max": 25.0}, cfg)
        self.assertEqual(out[0].parent_asin, "B")   # B boosted; A not pushed below by a penalty
        self.assertEqual([c.parent_asin for c in out], ["B", "A"])


class PostprocessTest(unittest.TestCase):
    def test_inert_when_everything_off_and_no_filters(self) -> None:
        res = RetrievalResult([_cand("A", 3.0), _cand("B", 2.0)], pool_size=2, dropped_constraints=[])
        out = postprocess(res, {}, {}, None, _cfg())
        self.assertEqual([c.parent_asin for c in out], ["A", "B"])
        self.assertEqual(out.pool_size, 2)
        self.assertEqual(out.dropped_constraints, [])

    def test_threads_pool_size_and_dropped(self) -> None:
        cands = [_cand("A", 3.0, cats=["Running Jacket"]), _cand("B", 2.0, cats=["Dress Shirt"])]
        idx = CategoryIndex.build({c.parent_asin: c.meta for c in cands})
        res = RetrievalResult(cands, pool_size=2, dropped_constraints=[])
        out = postprocess(res, {"category": "jacket", "size": "M"}, {}, idx,
                          _cfg(relaxation={"min_pool_size": 1, "priority": ["category"]}))
        self.assertEqual([c.parent_asin for c in out], ["A"])
        self.assertEqual(out.pool_size, 1)
        self.assertIn("size", out.dropped_constraints)


if __name__ == "__main__":
    unittest.main()
