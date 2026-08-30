"""Unit tests for src/retrieval/multiturn.py -- fabricated turn streams, no index, no catalog."""

from __future__ import annotations

import unittest

from src.contracts import ProductMeta, RetrievalRequest
from src.retrieval.multiturn import (
    SessionMemory,
    accumulate_query,
    blend_profile,
    build_effective_query,
    rocchio_terms,
)


def _cfg(**mt) -> dict:
    base = {"enabled": True, "window_turns": 0, "profile_blend": False, "rocchio": False,
            "rocchio_add_terms": 6, "rocchio_avoid_terms": 6}
    base.update(mt)
    return {"retrieval": {"max_query_terms": 40, "multiturn": base}}


def _meta(title, features=()) -> ProductMeta:
    return ProductMeta(title=title, price=None, categories=[], features=list(features), description=[],
                       store=None, details_brand=None, average_rating=0.0, rating_number=0)


def _req(**kw) -> RetrievalRequest:
    kw.setdefault("intent", "unknown")
    kw.setdefault("hard_filters", {})
    kw.setdefault("soft_prefs", {})
    kw.setdefault("top_k", 10)
    return RetrievalRequest(**kw)


class SessionMemoryTest(unittest.TestCase):
    def test_accumulates_turns_in_order(self) -> None:
        m = SessionMemory()
        m.observe("s", 1, "running shoes")
        m.observe("s", 2, "for that: cotton")
        self.assertEqual([t for t, _ in m.turns("s")], [1, 2])

    def test_turn_1_resets_a_reused_session_id(self) -> None:
        m = SessionMemory()
        m.observe("s", 1, "old query")
        m.observe("s", 2, "more")
        m.observe("s", 1, "brand new session")     # evaluator reuses ids across runs
        self.assertEqual(m.turns("s"), [(1, "brand new session")])

    def test_intent_change_wipes_prior_context(self) -> None:
        m = SessionMemory()
        m.observe("s", 1, "blue jacket")
        m.observe("s", 2, "actually a red dress", intent_changed=True)
        self.assertEqual(m.turns("s"), [(2, "actually a red dress")])

    def test_same_turn_reask_replaces_not_appends(self) -> None:
        m = SessionMemory()
        m.observe("s", 1, "first")
        m.observe("s", 1, "retry")
        self.assertEqual(m.turns("s"), [(1, "retry")])

    def test_lru_eviction_bounds_the_dict(self) -> None:
        m = SessionMemory(max_sessions=2)
        for i in range(4):
            m.observe(f"s{i}", 1, "q")
        self.assertEqual(m.turns("s0"), [])
        self.assertEqual([t for t, _ in m.turns("s3")], [1])


class AccumulateQueryTest(unittest.TestCase):
    def test_unions_terms_across_turns(self) -> None:
        q = accumulate_query([(1, "running shoes"), (2, "cotton lightweight")], _cfg())
        self.assertEqual(set(q.split()), {"running", "shoes", "cotton", "lightweight"})

    def test_recent_terms_win_when_capped(self) -> None:
        cfg = _cfg()
        cfg["retrieval"]["max_query_terms"] = 2
        q = accumulate_query([(1, "old alpha"), (2, "fresh beta")], cfg)
        self.assertEqual(set(q.split()), {"fresh", "beta"})

    def test_scaffolding_only_turn_contributes_nothing(self) -> None:
        # NullDialog's turn-2 message is pure stopwords/filler after tokenisation
        q = accumulate_query([(1, "leather boots"), (2, "those options are not quite right yet")], _cfg())
        self.assertIn("leather", q)
        self.assertIn("boots", q)

    def test_window_limits_turns_considered(self) -> None:
        q = accumulate_query([(1, "alpha"), (2, "beta"), (3, "gamma")], _cfg(window_turns=1))
        self.assertEqual(q.split(), ["gamma"])


class BlendProfileTest(unittest.TestCase):
    def test_off_by_default(self) -> None:
        self.assertEqual(blend_profile("shoes", {"preference_tags": ["warmth"]}, _cfg()), "shoes")

    def test_appends_preference_tags_when_on(self) -> None:
        out = blend_profile("shoes", {"preference_tags": ["warmth", "durability"]}, _cfg(profile_blend=True))
        self.assertEqual(out, "shoes warmth durability")

    def test_no_tags_is_a_noop(self) -> None:
        self.assertEqual(blend_profile("shoes", {}, _cfg(profile_blend=True)), "shoes")


class RocchioTest(unittest.TestCase):
    def test_add_terms_come_from_accepted_titles_and_features(self) -> None:
        accepted = [_meta("Merino Wool Base Layer", ["merino wool", "moisture wicking"])]
        add, avoid = rocchio_terms(accepted, [], _cfg())
        self.assertIn("merino", add)
        self.assertIn("wool", add)

    def test_avoid_terms_are_negative_only(self) -> None:
        accepted = [_meta("Wool Sweater")]
        negatives = [_meta("Polyester Fleece Jacket")]
        add, avoid = rocchio_terms(accepted, negatives, _cfg())
        self.assertIn("polyester", avoid)
        self.assertNotIn("wool", avoid)


class BuildEffectiveQueryTest(unittest.TestCase):
    def test_inert_when_disabled(self) -> None:
        m = SessionMemory()
        cfg = _cfg(); cfg["retrieval"]["multiturn"]["enabled"] = False
        out = build_effective_query(_req(canonical_query="raw", session_id="s", turn=2), m, cfg)
        self.assertEqual(out, "raw")
        self.assertEqual(m.turns("s"), [])  # not even recorded

    def test_inert_when_no_turn_or_session(self) -> None:
        m = SessionMemory()
        self.assertEqual(build_effective_query(_req(canonical_query="raw", turn=0), m, _cfg()), "raw")

    def test_turn_1_returns_raw_query_but_records_it(self) -> None:
        m = SessionMemory()
        out = build_effective_query(_req(canonical_query="running shoes", session_id="s", turn=1), m, _cfg())
        self.assertEqual(out, "running shoes")
        self.assertEqual(m.turns("s"), [(1, "running shoes")])

    def test_turn_2_accumulates_with_turn_1(self) -> None:
        m = SessionMemory()
        cfg = _cfg()
        build_effective_query(_req(canonical_query="running shoes", session_id="s", turn=1), m, cfg)
        out = build_effective_query(_req(canonical_query="for that what matters is cotton", session_id="s", turn=2), m, cfg)
        self.assertIn("running", out)
        self.assertIn("shoes", out)
        self.assertIn("cotton", out)

    def test_profile_blend_applies_when_flagged(self) -> None:
        m = SessionMemory()
        out = build_effective_query(
            _req(canonical_query="jacket", session_id="s", turn=1, profile={"preference_tags": ["warmth"]}),
            m, _cfg(profile_blend=True),
        )
        self.assertEqual(out, "jacket warmth")


if __name__ == "__main__":
    unittest.main()
