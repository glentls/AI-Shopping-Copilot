from __future__ import annotations

import math
import unittest

from src.confidence.confidence import compute_confidence
from src.confidence.fallback import safe_decide
from src.confidence.session_ledger import SessionLedger
from src.confidence.policy import DEFAULT_THETA, TURN_CUTOFF, decide
from src.reranker.types import RankResult


class FakeRanker:
    """Canned reranker returning a fixed RankResult (no retrieval dependency)."""

    def __init__(self, result: RankResult) -> None:
        self.result = result
        self.calls = 0

    def __call__(self) -> RankResult:
        self.calls += 1
        return self.result


def _rank(pool_size=100, max_coverage=1, crowd=10, ranked=None) -> RankResult:
    ranked = ranked if ranked is not None else [f"B{i:09d}" for i in range(10)]
    return RankResult(
        ranked=ranked,
        pool_size=pool_size,
        max_coverage=max_coverage,
        top_tier_crowd=crowd,
    )


class ConfidenceFunctionTest(unittest.TestCase):
    def test_score_in_unit_interval(self) -> None:
        score, _ = compute_confidence(_rank(max_coverage=4, crowd=1), 4)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_full_coverage_low_crowd_is_high(self) -> None:
        score, _ = compute_confidence(_rank(max_coverage=4, crowd=1), 4)
        self.assertGreater(score, DEFAULT_THETA)

    def test_high_crowd_lowers_confidence(self) -> None:
        low_crowd, _ = compute_confidence(_rank(max_coverage=2, crowd=1), 4)
        high_crowd, _ = compute_confidence(_rank(max_coverage=2, crowd=200), 4)
        self.assertGreater(low_crowd, high_crowd)

    def test_empty_pool_is_zero(self) -> None:
        score, reason = compute_confidence(_rank(pool_size=0), 3)
        self.assertEqual(score, 0.0)
        self.assertIn("no candidates", reason)


class PolicyMappingTest(unittest.TestCase):
    def test_zero_info_forces_clarify(self) -> None:
        ledger = SessionLedger("s", turn=1)
        payload = decide(_rank(max_coverage=4, crowd=1), ledger)
        self.assertTrue(payload.clarify)
        self.assertEqual(payload.ask_attribute, "other")

    def test_low_confidence_clarifies(self) -> None:
        ledger = SessionLedger("s", turn=2, constraints_known=["cotton", "black"])
        payload = decide(_rank(max_coverage=1, crowd=200), ledger, theta=0.9)
        self.assertTrue(payload.clarify)
        self.assertEqual(payload.ask_attribute, "other")

    def test_high_confidence_recommends_only(self) -> None:
        ledger = SessionLedger("s", turn=2, constraints_known=["cotton", "black", "size 10", "budget"])
        payload = decide(_rank(max_coverage=4, crowd=1), ledger, theta=0.3)
        self.assertFalse(payload.clarify)
        self.assertIsNone(payload.ask_attribute)


class EdgeCaseTest(unittest.TestCase):
    def test_override_resets_exhausted_and_resumes_clarify(self) -> None:
        ledger = SessionLedger("s", constraints_known=["cotton"])
        ledger.observe("I don't have an additional preference for color.", turn=2)
        self.assertTrue(ledger.exhausted)
        # Override arrives.
        ledger.observe("Actually, ignore my earlier preference. What I need is: leather.", turn=3)
        self.assertFalse(ledger.exhausted)
        self.assertTrue(ledger.override_seen)
        payload = decide(_rank(max_coverage=1, crowd=200), ledger, theta=0.9)
        self.assertTrue(payload.clarify)

    def test_boundary_brushoff_does_not_exhaust_and_clarify_continues(self) -> None:
        # Boundary brush-off ("no preference for THIS attribute; use your
        # judgment") must NOT latch exhaustion -- the customer may still have
        # other constraints, so clarification continues next turn.
        ledger = SessionLedger("s", turn=2, constraints_known=["cotton"])
        ledger.observe("I don't have a preference for color; please use your judgment.", turn=2)
        self.assertFalse(ledger.exhausted, "boundary brush-off must not exhaust")
        # Low-confidence rank -> policy should still clarify (not gated off).
        payload = decide(_rank(max_coverage=1, crowd=200), ledger, theta=0.9)
        self.assertTrue(payload.clarify)
        self.assertEqual(payload.ask_attribute, "other")

    def test_boundary_brushoff_does_not_count_as_no_progress(self) -> None:
        # A boundary brush-off is neutral: it should not advance no_progress_turns
        # (otherwise repeated brush-offs could trip the late-turn exhaustion rule).
        ledger = SessionLedger("s", turn=2, constraints_known=["cotton"])
        before = ledger.no_progress_turns
        ledger.observe("I don't have a preference for color; please use your judgment.", turn=2)
        self.assertEqual(ledger.no_progress_turns, before)

    def test_late_turn_no_progress_uses_shared_cutoff(self) -> None:
        # The late-turn exhaustion rule in the ledger must key off the same
        # TURN_CUTOFF the policy gate uses (single source of truth).
        ledger = SessionLedger("s", turn=2, constraints_known=["cotton"])
        ledger.observe("hmm", turn=TURN_CUTOFF)      # 1st no-progress at cutoff
        ledger.observe("still hmm", turn=TURN_CUTOFF) # 2nd -> exhausts
        self.assertTrue(ledger.exhausted)

    def test_exhausted_message_stops_clarify_forever(self) -> None:
        ledger = SessionLedger("s", constraints_known=["cotton", "black"])
        ledger.observe("I don't have an additional preference for material.", turn=4)
        self.assertTrue(ledger.exhausted)
        payload = decide(_rank(max_coverage=1, crowd=200), ledger, theta=0.9)
        self.assertFalse(payload.clarify)
        # Later turn, still exhausted.
        ledger.observe("still nothing", turn=5)
        self.assertTrue(ledger.exhausted)
        payload2 = decide(_rank(max_coverage=1, crowd=200), ledger, theta=0.9)
        self.assertFalse(payload2.clarify)

    def test_empty_pool_fallback_fires(self) -> None:
        ledger = SessionLedger("s", turn=1, constraints_known=["cotton"])
        fallback = [f"B{i:09d}" for i in range(10)]
        payload, recs = safe_decide(FakeRanker(_rank(pool_size=0, ranked=[])), ledger, fallback, DEFAULT_THETA)
        self.assertEqual(payload.score, 0.0)
        self.assertTrue(payload.clarify)
        self.assertEqual(recs, fallback)

    def test_exception_in_rank_no_raise(self) -> None:
        def boom() -> RankResult:
            raise ValueError("reranker exploded")

        ledger = SessionLedger("s", turn=1, constraints_known=["cotton"])
        fallback = [f"B{i:09d}" for i in range(10)]
        payload, recs = safe_decide(boom, ledger, fallback, DEFAULT_THETA)
        self.assertTrue(payload.clarify)
        self.assertEqual(recs, fallback)

    def test_determinism_identical_inputs(self) -> None:
        rank = _rank(max_coverage=2, crowd=37)
        a = decide(rank, SessionLedger("s", turn=3, constraints_known=["cotton", "black"]))
        b = decide(rank, SessionLedger("s", turn=3, constraints_known=["cotton", "black"]))
        self.assertEqual((a.score, a.clarify, a.ask_attribute), (b.score, b.clarify, b.ask_attribute))
        self.assertFalse(math.isnan(a.score))


if __name__ == "__main__":
    unittest.main()
