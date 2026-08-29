"""Proves the failure contract from docs/plan/architecture.md and CLAUDE.md: no component call
may raise or hang agent.respond(), reset() must never raise at all, and every response carries
exactly top_k valid, unique catalog parent_asin values -- even when every component is broken at
once, or the harness sends a turn number past the contractual maximum.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.contracts import ASK_ATTRIBUTES
from starter.agent import Agent

CATALOG_PATH = "data/catalog.jsonl"


def _boom(*args, **kwargs):
    raise RuntimeError("injected failure")


class FailureContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = Agent(CATALOG_PATH)
        cls.catalog_ids = set(cls.agent.index.products.keys())

    def _assert_valid_response(self, response: dict, top_k: int = 10) -> None:
        self.assertIsInstance(response, dict)
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], (None,) + ASK_ATTRIBUTES)
        recs = response["recommendations"]
        self.assertEqual(len(recs), top_k, f"expected exactly {top_k} recommendations, got {len(recs)}")
        asins = [item["parent_asin"] for item in recs]
        self.assertEqual(len(asins), len(set(asins)), "recommendations must be unique")
        for asin in asins:
            self.assertIn(asin, self.catalog_ids, f"{asin} is not a valid catalog parent_asin")

    def test_normal_turn_returns_exactly_k_valid_ids(self) -> None:
        self.agent.reset("normal-session", {"summary": "x"})
        response = self.agent.respond("normal-session", "I want cotton running shoes", 1, 10)
        self._assert_valid_response(response)

    def test_survives_dialog_exception(self) -> None:
        self.agent.reset("dialog-fail", {"summary": "x"})
        with patch("src.agent.dialog_primary", side_effect=_boom):
            response = self.agent.respond("dialog-fail", "hello", 1, 10)
        self._assert_valid_response(response)

    def test_survives_memory_exception(self) -> None:
        self.agent.reset("memory-fail", {"summary": "x"})
        with patch("src.agent.memory_primary", side_effect=_boom):
            response = self.agent.respond("memory-fail", "hello", 1, 10)
        self._assert_valid_response(response)

    def test_survives_retrieval_exception(self) -> None:
        self.agent.reset("retrieval-fail", {"summary": "x"})
        with patch("src.agent.retrieval_primary", side_effect=_boom):
            response = self.agent.respond("retrieval-fail", "hello", 1, 10)
        self._assert_valid_response(response)

    def test_survives_ranking_exception(self) -> None:
        self.agent.reset("ranking-fail", {"summary": "x"})
        with patch("src.agent.ranking_primary", side_effect=_boom):
            response = self.agent.respond("ranking-fail", "hello", 1, 10)
        self._assert_valid_response(response)

    def test_survives_every_component_failing_at_once(self) -> None:
        self.agent.reset("all-fail", {"summary": "x"})
        with patch("src.agent.dialog_primary", side_effect=_boom), \
             patch("src.agent.memory_primary", side_effect=_boom), \
             patch("src.agent.retrieval_primary", side_effect=_boom), \
             patch("src.agent.ranking_primary", side_effect=_boom):
            response = self.agent.respond("all-fail", "hello", 1, 10)
        self._assert_valid_response(response)

    def test_survives_respond_raising_entirely(self) -> None:
        """Even if agent.py's own glue code breaks (not just a component), respond() must still
        return a valid contract-shaped response instead of propagating."""
        self.agent.reset("glue-fail", {"summary": "x"})
        with patch.object(Agent, "_respond_unsafe", side_effect=_boom):
            response = self.agent.respond("glue-fail", "hello", 1, 10)
        self._assert_valid_response(response)

    def test_survives_forced_over_length_session(self) -> None:
        """turn=15 is outside the contract's declared range (max 10,
        docs/agent_api_contract.json:30) but the agent must not crash if the harness ever sends
        it anyway."""
        self.agent.reset("over-length", {"summary": "x"})
        response = self.agent.respond("over-length", "hello", 15, 10)
        self._assert_valid_response(response)

    def test_reset_never_raises_even_with_hostile_inputs(self) -> None:
        try:
            self.agent.reset("hostile-1", None)  # type: ignore[arg-type]
            self.agent.reset(12345, {"summary": "x"})  # type: ignore[arg-type]
            self.agent.reset("hostile-3", "not-a-dict")  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - this must never happen
            self.fail(f"reset() raised {exc!r}; the harness does not catch reset() exceptions")

    def test_reset_never_raises_even_if_session_state_construction_is_broken(self) -> None:
        with patch("src.agent.SessionState", side_effect=_boom):
            try:
                self.agent.reset("broken-sessionstate", {"summary": "x"})
            except Exception as exc:  # pragma: no cover - this must never happen
                self.fail(f"reset() raised {exc!r} even with SessionState broken")

    def test_respond_without_prior_reset_does_not_raise(self) -> None:
        response = self.agent.respond("never-reset", "hello", 1, 10)
        self._assert_valid_response(response)


if __name__ == "__main__":
    unittest.main()
