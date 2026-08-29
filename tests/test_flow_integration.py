"""Integration tests: message_parser -> intent_router -> ledger (via Agent.respond)."""
from __future__ import annotations

import unittest

from src.agent import Agent
from starter.ledger import LedgerService


class FlowIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = Agent()
        self.session = "test-session"
        self.agent.reset(self.session, {})

    def tearDown(self) -> None:
        ledger: LedgerService = self.agent._ledger
        if ledger.exists(self.session):
            ledger.delete(self.session)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _respond(self, message: str, turn: int = 1) -> dict:
        return self.agent.respond(self.session, message, turn=turn, top_k=5)

    def _state(self) -> dict:
        return self.agent._ledger.read(self.session)

    # ── intent routing ────────────────────────────────────────────────────────

    def test_structured_message_sets_buying_intent(self) -> None:
        self._respond("I want black leather boots, size 9")
        self.assertEqual(self._state()["intent"], "buying")

    def test_vague_message_sets_browsing_intent(self) -> None:
        self._respond("just looking around")
        self.assertEqual(self._state()["intent"], "browsing")

    def test_override_phrase_sets_intent_override_and_clears_constraints(self) -> None:
        self._respond("I want black leather boots", turn=1)
        self._respond("actually, start over", turn=2)
        state = self._state()
        self.assertEqual(state["intent"], "buying")
        self.assertEqual(state["constraints"], {})

    def test_no_preference_phrase_sets_boundary_intent(self) -> None:
        # Evaluator boundary phrasing used in production
        self._respond("I don't have a preference; please use your judgment.")
        self.assertEqual(self._state()["intent"], "boundary")

    # ── attribute extraction -> ledger constraints ────────────────────────────

    def test_color_extracted_and_stored_in_constraints(self) -> None:
        self._respond("I want red sneakers")
        self.assertIn("red", self._state()["constraints"].get("color", []))

    def test_material_extracted_and_stored(self) -> None:
        self._respond("looking for leather boots")
        self.assertIn("leather", self._state()["constraints"].get("material", []))

    def test_size_extracted_and_stored(self) -> None:
        self._respond("do you have this in size 10?")
        self.assertIn("10", self._state()["constraints"].get("size", []))

    def test_multiple_attributes_in_single_message(self) -> None:
        self._respond("black leather boots size 9")
        constraints = self._state()["constraints"]
        self.assertIn("black", constraints.get("color", []))
        self.assertIn("leather", constraints.get("material", []))
        self.assertIn("9", constraints.get("size", []))

    # ── price constraint ──────────────────────────────────────────────────────

    def test_price_upper_bound_stored(self) -> None:
        self._respond("something under $80")
        price = self._state().get("price_constraint")
        self.assertIsNotNone(price)
        self.assertIn(price["operator"], ("<", "<="))
        self.assertEqual(price["amount"], 80.0)

    def test_price_approximate_stored(self) -> None:
        self._respond("budget around $100")
        price = self._state().get("price_constraint")
        self.assertIsNotNone(price)
        self.assertEqual(price["operator"], "~")
        self.assertEqual(price["amount"], 100.0)

    def test_price_lower_bound_stored(self) -> None:
        self._respond("something over $50")
        price = self._state().get("price_constraint")
        self.assertIsNotNone(price)
        self.assertEqual(price["operator"], ">")
        self.assertEqual(price["amount"], 50.0)

    # ── turn counter and history ──────────────────────────────────────────────

    def test_turn_increments_on_each_respond(self) -> None:
        self._respond("hi", turn=1)
        self._respond("black shoes", turn=2)
        self.assertEqual(self._state()["turn"], 2)

    def test_history_accumulates(self) -> None:
        self._respond("first message", turn=1)
        self._respond("second message", turn=2)
        history = self._state()["history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["content"], "first message")
        self.assertEqual(history[1]["content"], "second message")

    # ── multi-turn constraint accumulation ────────────────────────────────────

    def test_constraints_accumulate_across_turns(self) -> None:
        self._respond("I want black boots", turn=1)
        self._respond("make it leather", turn=2)
        constraints = self._state()["constraints"]
        self.assertIn("black", constraints.get("color", []))
        self.assertIn("leather", constraints.get("material", []))

    def test_override_clears_previous_constraints(self) -> None:
        self._respond("I want black boots size 9", turn=1)
        self._respond("actually forget it, start fresh", turn=2)
        self.assertEqual(self._state()["constraints"], {})

    # ── search key reflects constraints ──────────────────────────────────────

    def test_search_key_reflects_constraints(self) -> None:
        self._respond("black leather boots")
        search_key = self._state()["search_key"]
        self.assertIn("color", search_key)
        self.assertIn("material", search_key)


if __name__ == "__main__":
    unittest.main()
