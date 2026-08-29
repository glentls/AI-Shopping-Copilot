"""Skeleton wiring tests. Each lane adds tests/test_lane_{a,b,c}.py alongside."""

from __future__ import annotations

import unittest

from src.contracts import ASK_ATTRIBUTES, ConversationState, SlotValue
from src.extract import detect_no_preference, detect_override, extract_slots
from src.policy.state import update


class ContractTest(unittest.TestCase):
    def test_active_ignores_retracted_values(self) -> None:
        state = ConversationState("s", {})
        state.add("material", SlotValue("leather", 0.9, 1))
        state.add("material", SlotValue("cotton", 0.9, 2, polarity=False))
        self.assertEqual(state.active("material"), ["leather"])
        self.assertEqual(state.excluded("material"), ["cotton"])


class ExtractTest(unittest.TestCase):
    def test_word_boundaries(self) -> None:
        state = ConversationState("s", {})
        found = extract_slots("an embroidered zippered jacket", 1, state)
        self.assertNotIn("color", found)

    def test_organizer_example(self) -> None:
        state = ConversationState("s", {})
        found = extract_slots("Water-resistant, comfortable and under $80", 2, state)
        self.assertIn("waterproof", [v.value for v in found["feature"]])
        self.assertIn("budget", found)

    def test_negation_flips_polarity(self) -> None:
        state = ConversationState("s", {})
        found = extract_slots("not leather please", 1, state)
        self.assertFalse(found["material"][0].polarity)

    def test_override_and_no_preference_cues(self) -> None:
        self.assertTrue(detect_override("Actually, ignore that - I need leather instead"))
        self.assertTrue(detect_no_preference("I don't have a preference for material"))
        self.assertFalse(detect_override("I need leather"))


class StateTest(unittest.TestCase):
    def test_override_retracts_the_old_value(self) -> None:
        state = ConversationState("s", {})
        update(state, "I want cotton", 1)
        self.assertEqual(state.active("material"), ["cotton"])
        update(state, "Actually, ignore that - I need leather instead", 3)
        self.assertEqual(state.active("material"), ["leather"])

    def test_boundary_marks_slot_unanswerable(self) -> None:
        state = ConversationState("s", {})
        state.last_asked = "color"
        update(state, "I don't have a preference; please use your judgment.", 2)
        self.assertIn("color", state.unanswerable)


class AgentContractTest(unittest.TestCase):
    def test_ask_attributes_match_the_evaluator(self) -> None:
        from evaluator.local_evaluator import ALLOWED_ATTRIBUTES

        self.assertEqual(ASK_ATTRIBUTES, frozenset(ALLOWED_ATTRIBUTES))


if __name__ == "__main__":
    unittest.main()
