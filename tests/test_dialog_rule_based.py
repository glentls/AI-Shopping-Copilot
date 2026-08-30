from __future__ import annotations

import unittest

from src.contracts import SessionState
from src.dialog.rule_based import update


class RuleBasedDialogTest(unittest.TestCase):
    def test_extracts_constraint_and_asks_next_attribute(self) -> None:
        state = SessionState("s", 1, "unknown", {}, {})
        result = update(state, "For that, what matters is: black; leather.")
        self.assertEqual(result.slots["color"], "black")
        self.assertEqual(result.slots["material"], "leather")
        self.assertEqual(result.ask_attribute, "material")

    def test_accumulates_slots_and_detects_override(self) -> None:
        state = SessionState("s", 3, "buy", {"color": "black"}, {}, asked_attributes=["category", "material"])
        result = update(state, "Actually, ignore my earlier preference. What I need is: red color.")
        self.assertTrue(result.intent_override)
        self.assertNotIn("black", result.slots.values())
        self.assertEqual(result.slots["color"], "red color")


if __name__ == "__main__":
    unittest.main()
