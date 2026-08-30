from __future__ import annotations

import unittest

from src.contracts import SessionState
from src.memory.distiller import distill


def _state(**kwargs) -> SessionState:
    values = {
        "session_id": "s1", "turn": 2, "intent": "buy", "slots": {},
        "slot_turn_added": {}, "history": [], "profile": {},
    }
    values.update(kwargs)
    return SessionState(**values)


class MemoryDistillerTest(unittest.TestCase):
    def test_distills_slots_and_profile_into_positive_boosts(self) -> None:
        result = distill(
            _state(slots={"color": "black", "material": ["leather"]}),
            {"preference_tags": ["water resistant"], "purchase_frequency": "3-4 prior purchases"},
        )
        self.assertEqual(result.boosts["color"]["black"], 1.0)
        self.assertEqual(result.boosts["material"]["leather"], 1.0)
        self.assertIn("water resistant", result.boosts["feature"])
        self.assertNotIn("purchase_frequency", result.summary)

    def test_distills_rejections_and_rejected_products(self) -> None:
        result = distill(
            _state(history=[{"user_message": "I do not want pink items."}], negatives=["B123"]),
            {},
        )
        self.assertLess(result.boosts["negative_terms"]["pink items"], 0)
        self.assertEqual(result.boosts["rejected_asins"], ["B123"])

    def test_malformed_inputs_degrade_to_valid_profile(self) -> None:
        result = distill(_state(slots=None, history=[None]), None)
        self.assertIsInstance(result.boosts, dict)
        self.assertIsInstance(result.summary, str)


if __name__ == "__main__":
    unittest.main()
