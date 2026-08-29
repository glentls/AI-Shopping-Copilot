from __future__ import annotations

import unittest

from starter.agent import ATTRIBUTES, LAST_ACTIONABLE_TURN, Agent


class AgentRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = Agent.__new__(Agent)
        self.state = {
            "intent": "browsing",
            "intent_locked": False,
            "exploratory_session": False,
        }

    def test_key_requirement_routes_to_buying(self) -> None:
        self.agent._route_intent(self.state, "A key requirement is: waterproof.")
        self.assertEqual(self.state["intent"], "buying")

    def test_exploratory_route_stays_broad_after_constraint(self) -> None:
        self.agent._route_intent(self.state, "I'm still exploring.")
        self.agent._route_intent(self.state, "For that, what matters is: blue.")
        self.assertEqual(self.state["intent"], "browsing")

    def test_override_can_leave_exploratory_route(self) -> None:
        self.agent._route_intent(self.state, "I'm still exploring.")
        self.agent._route_intent(self.state, "Actually, what I need is: leather.")
        self.assertEqual(self.state["intent"], "buying")


class QuestionPolicyTest(unittest.TestCase):
    """A null ask_attribute burns a turn, so every actionable turn must ask something."""

    def setUp(self) -> None:
        self.agent = Agent.__new__(Agent)
        self.state = {
            "profile": {},
            "asked": [],
            "slots": dict.fromkeys(ATTRIBUTES),
            "slot_status": {},
            "unconstrained": set(),
            "override_pending": False,
            "retrieval_feedback": {},
            "requirements_exhausted": False,
        }

    def test_asks_on_every_actionable_turn(self) -> None:
        for turn in range(1, LAST_ACTIONABLE_TURN):
            question = self.agent._choose_question(self.state, turn)
            self.assertIsNotNone(question, f"turn {turn} wasted a clarification")
            self.assertTrue(question[0])

    def test_open_question_used_before_requirements_are_exhausted(self) -> None:
        attribute, _ = self.agent._choose_question(self.state, 1)
        self.assertEqual(attribute, "other")

    def test_still_asks_after_requirements_are_exhausted(self) -> None:
        self.state["requirements_exhausted"] = True
        self.state["retrieval_feedback"] = {"missing_attributes": ["material"]}
        attribute, _ = self.agent._choose_question(self.state, 2)
        self.assertEqual(attribute, "material")

    def test_no_question_on_final_turn(self) -> None:
        self.assertIsNone(self.agent._choose_question(self.state, LAST_ACTIONABLE_TURN))


if __name__ == "__main__":
    unittest.main()
