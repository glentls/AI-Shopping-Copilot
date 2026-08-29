from __future__ import annotations

import unittest

from starter.agent import Agent


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


if __name__ == "__main__":
    unittest.main()
