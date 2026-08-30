from __future__ import annotations

import unittest

from src.agent import Agent

_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.2,
    "rating_style": "usually positive",
    "preference_tags": ["comfort", "fit"],
    "summary": "Prior purchases emphasize comfort and fit.",
}


class FollowUpMissingAttributesTest(unittest.TestCase):
    """A provided attribute must never keep showing up in the follow-up
    question as still-missing."""

    @classmethod
    def setUpClass(cls) -> None:
        # Building the agent (FTS5 index + catalog load) is the same
        # regardless of test case; build once per class, not per test.
        cls.agent = Agent()

    def setUp(self) -> None:
        self.agent = self.__class__.agent

    def test_dollar_sign_less_budget_is_recognised_as_known(self) -> None:
        # Regression: _parse_price_constraint (agent.py's own price regex,
        # used for search filtering) accepts "under 50" with no "$", but
        # extract_attributes()'s own budget regex requires a literal "$" --
        # so the two disagreed on whether budget was "known", and the
        # follow-up question kept asking about a budget the customer had
        # already given. Fixed by also counting price_constraint as known.
        session_id = "budget-regression"
        self.agent.reset(session_id, dict(_PROFILE))
        first = self.agent.respond(session_id, "i want a shirt", 1, 10)
        self.assertIn("budget", first["message"].lower())

        second = self.agent.respond(session_id, "under 50", 2, 10)
        self.assertNotIn("budget", second["message"].lower())

    def test_dollar_sign_budget_still_recognised_as_known(self) -> None:
        # Non-regression: the ordinary "$" phrasing must keep working too.
        session_id = "budget-dollar-sign"
        self.agent.reset(session_id, dict(_PROFILE))
        self.agent.respond(session_id, "i want a shirt", 1, 10)
        second = self.agent.respond(session_id, "under $50", 2, 10)
        self.assertNotIn("budget", second["message"].lower())

    def test_provided_material_and_color_drop_out_of_the_question(self) -> None:
        session_id = "material-color-regression"
        self.agent.reset(session_id, dict(_PROFILE))
        first = self.agent.respond(session_id, "i want a shirt", 1, 10)
        self.assertIn("material", first["message"].lower())
        self.assertIn("color", first["message"].lower())

        second = self.agent.respond(session_id, "black cotton", 2, 10)
        self.assertNotIn("material", second["message"].lower())
        self.assertNotIn("color", second["message"].lower())


if __name__ == "__main__":
    unittest.main()
