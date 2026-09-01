from __future__ import annotations

import unittest
from collections import Counter

from dialog.portfolio import portfolio_rerank
from dialog.posterior import RejectionTracker
from dialog.question_policy import choose_attribute
from dialog.slots import SlotState


class SlotStateTest(unittest.TestCase):
    def test_first_value_is_accumulated_not_flagged_as_override(self) -> None:
        state = SlotState()
        state.update("I want something in leather")
        self.assertEqual(state.values["material"], "leather")
        self.assertEqual(state.overridden_attributes, set())

    def test_conflicting_second_value_overrides_by_contradiction(self) -> None:
        state = SlotState()
        state.update("I want something in polyester")
        state.update("Actually, ignore my earlier preference. What I need is: leather.")
        self.assertEqual(state.values["material"], "leather")
        self.assertIn("material", state.overridden_attributes)

    def test_update_returns_only_attributes_overridden_this_call(self) -> None:
        state = SlotState()
        first_call_result = state.update("I want something in polyester")
        self.assertEqual(first_call_result, set())  # first value: accumulate, not override
        second_call_result = state.update("actually leather, and black")
        self.assertEqual(second_call_result, {"material"})  # color is a first value too
        third_call_result = state.update("still leather")
        self.assertEqual(third_call_result, set())  # repeat value: no override

    def test_repeating_the_same_value_is_not_an_override(self) -> None:
        state = SlotState()
        state.update("leather please")
        state.update("yes still leather")
        self.assertEqual(state.values["material"], "leather")
        self.assertEqual(state.overridden_attributes, set())

    def test_budget_and_color_extracted_independently(self) -> None:
        state = SlotState()
        state.update("looking for something black under $50")
        self.assertEqual(state.values["color"], "black")
        self.assertEqual(state.values["budget"], 50.0)

    def test_unfilled_attributes_excludes_blocklist(self) -> None:
        state = SlotState()
        unfilled = state.unfilled_attributes(("material", "color", "brand", "category", "budget"))
        self.assertNotIn("brand", unfilled)
        self.assertNotIn("category", unfilled)
        self.assertIn("material", unfilled)


class QuestionPolicyTest(unittest.TestCase):
    PRODUCTS = [
        {"title": "Black leather belt", "features": ["leather"], "price": 40.0},
        {"title": "Brown leather belt", "features": ["leather"], "price": 45.0},
        {"title": "Blue cotton shirt", "features": ["cotton"], "price": 20.0},
        {"title": "Red silk scarf", "features": ["silk"], "price": 200.0},
    ]

    def test_picks_most_diverse_unfilled_attribute(self) -> None:
        # material has 3 distinct values (leather/leather/cotton/silk), color has 3
        # distinct too (black/brown/blue/red) -- both diverse; either is a defensible
        # top pick, but it must not fall through to the low-signal fallback list.
        attribute = choose_attribute(self.PRODUCTS, filled_attributes=set())
        self.assertIn(attribute, ("material", "color", "budget"))

    def test_already_filled_attributes_are_skipped(self) -> None:
        attribute = choose_attribute(self.PRODUCTS, filled_attributes={"material", "color", "size", "budget"})
        self.assertIn(attribute, ("style", "use_case", "feature", "other"))

    def test_never_returns_blocklisted_attribute(self) -> None:
        for _ in range(10):
            attribute = choose_attribute(self.PRODUCTS, filled_attributes={"material", "color", "size", "budget"})
            self.assertNotIn(attribute, ("brand", "category"))


class PortfolioTest(unittest.TestCase):
    PRODUCTS = {
        "A": {"title": "Black leather belt", "features": ["leather"], "price": 40.0},
        "B": {"title": "Black leather wallet", "features": ["leather"], "price": 42.0},
        "C": {"title": "Blue cotton shirt", "features": ["cotton"], "price": 20.0},
        "D": {"title": "Red silk scarf", "features": ["silk"], "price": 200.0},
    }

    def test_rank_one_is_always_preserved(self) -> None:
        fused = ["A", "B", "C", "D"]
        result = portfolio_rerank(fused, self.PRODUCTS, top_k=3, turn=1)
        self.assertEqual(result[0], "A")

    def test_diversifies_rather_than_pure_greedy(self) -> None:
        # B is rank-2 by fusion but near-duplicate of A; C/D are more novel.
        fused = ["A", "B", "C", "D"]
        result = portfolio_rerank(fused, self.PRODUCTS, top_k=3, turn=1)
        self.assertIn("C", result)
        self.assertIn("D", result)
        self.assertNotIn("B", result)

    def test_rejected_values_are_downweighted(self) -> None:
        fused = ["A", "B", "C", "D"]
        counts = Counter({"leather": 10})
        result_with_penalty = portfolio_rerank(fused, self.PRODUCTS, top_k=2, turn=1, rejected_value_counts=counts)
        # B is a leather duplicate of A (already-selected rank-1) -- rejection penalty
        # should make it even less likely to be picked over C/D at slot 2.
        self.assertNotEqual(result_with_penalty[1], "B")

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(portfolio_rerank([], self.PRODUCTS, top_k=5, turn=1), [])


class RejectionTrackerTest(unittest.TestCase):
    def test_records_feature_values_from_shown_batch(self) -> None:
        tracker = RejectionTracker()
        products = {"A": {"title": "Black leather belt", "features": ["leather"], "price": 40.0}}
        tracker.record_rejected_batch(["A"], products)
        self.assertGreater(tracker.counts["leather"], 0)
        self.assertGreater(tracker.counts["black"], 0)


if __name__ == "__main__":
    unittest.main()
