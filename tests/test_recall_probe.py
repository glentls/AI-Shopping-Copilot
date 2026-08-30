"""Unit tests for eval/recall_probe.py's pure helpers -- fabricated data, no catalog, no index.
Milliseconds. Proves the vendored turn-1 query reconstruction and the recall summariser behave,
independent of the 50k catalog or the evaluator."""

from __future__ import annotations

import unittest

from eval.recall_probe import RECALL_KS, _coarse_category, _summarize, _turn1_message


class TurnOneMessageTest(unittest.TestCase):
    PRODUCT = {
        "title": "Acme Leather Tote Bag",
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Handbags & Wallets", "Totes"],
        "features": ["100% Leather", "Imported", "Adjustable strap"],
        "details": {"Department": "Womens"},
        "price": 42.0,
    }

    def test_buying_message_uses_category_and_first_hard_constraint(self) -> None:
        msg = _turn1_message({"scenario_type": "buying"}, self.PRODUCT)
        self.assertTrue(msg.startswith("I'm looking for Handbags & Wallets Totes."))
        self.assertIn("A key requirement is:", msg)
        self.assertIn("leather", msg.lower())

    def test_browsing_message_is_exploring_and_carries_no_constraint(self) -> None:
        msg = _turn1_message({"scenario_type": "browsing"}, self.PRODUCT)
        self.assertEqual(msg, "I'm looking for Handbags & Wallets Totes, but I'm still exploring.")

    def test_override_message_appends_a_soft_preference(self) -> None:
        msg = _turn1_message({"scenario_type": "intent_override"}, self.PRODUCT)
        self.assertTrue(msg.startswith("I'm looking for Handbags & Wallets Totes. "))
        self.assertNotIn("still exploring", msg)

    def test_coarse_category_drops_the_root_and_keeps_last_two(self) -> None:
        self.assertEqual(_coarse_category(self.PRODUCT["categories"]), "Handbags & Wallets Totes")
        self.assertEqual(_coarse_category([]), "clothing item")


class SummariseTest(unittest.TestCase):
    def test_recall_and_median_rank(self) -> None:
        # ranks: two hits inside 10, one at 60, one at 400, two misses
        summary = _summarize([3, 8, 60, 400, None, None])
        self.assertEqual(summary["n"], 6)
        self.assertEqual(summary["found"], 4)
        self.assertEqual(summary["recall@10"], round(2 / 6, 4))
        self.assertEqual(summary["recall@100"], round(3 / 6, 4))
        self.assertEqual(summary["recall@500"], round(4 / 6, 4))
        self.assertEqual(summary["median_rank_when_found"], 34)  # median(3,8,60,400)

    def test_all_misses_is_zero_not_error(self) -> None:
        summary = _summarize([None, None])
        self.assertEqual(summary["found"], 0)
        for k in RECALL_KS:
            self.assertEqual(summary[f"recall@{k}"], 0.0)
        self.assertIsNone(summary["median_rank_when_found"])


if __name__ == "__main__":
    unittest.main()
