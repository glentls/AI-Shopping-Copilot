from __future__ import annotations

import unittest

from retrieval.fusion import reciprocal_rank_fusion
from retrieval.lexical import LexicalRetriever
from retrieval.query import DialogState
from retrieval.structured import StructuredRetriever, extract_constraints


class LexicalRetrieverTest(unittest.TestCase):
    def test_ranks_matching_document_first(self) -> None:
        ids = ["A", "B", "C"]
        texts = [
            "blue running shoe for men lightweight mesh",
            "black leather winter boot waterproof",
            "red cotton t-shirt crew neck",
        ]
        retriever = LexicalRetriever(ids, texts)
        results = retriever.search("blue running shoe", k=3)
        self.assertEqual(results[0][0], "A")

    def test_empty_query_returns_empty(self) -> None:
        retriever = LexicalRetriever(["A"], ["blue shoe"])
        self.assertEqual(retriever.search("   ", k=3), [])


class StructuredRetrieverTest(unittest.TestCase):
    PRODUCTS = {
        "A": {"title": "Black leather belt", "features": ["leather"], "price": 40.0, "store": "Acme"},
        "B": {"title": "Blue cotton shirt", "features": ["cotton"], "price": 200.0, "store": "Other"},
        "C": {"title": "Generic item", "features": [], "price": None, "store": None},
    }

    def test_no_constraints_returns_empty(self) -> None:
        retriever = StructuredRetriever(self.PRODUCTS)
        self.assertEqual(retriever.search("just looking around", k=10), [])

    def test_material_and_budget_boost_matching_item(self) -> None:
        retriever = StructuredRetriever(self.PRODUCTS)
        results = retriever.search("I want something leather, budget around $50", k=10)
        asins = [asin for asin, _ in results]
        self.assertIn("A", asins)
        self.assertNotIn("C", asins)  # no matched terms in C at all -> score 0, excluded

    def test_missing_price_is_never_penalized(self) -> None:
        # C has no price and no material match -> shouldn't appear; but if it DID match a
        # material, a missing price must not incur the over-budget penalty.
        products = {**self.PRODUCTS, "C": {"title": "Leather item", "features": [], "price": None, "store": None}}
        retriever = StructuredRetriever(products)
        results = dict(retriever.search("leather, budget around $10", k=10))
        # A is leather and way over the $10 budget -> penalized but not excluded (soft).
        self.assertIn("A", results)
        # C is leather with unknown price -> gets the material boost, no budget penalty.
        self.assertGreater(results["C"], results["A"])

    def test_extract_constraints_parses_budget(self) -> None:
        constraints = extract_constraints("looking for something under $89.99 in navy")
        self.assertEqual(constraints["budget"], 89.99)
        self.assertIn("navy", constraints["colors"])


class FusionTest(unittest.TestCase):
    def test_weighted_rrf_favors_higher_weighted_route(self) -> None:
        lexical = ["X", "Y", "Z"]
        dense = ["Y", "X", "Z"]
        fused = reciprocal_rank_fusion([lexical, dense], weights=[2.0, 1.0], rrf_k=60)
        self.assertEqual(fused[0], "X")  # lexical rank-1 dominates with higher weight

    def test_zero_weight_route_is_ignored(self) -> None:
        fused = reciprocal_rank_fusion([["A", "B"], ["B", "A"]], weights=[1.0, 0.0], rrf_k=60)
        self.assertEqual(fused, ["A", "B"])

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["A"]], weights=[1.0, 2.0])


class DialogStateTest(unittest.TestCase):
    def test_latest_turn_is_repeated_and_weighted(self) -> None:
        state = DialogState()
        state.add_turn("looking for boots")
        state.add_turn("actually leather ones")
        query = state.build_query()
        self.assertEqual(query.count("actually leather ones"), 2)
        self.assertIn("looking for boots", query)

    def test_empty_state_returns_empty_query(self) -> None:
        self.assertEqual(DialogState().build_query(), "")

    def test_profile_preference_tags_included(self) -> None:
        state = DialogState()
        state.add_turn("hi")
        query = state.build_query({"preference_tags": ["eco-friendly", "wide-fit"]})
        self.assertIn("eco-friendly", query)


if __name__ == "__main__":
    unittest.main()
