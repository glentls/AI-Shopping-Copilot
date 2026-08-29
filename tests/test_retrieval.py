from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.retrieval import (
    CatalogStore,
    HybridSearcher,
    ProductMeta,
    SearchResult,
    apply_metadata_filters,
    build_fts_expression,
    build_retrieval_feedback,
    extract_constraint_phrases,
    missing_attributes,
    rerank_candidates,
    strip_boilerplate,
)
from starter.retrieval.search import _slot_values_from_filters


SAMPLE_CATALOG = [
    {
        "parent_asin": "A",
        "title": "Blue running shoe lightweight mesh",
        "features": ["mesh upper", "cushioned sole"],
        "details": {"department": "womens"},
        "description": ["running shoe"],
        "categories": ["Clothing", "Shoes", "Running"],
        "store": "RunCo",
        "average_rating": 4.2,
        "rating_number": 10,
        "price": 49.0,
    },
    {
        "parent_asin": "B",
        "title": "Black winter boot leather",
        "features": ["leather upper"],
        "details": {"department": "womens"},
        "description": ["winter boot"],
        "categories": ["Clothing", "Boots"],
        "store": "BootCo",
        "average_rating": 4.4,
        "rating_number": 12,
        "price": 89.0,
    },
]


class QueryBuilderTest(unittest.TestCase):
    def test_extract_phrases_from_what_matters_is(self) -> None:
        text = "For that, what matters is: lightweight mesh upper; cushioned sole."
        self.assertEqual(
            extract_constraint_phrases(text),
            ["lightweight mesh upper", "cushioned sole"],
        )

    def test_extract_phrases_from_key_requirement(self) -> None:
        text = "I'm looking for running shoes. A key requirement is: 100% cotton."
        self.assertEqual(extract_constraint_phrases(text), ["100% cotton"])

    def test_strip_boilerplate_removes_evaluator_noise(self) -> None:
        text = "Those options are not quite right yet. Ask me about one specific attribute."
        cleaned = strip_boilerplate(text)
        self.assertNotIn("not quite right yet", cleaned.lower())
        self.assertNotIn("ask me about", cleaned.lower())

    def test_browsing_expression_uses_or(self) -> None:
        expr = build_fts_expression("running shoes mesh", mode="browsing")
        self.assertIn(" OR ", expr)
        self.assertNotIn(" AND ", expr)

    def test_buying_expression_uses_and_for_phrases(self) -> None:
        text = "I'm looking for running shoes.\nFor that, what matters is: cotton; blue."
        expr = build_fts_expression(text, mode="buying")
        self.assertIn(" AND ", expr)
        self.assertIn('"cotton"', expr)
        self.assertIn('"blue"', expr)

    def test_constraint_with_internal_period_survives(self) -> None:
        text = "For that, what matters is: heel height: 3.5 inches; machine wash cold."
        self.assertEqual(
            extract_constraint_phrases(text),
            ["heel height: 3.5 inches", "machine wash cold"],
        )

    def test_phrases_do_not_leak_across_turns(self) -> None:
        text = "A key requirement is: 100% cotton.\nFor that, what matters is: crew neck."
        self.assertEqual(
            extract_constraint_phrases(text),
            ["100% cotton", "crew neck"],
        )

    def test_slot_values_included_as_phrases(self) -> None:
        expr = build_fts_expression("running shoes", mode="buying", slot_values=["lightweight mesh upper"])
        self.assertIn('"lightweight mesh upper"', expr)


class FiltersTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self._directory.name) / "catalog.jsonl"
        self.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in SAMPLE_CATALOG),
            encoding="utf-8",
        )
        self.store = CatalogStore(self.catalog_path)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_price_filter_excludes_expensive_product(self) -> None:
        filtered = apply_metadata_filters(
            ["A", "B"],
            {"max_price": 50.0},
            self.store,
            mode="buying",
        )
        self.assertEqual(filtered, ["A"])

    def test_browsing_filter_keeps_candidates_when_nothing_matches(self) -> None:
        filtered = apply_metadata_filters(
            ["A", "B"],
            {"material": ["impossible-material-xyz"]},
            self.store,
            mode="browsing",
        )
        self.assertEqual(filtered, ["A", "B"])


class RerankerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self._directory.name) / "catalog.jsonl"
        self.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in SAMPLE_CATALOG),
            encoding="utf-8",
        )
        self.store = CatalogStore(self.catalog_path)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_rerank_promotes_exact_phrase_match(self) -> None:
        ranked = rerank_candidates(
            ["B", "A"],
            {},
            "what matters is: lightweight mesh upper.",
            ["lightweight mesh upper"],
            self.store,
        )
        self.assertEqual(ranked[0], "A")


class FeedbackTest(unittest.TestCase):
    def test_missing_attributes_excludes_unconstrained_and_asked(self) -> None:
        filters = {
            "material": ["cotton"],
            "unconstrained": ["color"],
            "asked": ["category"],
            "slot_status": {"material": "confirmed"},
        }
        missing = missing_attributes(filters)
        self.assertNotIn("material", missing)
        self.assertNotIn("color", missing)
        self.assertNotIn("category", missing)
        self.assertIn("use_case", missing)

    def test_build_feedback_marks_overloaded(self) -> None:
        feedback = build_retrieval_feedback(842, {})
        self.assertTrue(feedback["overloaded"])
        self.assertEqual(feedback["candidate_count"], 842)
        self.assertIn("category", feedback["missing_attributes"])

    def test_build_feedback_not_overloaded(self) -> None:
        feedback = build_retrieval_feedback(12, {"category_tokens": ["running"]})
        self.assertFalse(feedback["overloaded"])

    def test_search_feedback_respects_dialog_hints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text(
                "".join(json.dumps(row) + "\n" for row in SAMPLE_CATALOG),
                encoding="utf-8",
            )
            searcher = HybridSearcher(catalog_path)
            result = searcher.search(
                "running",
                mode="browsing",
                filters={"asked": ["category"], "unconstrained": ["brand"]},
                top_k=10,
            )
            self.assertNotIn("category", result.feedback["missing_attributes"])
            self.assertNotIn("brand", result.feedback["missing_attributes"])


class HybridSearcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self._directory.name) / "catalog.jsonl"
        self.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in SAMPLE_CATALOG),
            encoding="utf-8",
        )
        self.searcher = HybridSearcher(self.catalog_path)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_search_returns_ten_valid_asins(self) -> None:
        result = self.searcher.search("running shoes mesh", mode="browsing", filters={}, top_k=10)
        self.assertIsInstance(result, SearchResult)
        self.assertLessEqual(len(result.asins), 10)
        self.assertGreater(len(result.asins), 0)
        self.assertEqual(result.asins[0], "A")

    def test_search_includes_feedback(self) -> None:
        result = self.searcher.search("running", mode="browsing", filters={}, top_k=10)
        self.assertIn("candidate_count", result.feedback)
        self.assertIn("overloaded", result.feedback)
        self.assertIn("missing_attributes", result.feedback)
        self.assertIsInstance(result.feedback["missing_attributes"], list)

    def test_empty_query_still_returns_results(self) -> None:
        result = self.searcher.search("", mode="browsing", filters={}, top_k=10)
        self.assertGreater(len(result.asins), 0)

    def test_filters_slot_values_affect_ranking(self) -> None:
        result = self.searcher.search(
            "shoe",
            mode="buying",
            filters={"material": ["mesh upper"]},
            top_k=10,
        )
        self.assertIn("A", result.asins)

    def test_buying_falls_back_when_strict_query_misses(self) -> None:
        result = self.searcher.search(
            "what matters is: impossible phrase xyz; another missing phrase.",
            mode="buying",
            filters={},
            top_k=10,
        )
        self.assertGreater(len(result.asins), 0)

    def test_full_utterances_never_become_slot_phrases(self) -> None:
        utterance = "I'm looking for shoes. A key requirement is: blue."
        for mode in ("buying", "browsing"):
            self.assertEqual(_slot_values_from_filters({"color": utterance}, mode), [])

    def test_concise_slot_values_are_kept_in_both_modes(self) -> None:
        for mode in ("buying", "browsing"):
            self.assertEqual(_slot_values_from_filters({"color": "blue"}, mode), ["blue"])


class CatalogStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self._directory.name) / "catalog.jsonl"
        self.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in SAMPLE_CATALOG),
            encoding="utf-8",
        )
        self.store = CatalogStore(self.catalog_path)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_get_returns_product_metadata(self) -> None:
        product = self.store.get("A")
        self.assertIsInstance(product, ProductMeta)
        assert product is not None
        self.assertEqual(product.parent_asin, "A")
        self.assertEqual(product.price, 49.0)
        self.assertIn("mesh upper", product.searchable_text)

    def test_bm25_search_returns_ranked_asins(self) -> None:
        expression = '"running" OR "mesh"'
        results = self.store.bm25_search(expression, limit=10)
        self.assertEqual(results[0], "A")
        self.assertIn("A", results)
        self.assertNotIn("bad-id", results)

    def test_count_matches_reports_hits(self) -> None:
        expression = '"running" OR "mesh"'
        self.assertEqual(self.store.count_matches(expression), 1)
        self.assertEqual(self.store.count_matches('"leather"'), 1)

    def test_empty_expression_returns_no_results(self) -> None:
        self.assertEqual(self.store.bm25_search("", limit=10), [])
        self.assertEqual(self.store.count_matches(""), 0)


if __name__ == "__main__":
    unittest.main()
