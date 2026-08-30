"""Unit tests for src/retrieval/embed_index.py's pure parts -- fabricated data, no model,
no catalog, no torch. The Encoder batching/normalisation contract is checked with a fake
encoder that returns deterministic un-normalised vectors."""

from __future__ import annotations

import unittest

import numpy as np

from src.retrieval.embed_index import (
    Encoder,
    build_doc_text,
    cache_key,
)

TEMPLATE = {
    "version": "test1",
    "field_sep": " | ",
    "category_slice": [1, 4],
    "max_features": 3,
    "include_details": True,
    "details_keys": ["Material", "Department"],
    "description_chars": 40,
}

PRODUCT = {
    "parent_asin": "B1",
    "title": "Acme  Leather   Tote",
    "categories": ["Clothing, Shoes & Jewelry", "Women", "Handbags & Wallets", "Totes", "Shoulder Bags"],
    "store": "Acme",
    "features": ["100% Leather", "Imported", "Adjustable strap", "Gold hardware"],
    "details": {"Material": "Leather", "Department": "Womens", "Date First Available": "2020"},
    "description": ["A very long description that should be truncated well before this point ends."],
}


class DocTextTest(unittest.TestCase):
    def test_sections_in_order_and_whitespace_collapsed(self) -> None:
        text = build_doc_text(PRODUCT, TEMPLATE)
        sections = text.split(" | ")
        self.assertEqual(sections[0], "Acme Leather Tote")
        self.assertEqual(sections[1], "Women Handbags & Wallets Totes")  # categories[1:4]
        self.assertEqual(sections[2], "Acme")
        self.assertEqual(sections[3], "100% Leather Imported Adjustable strap")  # max_features=3
        self.assertIn("Material: Leather", text)
        self.assertIn("Department: Womens", text)
        self.assertNotIn("Date First Available", text)  # not in details_keys

    def test_description_truncated(self) -> None:
        desc_section = build_doc_text(PRODUCT, TEMPLATE).split(" | ")[-1]
        self.assertLessEqual(len(desc_section), 40)

    def test_include_details_false_drops_the_section(self) -> None:
        text = build_doc_text(PRODUCT, {**TEMPLATE, "include_details": False})
        self.assertNotIn("Material: Leather", text)

    def test_missing_fields_are_skipped_not_errored(self) -> None:
        sparse = {"parent_asin": "B2", "title": "Bare", "categories": [], "features": [],
                  "description": [], "details": {}}
        self.assertEqual(build_doc_text(sparse, TEMPLATE), "Bare")


class CacheKeyTest(unittest.TestCase):
    def test_key_changes_with_each_input(self) -> None:
        base = cache_key("m", "v1", "sha")
        self.assertEqual(base, cache_key("m", "v1", "sha"))
        self.assertNotEqual(base, cache_key("m2", "v1", "sha"))
        self.assertNotEqual(base, cache_key("m", "v2", "sha"))
        self.assertNotEqual(base, cache_key("m", "v1", "sha2"))
        self.assertEqual(len(base), 16)


class _FakeEncoder(Encoder):
    name = "fake"
    dim = 2

    def _load(self) -> None:
        self._loaded = True

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        # un-normalised, length-dependent so we can detect mis-ordering
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)


class EncoderContractTest(unittest.TestCase):
    def test_output_is_normalised_and_row_aligned(self) -> None:
        enc = _FakeEncoder(batch_size=2)
        texts = ["a", "abcd", "ab", "abcdef", "abc"]
        out = enc.encode(texts)
        self.assertEqual(out.shape, (5, 2))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(5), atol=1e-5)
        # row i must still correspond to texts[i] despite internal length-sorting
        expected_first_component = np.array([len(t) for t in texts], dtype=np.float32)
        expected_first_component /= np.sqrt(expected_first_component**2 + 1.0)
        np.testing.assert_allclose(out[:, 0], expected_first_component, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
