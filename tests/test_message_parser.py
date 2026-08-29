from __future__ import annotations

import unittest

from src.message_parser import MessageParser, load_catalog_vocab


class MessageParserAttributeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = MessageParser()

    def test_extracts_material_color_size_budget(self) -> None:
        parsed = self.parser.parse("I'm looking for black leather boots, size 9, under $80.")
        self.assertEqual(parsed.attributes["material"], "leather")
        self.assertEqual(parsed.attributes["color"], "black")
        self.assertEqual(parsed.attributes["size"], "9")
        self.assertEqual(parsed.attributes["budget"], "80")

    def test_extracts_letter_size(self) -> None:
        parsed = self.parser.parse("Do you have this in a large?")
        self.assertEqual(parsed.attributes["size"], "LARGE")

    def test_extracts_use_case_and_style(self) -> None:
        parsed = self.parser.parse("Need something casual for running in the winter.")
        self.assertEqual(parsed.attributes["style"], "casual")
        self.assertEqual(parsed.attributes["use_case"], "running")

    def test_no_preference_signal_suppresses_attribute_extraction(self) -> None:
        # Real evaluator boundary phrasing (evaluator/local_evaluator.py customer_reply):
        # "I don't have a preference for {attribute}; please use your judgment."
        parsed = self.parser.parse("I don't have a preference for material; please use your judgment.")
        self.assertTrue(parsed.is_no_preference)
        self.assertNotIn("material", parsed.attributes)

    def test_no_preference_alternate_phrasing(self) -> None:
        for phrase in [
            "Honestly it doesn't matter to me.",
            "No strong preference either way.",
            "Whatever works, you decide.",
        ]:
            with self.subTest(phrase=phrase):
                self.assertTrue(self.parser.parse(phrase).is_no_preference)

    def test_override_signal(self) -> None:
        # Real evaluator override phrasing (evaluator/local_evaluator.py behavior_for):
        # "Actually, ignore my earlier preference. What I need is: {new_value}."
        parsed = self.parser.parse("Actually, ignore my earlier preference. What I need is: leather.")
        self.assertTrue(parsed.is_override)
        self.assertEqual(parsed.attributes.get("material"), "leather")

    def test_override_alternate_phrasing(self) -> None:
        for phrase in [
            "Scratch that, I changed my mind.",
            "On second thought, let's go with something else.",
            "Forget what I said earlier.",
        ]:
            with self.subTest(phrase=phrase):
                self.assertTrue(self.parser.parse(phrase).is_override)

    def test_vague_message_flagged(self) -> None:
        # Real evaluator browsing opener: "I'm looking for {category}, but I'm still exploring."
        parsed = self.parser.parse("I'm looking for shoes, but I'm still exploring.")
        self.assertTrue(parsed.is_vague)

    def test_specific_message_not_flagged_vague(self) -> None:
        parsed = self.parser.parse("I'm looking for black leather boots, size 9, under $80.")
        self.assertFalse(parsed.is_vague)

    def test_generic_reprompt_flagged_vague_not_feature(self) -> None:
        # Real evaluator reprompt (customer_reply, sent when ask_attribute was null):
        # "Those options are not quite right yet. Ask me about one specific attribute."
        parsed = self.parser.parse(
            "Those options are not quite right yet. Ask me about one specific attribute."
        )
        self.assertTrue(parsed.is_vague)
        self.assertNotIn("feature", parsed.attributes)

    def test_feature_fallback_for_unclassified_but_meaningful_text(self) -> None:
        parsed = self.parser.parse("Looking for something with reinforced stitching and a padded collar.")
        self.assertIn("feature", parsed.attributes)
        self.assertTrue(parsed.attributes["feature"])

    def test_keywords_deduped_and_stopwords_removed(self) -> None:
        parsed = self.parser.parse("I want a black black jacket for the office")
        self.assertEqual(parsed.keywords.count("black"), 1)
        self.assertNotIn("for", parsed.keywords)
        self.assertNotIn("the", parsed.keywords)

    def test_empty_message_does_not_crash(self) -> None:
        parsed = self.parser.parse("")
        self.assertEqual(parsed.attributes, {})
        self.assertEqual(parsed.keywords, [])

    def test_to_dict_shape(self) -> None:
        parsed = self.parser.parse("black leather boots size 9")
        payload = parsed.to_dict()
        self.assertEqual(set(payload.keys()), {"raw_text", "keywords", "attributes", "signals"})
        self.assertEqual(
            set(payload["signals"].keys()), {"is_override", "is_no_preference", "is_vague"}
        )


class CatalogVocabTest(unittest.TestCase):
    def test_vocab_improves_category_and_brand_matching(self) -> None:
        categories, brands = load_catalog_vocab("data/catalog.jsonl")
        self.assertIn("boots", categories)
        self.assertIn("earrings", categories)
        self.assertGreater(len(brands), 1000)

        parser = MessageParser(known_categories=categories, known_brands=brands)
        parsed = parser.parse("I need a pair of boots.")
        self.assertEqual(parsed.attributes.get("category"), "boots")


if __name__ == "__main__":
    unittest.main()
