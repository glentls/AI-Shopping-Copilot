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

    def test_short_reply_with_real_attribute_not_vague(self) -> None:
        # A short single-word reply is not "vague" if it carries a real,
        # actionable structured attribute (this is what a customer answering
        # a clarifying question looks like: "Size 10", "Casual", "blue").
        for message, expected_key in [("blue", "color"), ("Size 10", "size"), ("Casual", "style")]:
            with self.subTest(message=message):
                parsed = self.parser.parse(message)
                self.assertIn(expected_key, parsed.attributes)
                self.assertFalse(parsed.is_vague)

    def test_negation_suppresses_the_negated_attribute(self) -> None:
        for message in ["I don't want red", "no leather please", "anything but blue"]:
            with self.subTest(message=message):
                parsed = self.parser.parse(message)
                self.assertNotIn("material", parsed.attributes)
                self.assertNotIn("color", parsed.attributes)

    def test_jewelry_dimension_not_mistaken_for_garment_size(self) -> None:
        # Raw catalog text labels a physical measurement the same way it
        # labels a real size ("Size: N") -- a unit marker right after the
        # number distinguishes a dimension/gauge from a real garment size.
        for message in [
            "Size: 2.5'' in length weight 0.2oz a pair",
            'Waist Size:25.2", Hip Size:54.8"',
            "hoop ring size: 14 gauge (ring thickness)",
        ]:
            with self.subTest(message=message):
                parsed = self.parser.parse(message)
                self.assertNotIn("size", parsed.attributes)

    def test_real_garment_size_still_extracted(self) -> None:
        for message, expected in [("Black Size 10", "10"), ("size 9", "9")]:
            with self.subTest(message=message):
                parsed = self.parser.parse(message)
                self.assertEqual(parsed.attributes.get("size"), expected)

    def test_negation_does_not_cross_clause_boundary(self) -> None:
        # A negation earlier in the sentence must not suppress a genuinely
        # positive, unrelated mention in a later clause.
        parsed = self.parser.parse("I don't want polyester, I love cotton")
        self.assertEqual(parsed.attributes.get("material"), "cotton")

    def test_no_structured_attribute_is_vague_even_without_pattern(self) -> None:
        # No uncertainty phrase here, but nothing structured was extracted
        # either (only the loose `feature` catch-all) -- still vague.
        parsed = self.parser.parse("I want something comfortable")
        self.assertEqual(set(parsed.attributes), {"feature"})
        self.assertTrue(parsed.is_vague)

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
        self.assertEqual(
            set(payload.keys()),
            {"raw_text", "keywords", "attributes", "intent", "category", "product", "signals"},
        )
        self.assertEqual(
            set(payload["signals"].keys()), {"is_override", "is_no_preference", "is_vague"}
        )


class CatalogVocabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Scanning the 50k-row catalog is the same regardless of test case;
        # load once per class instead of once per test method.
        cls.categories, cls.brands = load_catalog_vocab("data/catalog.jsonl")

    def setUp(self) -> None:
        self.parser = MessageParser(known_categories=self.categories, known_brands=self.brands)

    def test_vocab_improves_category_and_brand_matching(self) -> None:
        self.assertIn("boots", self.categories)
        self.assertIn("earrings", self.categories)
        self.assertGreater(len(self.brands), 1000)
        parsed = self.parser.parse("I need a pair of boots.")
        self.assertEqual(parsed.attributes.get("category"), "boots")

    def test_ambiguous_material_category_terms_not_double_assigned(self) -> None:
        # "cotton"/"denim"/"fleece" are real materials AND real catalog
        # categories; "bamboo"/"canvas" are real materials AND real store
        # names. Only material should be assigned.
        for word in ["cotton", "denim", "fleece", "bamboo", "canvas"]:
            with self.subTest(word=word):
                parsed = self.parser.parse(f"I want something in {word}.")
                self.assertEqual(parsed.attributes.get("material"), word)
                self.assertNotIn("category", parsed.attributes)
                self.assertNotIn("brand", parsed.attributes)

    def test_generic_word_brand_false_positive_blocked(self) -> None:
        # Real catalog store names "Key" and "Not" would otherwise false
        # positive on ordinary sentences.
        parsed = self.parser.parse(
            "Those options are not quite right yet. Ask me about one specific attribute."
        )
        self.assertNotIn("brand", parsed.attributes)

    def test_hyphenated_category_matches_despite_tokenization(self) -> None:
        # Catalog stores "t-shirts" with a literal hyphen; query tokenization
        # strips punctuation, so both sides must normalize the same way.
        for message in ["I need a t-shirt.", "I need a t shirt."]:
            with self.subTest(message=message):
                parsed = self.parser.parse(message)
                self.assertEqual(parsed.attributes.get("category"), "t shirts")

    def test_merged_compound_word_category_alias(self) -> None:
        cases = {
            "tshirt": "t shirts",
            "flipflops": "flip flops",
            "carryon luggage": "carry ons",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                parsed = self.parser.parse(message)
                self.assertEqual(parsed.attributes.get("category"), expected)

    def test_brand_distinctiveness_filters_generic_words(self) -> None:
        # Real catalog quirk: "machine" is the literal store name for 1
        # product but appears 10,975 times catalog-wide (almost always from
        # "Machine Wash" care instructions) -- must not be trusted as brand.
        # "skechers" is store name for 375 products and appears 388 times
        # total -- a real, distinctive brand -- must be kept.
        for noisy in ["machine", "waterproof", "simple", "seasons", "goddess"]:
            self.assertNotIn(noisy, self.brands, f"{noisy!r} should be filtered as non-distinctive")
        for real in ["skechers", "creepyparty", "efixtk"]:
            self.assertIn(real, self.brands, f"{real!r} should survive as a distinctive brand")

        parsed = self.parser.parse("Rubber sole; Skechers Go Walk 5 shoe is designed for comfort.")
        self.assertEqual(parsed.attributes.get("brand"), "skechers")
        parsed = self.parser.parse("Care instructions: Machine Wash, tumble dry low.")
        self.assertNotIn("brand", parsed.attributes)

    def test_leftover_use_case_keyword_not_leaked_as_brand(self) -> None:
        # Real catalog quirk: "work" is a valid use_case keyword AND happens
        # to be a literal (if non-distinctive) store name. Previously only
        # the first use_case hit ("outdoor") was claimed, leaving "work"
        # free for the brand matcher to grab.
        parsed = self.parser.parse("I'm looking for Outdoor & Work Snow & Cold Weather gear.")
        self.assertEqual(parsed.attributes.get("use_case"), "outdoor")
        self.assertNotEqual(parsed.attributes.get("brand"), "work")

    def test_compound_alias_does_not_leak_fragment_as_brand(self) -> None:
        # "crossbody" expands toward "cross body" for category matching, but
        # the split fragment "cross" must not separately false-positive
        # against an unrelated store also named "Cross".
        parsed = self.parser.parse("I want a crossbody bag.")
        self.assertNotEqual(parsed.attributes.get("brand"), "cross")


if __name__ == "__main__":
    unittest.main()