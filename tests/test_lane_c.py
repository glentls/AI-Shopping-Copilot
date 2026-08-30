"""Lane C: dialogue policy.

The four behaviours the brief names, plus the ones that actually broke while
building it. Everything but the last class runs on synthetic fixtures so the
suite stays in milliseconds; TestAgentContract builds a real Agent over a
miniature catalog to prove the wiring holds end to end.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.attributes import AttributeTable
from src.contracts import Candidate, ConversationState
from src.policy.message import compose_message
from src.policy.question import (
    DECLINE_PATIENCE,
    choose_question,
    other_value,
    score_slots,
    wildcard_declines,
)
from src.policy.state import learned_on, update


def make_state(**kwargs) -> ConversationState:
    return ConversationState(session_id="t", user_profile={}, **kwargs)


def make_table(inverted: dict[str, dict[str, list[str]]], total: int = 100) -> AttributeTable:
    return AttributeTable(
        {slot: {value: set(asins) for value, asins in values.items()}
         for slot, values in inverted.items()},
        total,
    )


def make_candidates(n: int = 20) -> list[Candidate]:
    return [Candidate(parent_asin=f"A{i:03}", score=float(-i)) for i in range(n)]


def say(state: ConversationState, message: str, turn: int, asked: str | None = None) -> None:
    """One turn, in the order starter/agent.py runs it: update, then record."""
    update(state, message, turn)
    if asked:
        state.asked.append(asked)
        state.question_history.append((turn, asked))
        state.last_asked = asked


TABLE = make_table({
    "material": {"cotton": ["A001", "A002"], "leather": ["A003"]},
    "color": {"black": ["A001"], "blue": ["A002", "A003"]},
    "use_case": {"hiking": ["A001"], "work": ["A002"]},
    "style": {"slim": ["A001"]},
    "feature": {"waterproof": ["A002"]},
    "brand": {"acme": ["A001"], "other": ["A002"]},
    "category": {"boots": ["A001", "A002", "A003"]},
    "size": {"wide": ["A001"]},
    "budget": {"50": ["A001"]},
})


class TestOverride(unittest.TestCase):
    """15% of sessions replace a preference on turn 3 or 4."""

    def test_override_retracts_the_old_value(self):
        state = make_state()
        say(state, "I want a cotton shirt", 1, asked="other")
        self.assertEqual(state.active("material"), ["cotton"])

        say(state, "Actually, ignore that — I need leather instead", 2)
        self.assertEqual(state.active("material"), ["leather"])
        self.assertIn("cotton", state.excluded("material"))

    def test_override_keeps_a_value_it_merely_restates(self):
        """"Actually, what I need is leather" when we already had leather is
        the customer stressing a priority, not changing one."""
        state = make_state()
        say(state, "I need a leather belt", 1, asked="other")
        say(state, "Actually, ignore my earlier preference. What I need is: leather.", 2)
        self.assertEqual(state.active("material"), ["leather"])
        self.assertEqual(state.excluded("material"), [])

    def test_broad_override_replaces_initial_preference_across_slots(self):
        state = make_state()
        say(state, "I'm looking for watches. Stainless Steel Band", 1, asked="other")
        say(state, "For that, what matters is: waterproof.", 2, asked="other")

        say(
            state,
            "Actually, ignore my earlier preference. What I need is: black.",
            3,
        )

        self.assertIn("stainless steel", state.excluded("material"))
        self.assertEqual(state.active("color"), ["black"])
        self.assertIn("waterproof", state.active("feature"))

    def test_broad_override_does_not_erase_later_values_in_the_new_slot(self):
        state = make_state()
        say(state, "I'm looking for boots. cotton", 1, asked="other")
        say(state, "For that, what matters is: comfortable.", 2, asked="other")
        say(
            state,
            "Actually, ignore my earlier preference. What I need is: waterproof.",
            3,
        )

        self.assertIn("cotton", state.excluded("material"))
        self.assertCountEqual(state.active("feature"), ["comfortable", "waterproof"])

    def test_unmapped_broad_override_preserves_later_material_details(self):
        """Regression for public_0046: do not guess the old slot from wool."""
        state = make_state()
        say(state, "I'm looking for socks. No Closure closure", 1, asked="other")
        say(
            state,
            "For that, what matters is: wool; 44% Acrylic, 28% Cotton, "
            "20% Merino Wool, 8% Polyester.",
            2,
            asked="other",
        )

        say(
            state,
            "Actually, ignore my earlier preference. What I need is: wool.",
            3,
        )

        self.assertCountEqual(
            state.active("material"),
            ["wool", "acrylic", "cotton", "polyester"],
        )

    def test_retracted_value_can_be_revived(self):
        state = make_state()
        say(state, "cotton please", 1, asked="other")
        say(state, "actually, leather instead", 2)
        self.assertIn("cotton", state.excluded("material"))
        say(state, "on reflection, cotton", 3)
        self.assertIn("cotton", state.active("material"))
        self.assertNotIn("cotton", state.excluded("material"))

    def test_override_starts_a_fresh_recommendation_epoch(self):
        state = make_state(shown_recommendations={"A001", "A002"})
        say(state, "leather please", 1, asked="other")
        self.assertEqual(state.shown_recommendations, {"A001", "A002"})

        say(state, "Actually, ignore that — I need cotton instead", 2)
        self.assertEqual(state.shown_recommendations, set())


class TestRepeats(unittest.TestCase):
    def test_repeat_folds_instead_of_duplicating(self):
        """One fact said twice is one fact. The reranker scores a point per
        live value, so a duplicate quietly doubles its weight."""
        state = make_state()
        say(state, "something in cotton", 1, asked="other")
        say(state, "cotton, and it should be waterproof", 2, asked="other")
        self.assertEqual(state.active("material"), ["cotton"])
        self.assertEqual(len(state.slots["material"]), 1)

    def test_repeat_is_not_a_newly_learned_fact(self):
        """SlotValue.turn is the turn a fact was LEARNED on; a repeat must not
        restamp it, or a silent turn reads as a productive one."""
        state = make_state()
        say(state, "cotton", 1, asked="other")
        say(state, "cotton again", 2, asked="other")
        self.assertEqual(learned_on(state, 1), 1)
        self.assertEqual(learned_on(state, 2), 0)

    def test_repeat_raises_confidence(self):
        state = make_state()
        say(state, "cotton", 1, asked="other")
        first = state.slots["material"][0].confidence
        say(state, "cotton", 2, asked="other")
        self.assertGreater(state.slots["material"][0].confidence, first)


class TestBoundary(unittest.TestCase):
    """5% of sessions answer 'I have no preference for what you asked'."""

    def test_no_preference_marks_the_asked_slot_unanswerable(self):
        state = make_state()
        say(state, "I need boots", 1, asked="material")
        say(state, "I don't have a preference for material; please use your judgment.", 2)
        self.assertIn("material", state.unanswerable)

    def test_implicit_no_preference_resolves_to_what_we_asked(self):
        state = make_state()
        say(state, "I need boots", 1, asked="color")
        say(state, "You decide.", 2)
        self.assertIn("color", state.unanswerable)

    def test_an_unanswerable_slot_is_never_asked_again(self):
        state = make_state(unanswerable={"material"}, turn=2)
        ranked = dict(score_slots(state, make_candidates(), TABLE))
        self.assertNotIn("material", ranked)

        attribute, extras = choose_question(state, make_candidates(), TABLE)
        self.assertNotIn("material", [attribute, *extras])


class TestQuestionChoice(unittest.TestCase):
    def test_never_repeats_a_slot_already_asked(self):
        state = make_state(asked=["material", "color"], turn=3)
        ranked = dict(score_slots(state, make_candidates(), TABLE))
        self.assertNotIn("material", ranked)
        self.assertNotIn("color", ranked)

    def test_never_asks_a_slot_we_already_know(self):
        state = make_state()
        say(state, "I want cotton", 1)
        ranked = dict(score_slots(state, make_candidates(), TABLE))
        self.assertNotIn("material", ranked)

    def test_stops_asking_after_every_preference_is_exhausted(self):
        state = make_state(asked=list(TABLE.slots()), unanswerable=set(TABLE.slots()))
        say(state, "I'm looking for boots", 1, asked="other")
        say(state, "I don't have an additional preference for other.", 2, asked="brand")
        say(state, "I don't have a preference for brand.", 3, asked="other")
        say(state, "I don't have an additional preference for other.", 4)

        attribute, _ = choose_question(state, make_candidates(), TABLE)
        self.assertIsNone(attribute)

    def test_bundles_extra_topics(self):
        """Only ask_attribute is scored, so the prose is free upside."""
        state = make_state(turn=1)
        attribute, extras = choose_question(state, make_candidates(), TABLE)
        self.assertTrue(extras)
        self.assertNotIn(attribute, extras)


class TestWildcardPricing(unittest.TestCase):
    def test_wildcard_leads_on_a_fresh_session(self):
        state = make_state(turn=1)
        attribute, _ = choose_question(state, make_candidates(), TABLE)
        self.assertEqual(attribute, "other")

    def test_yield_falls_when_the_wildcard_stops_paying(self):
        productive = make_state()
        say(productive, "I need boots", 1, asked="other")
        say(productive, "cotton and waterproof", 2, asked="other")

        silent = make_state()
        say(silent, "I need boots", 1, asked="other")
        say(silent, "nothing more to add", 2, asked="other")
        say(silent, "nothing more to add", 3, asked="other")

        self.assertGreater(other_value(productive), other_value(silent))

    def test_one_refusal_does_not_stand_the_wildcard_down(self):
        """A boundary customer refuses whatever we ask first, then answers
        normally. Giving up on one refusal costs boundary MTTC 4.00 -> 4.90."""
        state = make_state()
        say(state, "I'm looking for boots, but I'm still exploring.", 1, asked="other")
        say(state, "I don't have a preference for other; please use your judgment.", 2)
        self.assertEqual(wildcard_declines(state), 1)
        self.assertGreater(other_value(state), 0.0)

    def test_repeated_refusals_stand_the_wildcard_down(self):
        state = make_state()
        say(state, "I'm looking for boots", 1, asked="other")
        for turn in range(2, 2 + DECLINE_PATIENCE):
            say(state, "I don't have an additional preference for other.", turn, asked="other")
        self.assertGreaterEqual(wildcard_declines(state), DECLINE_PATIENCE)
        self.assertEqual(other_value(state), 0.0)

        attribute, _ = choose_question(state, make_candidates(), TABLE)
        self.assertNotEqual(attribute, "other")

    def test_interleaved_refusals_still_stand_the_wildcard_down(self):
        state = make_state()
        say(state, "I'm looking for boots", 1, asked="other")
        say(state, "I don't have an additional preference for other.", 2, asked="brand")
        say(state, "I don't have a preference for brand.", 3, asked="other")
        say(state, "I don't have an additional preference for other.", 4)

        self.assertEqual(wildcard_declines(state), 2)
        self.assertEqual(other_value(state), 0.0)

    def test_override_reopens_the_wildcard_for_the_new_intent(self):
        state = make_state()
        say(state, "I'm looking for boots", 1, asked="other")
        say(state, "I don't have an additional preference for other.", 2, asked="other")
        say(state, "I don't have an additional preference for other.", 3)
        self.assertEqual(other_value(state), 0.0)

        say(state, "Actually, ignore my earlier preference. I need cotton.", 4)
        self.assertEqual(wildcard_declines(state), 0)
        self.assertGreater(other_value(state), 0.0)


class TestMessage(unittest.TestCase):
    def test_acknowledges_only_what_this_turn_taught(self):
        state = make_state()
        say(state, "I need cotton", 1, asked="other")
        say(state, "and waterproof", 2, asked="other")
        text = compose_message(state, make_candidates(), "other", ["material"])
        self.assertIn("waterproof", text)
        self.assertNotIn("cotton", text)

    def test_names_the_constraint_an_override_restates(self):
        """The restated value folds rather than re-learns, so there is nothing
        newly learned to name -- but the customer just said it out loud."""
        state = make_state()
        say(state, "I need a leather belt", 1, asked="other")
        say(state, "Actually, ignore my earlier preference. What I need is: leather.", 2,
            asked="other")
        text = compose_message(state, make_candidates(), "other", [])
        self.assertIn("leather", text)
        self.assertNotIn("the new requirement", text)

    def test_does_not_promise_to_drop_a_topic_it_is_still_asking(self):
        state = make_state()
        say(state, "I need boots", 1, asked="other")
        say(state, "I don't have an additional preference for other.", 2, asked="other")
        text = compose_message(state, make_candidates(), "other", [])
        self.assertNotIn("stop asking", text)

    def test_exhausted_customer_gets_recommendations_without_another_question(self):
        state = make_state()
        say(state, "I need boots", 1, asked="other")
        say(state, "I don't have an additional preference for other.", 2)
        text = compose_message(state, make_candidates(), None, [])
        self.assertNotIn("Could you tell me", text)
        self.assertNotIn("It would help to know", text)

    def test_budget_is_spoken_not_echoed_raw(self):
        """Budget is stored as a bare number. "Got it -- comfortable and 80"
        is the kind of line a judge remembers for the wrong reason."""
        state = make_state()
        say(state, "something comfortable under $80", 1, asked="other")
        text = compose_message(state, make_candidates(), "other", [])
        self.assertIn("under $80", text)
        self.assertNotIn(", and 80", text)

    def test_acknowledgement_lists_with_and_not_or(self):
        """These are things the customer HAS said, not alternatives on offer."""
        state = make_state()
        say(state, "waterproof, comfortable, cotton", 1, asked="other")
        text = compose_message(state, make_candidates(), "other", [])
        opening = text.split(".")[0]
        self.assertIn(" and ", opening)
        self.assertNotIn(", or ", opening)

    def test_never_raises_on_a_bare_state(self):
        for message in ("", "hello", "Actually, ignore that.", "You decide."):
            state = make_state()
            say(state, message, 1, asked="other")
            self.assertIsInstance(compose_message(state, [], "other", ["material"]), str)

    def test_wording_moves_across_a_frozen_state(self):
        """Nothing new arrives after the customer dries up, so identical
        wording every turn is the default failure. It must not be."""
        state = make_state()
        say(state, "I need boots", 1, asked="other")
        seen = set()
        for turn in range(2, 8):
            say(state, "I don't have an additional preference for other.", turn, asked="other")
            seen.add(compose_message(state, make_candidates(), "other", []))
        self.assertGreater(len(seen), 1)


MINI_CATALOG = [
    {"parent_asin": "B001", "title": "Waterproof Leather Hiking Boot",
     "features": ["waterproof", "leather upper"], "description": ["built for hiking"],
     "details": {"Material": "leather"}, "categories": ["Shoes", "Boots"],
     "store": "Acme", "price": 90, "rating_number": 400},
    {"parent_asin": "B002", "title": "Cotton Crew Neck T-Shirt",
     "features": ["breathable cotton"], "description": ["everyday tee"],
     "details": {"Material": "cotton"}, "categories": ["Clothing", "Shirts"],
     "store": "Basics", "price": 20, "rating_number": 900},
    {"parent_asin": "B003", "title": "Black Wool Winter Coat",
     "features": ["insulated", "wool"], "description": ["warm winter coat"],
     "details": {"Material": "wool"}, "categories": ["Clothing", "Coats"],
     "store": "Northline", "price": 200, "rating_number": 150},
] + [
    {"parent_asin": f"B{index:03}", "title": f"Generic Clothing Item {index}",
     "features": ["everyday basic"], "description": ["general purpose clothing"],
     "details": {}, "categories": ["Clothing"], "store": "Generic",
     "price": 25 + index, "rating_number": index}
    for index in range(4, 21)
]


class TestAgentContract(unittest.TestCase):
    """Every turn recommends; useful questions accompany those results."""

    @classmethod
    def setUpClass(cls):
        from src.contracts import ASK_ATTRIBUTES
        from starter.agent import Agent

        cls.ASK_ATTRIBUTES = ASK_ATTRIBUTES
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        catalog = root / "catalog.jsonl"
        catalog.write_text(
            "\n".join(json.dumps(product) for product in MINI_CATALOG) + "\n",
            encoding="utf-8",
        )
        cls.agent = Agent(catalog, root / "artifacts")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_turn_returns_a_question_and_recommendations(self):
        self.agent.reset("s1", {})
        messages = [
            "I'm looking for Shoes Boots. A key requirement is: waterproof.",
            "For that, what matters is: leather.",
            "Actually, ignore my earlier preference. What I need is: cotton.",
            "I don't have a preference for material; please use your judgment.",
            "I don't have an additional preference for other.",
        ]
        for turn, message in enumerate(messages, start=1):
            with self.subTest(turn=turn):
                response = self.agent.respond("s1", message, turn, top_k=3)
                self.assertIsInstance(response["message"], str)
                self.assertTrue(response["message"].strip())
                self.assertIn(response["ask_attribute"], self.ASK_ATTRIBUTES)
                self.assertTrue(response["recommendations"])
                for item in response["recommendations"]:
                    self.assertIn("parent_asin", item)

    def test_recommendations_are_padded_to_top_k(self):
        self.agent.reset("s2", {})
        response = self.agent.respond("s2", "I'm still exploring.", 1, top_k=3)
        self.assertEqual(len(response["recommendations"]), 3)

    def test_recommendations_do_not_repeat_within_an_intent(self):
        self.agent.reset("s4", {})
        seen: set[str] = set()
        messages = [
            "I'm looking for everyday clothing.",
            "Something comfortable.",
            "I don't have an additional preference for other.",
            "Show me some more options.",
        ]
        for turn, message in enumerate(messages, start=1):
            response = self.agent.respond("s4", message, turn, top_k=3)
            current = {item["parent_asin"] for item in response["recommendations"]}
            self.assertEqual(len(current), 3)
            self.assertTrue(current.isdisjoint(seen))
            seen.update(current)

    def test_question_and_message_use_only_unseen_candidates(self):
        self.agent.reset("s5", {})
        state = self.agent._states["s5"]
        state.shown_recommendations.add("B001")
        candidates = [
            Candidate("B001", 2.0, why="it is the already shown option"),
            Candidate("B002", 1.0, why="it matches the unseen cotton option"),
        ]

        with (
            patch.object(self.agent.retriever, "search", return_value=candidates),
            patch.object(self.agent.retriever, "rerank", return_value=candidates),
            patch("starter.agent.choose_question", return_value=(None, [])) as choose,
        ):
            response = self.agent.respond("s5", "Show me another option", 1, top_k=1)

        passed_candidates = choose.call_args.args[1]
        self.assertEqual([candidate.parent_asin for candidate in passed_candidates], ["B002"])
        self.assertEqual(response["recommendations"][0]["parent_asin"], "B002")
        self.assertIn("unseen cotton option", response["message"])
        self.assertNotIn("already shown option", response["message"])

    def test_override_moves_the_ranking(self):
        """The whole point of retraction: after an override the ranking must
        reflect the new constraint, not both constraints at once."""
        self.agent.reset("s3", {})
        self.agent.respond("s3", "I need a leather boot", 1, top_k=3)
        after = self.agent.respond(
            "s3", "Actually, ignore that — I need a cotton shirt instead.", 2, top_k=3
        )
        state = self.agent._states["s3"]
        self.assertEqual(state.active("material"), ["cotton"])
        self.assertIn("leather", state.excluded("material"))
        self.assertEqual(after["recommendations"][0]["parent_asin"], "B002")


if __name__ == "__main__":
    unittest.main()
