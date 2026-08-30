from __future__ import annotations

import unittest

from src.contracts import Candidate, ConversationState, SlotValue
from src.orchestration import (
    BROAD_CANDIDATE_CUTOFF,
    candidate_pool_overloaded,
    compile_context_program,
)
from src.retrieval.blend import lock_hard_constraints


class StubTable:
    def __init__(self, values: dict[tuple[str, str], list[str]]) -> None:
        self._values = values

    def values(self, asin: str, slot: str) -> list[str]:
        return self._values.get((asin, slot), [])


class ContextProgrammingTest(unittest.TestCase):
    def test_broad_opener_uses_personalized_browsing_program(self) -> None:
        state = ConversationState(
            "s",
            {"preference_tags": ["Comfort", "fit", "comfort", "style"]},
        )
        state.history.append((
            "customer",
            "I'm looking for shoes, but I'm still exploring.",
        ))
        state.add("category", SlotValue("shoes", 0.95, 1))

        program = compile_context_program(state)

        self.assertEqual(program.route, "browsing")
        self.assertTrue(program.over_general)
        self.assertEqual(program.candidate_cutoff, BROAD_CANDIDATE_CUTOFF)
        self.assertEqual(program.profile_terms, ("comfort", "fit", "style"))
        self.assertGreater(program.profile_weight, 0.0)
        self.assertTrue(candidate_pool_overloaded(program, 200))

    def test_new_constraint_reorchestrates_browsing_to_buying(self) -> None:
        state = ConversationState("s", {"preference_tags": ["comfort"]})
        state.history.extend([
            ("customer", "I'm looking for shoes, but I'm still exploring."),
            ("agent", "Do you have a material preference?"),
            ("customer", "I need leather."),
        ])
        state.add("category", SlotValue("shoes", 0.95, 1))
        state.add("material", SlotValue("leather", 0.95, 2))

        program = compile_context_program(state)

        self.assertEqual(program.route, "buying")
        self.assertFalse(program.over_general)
        self.assertEqual(program.profile_weight, 0.0)
        self.assertTrue(program.lock_hard_constraints)
        self.assertIn(("material", ("leather",)), program.hard_constraints)

    def test_exploratory_detail_switches_route_without_forcing_a_hard_lock(self) -> None:
        state = ConversationState("s", {})
        state.history.extend([
            ("customer", "I'm looking for shoes, but I'm still exploring."),
            ("customer", "For that, what matters is: leather."),
        ])
        state.add("category", SlotValue("shoes", 0.95, 1))
        state.add("material", SlotValue("leather", 0.95, 2))

        program = compile_context_program(state)

        self.assertEqual(program.route, "buying")
        self.assertFalse(program.lock_hard_constraints)

    def test_retracted_values_are_not_programmed_as_hard_constraints(self) -> None:
        state = ConversationState("s", {})
        state.history.append(("customer", "I need cotton instead."))
        state.add("material", SlotValue("leather", 0.95, 1, polarity=False))
        state.add("material", SlotValue("cotton", 0.95, 2))

        program = compile_context_program(state)

        self.assertEqual(program.active_terms, ("cotton",))
        self.assertEqual(program.hard_constraints, (("material", ("cotton",)),))


class BuyingConstraintLockTest(unittest.TestCase):
    def test_complete_hard_matches_fill_the_top_ten(self) -> None:
        candidates = [Candidate(f"P{index:02}", -float(index)) for index in range(12)]
        # Put the two non-matches first to prove the Buying lock changes the
        # customer-visible top ten rather than merely annotating candidates.
        candidates = candidates[10:] + candidates[:10]
        table = StubTable({
            (f"P{index:02}", "material"): ["leather"]
            for index in range(10)
        })

        ranked = lock_hard_constraints(
            candidates, (("material", ("leather",)),), table
        )

        self.assertEqual(
            [candidate.parent_asin for candidate in ranked[:10]],
            [f"P{index:02}" for index in range(10)],
        )
        self.assertTrue(all(
            candidate.components.get("hard_filter") == 1.0
            for candidate in ranked[:10]
        ))

    def test_sparse_metadata_backs_off_without_losing_recall(self) -> None:
        candidates = [Candidate(f"P{index:02}", -float(index)) for index in range(12)]
        table = StubTable({("P11", "material"): ["leather"]})

        ranked = lock_hard_constraints(
            candidates, (("material", ("leather",)),), table
        )

        self.assertEqual(
            [candidate.parent_asin for candidate in ranked],
            [candidate.parent_asin for candidate in candidates],
        )
        self.assertTrue(all(
            candidate.components.get("hard_filter_backoff") == 1.0
            for candidate in ranked
        ))


if __name__ == "__main__":
    unittest.main()
