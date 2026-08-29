from __future__ import annotations

import unittest

from starter.ledger import LedgerService


class LedgerCRUDTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = LedgerService()
        self.ledger.create("s1", {"summary": "test"})

    def test_create_initialises_empty_state(self) -> None:
        state = self.ledger.read("s1")
        self.assertEqual(state["turn"], 0)
        self.assertIsNone(state["intent"])
        self.assertEqual(state["constraints"], {})
        self.assertEqual(state["soft_preferences"], [])
        self.assertEqual(state["asked_attributes"], [])
        self.assertEqual(state["search_key"], {})

    def test_exists_true_after_create(self) -> None:
        self.assertTrue(self.ledger.exists("s1"))

    def test_exists_false_after_delete(self) -> None:
        self.ledger.delete("s1")
        self.assertFalse(self.ledger.exists("s1"))

    def test_read_returns_deep_copy(self) -> None:
        state = self.ledger.read("s1")
        state["turn"] = 99
        self.assertEqual(self.ledger.read("s1")["turn"], 0)

    def test_read_unknown_session_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.ledger.read("does_not_exist")


class LedgerHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = LedgerService()
        self.ledger.create("s1", {"summary": "test"})

    def test_increment_turn(self) -> None:
        self.ledger.increment_turn("s1")
        self.ledger.increment_turn("s1")
        self.assertEqual(self.ledger.read("s1")["turn"], 2)

    def test_set_intent(self) -> None:
        self.ledger.set_intent("s1", "buying")
        self.assertEqual(self.ledger.read("s1")["intent"], "buying")

    def test_add_constraint_appends(self) -> None:
        self.ledger.add_constraint("s1", "color", "black")
        self.ledger.add_constraint("s1", "color", "red")
        self.assertEqual(self.ledger.read("s1")["constraints"]["color"], ["black", "red"])

    def test_add_constraint_no_duplicates(self) -> None:
        self.ledger.add_constraint("s1", "color", "black")
        self.ledger.add_constraint("s1", "color", "black")
        self.assertEqual(self.ledger.read("s1")["constraints"]["color"], ["black"])

    def test_set_constraint_overwrites(self) -> None:
        self.ledger.add_constraint("s1", "color", "black")
        self.ledger.add_constraint("s1", "color", "red")
        self.ledger.set_constraint("s1", "color", "white")
        self.assertEqual(self.ledger.read("s1")["constraints"]["color"], ["white"])

    def test_add_constraint_unknown_attribute_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.ledger.add_constraint("s1", "unknown_attr", "value")

    def test_clear_constraints_wipes_constraints_and_soft_preferences(self) -> None:
        self.ledger.add_constraint("s1", "color", "black")
        self.ledger.add_soft_preference("s1", "boots")
        self.ledger.clear_constraints("s1")
        state = self.ledger.read("s1")
        self.assertEqual(state["constraints"], {})
        self.assertEqual(state["soft_preferences"], [])

    def test_add_soft_preference_no_duplicates(self) -> None:
        self.ledger.add_soft_preference("s1", "boots")
        self.ledger.add_soft_preference("s1", "boots")
        self.assertEqual(self.ledger.read("s1")["soft_preferences"], ["boots"])

    def test_mark_attribute_asked_no_duplicates(self) -> None:
        self.ledger.mark_attribute_asked("s1", "color")
        self.ledger.mark_attribute_asked("s1", "color")
        self.assertEqual(self.ledger.read("s1")["asked_attributes"], ["color"])

    def test_set_search_key(self) -> None:
        key = {"color": ["black"], "material": ["leather"]}
        self.ledger.set_search_key("s1", key)
        self.assertEqual(self.ledger.read("s1")["search_key"], key)

    def test_next_unasked_attribute_respects_priority(self) -> None:
        # category is highest priority and nothing is asked yet
        self.assertEqual(self.ledger.next_unasked_attribute("s1"), "category")

    def test_next_unasked_attribute_skips_constrained(self) -> None:
        self.ledger.add_constraint("s1", "category", "boots")
        # next should be use_case
        self.assertEqual(self.ledger.next_unasked_attribute("s1"), "use_case")

    def test_next_unasked_attribute_skips_asked(self) -> None:
        self.ledger.mark_attribute_asked("s1", "category")
        self.assertEqual(self.ledger.next_unasked_attribute("s1"), "use_case")

    def test_next_unasked_attribute_returns_none_when_all_covered(self) -> None:
        from starter.ledger import ATTRIBUTE_PRIORITY
        for attr in ATTRIBUTE_PRIORITY:
            self.ledger.mark_attribute_asked("s1", attr)
        self.assertIsNone(self.ledger.next_unasked_attribute("s1"))


class LedgerContextManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = LedgerService()
        self.ledger.create("s1", {"summary": "test"})

    def test_session_writes_back_on_clean_exit(self) -> None:
        with self.ledger.session("s1") as state:
            state["turn"] = 5
            state["intent"] = "browsing"
        result = self.ledger.read("s1")
        self.assertEqual(result["turn"], 5)
        self.assertEqual(result["intent"], "browsing")

    def test_session_does_not_write_back_on_exception(self) -> None:
        with self.assertRaises(ValueError):
            with self.ledger.session("s1") as state:
                state["turn"] = 99
                raise ValueError("abort")
        self.assertEqual(self.ledger.read("s1")["turn"], 0)

    def test_managed_session_deletes_after_exit(self) -> None:
        with self.ledger.managed_session("s2", {"summary": "x"}):
            self.assertTrue(self.ledger.exists("s2"))
        self.assertFalse(self.ledger.exists("s2"))

    def test_managed_session_deletes_even_on_exception(self) -> None:
        with self.assertRaises(RuntimeError):
            with self.ledger.managed_session("s3", {"summary": "x"}):
                raise RuntimeError("crash")
        self.assertFalse(self.ledger.exists("s3"))


if __name__ == "__main__":
    unittest.main()