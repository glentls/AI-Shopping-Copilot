"""Tests for the conversation-trace harness in ``tools.trace_runner``.

These run against a small synthetic catalog and 100 synthetic sessions, so the
whole suite finishes in seconds without needing ``data/catalog.jsonl``. A final
integration test runs a handful of *real* public-set sessions when the catalog
is present.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import tempfile
import unittest
from pathlib import Path

from tools import trace_runner
from tools.trace_runner import (
    agent_state_snapshot,
    render_transcript,
    run,
    run_session,
    select_samples,
    summarize,
)
from evaluator.local_evaluator import MAX_TURNS, TOP_K, catalog_index, load_jsonl
from state.state_manager import HIGHEST_YIELD_ATTRIBUTE
from starter.agent import Agent


# =============================================================================
# SYNTHETIC FIXTURES
# =============================================================================

COLORS = ["black", "white", "blue", "red", "green", "brown", "pink", "gray"]
MATERIALS = ["cotton", "leather", "wool", "nylon", "polyester", "silk", "rayon", "spandex"]
KINDS = [
    ("Shoes", "Running"),
    ("Shoes", "Boots"),
    ("Clothing", "Shirts"),
    ("Clothing", "Jackets"),
    ("Accessories", "Belts"),
    ("Jewelry", "Necklaces"),
    ("Clothing", "Dresses"),
    ("Accessories", "Hats"),
]
SCENARIOS = ["buying", "browsing", "intent_override", "boundary"]


def synthetic_catalog(count: int = 40) -> list[dict]:
    rows = []
    for index in range(count):
        color = COLORS[index % len(COLORS)]
        material = MATERIALS[(index // 2) % len(MATERIALS)]
        group, kind = KINDS[index % len(KINDS)]
        rows.append(
            {
                "parent_asin": f"P{index:04d}",
                "title": f"{color.title()} {material} {kind[:-1]} model {index}",
                "features": [f"{material} upper", f"color: {color}", f"{kind.lower()} use"],
                "details": {"Department": "unisex", "Material": material},
                "description": [f"A {color} {material} {kind.lower()} item, index {index}."],
                "categories": ["Clothing, Shoes & Jewelry", group, kind],
                "store": f"Store{index % 7}",
                "average_rating": 3.0 + (index % 21) / 10.0,
                "rating_number": 10 + index * 7,
                "price": 15.0 + index * 3.5,
            }
        )
    return rows


def synthetic_samples(catalog: list[dict], count: int = 100) -> list[dict]:
    samples = []
    for index in range(count):
        product = catalog[index % len(catalog)]
        samples.append(
            {
                "sample_id": f"syn_{index:04d}",
                "scenario_type": SCENARIOS[index % len(SCENARIOS)],
                "category_bucket": "clothing",
                "difficulty_bucket": ["easy", "medium", "hard"][index % 3],
                "ground_truth": {"parent_asin": product["parent_asin"]},
                "user_profile": {
                    "average_prior_rating": 4.0,
                    "preference_tags": ["fit", "comfort"],
                    "purchase_frequency": "3-4 prior purchases",
                    "rating_style": "usually positive",
                    "summary": "Prior purchases emphasize fit and comfort.",
                },
            }
        )
    return samples


def write_catalog(root: Path, rows: list[dict]) -> Path:
    path = root / "catalog.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def strip_timing(trace: dict) -> dict:
    """Remove wall-clock fields so two runs can be compared for determinism."""
    clone = json.loads(json.dumps(trace, default=str))
    clone.pop("duration_ms", None)
    for turn in clone["turns"]:
        turn.pop("latency_ms", None)
    return clone


class BrokenAgent:
    """Agent that always raises, to prove the harness records instead of crashing."""

    def reset(self, session_id: str, user_profile: dict) -> None:
        pass

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        raise ValueError("boom")


class MalformedAgent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        pass

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {"message": None, "recommendations": "not-a-list"}


# =============================================================================
# SELECTION
# =============================================================================


class SelectSamplesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = synthetic_samples(synthetic_catalog(), 200)

    def test_head_selection_is_file_order(self) -> None:
        picked = select_samples(self.samples, 10, "head")
        self.assertEqual([item["sample_id"] for item in picked], [item["sample_id"] for item in self.samples[:10]])

    def test_stratified_selection_returns_exact_limit(self) -> None:
        picked = select_samples(self.samples, 100, "stratified")
        self.assertEqual(len(picked), 100)

    def test_stratified_selection_covers_every_scenario(self) -> None:
        picked = select_samples(self.samples, 100, "stratified")
        covered = {item["scenario_type"] for item in picked}
        self.assertEqual(covered, set(SCENARIOS))

    def test_stratified_is_proportional_on_a_skewed_set(self) -> None:
        skewed = (
            [{"sample_id": f"a{i}", "scenario_type": "buying"} for i in range(80)]
            + [{"sample_id": f"b{i}", "scenario_type": "browsing"} for i in range(80)]
            + [{"sample_id": f"c{i}", "scenario_type": "intent_override"} for i in range(30)]
            + [{"sample_id": f"d{i}", "scenario_type": "boundary"} for i in range(10)]
        )
        picked = select_samples(skewed, 100, "stratified")
        counts: dict[str, int] = {}
        for item in picked:
            counts[item["scenario_type"]] = counts.get(item["scenario_type"], 0) + 1
        self.assertEqual(counts, {"buying": 40, "browsing": 40, "intent_override": 15, "boundary": 5})

    def test_limit_above_population_returns_everything(self) -> None:
        picked = select_samples(self.samples, 10_000, "stratified")
        self.assertEqual(len(picked), len(self.samples))

    def test_selection_preserves_original_file_order(self) -> None:
        picked = select_samples(self.samples, 50, "stratified")
        order = {item["sample_id"]: index for index, item in enumerate(self.samples)}
        indices = [order[item["sample_id"]] for item in picked]
        self.assertEqual(indices, sorted(indices))


# =============================================================================
# ONE-SESSION TRACING
# =============================================================================


class SyntheticFixture:
    """Shared synthetic catalog/agent; not a TestCase so it is not collected."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.catalog_rows = synthetic_catalog()
        cls.catalog_path = write_catalog(root, cls.catalog_rows)
        cls.ids, cls.categories, cls.products = catalog_index(cls.catalog_path)
        cls.agent = Agent(cls.catalog_path)
        cls.samples = synthetic_samples(cls.catalog_rows, 100)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def trace(self, sample: dict, agent=None) -> dict:
        return run_session(agent or self.agent, sample, self.ids, self.categories, self.products)


class TraceStructureTest(SyntheticFixture, unittest.TestCase):

    def test_trace_has_required_top_level_keys(self) -> None:
        trace = self.trace(self.samples[0])
        for key in (
            "sample_id", "scenario_type", "target", "intent_card", "behavior",
            "hit", "first_hit_turn", "best_rank", "reciprocal_rank",
            "end_reason", "turn_count", "usage", "final_state", "state_history", "turns",
        ):
            self.assertIn(key, trace)

    def test_turn_count_is_within_the_ten_turn_budget(self) -> None:
        for sample in self.samples[:20]:
            with self.subTest(sample["sample_id"]):
                trace = self.trace(sample)
                self.assertGreaterEqual(trace["turn_count"], 1)
                self.assertLessEqual(trace["turn_count"], MAX_TURNS)

    def test_every_turn_records_both_sides_of_the_conversation(self) -> None:
        trace = self.trace(self.samples[0])
        for record in trace["turns"]:
            self.assertIsInstance(record["user_message"], str)
            self.assertTrue(record["user_message"])
            self.assertIsInstance(record["agent_message"], str)

    def test_turn_numbers_are_contiguous_from_one(self) -> None:
        trace = self.trace(self.samples[1])
        self.assertEqual([record["turn"] for record in trace["turns"]], list(range(1, trace["turn_count"] + 1)))

    def test_recommendations_are_valid_unique_and_capped(self) -> None:
        for sample in self.samples[:20]:
            trace = self.trace(sample)
            for record in trace["turns"]:
                with self.subTest(sample=sample["sample_id"], turn=record["turn"]):
                    recs = record["recommendations"]
                    self.assertLessEqual(len(recs), TOP_K)
                    self.assertEqual(len(recs), len(set(recs)))
                    self.assertTrue(set(recs).issubset(self.ids))

    def test_state_snapshot_is_logged_each_turn(self) -> None:
        trace = self.trace(self.samples[0])
        for record in trace["turns"]:
            state = record["state"]
            self.assertIsInstance(state, dict)
            for key in ("intent", "constraints", "no_preference", "asked_attributes", "search_query", "next_action"):
                self.assertIn(key, state)

    def test_constraints_accumulate_across_turns(self) -> None:
        multi_turn = [self.trace(sample) for sample in self.samples[:12]]
        grew = [
            trace for trace in multi_turn
            if trace["turn_count"] >= 2
            and len(trace["turns"][-1]["state"]["constraints"]) >= len(trace["turns"][0]["state"]["constraints"])
        ]
        self.assertTrue(grew, "no session accumulated or held constraints across turns")

    def test_state_history_is_captured_for_debugging(self) -> None:
        trace = self.trace(self.samples[0])
        self.assertIsInstance(trace["state_history"], list)
        if trace["state_history"]:
            self.assertIn("turn", trace["state_history"][0])

    def test_target_rank_matches_recommendation_position(self) -> None:
        for sample in self.samples[:25]:
            trace = self.trace(sample)
            target = trace["target"]["parent_asin"]
            for record in trace["turns"]:
                with self.subTest(sample=sample["sample_id"], turn=record["turn"]):
                    if target in record["recommendations"]:
                        self.assertEqual(record["target_rank"], record["recommendations"].index(target) + 1)
                    else:
                        self.assertIsNone(record["target_rank"])

    def test_hit_stops_the_session_at_the_hit_turn(self) -> None:
        hits = [self.trace(sample) for sample in self.samples[:30]]
        hits = [trace for trace in hits if trace["hit"]]
        self.assertTrue(hits, "expected at least one hit on the synthetic catalog")
        for trace in hits:
            with self.subTest(trace["sample_id"]):
                self.assertEqual(trace["first_hit_turn"], trace["turn_count"])
                self.assertEqual(trace["end_reason"], "hit")
                self.assertEqual(trace["turns"][-1]["event"], "hit")

    def test_reciprocal_rank_matches_best_rank(self) -> None:
        for sample in self.samples[:25]:
            trace = self.trace(sample)
            with self.subTest(sample["sample_id"]):
                if trace["best_rank"] is None:
                    self.assertEqual(trace["reciprocal_rank"], 0.0)
                else:
                    self.assertAlmostEqual(trace["reciprocal_rank"], 1.0 / trace["best_rank"])

    def test_miss_records_turn_limit_end_reason(self) -> None:
        misses = [self.trace(sample) for sample in self.samples[:30]]
        misses = [trace for trace in misses if not trace["hit"]]
        for trace in misses:
            with self.subTest(trace["sample_id"]):
                self.assertEqual(trace["end_reason"], "turn_limit")
                self.assertEqual(trace["turn_count"], MAX_TURNS)
                self.assertIsNone(trace["best_rank"])

    def test_ask_attribute_is_contract_legal(self) -> None:
        legal = {
            "category", "material", "color", "size", "style",
            "brand", "budget", "feature", "use_case", "other", None,
        }
        for sample in self.samples[:20]:
            trace = self.trace(sample)
            for record in trace["turns"]:
                with self.subTest(sample=sample["sample_id"], turn=record["turn"]):
                    self.assertIn(record["ask_attribute"], legal)

    def test_agent_never_repeats_a_classified_attribute(self) -> None:
        """Every attribute except the wildcard must be asked at most once.

        ``HIGHEST_YIELD_ATTRIBUTE`` is exempt by design: the simulator answers
        it without classifying, so it returns undisclosed constraints of any
        class and stays productive on every turn.
        """
        for sample in self.samples[:20]:
            trace = self.trace(sample)
            asked = [
                record["ask_attribute"]
                for record in trace["turns"]
                if record["ask_attribute"] and record["ask_attribute"] != HIGHEST_YIELD_ATTRIBUTE
            ]
            with self.subTest(sample["sample_id"]):
                self.assertEqual(len(asked), len(set(asked)))

    def test_usage_is_reported_per_turn_and_totalled(self) -> None:
        trace = self.trace(self.samples[0])
        total = sum(record["usage"]["prompt_tokens"] for record in trace["turns"])
        self.assertEqual(total, trace["usage"]["prompt_tokens"])
        self.assertEqual(
            trace["usage"]["total_tokens"],
            trace["usage"]["prompt_tokens"] + trace["usage"]["completion_tokens"],
        )

    def test_latency_is_recorded_per_turn(self) -> None:
        trace = self.trace(self.samples[0])
        for record in trace["turns"]:
            self.assertIsInstance(record["latency_ms"], float)
            self.assertGreaterEqual(record["latency_ms"], 0.0)


# =============================================================================
# SCENARIO BEHAVIOUR
# =============================================================================


class ScenarioTest(SyntheticFixture, unittest.TestCase):
    def scenario_samples(self, scenario: str, count: int = 6) -> list[dict]:
        return [item for item in self.samples if item["scenario_type"] == scenario][:count]

    def test_buying_opens_with_a_key_requirement(self) -> None:
        for sample in self.scenario_samples("buying"):
            with self.subTest(sample["sample_id"]):
                self.assertIn("key requirement", self.trace(sample)["turns"][0]["user_message"].lower())

    def test_browsing_opens_as_still_exploring(self) -> None:
        for sample in self.scenario_samples("browsing"):
            with self.subTest(sample["sample_id"]):
                self.assertIn("still exploring", self.trace(sample)["turns"][0]["user_message"].lower())

    def test_override_message_is_injected_at_the_scheduled_turn(self) -> None:
        for sample in self.scenario_samples("intent_override"):
            trace = self.trace(sample)
            override_turn = int(trace["behavior"]["override"]["turn"])
            with self.subTest(sample["sample_id"]):
                self.assertIn(override_turn, (3, 4))
                if trace["turn_count"] >= override_turn:
                    self.assertIn("ignore my earlier preference", trace["turns"][override_turn - 1]["user_message"].lower())
                    self.assertEqual(trace["turns"][override_turn - 2]["event"], "override_injected_next_turn")

    def test_override_turns_before_the_switch_are_marked_unscored(self) -> None:
        for sample in self.scenario_samples("intent_override"):
            trace = self.trace(sample)
            override_turn = int(trace["behavior"]["override"]["turn"])
            with self.subTest(sample["sample_id"]):
                for record in trace["turns"][: override_turn - 1]:
                    self.assertFalse(record["scored"])
                for record in trace["turns"][override_turn - 1:]:
                    self.assertTrue(record["scored"])

    def test_override_hit_can_never_land_before_the_override(self) -> None:
        for sample in self.scenario_samples("intent_override"):
            trace = self.trace(sample)
            with self.subTest(sample["sample_id"]):
                if trace["hit"]:
                    self.assertGreaterEqual(trace["first_hit_turn"], int(trace["behavior"]["override"]["turn"]))

    def test_boundary_customer_declines_exactly_once(self) -> None:
        for sample in self.scenario_samples("boundary"):
            trace = self.trace(sample)
            declines = [
                record for record in trace["turns"]
                if "please use your judgment" in record["user_message"].lower()
            ]
            with self.subTest(sample["sample_id"]):
                self.assertLessEqual(len(declines), 1)

    def test_boundary_no_preference_lands_in_state(self) -> None:
        for sample in self.scenario_samples("boundary"):
            trace = self.trace(sample)
            declined = [
                record for record in trace["turns"]
                if "don't have a preference for" in record["user_message"].lower()
            ]
            with self.subTest(sample["sample_id"]):
                if declined:
                    self.assertTrue(declined[0]["state"]["no_preference"])

    def test_no_preference_attribute_is_not_asked_again(self) -> None:
        """Except the wildcard, which the simulator keeps answering regardless."""
        for sample in self.scenario_samples("boundary"):
            trace = self.trace(sample)
            for record in trace["turns"]:
                no_pref = set(record["state"]["no_preference"]) - {HIGHEST_YIELD_ATTRIBUTE}
                with self.subTest(sample=sample["sample_id"], turn=record["turn"]):
                    self.assertNotIn(record["ask_attribute"], no_pref)


# =============================================================================
# ROBUSTNESS
# =============================================================================


class RobustnessTest(SyntheticFixture, unittest.TestCase):
    def test_agent_exception_is_logged_not_raised(self) -> None:
        with self.assertLogs("trace", level="ERROR") as captured:
            trace = self.trace(self.samples[0], agent=BrokenAgent())
        self.assertEqual(len(captured.records), MAX_TURNS)
        self.assertEqual(trace["turn_count"], MAX_TURNS)
        self.assertFalse(trace["hit"])
        for record in trace["turns"]:
            self.assertIn("ValueError: boom", record["error"] or "")

    def test_malformed_response_is_coerced_and_flagged(self) -> None:
        trace = self.trace(self.samples[0], agent=MalformedAgent())
        self.assertEqual(trace["turn_count"], MAX_TURNS)
        for record in trace["turns"]:
            self.assertEqual(record["recommendations"], [])
            self.assertIsNotNone(record["error"])

    def test_state_snapshot_is_none_for_agents_without_a_manager(self) -> None:
        self.assertIsNone(agent_state_snapshot(BrokenAgent(), "s"))

    def test_tracing_is_deterministic_across_repeat_runs(self) -> None:
        for sample in self.samples[:8]:
            first = strip_timing(self.trace(sample))
            second = strip_timing(self.trace(sample))
            with self.subTest(sample["sample_id"]):
                self.assertEqual(first, second)

    def test_transcript_renders_without_error_for_hits_and_misses(self) -> None:
        for sample in self.samples[:8]:
            trace = self.trace(sample)
            text = render_transcript(trace)
            with self.subTest(sample["sample_id"]):
                self.assertIn(trace["sample_id"], text)
                self.assertIn("RESULT :", text)
                self.assertIn("Turn 1", text)


# =============================================================================
# AGGREGATION
# =============================================================================


class SummaryTest(unittest.TestCase):
    def traces(self) -> list[dict]:
        return [
            {
                "sample_id": "a", "scenario_type": "buying", "hit": True,
                "first_hit_turn": 2, "best_rank": 2, "reciprocal_rank": 0.5,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            {
                "sample_id": "b", "scenario_type": "browsing", "hit": False,
                "first_hit_turn": None, "best_rank": None, "reciprocal_rank": 0.0,
                "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            },
        ]

    def test_summary_matches_the_official_metric_definitions(self) -> None:
        summary = summarize(self.traces())
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["hit_rate_at_10"], 0.5)
        self.assertEqual(summary["mrr"], 0.25)
        self.assertEqual(summary["mttc"], 6.5)  # miss counts as turn 11
        self.assertAlmostEqual(summary["efficiency"], 0.45)
        self.assertAlmostEqual(summary["recommended_technical_score"], 0.5 * 0.5 + 0.3 * 0.25 + 0.2 * 0.45)

    def test_summary_totals_reported_tokens(self) -> None:
        usage = summarize(self.traces())["reported_token_usage"]
        self.assertEqual(usage, {"prompt_tokens": 14, "completion_tokens": 6, "total_tokens": 20})

    def test_summary_breaks_down_by_scenario(self) -> None:
        scenarios = summarize(self.traces())["scenario_metrics"]
        self.assertEqual(set(scenarios), {"buying", "browsing"})
        self.assertEqual(scenarios["buying"]["hit_rate_at_10"], 1.0)
        self.assertEqual(scenarios["browsing"]["hit_rate_at_10"], 0.0)


# =============================================================================
# END-TO-END: 100 CASES + LOG ARTEFACTS
# =============================================================================


class HundredCaseRunTest(unittest.TestCase):
    CASES = 100

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        catalog_rows = synthetic_catalog()
        cls.catalog_path = write_catalog(root, catalog_rows)
        cls.samples = synthetic_samples(catalog_rows, cls.CASES)
        cls.out_dir = root / "logs"
        trace_runner.configure_logging(cls.out_dir / "trace_run.log", verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.summary = run(cls.samples, cls.catalog_path, cls.out_dir)
        for handler in list(trace_runner.LOGGER.handlers):
            handler.close()
        cls.traces = [json.loads(line) for line in (cls.out_dir / "conversations.jsonl").read_text().splitlines()]

    @classmethod
    def tearDownClass(cls) -> None:
        logging.getLogger("trace").handlers.clear()
        cls._tmp.cleanup()

    def test_exactly_one_hundred_cases_are_traced(self) -> None:
        self.assertEqual(len(self.traces), self.CASES)
        self.assertEqual(self.summary["sample_count"], self.CASES)

    def test_every_sample_id_appears_exactly_once(self) -> None:
        ids = [trace["sample_id"] for trace in self.traces]
        self.assertEqual(len(set(ids)), self.CASES)
        self.assertEqual(ids, [sample["sample_id"] for sample in self.samples])

    def test_all_four_scenario_types_are_exercised(self) -> None:
        self.assertEqual({trace["scenario_type"] for trace in self.traces}, set(SCENARIOS))

    def test_jsonl_conversation_log_is_complete(self) -> None:
        for trace in self.traces:
            with self.subTest(trace["sample_id"]):
                self.assertTrue(trace["turns"])
                self.assertEqual(len(trace["turns"]), trace["turn_count"])

    def test_markdown_transcript_covers_every_session(self) -> None:
        text = (self.out_dir / "conversations.md").read_text(encoding="utf-8")
        self.assertIn(f"# Conversation traces ({self.CASES} sessions)", text)
        for trace in self.traces:
            with self.subTest(trace["sample_id"]):
                self.assertIn(f"## {trace['sample_id']} ", text)

    def test_run_log_records_every_session_and_turn(self) -> None:
        log_text = (self.out_dir / "trace_run.log").read_text(encoding="utf-8")
        self.assertIn("catalog indexed:", log_text)
        for trace in self.traces:
            with self.subTest(trace["sample_id"]):
                self.assertIn(f"=== {trace['sample_id']} ", log_text)
        self.assertIn("USER", log_text)
        self.assertIn("AGENT", log_text)
        self.assertIn("STATE", log_text)
        self.assertIn("QUERY", log_text)

    def test_summary_file_matches_returned_summary(self) -> None:
        on_disk = json.loads((self.out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["sample_count"], self.summary["sample_count"])
        self.assertEqual(on_disk["hit_rate_at_10"], self.summary["hit_rate_at_10"])
        self.assertEqual(on_disk["mrr"], self.summary["mrr"])

    def test_hit_rate_matches_the_traced_sessions(self) -> None:
        hits = sum(1 for trace in self.traces if trace["hit"])
        self.assertAlmostEqual(self.summary["hit_rate_at_10"], round(hits / self.CASES, 6))

    def test_metrics_stay_inside_their_valid_ranges(self) -> None:
        self.assertGreaterEqual(self.summary["hit_rate_at_10"], 0.0)
        self.assertLessEqual(self.summary["hit_rate_at_10"], 1.0)
        self.assertGreaterEqual(self.summary["mrr"], 0.0)
        self.assertLessEqual(self.summary["mrr"], 1.0)
        self.assertGreaterEqual(self.summary["mttc"], 1.0)
        self.assertLessEqual(self.summary["mttc"], MAX_TURNS + 1)
        self.assertGreaterEqual(self.summary["efficiency"], 0.0)
        self.assertLessEqual(self.summary["efficiency"], 1.0)

    def test_no_session_leaks_state_into_the_next(self) -> None:
        """Sessions share one Agent instance; turn 1 must always start clean."""
        for trace in self.traces:
            first = trace["turns"][0]["state"]
            with self.subTest(trace["sample_id"]):
                self.assertEqual(first["turn"], 1)
                self.assertLessEqual(len(first["asked_attributes"]), 1)


# =============================================================================
# INTEGRATION AGAINST THE REAL PUBLIC SET
# =============================================================================

REAL_CATALOG = Path("data/catalog.jsonl")
REAL_DATASET = Path("data/public_set.jsonl")


@unittest.skipUnless(
    REAL_CATALOG.exists() and REAL_DATASET.exists(),
    "data/catalog.jsonl not downloaded; see README",
)
class PublicSetIntegrationTest(unittest.TestCase):
    def test_real_sessions_produce_readable_conversation_history(self) -> None:
        samples = select_samples(load_jsonl(REAL_DATASET), 8, "stratified")
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory) / "logs"
            trace_runner.configure_logging(out_dir / "trace_run.log", verbose=False)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run(samples, REAL_CATALOG, out_dir)
            for handler in list(trace_runner.LOGGER.handlers):
                handler.close()
            logging.getLogger("trace").handlers.clear()

            self.assertEqual(summary["sample_count"], len(samples))
            traces = [json.loads(line) for line in (out_dir / "conversations.jsonl").read_text().splitlines()]
            self.assertEqual(len(traces), len(samples))
            for trace in traces:
                with self.subTest(trace["sample_id"]):
                    self.assertTrue(trace["target"]["title"])
                    self.assertTrue(trace["turns"][0]["user_message"].startswith("I'm looking for"))
                    for record in trace["turns"]:
                        self.assertIsInstance(record["state"]["search_query"], str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
