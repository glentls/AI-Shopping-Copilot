"""Paraphrase stress test: reword the evaluator's four reply templates and
confirm the pipeline degrades gracefully rather than collapsing.

The public set uses fixed template strings; the private 800 explicitly reserves
the right to paraphrase (docs/competition_specification.md:40). This harness
monkeypatches the evaluator's ``initial_message`` and ``customer_reply`` to emit
reworded variants -- dropped markers, reordered clauses, synonym wrappers --
then runs the full protocol. The bucket ladder + transcript fallback should
keep the score well above a collapse.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluator.local_evaluator as ev  # noqa: E402
from src.agent import Agent  # noqa: E402


def _reword_opening(sample, category, disclosed):
    scenario = sample["scenario_type"]
    if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
        c = str(sample["intent_card"]["hard_constraints"][0])
        disclosed.add(c)
        # Marker dropped, wrapper reworded, clause reordered.
        return f"Hey, I really need {c} and I'm after some {category} today."
    if scenario == "intent_override":
        old = str(sample["behavior"]["override"]["old_value"])
        return f"Hi there, show me some {category} - {old}"
    return f"just browsing for {category} at the moment"


def _reword_reply(sample, ask_attribute, disclosed, boundary_used):
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
        return f"no strong feeling on {attribute}, your call", True
    if not attribute:
        return "hmm those aren't right, ask me something specific", boundary_used
    if attribute not in ev.ALLOWED_ATTRIBUTES:
        attribute = "other"
    constraints = [
        *[str(v) for v in sample["intent_card"].get("hard_constraints", [])],
        *[str(v) for v in sample["intent_card"].get("soft_preferences", [])],
    ]
    matches = [
        v for v in constraints
        if v not in disclosed and (attribute == "other" or ev.classify_constraint(v) == attribute)
    ][:2]
    if not matches:
        return f"nothing else on {attribute} sorry", boundary_used
    disclosed.update(matches)
    # Marker "For that, what matters is:" dropped; joined with "and" not ";".
    return "well " + " and also ".join(matches) + " would be great", boundary_used


def main() -> None:
    ev.initial_message = _reword_opening
    ev.customer_reply = _reword_reply

    catalog = "data/catalog.jsonl"
    samples = ev.load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = ev.catalog_index(catalog)
    result = ev.evaluate(Agent(catalog), samples, catalog_ids, categories, products)
    print("paraphrased-template run:")
    for k in ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score"):
        print(f"  {k:32s} {result[k]}")
    score = result["recommended_technical_score"]
    print(f"\n{'PASS' if score >= 0.80 else 'BELOW 0.80'}: degraded score {score:.4f} (target >= 0.80)")


if __name__ == "__main__":
    main()
