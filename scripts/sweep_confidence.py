"""Ship-gate sweep harness (Step 3).

Runs the 200-session evaluator across policy arms and prints one comparison
table. The evaluator is left untouched: we import ``evaluate`` and construct the
agent per arm.

    P0: always-ask-until-exhausted (champion baseline)
    P1: theta=0.3   P2: theta=0.5   P3: theta=0.7

SHIP RULE:
    - best P-theta >= P0 (within TOL) -> ship the confidence policy, record theta
    - all P-theta < P0                -> keep always-ask; ship confidence as a
                                         reported-only introspection signal

NOTE: This requires a champion Agent that accepts a policy configuration
(``policy="always_ask"`` or ``policy="confidence", theta=...``). Until that
agent lands on the integration branch, ``--agent-module`` defaults to the
starter baseline purely so the harness itself is runnable; the *numbers* only
become meaningful against the real reranker-backed agent.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

TOL = 0.002
TECH_SCORE_KEY = "recommended_technical_score"


def _build_agent(agent_module: str, catalog_path: str, **agent_kwargs):
    """Construct an agent from ``module:ClassName`` with optional kwargs."""
    module_name, _, class_name = agent_module.partition(":")
    class_name = class_name or "Agent"
    module = importlib.import_module(module_name)
    agent_cls = getattr(module, class_name)
    try:
        return agent_cls(catalog_path, **agent_kwargs)
    except TypeError:
        # Baseline agent takes only the catalog path.
        if agent_kwargs:
            return agent_cls(catalog_path)
        raise


ARMS: list[tuple[str, dict]] = [
    ("P0_always_ask", {"policy": "always_ask"}),
    ("P1_theta_0.3", {"policy": "confidence", "theta": 0.3}),
    ("P2_theta_0.5", {"policy": "confidence", "theta": 0.5}),
    ("P3_theta_0.7", {"policy": "confidence", "theta": 0.7}),
]


def run(agent_module: str, catalog: str, dataset: str, output: str) -> dict:
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)

    arm_results: dict[str, dict] = {}
    for arm_name, kwargs in ARMS:
        agent = _build_agent(agent_module, catalog, **kwargs)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        arm_results[arm_name] = {
            k: result[k]
            for k in ("hit_rate_at_10", "mrr", "mttc", "efficiency", TECH_SCORE_KEY)
        }
        arm_results[arm_name]["scenario_metrics"] = result["scenario_metrics"]

    decision = _apply_ship_rule(arm_results)
    report = {"arms": arm_results, "decision": decision}
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_table(arm_results, decision)
    return report


def _apply_ship_rule(arm_results: dict) -> dict:
    p0 = arm_results["P0_always_ask"][TECH_SCORE_KEY]
    theta_arms = {k: v for k, v in arm_results.items() if k != "P0_always_ask"}
    best_arm = max(theta_arms, key=lambda k: theta_arms[k][TECH_SCORE_KEY])
    best_score = theta_arms[best_arm][TECH_SCORE_KEY]

    if best_score >= p0 - TOL:
        return {
            "ship": "confidence_policy",
            "chosen_arm": best_arm,
            "best_technical_score": best_score,
            "p0_technical_score": p0,
            "note": f"{best_arm} >= P0 within {TOL}; ship confidence policy.",
        }
    return {
        "ship": "always_ask_scored__confidence_reported_only",
        "chosen_arm": "P0_always_ask",
        "best_technical_score": best_score,
        "p0_technical_score": p0,
        "note": "All theta arms below P0; keep always-ask, report confidence as introspection.",
    }


def _print_table(arm_results: dict, decision: dict) -> None:
    header = f"{'arm':<18}{'TS':>10}{'HR@10':>10}{'MRR':>10}{'MTTC':>10}"
    print(header)
    print("-" * len(header))
    for arm, m in arm_results.items():
        print(
            f"{arm:<18}{m[TECH_SCORE_KEY]:>10.4f}{m['hit_rate_at_10']:>10.4f}"
            f"{m['mrr']:>10.4f}{m['mttc']:>10.4f}"
        )
    print("-" * len(header))
    print(f"DECISION: {decision['ship']} ({decision['note']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence policy ship-gate sweep")
    parser.add_argument("--agent-module", default="starter.agent:Agent",
                        help="module:ClassName of the champion agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="sweep_results.json")
    args = parser.parse_args()
    run(args.agent_module, args.catalog, args.dataset, args.output)


if __name__ == "__main__":
    main()
