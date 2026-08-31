"""Shared session-replay logic for the diagnostic scripts in this package.

Everything here is read-only with respect to evaluator/ and data/: it imports the
official evaluator's own primitives (materialize_hidden_fields, initial_message,
customer_reply, normalize_recommendations, ...) instead of re-deriving the
dialogue-simulation rules, so behavior stays identical to evaluator.local_evaluator
whenever it's driven at top_k=10.

run_session() extends the official loop with two things the evaluator itself doesn't
expose: a full per-turn transcript, and a "recall" check against a candidate pool
deeper than the scored Top-10 (recall_probe.py drives this with top_k > 10). That
second use is a diagnostic-only extension of the Agent contract -- the real evaluator
always calls respond(..., top_k=10) (see evaluator/local_evaluator.py:240). It assumes
an Agent's ranking and ask_attribute choice do not depend on the top_k value it was
asked for (top_k only truncates output length). That holds for retrieval/portfolio
agents; re-check it if a component's behavior ever becomes top_k-sensitive.
"""

from __future__ import annotations

import importlib
import uuid
from typing import Any

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
)


def load_agent(spec: str, catalog_path: str) -> Any:
    """Import "module:ClassName" and construct it as ClassName(catalog_path), matching
    how evaluator.local_evaluator.main() constructs the Agent (evaluator/local_evaluator.py:306)."""
    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"--agent-import must be 'module:ClassName', got {spec!r}")
    module = importlib.import_module(module_name)
    agent_cls = getattr(module, class_name)
    return agent_cls(catalog_path)


def normalize_recommendations_k(payload: object, catalog_ids: set[str], k: int) -> list[str]:
    """Same de-dup/validity rules as evaluator.local_evaluator.normalize_recommendations,
    parameterized by cutoff instead of the hardcoded TOP_K, so probes can look deeper
    than the scored Top-10 without touching evaluator/."""
    if not isinstance(payload, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in payload:
        value = item.get("parent_asin", "") if isinstance(item, dict) else item
        parent_asin = str(value).strip()
        if not parent_asin or parent_asin in seen or parent_asin not in catalog_ids:
            continue
        seen.add(parent_asin)
        result.append(parent_asin)
        if len(result) >= k:
            break
    return result


def run_session(
    agent: Any,
    sample: dict,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    top_k: int,
) -> dict:
    """Replay one session, mirroring evaluator.local_evaluator.evaluate()'s per-sample
    loop turn for turn, plus a captured transcript and a recall check at `top_k` depth.

    At top_k=10 this reproduces the official hit/best_rank/reciprocal_rank exactly.
    """
    session_id = f"probe_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)

    hit_turn: int | None = None
    best_rank: int | None = None
    recall_hit_turn: int | None = None
    turns: list[dict] = []

    for turn in range(1, MAX_TURNS + 1):
        try:
            response = agent.respond(session_id, user_message, turn, top_k)
        except Exception:
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            response = {"message": "", "ask_attribute": None, "recommendations": []}

        candidates = normalize_recommendations_k(response.get("recommendations"), catalog_ids, top_k)
        scored = candidates[:TOP_K]

        if override_applied and recall_hit_turn is None and target in candidates:
            recall_hit_turn = turn
        if override_applied and target in scored:
            best_rank = scored.index(target) + 1
            hit_turn = turn

        turns.append({
            "turn": turn,
            "user_message": user_message,
            "message": response.get("message", ""),
            "ask_attribute": response.get("ask_attribute"),
            "recommendations_shown": scored,
            "candidate_pool_size": len(candidates),
            "target_in_candidate_pool": target in candidates,
            "target_in_shown_top10": target in scored,
        })

        if hit_turn is not None or turn == MAX_TURNS:
            break

        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                effective_sample, response.get("ask_attribute"), disclosed, boundary_used
            )

    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "target": target,
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        "recall_hit": recall_hit_turn is not None,
        "first_recall_turn": recall_hit_turn,
        "turns": turns,
    }
