"""Run N labelled sessions through the Agent and log the full conversation history.

This is a *diagnostic* wrapper, not a replacement scorer. It re-uses the exact
simulator primitives from ``evaluator.local_evaluator`` (initial message,
customer reply, override injection, recommendation normalisation, metrics) so a
trace reproduces what the official evaluator would have seen, while additionally
recording every turn: user message, agent message, asked attribute, accumulated
state, generated search query, the ranked Top-K and the rank of the target.

Outputs (default ``logs/``):

    conversations.jsonl   one JSON object per session, full turn-by-turn history
    conversations.md      human-readable transcripts
    summary.json          aggregate + per-scenario metrics for the traced subset
    trace_run.log         the run log

Usage::

    python3 -m tools.trace_runner                 # 100 stratified sessions
    python3 -m tools.trace_runner -v              # stream every turn to stderr
    python3 -m tools.trace_runner --limit 20 --select head
    python3 -m tools.trace_runner --scenario boundary --limit 10
    python3 -m tools.trace_runner --sample-ids public_0001,public_0002
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)
from starter.agent import Agent
from tools.customer_sim import RealisticCustomer


LOGGER = logging.getLogger("trace")
# Keeps stderr quiet when the harness is imported without configure_logging().
LOGGER.addHandler(logging.NullHandler())

SCENARIO_ORDER = ("buying", "browsing", "intent_override", "boundary")


# =============================================================================
# SAMPLE SELECTION
# =============================================================================


def select_samples(samples: list[dict], limit: int, mode: str = "stratified") -> list[dict]:
    """Pick ``limit`` samples deterministically.

    ``head``        first ``limit`` rows in file order.
    ``stratified``  proportional per ``scenario_type`` (largest-remainder), so a
                    100-case run still covers buying / browsing / override /
                    boundary in roughly their public-set proportions.
    """

    if limit >= len(samples):
        return list(samples)
    if mode == "head":
        return samples[:limit]

    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        groups[str(sample.get("scenario_type", "unknown"))].append(sample)

    names = sorted(groups, key=lambda name: (SCENARIO_ORDER.index(name) if name in SCENARIO_ORDER else 99, name))
    exact = {name: len(groups[name]) * limit / len(samples) for name in names}
    quota = {name: int(exact[name]) for name in names}

    remaining = limit - sum(quota.values())
    for name in sorted(names, key=lambda name: (-(exact[name] - quota[name]), name)):
        if remaining <= 0:
            break
        if quota[name] < len(groups[name]):
            quota[name] += 1
            remaining -= 1

    chosen_ids = {
        str(sample["sample_id"])
        for name in names
        for sample in groups[name][: quota[name]]
    }
    return [sample for sample in samples if str(sample["sample_id"]) in chosen_ids]


# =============================================================================
# STATE INTROSPECTION
# =============================================================================


def agent_state_snapshot(agent: Any, session_id: str) -> dict | None:
    """Best-effort read of the agent's accumulated state. Never raises."""

    manager = getattr(agent, "manager", None)
    if manager is None or not hasattr(manager, "export"):
        return None
    try:
        return manager.export(session_id)
    except Exception:  # pragma: no cover - diagnostics must not break the run
        LOGGER.warning("state export failed for %s", session_id, exc_info=True)
        return None


def state_history(agent: Any, session_id: str) -> list[dict]:
    manager = getattr(agent, "manager", None)
    if manager is None or not hasattr(manager, "get"):
        return []
    try:
        return list(getattr(manager.get(session_id), "history", []) or [])
    except Exception:  # pragma: no cover
        return []


# =============================================================================
# CUSTOMER ADAPTERS
# =============================================================================


class OfficialCustomer:
    """The evaluator's own deterministic customer policy, unchanged."""

    kind = "official"

    def __init__(self, effective_sample: dict, coarse: str) -> None:
        self.sample = effective_sample
        self.coarse = coarse
        self.disclosed: set[str] = set()
        self.boundary_used = False
        self._override = (effective_sample.get("behavior") or {}).get("override") or {}
        self.override_turn = int(self._override.get("turn", 3)) if self._override else None

    def opening(self) -> str:
        return initial_message(self.sample, self.coarse, self.disclosed)

    def override_message(self) -> tuple[str, str]:
        new_value = str(self._override.get("new_value", ""))
        if new_value:
            self.disclosed.add(new_value)
        message = str(self._override.get("message", "Actually, please ignore my earlier preference."))
        return message, new_value

    def reply(self, ask_attribute: object) -> str:
        message, self.boundary_used = customer_reply(
            self.sample, ask_attribute, self.disclosed, self.boundary_used
        )
        return message

    def describe(self) -> dict:
        return {
            "intent_card": self.sample.get("intent_card"),
            "behavior": self.sample.get("behavior"),
        }


class RealisticCustomerAdapter:
    """Human-phrased shopper from ``tools.customer_sim``, same interface."""

    kind = "realistic"

    def __init__(self, sample: dict, product: dict, coarse: str) -> None:
        self.inner = RealisticCustomer(sample, product, coarse)
        self.override_turn = (
            self.inner.override_turn if sample.get("scenario_type") == "intent_override" else None
        )

    @property
    def boundary_used(self) -> bool:
        return self.inner.boundary_used

    def opening(self) -> str:
        return self.inner.opening()

    def override_message(self) -> tuple[str, str]:
        return self.inner.override_message()

    def reply(self, ask_attribute: object) -> str:
        return self.inner.reply(ask_attribute)

    def describe(self) -> dict:
        return {
            "intent_card": {
                "target_category": self.inner.category,
                "hard_constraints": list(self.inner.facets.values())[:2],
                "soft_preferences": list(self.inner.facets.values())[2:4],
            },
            "behavior": {
                "scenario_type": self.inner.scenario,
                "override": (
                    {"turn": self.inner.override_turn, "new_value": self.inner._override_value}
                    if self.override_turn
                    else None
                ),
            },
            "facets": dict(self.inner.facets),
            "persona": dict(self.inner.persona),
            "mission_type": self.inner.mission,
        }


def build_customer(
    simulator: str,
    sample: dict,
    products: dict[str, dict],
    target: str,
    coarse: str,
) -> OfficialCustomer | RealisticCustomerAdapter:
    if simulator == "realistic":
        return RealisticCustomerAdapter(sample, products.get(target, {}), coarse)
    intent_card, behavior = materialize_hidden_fields(sample, products)
    return OfficialCustomer({**sample, "intent_card": intent_card, "behavior": behavior}, coarse)


# =============================================================================
# ONE SESSION
# =============================================================================


def run_session(
    agent: Any,
    sample: dict,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    simulator: str = "official",
) -> dict:
    """Replay one labelled session and return its full trace.

    The control flow mirrors ``evaluator.local_evaluator.evaluate`` turn for turn
    so the recorded hit / rank / MTTC match the official run.
    """

    sample_id = str(sample["sample_id"])
    scenario = str(sample["scenario_type"])
    session_id = f"trace_{sample_id}"
    target = str(sample["ground_truth"]["parent_asin"])

    agent.reset(session_id, sample["user_profile"])

    coarse = coarse_category(categories.get(target, []))
    customer = build_customer(simulator, sample, products, target, coarse)
    override_applied = scenario != "intent_override"
    user_message = customer.opening()

    turns: list[dict] = []
    hit_turn: int | None = None
    best_rank: int | None = None
    end_reason = "turn_limit"
    prompt_tokens = 0
    completion_tokens = 0
    started = time.perf_counter()

    LOGGER.info("=== %s | %s | target=%s (%s)", sample_id, scenario, target, coarse)

    for turn in range(1, MAX_TURNS + 1):
        error: str | None = None
        turn_started = time.perf_counter()
        try:
            response = agent.respond(session_id, user_message, turn, TOP_K)
        except Exception as exc:  # evaluator swallows this too; we record it
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("%s turn %d: agent raised %s", sample_id, turn, error, exc_info=True)
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            error = error or "malformed response coerced to empty"
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        latency_ms = round((time.perf_counter() - turn_started) * 1000, 3)

        usage = response.get("usage")
        turn_prompt = turn_completion = 0
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] >= 0:
                turn_prompt = usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), int) and usage["completion_tokens"] >= 0:
                turn_completion = usage["completion_tokens"]
        prompt_tokens += turn_prompt
        completion_tokens += turn_completion

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        target_rank = ranked.index(target) + 1 if target in ranked else None
        state = agent_state_snapshot(agent, session_id)

        record = {
            "turn": turn,
            "user_message": user_message,
            "agent_message": response.get("message", ""),
            "ask_attribute": response.get("ask_attribute"),
            "recommendations": ranked,
            "recommendation_titles": [
                str(products.get(pid, {}).get("title", ""))[:80] for pid in ranked[:3]
            ],
            "target_rank": target_rank,
            "target_in_top_k": target_rank is not None,
            "scored": override_applied,
            "state": state,
            "usage": {"prompt_tokens": turn_prompt, "completion_tokens": turn_completion},
            "latency_ms": latency_ms,
            "error": error,
            "event": None,
        }
        turns.append(record)

        LOGGER.info(
            "  t%-2d USER  %s",
            turn,
            user_message[:140],
        )
        if state:
            LOGGER.info(
                "      STATE intent=%s next=%s constraints=%s no_pref=%s",
                state.get("intent"),
                state.get("next_action"),
                state.get("constraints"),
                state.get("no_preference"),
            )
            LOGGER.info("      QUERY %r", state.get("search_query"))
        LOGGER.info(
            "      AGENT ask=%s | %s",
            response.get("ask_attribute"),
            str(response.get("message", ""))[:120],
        )
        LOGGER.info(
            "      TOP%-2d %s%s",
            TOP_K,
            " ".join(ranked[:TOP_K]) or "(empty)",
            f" | target rank {target_rank}" if target_rank else "",
        )

        # --- scoring: only counts once the override (if any) has been issued ---
        if override_applied and target in ranked:
            best_rank = target_rank
            hit_turn = turn
            end_reason = "hit"
            record["event"] = "hit"
            LOGGER.info("  --> HIT %s at turn %d rank %d", sample_id, turn, best_rank)
            break

        if turn == MAX_TURNS:
            break

        # --- build the next user message the same way the evaluator does ---
        if (
            not override_applied
            and customer.override_turn is not None
            and turn + 1 == customer.override_turn
        ):
            override_applied = True
            user_message, _ = customer.override_message()
            record["event"] = "override_injected_next_turn"
            LOGGER.info("      EVENT override injected for turn %d", turn + 1)
        else:
            before = customer.boundary_used
            user_message = customer.reply(response.get("ask_attribute"))
            if customer.boundary_used and not before:
                record["event"] = "boundary_reply_next_turn"

    return {
        "sample_id": sample_id,
        "scenario_type": scenario,
        "difficulty_bucket": sample.get("difficulty_bucket"),
        "category_bucket": sample.get("category_bucket"),
        "user_profile": sample.get("user_profile"),
        "target": {
            "parent_asin": target,
            "title": str(products.get(target, {}).get("title", "")),
            "price": products.get(target, {}).get("price"),
            "coarse_category": coarse,
        },
        "simulator": customer.kind,
        **customer.describe(),
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        "end_reason": end_reason,
        "turn_count": len(turns),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "final_state": agent_state_snapshot(agent, session_id),
        "state_history": state_history(agent, session_id),
        "turns": turns,
    }


# =============================================================================
# AGGREGATION + RENDERING
# =============================================================================


def summarize(traces: list[dict]) -> dict:
    sessions = [
        {
            "sample_id": trace["sample_id"],
            "scenario_type": trace["scenario_type"],
            "hit": trace["hit"],
            "first_hit_turn": trace["first_hit_turn"],
            "best_rank": trace["best_rank"],
            "reciprocal_rank": trace["reciprocal_rank"],
        }
        for trace in traces
    ]
    overall = metric_summary(sessions)
    efficiency = 0.0
    if overall["mttc"] is not None:
        efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency

    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)

    prompt_tokens = sum(trace["usage"]["prompt_tokens"] for trace in traces)
    completion_tokens = sum(trace["usage"]["completion_tokens"] for trace in traces)

    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "reported_token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
        "sessions": sessions,
    }


def render_transcript(trace: dict) -> str:
    lines: list[str] = []
    header = f"## {trace['sample_id']} · {trace['scenario_type']} · {trace.get('difficulty_bucket') or '-'}"
    lines.append(header)
    target = trace["target"]
    lines.append(f"target : {target['parent_asin']} — {target['title'][:100]}")
    lines.append(f"coarse : {target['coarse_category']}")
    if trace.get("difficulty_bucket"):
        lines.append(f"level  : {trace['difficulty_bucket']}"
                     + (f" · {trace['mission_type']}" if trace.get("mission_type") else ""))
    if trace.get("facets"):
        lines.append(f"facets : {trace['facets']}")
        persona = trace.get("persona") or {}
        if persona:
            lines.append(f"persona: {persona}")
    else:
        card = trace.get("intent_card") or {}
        lines.append(f"hard   : {card.get('hard_constraints')}")
        lines.append(f"soft   : {card.get('soft_preferences')}")
    override = (trace.get("behavior") or {}).get("override")
    if override:
        lines.append(f"override@turn {override.get('turn')} -> {override.get('new_value')}")
    lines.append("")

    for record in trace["turns"]:
        lines.append(f"Turn {record['turn']}")
        lines.append(f"  USER  : {record['user_message']}")
        state = record.get("state") or {}
        if state:
            lines.append(
                "  STATE : intent={intent} next={next_action} constraints={constraints}".format(
                    intent=state.get("intent"),
                    next_action=state.get("next_action"),
                    constraints=state.get("constraints"),
                )
            )
            lines.append(
                f"          no_pref={state.get('no_preference')} asked={state.get('asked_attributes')}"
            )
            lines.append(f"  QUERY : {state.get('search_query')!r}")
        lines.append(f"  AGENT : {record['agent_message']}   [ask={record['ask_attribute']}]")
        rank = f"  <- target @ {record['target_rank']}" if record["target_rank"] else ""
        lines.append(f"  TOP{TOP_K}: {' '.join(record['recommendations']) or '(empty)'}{rank}")
        for title in record["recommendation_titles"]:
            lines.append(f"          · {title}")
        if not record["scored"]:
            lines.append("  NOTE  : pre-override turn — hits do not score here")
        if record["error"]:
            lines.append(f"  ERROR : {record['error']}")
        if record["event"]:
            lines.append(f"  EVENT : {record['event']}")
        lines.append("")

    if trace["hit"]:
        lines.append(f"RESULT : HIT at turn {trace['first_hit_turn']}, rank {trace['best_rank']}, RR={trace['reciprocal_rank']:.4f}")
    else:
        lines.append(f"RESULT : MISS after {trace['turn_count']} turns ({trace['end_reason']})")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def configure_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(stream_handler)


# =============================================================================
# ENTRY POINT
# =============================================================================


def run(
    samples: list[dict],
    catalog_path: str | Path,
    out_dir: Path,
    simulator: str = "official",
) -> dict:
    catalog_ids, categories, products = catalog_index(catalog_path)
    LOGGER.info("catalog indexed: %d products | simulator=%s", len(catalog_ids), simulator)

    build_started = time.perf_counter()
    agent = Agent(catalog_path)
    LOGGER.info("agent built in %.2fs", time.perf_counter() - build_started)

    traces: list[dict] = []
    for index, sample in enumerate(samples, start=1):
        trace = run_session(agent, sample, catalog_ids, categories, products, simulator)
        traces.append(trace)
        print(
            f"[{index:>3}/{len(samples)}] {trace['sample_id']} {trace['scenario_type']:<15} "
            f"{'HIT ' if trace['hit'] else 'MISS'} "
            f"turn={trace['first_hit_turn'] or '-':<3} rank={trace['best_rank'] or '-':<3} "
            f"{trace['duration_ms']:.0f}ms",
            flush=True,
        )

    summary = summarize(traces)

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "conversations.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")

    md_path = out_dir / "conversations.md"
    md_path.write_text(
        f"# Conversation traces ({len(traces)} sessions)\n\n" + "".join(render_transcript(t) for t in traces),
        encoding="utf-8",
    )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    LOGGER.info("wrote %s, %s, %s", jsonl_path, md_path, summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run labelled sessions and log full conversation history")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--out-dir", default="logs")
    parser.add_argument("--limit", type=int, default=100, help="number of test cases (default 100)")
    parser.add_argument("--select", choices=("stratified", "head"), default="stratified")
    parser.add_argument(
        "--simulator",
        choices=("official", "realistic"),
        default="official",
        help="official = evaluator templates; realistic = human-phrased shopper",
    )
    parser.add_argument("--scenario", default=None, help="filter to one scenario_type before selection")
    parser.add_argument("--sample-ids", default=None, help="comma-separated sample_ids; overrides --limit/--select")
    parser.add_argument("-v", "--verbose", action="store_true", help="stream every turn to stderr")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    configure_logging(out_dir / "trace_run.log", args.verbose)

    samples = load_jsonl(args.dataset)
    if args.scenario:
        samples = [s for s in samples if str(s.get("scenario_type")) == args.scenario]
    if args.sample_ids:
        wanted = {item.strip() for item in args.sample_ids.split(",") if item.strip()}
        samples = [s for s in samples if str(s["sample_id"]) in wanted]
    else:
        samples = select_samples(samples, args.limit, args.select)

    if not samples:
        parser.error("no samples selected")

    LOGGER.info("selected %d samples | simulator=%s", len(samples), args.simulator)
    summary = run(samples, args.catalog, out_dir, args.simulator)

    print(json.dumps({k: v for k, v in summary.items() if k != "sessions"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
