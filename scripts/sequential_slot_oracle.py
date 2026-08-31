#!/usr/bin/env python3
"""Exact, dependency-free oracle for ShopLens's scored recommendation slots.

This is a decision-support tool, not an online policy.  Its allocation result
is optimal only when candidate beliefs stay fixed and a product is shown at
most once.  See ``docs/sequential-slate-allocation.md`` for the assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence


MAX_TURNS = 10
MAX_RANK = 10
MISS_TURN = 11

HIT_RATE_WEIGHT = Fraction(1, 2)
MRR_WEIGHT = Fraction(3, 10)
EFFICIENCY_WEIGHT = Fraction(1, 5)


def _bounded_integer(value: int, *, name: str, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 1 <= value <= upper:
        raise ValueError(f"{name} must be between 1 and {upper}")
    return value


def hit_utility(turn: int, rank: int) -> Fraction:
    """Return one hit's exact contribution to mean TechnicalScore.

    The evaluator assigns a miss turn of 11.  For a hit at ``(turn, rank)``,
    the per-session contribution is therefore

        1/2 + (3/10) / rank + (1/5) * (11 - turn) / 10.

    A ``Fraction`` is returned so comparisons and tie handling are exact.
    """

    valid_turn = _bounded_integer(turn, name="turn", upper=MAX_TURNS)
    valid_rank = _bounded_integer(rank, name="rank", upper=MAX_RANK)
    return (
        HIT_RATE_WEIGHT
        + MRR_WEIGHT / valid_rank
        + EFFICIENCY_WEIGHT * Fraction(MISS_TURN - valid_turn, MAX_TURNS)
    )


def session_utility(
    *, hit: bool, turn: int | None = None, rank: int | None = None
) -> Fraction:
    """Return exact per-session utility, with every miss worth exactly zero."""

    if not hit:
        if turn is not None or rank is not None:
            raise ValueError("a miss must not supply a turn or rank")
        return Fraction(0)
    if turn is None or rank is None:
        raise ValueError("a hit requires both turn and rank")
    return hit_utility(turn, rank)


@dataclass(frozen=True, slots=True)
class Slot:
    turn: int
    rank: int
    utility: Fraction


@dataclass(frozen=True, slots=True)
class Assignment:
    """One candidate belief assigned to one scored opportunity."""

    candidate: int
    belief: float
    slot: Slot

    @property
    def contribution(self) -> float:
        return self.belief * float(self.slot.utility)


@dataclass(frozen=True, slots=True)
class Headroom:
    """Idealized score decomposition reconstructed from aggregate metrics."""

    efficiency: float
    reconstructed_score: float
    ranking: float
    timing: float
    fixed_membership_oracle: float
    membership_to_perfect: float
    total_to_perfect: float


def slots_by_utility(turns: int = MAX_TURNS, ranks: int = MAX_RANK) -> list[Slot]:
    """Return slots in the order required by the fixed-belief optimum."""

    valid_turns = _bounded_integer(turns, name="turns", upper=MAX_TURNS)
    valid_ranks = _bounded_integer(ranks, name="ranks", upper=MAX_RANK)
    slots = [
        Slot(turn, rank, hit_utility(turn, rank))
        for turn in range(1, valid_turns + 1)
        for rank in range(1, valid_ranks + 1)
    ]
    return sorted(slots, key=lambda slot: (-slot.utility, slot.turn, slot.rank))


def chronological_slots(
    turns: int = MAX_TURNS, ranks: int = MAX_RANK
) -> list[Slot]:
    """Return the tempting turn-first order used as a comparison, not an oracle."""

    valid_turns = _bounded_integer(turns, name="turns", upper=MAX_TURNS)
    valid_ranks = _bounded_integer(ranks, name="ranks", upper=MAX_RANK)
    return [
        Slot(turn, rank, hit_utility(turn, rank))
        for turn in range(1, valid_turns + 1)
        for rank in range(1, valid_ranks + 1)
    ]


def _candidate_beliefs(values: Sequence[float]) -> list[tuple[int, float]]:
    if not values:
        raise ValueError("at least one candidate belief is required")
    beliefs: list[tuple[int, float]] = []
    for candidate, raw_value in enumerate(values, start=1):
        if isinstance(raw_value, bool):
            raise ValueError(f"belief {candidate} must be a finite non-negative number")
        try:
            belief = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"belief {candidate} must be a finite non-negative number"
            ) from exc
        if not math.isfinite(belief) or belief < 0:
            raise ValueError(f"belief {candidate} must be a finite non-negative number")
        beliefs.append((candidate, belief))
    # Stable candidate index is an explicit, deterministic tie-break.  It does
    # not change the objective when beliefs are equal.
    return sorted(beliefs, key=lambda item: (-item[1], item[0]))


def _allocate(values: Sequence[float], slots: Sequence[Slot]) -> list[Assignment]:
    candidates = _candidate_beliefs(values)
    return [
        Assignment(candidate=candidate, belief=belief, slot=slot)
        for (candidate, belief), slot in zip(candidates, slots)
    ]


def optimal_allocation(
    beliefs: Sequence[float], turns: int = MAX_TURNS, ranks: int = MAX_RANK
) -> list[Assignment]:
    """Pair descending beliefs with descending slot utilities globally.

    If there are more candidates than slots, the lowest-belief candidates are
    left unassigned.  If there are fewer, only the highest-utility slots are
    used.  Beliefs may be unnormalised relevance weights; the objective has an
    expected-score interpretation only when they are mutually exclusive target
    probabilities whose total is at most one.
    """

    return _allocate(beliefs, slots_by_utility(turns, ranks))


def chronological_allocation(
    beliefs: Sequence[float], turns: int = MAX_TURNS, ranks: int = MAX_RANK
) -> list[Assignment]:
    """Fill turn 1 before turn 2, for comparison with the global optimum."""

    return _allocate(beliefs, chronological_slots(turns, ranks))


def allocation_value(assignments: Sequence[Assignment]) -> float:
    return math.fsum(item.contribution for item in assignments)


def score_headroom(hit_rate: float, mrr: float, mttc: float) -> Headroom:
    """Decompose the gap above aggregate metrics into idealized components.

    ``fixed_membership_oracle`` moves every existing hit to turn 1, rank 1 but
    leaves misses as misses. ``membership_to_perfect`` then recovers every
    remaining miss at turn 1, rank 1. These are ceilings, not forecasts.
    """

    values = {"hit_rate": hit_rate, "mrr": mrr, "mttc": mttc}
    converted: dict[str, float] = {}
    for name, raw_value in values.items():
        if isinstance(raw_value, bool):
            raise ValueError(f"{name} must be finite")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        converted[name] = value

    h = converted["hit_rate"]
    reciprocal_rank = converted["mrr"]
    mean_turn = converted["mttc"]
    if not 0 <= h <= 1:
        raise ValueError("hit_rate must be between 0 and 1")
    if not 0 <= reciprocal_rank <= h + 1e-6:
        raise ValueError("mrr must be between 0 and hit_rate")
    if reciprocal_rank + 1e-6 < h / MAX_RANK:
        raise ValueError("metrics are inconsistent: mrr is too low for the hit rate")
    if not 1 <= mean_turn <= MISS_TURN:
        raise ValueError(f"mttc must be between 1 and {MISS_TURN}")

    efficiency = (MISS_TURN - mean_turn) / MAX_TURNS
    # Rounded aggregate inputs can disagree in the seventh decimal place.
    if efficiency > h + 1e-6:
        raise ValueError("metrics are inconsistent: efficiency exceeds hit_rate")
    if efficiency + 1e-6 < h / MAX_TURNS:
        raise ValueError("metrics are inconsistent: efficiency is too low for the hit rate")
    score = 0.50 * h + 0.30 * reciprocal_rank + 0.20 * efficiency
    ranking = 0.30 * (h - reciprocal_rank)
    timing = 0.20 * (h - efficiency)
    # Algebraically this equals score + ranking + timing. Use h directly so
    # the invariant stays visible despite binary floating-point arithmetic.
    fixed_membership_oracle = h
    membership_to_perfect = 1.0 - h
    return Headroom(
        efficiency=efficiency,
        reconstructed_score=score,
        ranking=ranking,
        timing=timing,
        fixed_membership_oracle=fixed_membership_oracle,
        membership_to_perfect=membership_to_perfect,
        total_to_perfect=1.0 - score,
    )


def _parse_beliefs(raw_value: str) -> list[float]:
    parts = [part.strip() for part in raw_value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError("beliefs must be comma-separated numbers")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("beliefs must be comma-separated numbers") from exc
    try:
        _candidate_beliefs(values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return values


def _assignment_dict(item: Assignment) -> dict[str, object]:
    return {
        "candidate": item.candidate,
        "belief": item.belief,
        "turn": item.slot.turn,
        "rank": item.slot.rank,
        "utility": float(item.slot.utility),
        "utility_exact": str(item.slot.utility),
        "contribution": item.contribution,
    }


def _print_allocation_table(title: str, assignments: Sequence[Assignment]) -> None:
    print(title)
    print("candidate  belief    turn  rank  utility  contribution")
    for item in assignments:
        print(
            f"{item.candidate:>9}  {item.belief:>7.4f}  "
            f"{item.slot.turn:>4}  {item.slot.rank:>4}  "
            f"{float(item.slot.utility):>7.4f}  {item.contribution:>12.6f}"
        )
    print(f"objective: {allocation_value(assignments):.6f}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact fixed-belief slot oracle for the ShopLens TechnicalScore"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    utility = subparsers.add_parser("utility", help="score one hit or a miss")
    utility.add_argument("--miss", action="store_true", help="return the miss utility")
    utility.add_argument("--turn", type=int)
    utility.add_argument("--rank", type=int)
    utility.add_argument("--format", choices=("text", "json"), default="text")

    allocate = subparsers.add_parser(
        "allocate", help="compare global utility ordering with chronological fill"
    )
    allocate.add_argument("--beliefs", required=True, type=_parse_beliefs)
    allocate.add_argument("--turns", type=int, default=MAX_TURNS)
    allocate.add_argument("--ranks", type=int, default=MAX_RANK)
    allocate.add_argument("--format", choices=("text", "json"), default="text")

    headroom = subparsers.add_parser(
        "headroom", help="decompose idealized headroom from aggregate metrics"
    )
    headroom.add_argument("--hit-rate", required=True, type=float)
    headroom.add_argument("--mrr", required=True, type=float)
    headroom.add_argument("--mttc", required=True, type=float)
    headroom.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "utility":
            if args.miss:
                if args.turn is not None or args.rank is not None:
                    parser.error("--miss cannot be combined with --turn or --rank")
                value = session_utility(hit=False)
                payload = {"hit": False, "utility": 0.0, "utility_exact": "0"}
            else:
                if args.turn is None or args.rank is None:
                    parser.error("a hit requires --turn and --rank")
                value = session_utility(hit=True, turn=args.turn, rank=args.rank)
                payload = {
                    "hit": True,
                    "turn": args.turn,
                    "rank": args.rank,
                    "utility": float(value),
                    "utility_exact": str(value),
                }
            if args.format == "json":
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"utility = {float(value):.6f} (exactly {value})")
            return 0

        if args.command == "allocate":
            optimal = optimal_allocation(args.beliefs, args.turns, args.ranks)
            chronological = chronological_allocation(
                args.beliefs, args.turns, args.ranks
            )
            optimal_value = allocation_value(optimal)
            chronological_value = allocation_value(chronological)
            payload = {
                "assumption": "fixed beliefs; each candidate shown at most once",
                "belief_sum": math.fsum(args.beliefs),
                "expected_score_interpretation": math.fsum(args.beliefs) <= 1.0 + 1e-12,
                "optimal": {
                    "objective": optimal_value,
                    "assignments": [_assignment_dict(item) for item in optimal],
                },
                "chronological": {
                    "objective": chronological_value,
                    "assignments": [_assignment_dict(item) for item in chronological],
                },
                "optimality_gap": optimal_value - chronological_value,
            }
            if args.format == "json":
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_allocation_table("Global utility order (optimal)", optimal)
                print()
                _print_allocation_table("Chronological turn-first fill", chronological)
                print(f"optimality gap: {payload['optimality_gap']:.6f}")
                if not payload["expected_score_interpretation"]:
                    print(
                        "note: beliefs sum above one; ordering remains valid, but the "
                        "objective is not an expected TechnicalScore"
                    )
            return 0

        metrics = score_headroom(args.hit_rate, args.mrr, args.mttc)
        payload = {
            "efficiency": metrics.efficiency,
            "reconstructed_score": metrics.reconstructed_score,
            "ranking_headroom_fixed_hits": metrics.ranking,
            "timing_headroom_fixed_hits": metrics.timing,
            "fixed_membership_oracle_score": metrics.fixed_membership_oracle,
            "membership_headroom_to_perfect": metrics.membership_to_perfect,
            "total_headroom_to_perfect": metrics.total_to_perfect,
        }
        if args.format == "json":
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for name, value in payload.items():
                print(f"{name}: {value:.6f}")
        return 0
    except ValueError as exc:
        parser.error(str(exc))
    return 2  # argparse.error raises; this keeps the return type explicit.


if __name__ == "__main__":
    raise SystemExit(main())
