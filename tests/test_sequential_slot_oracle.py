from __future__ import annotations

import itertools
import json
from fractions import Fraction

import pytest

from scripts.sequential_slot_oracle import (
    allocation_value,
    chronological_allocation,
    hit_utility,
    main,
    optimal_allocation,
    score_headroom,
    session_utility,
    slots_by_utility,
)


def test_hit_utility_matches_the_evaluator_formula_exactly() -> None:
    assert hit_utility(1, 1) == Fraction(1, 1)
    assert hit_utility(10, 10) == Fraction(11, 20)
    assert session_utility(hit=False) == 0


def test_average_session_utility_equals_metric_aggregation() -> None:
    outcomes = [
        session_utility(hit=True, turn=1, rank=1),
        session_utility(hit=True, turn=4, rank=2),
        session_utility(hit=False),
    ]
    direct_average = sum(outcomes, Fraction(0)) / len(outcomes)

    hit_rate = Fraction(2, 3)
    mrr = Fraction(1, 3) * (Fraction(1, 1) + Fraction(1, 2))
    mttc = Fraction(1 + 4 + 11, 3)
    efficiency = (11 - mttc) / 10
    metric_score = Fraction(1, 2) * hit_rate + Fraction(3, 10) * mrr + Fraction(1, 5) * efficiency

    assert direct_average == metric_score


def test_next_turn_rank_one_dominates_current_turn_rank_ten() -> None:
    for turn in range(1, 10):
        assert hit_utility(turn + 1, 1) - hit_utility(turn, 10) == Fraction(1, 4)


def test_global_slot_order_is_not_chronological() -> None:
    first_four = [(slot.turn, slot.rank) for slot in slots_by_utility(2, 2)]
    assert first_four == [(1, 1), (2, 1), (1, 2), (2, 2)]


def test_global_allocation_beats_turn_first_fill() -> None:
    beliefs = [0.4, 0.3, 0.2, 0.1]
    optimal = optimal_allocation(beliefs, turns=2, ranks=2)
    chronological = chronological_allocation(beliefs, turns=2, ranks=2)

    assert allocation_value(optimal) == pytest.approx(0.947)
    assert allocation_value(chronological) == pytest.approx(0.934)
    assert allocation_value(optimal) > allocation_value(chronological)


def test_rearrangement_solution_matches_brute_force_optimum() -> None:
    beliefs = [0.4, 0.3, 0.2, 0.1]
    slots = slots_by_utility(2, 2)
    brute_force = max(
        sum(belief * float(slot.utility) for belief, slot in zip(beliefs, permutation))
        for permutation in itertools.permutations(slots)
    )

    assert allocation_value(optimal_allocation(beliefs, 2, 2)) == pytest.approx(brute_force)


def test_candidate_identity_survives_belief_sorting() -> None:
    assignments = optimal_allocation([0.1, 0.7, 0.2], turns=1, ranks=3)

    assert [item.candidate for item in assignments] == [2, 3, 1]
    assert [(item.slot.turn, item.slot.rank) for item in assignments] == [
        (1, 1),
        (1, 2),
        (1, 3),
    ]


def test_t_dev_headroom_reconstructs_the_reported_score() -> None:
    headroom = score_headroom(0.941667, 0.795913, 3.141667)

    assert headroom.efficiency == pytest.approx(0.7858333)
    assert headroom.reconstructed_score == pytest.approx(0.86677406)
    assert headroom.ranking == pytest.approx(0.0437262)
    assert headroom.timing == pytest.approx(0.03116674)
    assert headroom.fixed_membership_oracle == pytest.approx(0.941667)
    assert headroom.membership_to_perfect == pytest.approx(0.058333)
    assert headroom.total_to_perfect == pytest.approx(
        headroom.ranking + headroom.timing + headroom.membership_to_perfect
    )


@pytest.mark.parametrize(
    ("hit_rate", "mrr", "mttc", "message"),
    [
        (1.0, 0.09, 10.0, "mrr is too low"),
        (1.0, 0.10, 10.1, "efficiency is too low"),
    ],
)
def test_impossible_metric_combinations_are_rejected(
    hit_rate: float, mrr: float, mttc: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        score_headroom(hit_rate, mrr, mttc)


@pytest.mark.parametrize(
    ("beliefs", "message"),
    [([], "at least one"), ([-0.1], "non-negative"), ([float("nan")], "finite")],
)
def test_invalid_beliefs_are_rejected(beliefs: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        optimal_allocation(beliefs)


def test_json_cli_exposes_assumptions_and_comparator(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "allocate",
            "--beliefs",
            "0.4,0.3,0.2,0.1",
            "--turns",
            "2",
            "--ranks",
            "2",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["expected_score_interpretation"] is True
    assert payload["optimal"]["objective"] == pytest.approx(0.947)
    assert payload["optimality_gap"] == pytest.approx(0.013)
