"""Run the adversarial input corpus and report what breaks.

    python3 -m tools.robustness                      # pass rate by characteristic
    python3 -m tools.robustness --failures            # every failing case
    python3 -m tools.robustness --tag numeric_not_budget --failures
    python3 -m tools.robustness --soft                # include debatable cases

The public evaluator never feeds the agent a sentence a person would type, so
none of this is visible in the TechnicalScore. A failure here is a bug the
scored loop cannot see.

Hard cases assert behaviour that should not be controversial. Cases marked
``soft`` in the corpus are genuinely debatable -- non-USD currency, spelled-out
numbers, idioms -- and are reported separately so they never gate CI.
"""

from __future__ import annotations

import argparse
import collections
import sys

from src.contracts import ConversationState
from src.extract import parse_budget
from src.policy.state import record_question, update
from src.policy.question import choose_question
from tests.robustness_cases import CASES, DIALOGUES, SKIP, Case, Dialogue


def evaluate(case: Case) -> list[str]:
    """Every way this case failed. Empty list means it passed."""
    problems: list[str] = []

    state = ConversationState(session_id="robustness", user_profile={})
    try:
        update(state, case.text, 1)
    except Exception as error:  # a raised exception is a MISS in the evaluator
        return [f"RAISED {type(error).__name__}: {error}"]

    for slot, value in case.expect:
        active = state.active(slot)
        if value not in active:
            problems.append(f"missing {slot}={value!r} (got {active or 'nothing'})")

    for rule in case.forbid:
        slot, _, value = rule.partition(":")
        active = state.active(slot)
        if value:
            if value in active:
                problems.append(f"should not hold {slot}={value!r}")
        elif active:
            problems.append(f"{slot} should be empty (got {active})")

    if case.budget is not SKIP and case.budget != SKIP:
        try:
            actual = parse_budget(case.text)
        except Exception as error:
            return problems + [f"parse_budget RAISED {type(error).__name__}: {error}"]
        if case.budget is None and actual is not None:
            problems.append(f"invented a budget of ${actual:g} from a non-price number")
        elif case.budget is not None and actual != case.budget:
            problems.append(f"budget {actual!r}, expected {case.budget!r}")

    return problems


def evaluate_dialogue(dialogue: Dialogue, table=None) -> list[str]:
    """Replay a conversation through the real policy loop."""
    problems: list[str] = []
    state = ConversationState(session_id="robustness", user_profile={})
    for turn, message in enumerate(dialogue.turns, start=1):
        try:
            update(state, message, turn)
        except Exception as error:
            return [f"turn {turn} RAISED {type(error).__name__}: {error}"]
        if table is not None:
            try:
                attribute, extras = choose_question(state, [], table)
            except Exception as error:
                return [f"turn {turn} choose_question RAISED {type(error).__name__}: {error}"]
            if attribute is None:
                # The policy deliberately stops asking once every concrete
                # topic has appeared and the open-ended action is exhausted.
                # Recommendations still make this a useful customer turn.
                continue
            if attribute != "other" and attribute in state.asked:
                problems.append(f"turn {turn} re-asked {attribute!r}")
            elif attribute in state.unanswerable and attribute != "other":
                problems.append(f"turn {turn} asked {attribute!r} after it was declined")
            record_question(state, turn, attribute, extras)

    for slot, value in dialogue.expect:
        if value not in state.active(slot):
            problems.append(
                f"missing {slot}={value!r} at end (got {state.active(slot) or 'nothing'})")
    for rule in dialogue.forbid:
        slot, _, value = rule.partition(":")
        if value:
            if value in state.active(slot):
                problems.append(f"still holds {slot}={value!r} at end")
        elif state.active(slot):
            problems.append(f"{slot} should be empty at end (got {state.active(slot)})")
    for slot in dialogue.unanswerable:
        if slot not in state.unanswerable:
            problems.append(f"{slot!r} was declined but not retired")
    return problems


def _table():
    """Real attribute table when the artifacts exist; None otherwise.

    The dialogue checks that matter (retraction, retirement) need no catalog.
    The question-repetition check does, so it is skipped rather than faked when
    the index has not been built.
    """
    try:
        from pathlib import Path

        from src.attributes import load_attribute_table
        if not Path("artifacts/attributes.json").exists():
            return None
        return load_attribute_table("artifacts", "data/catalog.jsonl")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial input robustness")
    parser.add_argument("--failures", action="store_true", help="print every failing case")
    parser.add_argument("--tag", help="restrict to one characteristic")
    parser.add_argument("--soft", action="store_true",
                        help="include debatable cases in the headline number")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any hard case fails")
    parser.add_argument("--dialogues", action="store_true",
                        help="only the multi-turn dialogues")
    args = parser.parse_args()

    cases = [c for c in CASES if not args.tag or c.tag == args.tag]
    if args.dialogues:
        cases = []
    if not cases:
        sys.exit(f"no cases tagged {args.tag!r}")

    table = _table()
    dialogues = [dlg for dlg in DIALOGUES if not args.tag or dlg.tag == args.tag]
    results = [(case, evaluate(case)) for case in cases]
    results += [(dlg, evaluate_dialogue(dlg, table)) for dlg in dialogues]
    if dialogues and table is None:
        print("  (no artifacts/attributes.json -- question-repetition checks skipped;"
              " run tools.build_index)")
    hard = [(c, p) for c, p in results if not c.soft]
    soft = [(c, p) for c, p in results if c.soft]

    def rate(rows):
        passed = sum(1 for _, problems in rows if not problems)
        return passed, len(rows)

    by_tag: dict[str, list] = collections.defaultdict(list)
    for case, problems in results:
        by_tag[case.tag].append((case, problems))

    print(f"\n  {'characteristic':22} {'hard':>9} {'soft':>9}   {'':<6}")
    for tag in sorted(by_tag):
        rows = by_tag[tag]
        hp, ht = rate([r for r in rows if not r[0].soft])
        sp, st = rate([r for r in rows if r[0].soft])
        hard_cell = f"{hp}/{ht}" if ht else "-"
        soft_cell = f"{sp}/{st}" if st else "-"
        flag = ""
        if ht and hp < ht:
            flag = "  <-- " + ("FAIL" if hp < ht else "")
        print(f"  {tag:22} {hard_cell:>9} {soft_cell:>9}   {flag}")

    hp, ht = rate(hard)
    sp, st = rate(soft)
    print(f"\n  hard cases  {hp}/{ht} = {hp / max(1, ht):.1%}"
          f"   (behaviour that should not be controversial)")
    print(f"  soft cases  {sp}/{st} = {sp / max(1, st):.1%}"
          f"   (debatable; never gates CI)")

    if args.failures:
        shown = results if args.soft else hard
        failing = [(c, p) for c, p in shown if p]
        print(f"\n  {'=' * 72}\n  {len(failing)} FAILING CASES\n  {'=' * 72}")
        for case, problems in failing:
            marker = " [soft]" if case.soft else ""
            label = case.text if isinstance(case, Case) else " | ".join(case.turns)
            print(f"\n  [{case.tag}]{marker} {label!r}")
            for problem in problems:
                print(f"      {problem}")
            if case.note:
                print(f"      note: {case.note}")

    if args.strict and hp < ht:
        sys.exit(1)


if __name__ == "__main__":
    main()
