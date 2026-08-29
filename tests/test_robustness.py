"""CI gate for the adversarial input corpus.

The corpus in `tests/robustness_cases.py` asserts what SHOULD happen, so some
of it fails today by design -- those failures are the bug list, not a broken
build. This module keeps CI honest without freezing the bugs in place:

  1. No input may ever raise. The evaluator scores a raised exception as a
     MISS, so this is absolute and applies to every case, soft included.
  2. No hard failure may appear at all. KNOWN_FAILURES is an allowlist and is
     currently empty -- all 283 hard cases pass.

`python3 -m tools.robustness --failures` shows the detail.
"""

from __future__ import annotations

import unittest

from tests.robustness_cases import CASES, DIALOGUES
from tools.robustness import _table, evaluate, evaluate_dialogue

# Empty, and it should stay that way. Every case in the corpus asserts
# behaviour that should not be controversial, so a hard failure is a bug. If
# you add a case the pipeline cannot yet satisfy, either mark it soft=True in
# the corpus (genuinely debatable) or fix the pipeline -- do not park it here.
KNOWN_FAILURES: set[str] = set()


class TestNeverRaises(unittest.TestCase):
    """An exception is a MISS. This holds for every input, always."""

    def test_no_single_message_raises(self):
        for case in CASES:
            with self.subTest(text=case.text):
                for problem in evaluate(case):
                    self.assertNotIn("RAISED", problem, f"{case.text!r}: {problem}")

    def test_no_dialogue_raises(self):
        table = _table()
        for dialogue in DIALOGUES:
            with self.subTest(turns=dialogue.turns):
                for problem in evaluate_dialogue(dialogue, table):
                    self.assertNotIn("RAISED", problem, f"{dialogue.turns}: {problem}")


class TestNoNewRegressions(unittest.TestCase):
    def test_no_unexpected_hard_failure(self):
        table = _table()
        failing: dict[str, list[str]] = {}
        for case in CASES:
            if case.soft:
                continue
            problems = evaluate(case)
            if problems:
                failing[case.text] = problems
        for dialogue in DIALOGUES:
            if dialogue.soft:
                continue
            problems = evaluate_dialogue(dialogue, table)
            if problems:
                failing[" | ".join(dialogue.turns)] = problems

        unexpected = {k: v for k, v in failing.items() if k not in KNOWN_FAILURES}
        self.assertFalse(
            unexpected,
            "new robustness failures:\n" + "\n".join(
                f"  {text!r}: {'; '.join(problems)}" for text, problems in unexpected.items()
            ),
        )

    def test_known_failures_are_still_real(self):
        """A fixed bug must be removed from the allowlist, or it rots."""
        table = _table()
        by_text = {c.text: (c, evaluate) for c in CASES if not c.soft}
        by_text.update({
            " | ".join(d.turns): (d, lambda x: evaluate_dialogue(x, table))
            for d in DIALOGUES if not d.soft
        })
        fixed = [
            text for text in KNOWN_FAILURES
            if text in by_text and not by_text[text][1](by_text[text][0])
        ]
        self.assertFalse(
            fixed,
            "these now pass -- delete them from KNOWN_FAILURES:\n  " + "\n  ".join(fixed),
        )


if __name__ == "__main__":
    unittest.main()
