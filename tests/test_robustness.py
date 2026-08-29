"""CI gate for the adversarial input corpus.

The corpus in `tests/robustness_cases.py` asserts what SHOULD happen, so some
of it fails today by design -- those failures are the bug list, not a broken
build. This module keeps CI honest without freezing the bugs in place:

  1. No input may ever raise. The evaluator scores a raised exception as a
     MISS, so this is absolute and applies to every case, soft included.
  2. No NEW hard failure may appear. KNOWN_FAILURES is an allowlist; anything
     failing outside it fails the build.

When you fix one, delete its line from KNOWN_FAILURES. The list may only ever
get shorter. `python3 -m tools.robustness --failures` shows the detail.
"""

from __future__ import annotations

import unittest

from tests.robustness_cases import CASES, DIALOGUES
from tools.robustness import _table, evaluate, evaluate_dialogue

# Known gaps, all in the cue lexicons (src/lexicons/__init__.py). See
# docs/lane_c_robustness.md for the analysis. Shrink this list, never grow it.
KNOWN_FAILURES = {
    # NEGATION_CUES misses "nothing", "hate", "never", "can't". The negated
    # value stays an ACTIVE preference, so the ranker chases what the customer
    # just rejected -- worse than not extracting it at all.
    "nothing in pink",
    "I hate the colour orange",
    "nothing sleeveless",
    "I can't wear nylon",
    "never black",
    "nothing waterproof, I don't need it",
    "I hate polyester",
    "can't stand synthetic fabrics",
    # OVERRIDE_CUES misses "no wait".
    "show me sneakers | actually boots | no wait, sandals",
    # A bare number trailing a model name reads as a price ceiling.
    "air max 90",
    # use_case lexicon has "running" but not the bare verb "run".
    "for my morning run",
}


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
