"""Genuine-contradiction stress test for the supersession policy.

The public evaluator never produces a real mind-change: its "override" restates
a constraint the target still satisfies. This harness constructs the case the
public set cannot -- a turn-1 preference that is a DECOY the target provably
does not satisfy, contradicted at the override turn by the true constraint --
and confirms ``evict_on_conflict`` beats ``keep`` under it while staying
identical on the unmodified set (see plan verification section 5).

We isolate the memory + constraint scorer (no full evaluator loop) so the only
variable is the supersession policy. Each trial: a target product with a known
material, a decoy material it does NOT have, and a distractor in the same
bucket that DOES have the decoy material. Correct behaviour: after the override
discloses the true material, the target must outrank the distractor.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent_router.constraint_memory import ConstraintMemory  # noqa: E402
from src.retrieval.constraint_index import prepare  # noqa: E402


class _Idx:
    """Minimal stand-in for ConstraintIndex.score over two hand-built products."""

    def __init__(self, products: dict[str, set[str]]):
        self.products = products

    def score(self, asin: str, constraints) -> float:
        attrs = self.products[asin]
        total = 0.0
        for norm, toks, w in constraints:
            if w <= 0 or not norm:
                continue
            if norm in attrs:
                total += w * 3.0
            elif any(norm in a for a in attrs):
                total += w * 1.0
        return total


def _trial(policy: str) -> tuple[bool, float]:
    """One genuine-contradiction trial. Returns (target_wins, target_rr)."""
    os.environ["OVERRIDE_POLICY"] = policy
    target = {"leather"}          # true material
    distractor = {"cotton"}       # has the decoy material instead
    idx = _Idx({"TARGET": target, "DISTRACT": distractor})

    mem = ConstraintMemory()
    # Turn 1: decoy preference (target does NOT satisfy it).
    mem.add_message("I'm looking for Boots. A key requirement is: cotton.", 1)
    # Override turn: the real material, contradicting the decoy.
    mem.add_message("Actually, ignore my earlier preference. What I need is: leather.", 3)

    prepared = [(*prepare(c), 1.0) for c in mem.constraints]
    st = idx.score("TARGET", prepared)
    sd = idx.score("DISTRACT", prepared)
    ordered = sorted(["TARGET", "DISTRACT"], key=lambda a: -idx.score(a, prepared))
    rank = ordered.index("TARGET") + 1
    return st > sd, 1.0 / rank


def main() -> None:
    for policy in ("keep", "evict_on_conflict", "evict_all"):
        wins, rr = _trial(policy)
        print(f"{policy:20s} target_outranks_distractor={wins!s:5s} target_RR={rr:.3f}")


if __name__ == "__main__":
    main()
