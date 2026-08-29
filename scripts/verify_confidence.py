"""End-to-end functional walkthrough of the confidence component.

Simulates the evaluator's conversation loop (message in -> ledger.observe ->
reranker RankResult -> confidence.decide) across the four scenario types, plus
the fallback paths. This exercises the real code paths without needing the
not-yet-built champion agent.

Run: python3 -m scripts.verify_confidence
"""

from __future__ import annotations

from src.confidence import (
    SessionLedger,
    always_ask,
    decide,
    popularity_top10,
    safe_decide,
)
from src.reranker import RankResult


def rr(pool, cov, crowd, n=10):
    return RankResult(
        ranked=[f"B{i:09d}" for i in range(n)],
        pool_size=pool,
        max_coverage=cov,
        top_tier_crowd=crowd,
    )


def show(turn, msg, payload):
    tag = f"ASK({payload.ask_attribute})" if payload.clarify else "RECOMMEND"
    score = "nan" if payload.score != payload.score else f"{payload.score:.3f}"
    print(f"  turn {turn}: user={msg!r}")
    print(f"          -> conf={score} {tag}  [{payload.reason}]")


def scenario_buying():
    print("\n=== BUYING: hard constraint disclosed early ===")
    led = SessionLedger("buy")
    # Turn 1: two constraints already in opening message.
    led.observe("I'm looking for shoes. A key requirement is: leather.", turn=1)
    led.add_constraint("leather")
    led.add_constraint("shoes")
    p = decide(rr(pool=40, cov=2, crowd=3), led)
    show(1, "leather shoes", p)
    assert not p.clarify, "buying with 2 satisfied constraints + tiny crowd should recommend"


def scenario_browsing():
    print("\n=== BROWSING: starts vague, narrows over turns ===")
    led = SessionLedger("brw")
    # Turn 1: zero info -> forced clarify.
    led.observe("I'm looking for clothing, but I'm still exploring.", turn=1)
    p = decide(rr(pool=2000, cov=0, crowd=800), led)
    show(1, "vague", p)
    assert p.clarify and p.ask_attribute == "other", "zero-info must force clarify"

    # Turn 2: one satisfied constraint. NOTE (finding for Step 3 sweep):
    # with a single constraint s1 saturates at 1.0, so conf floors at
    # W1 + W3*s3 = 0.45 + 0.20*0.25 = 0.50 regardless of crowd size. At the
    # default theta=0.5 this tips to RECOMMEND even with a large tie-crowd.
    # This is the documented formula behavior; the theta sweep will measure
    # whether it costs us on browsing scenarios.
    led.observe("For that, what matters is: cotton.", turn=2)
    led.add_constraint("cotton")
    led.reset_progress()
    p = decide(rr(pool=500, cov=1, crowd=300), led)
    show(2, "cotton", p)
    assert p.score == 0.5, "single satisfied constraint floors conf at 0.50"
    assert not p.clarify, "at theta=0.5, conf==0.5 is not < theta -> recommend"

    # Turn 3: raising theta above the floor makes the same state clarify,
    # confirming the crowd still influences the decision via theta choice.
    p_hi = decide(rr(pool=500, cov=1, crowd=300), led, theta=0.7)
    show(3, "cotton (theta=0.7)", p_hi)
    assert p_hi.clarify, "higher theta re-enables clarify for the ambiguous crowd"

    # Turn 4: enough constraints, crowd collapsed -> recommend.
    led.observe("For that, what matters is: blue.", turn=4)
    led.add_constraint("blue")
    led.add_constraint("size 10")
    led.reset_progress()
    p = decide(rr(pool=30, cov=3, crowd=2), led)
    show(4, "narrowed", p)
    assert not p.clarify, "narrowed pool with satisfied constraints should recommend"


def scenario_override():
    print("\n=== INTENT OVERRIDE: exhausted then override resumes clarify ===")
    led = SessionLedger("ovr", constraints_known=["cotton"])
    led.observe("I don't have an additional preference for color.", turn=2)
    p = decide(rr(pool=200, cov=1, crowd=200), led)
    show(2, "no more prefs", p)
    assert not p.clarify, "exhausted -> recommend only"

    led.observe("Actually, ignore my earlier preference. What I need is: leather.", turn=3)
    led.add_constraint("leather")
    p = decide(rr(pool=200, cov=1, crowd=200), led)
    show(3, "override", p)
    assert p.clarify, "override must reset exhaustion and resume clarify"


def scenario_boundary():
    print("\n=== BOUNDARY: brush-off, confidence score unaffected ===")
    led = SessionLedger("bnd", turn=2, constraints_known=["shoes"])
    a = decide(rr(pool=100, cov=1, crowd=50), led)
    b = decide(rr(pool=100, cov=1, crowd=50), led)
    show(2, "same inputs x2", a)
    assert (a.score, a.clarify) == (b.score, b.clarify), "score must be deterministic"


def fallbacks():
    print("\n=== FALLBACKS: empty pool + reranker exception ===")
    pop = popularity_top10("data/catalog.jsonl")
    led = SessionLedger("fb", turn=1, constraints_known=["cotton"])

    p, recs = safe_decide(lambda: rr(pool=0, cov=0, crowd=0, n=0), led, pop, 0.5)
    print(f"  empty pool -> clarify={p.clarify} recs={len(recs)} [{p.reason}]")
    assert p.clarify and recs == pop, "empty pool -> popularity fallback"

    def boom():
        raise RuntimeError("reranker exploded")

    p, recs = safe_decide(boom, led, pop, 0.5)
    print(f"  exception  -> clarify={p.clarify} recs={len(recs)} [{p.reason}]")
    assert p.clarify and recs == pop, "exception -> popularity fallback, no raise"


def ship_arm_p0():
    print("\n=== P0 ARM: always-ask-until-exhausted ===")
    led = SessionLedger("p0", turn=2, constraints_known=["cotton", "black"])
    p = always_ask(led)
    print(f"  not exhausted -> clarify={p.clarify} [{p.reason}]")
    assert p.clarify
    led.exhausted = True
    p = always_ask(led)
    print(f"  exhausted     -> clarify={p.clarify}")
    assert not p.clarify


def main():
    scenario_buying()
    scenario_browsing()
    scenario_override()
    scenario_boundary()
    fallbacks()
    ship_arm_p0()
    print("\nAll functional checks passed.")


if __name__ == "__main__":
    main()
