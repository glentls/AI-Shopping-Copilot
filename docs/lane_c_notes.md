# Lane C — dialogue policy

Owns `src/policy/{state,question,message}.py`, `tools/bench.py`,
`tests/test_lane_c.py`. Reproduce everything below with:

```bash
python3 -m tools.build_index      # ~28s -> artifacts/
python3 -m tools.bench            # metrics, per scenario, first-hit histogram
python3 -m tools.bench --depth    # where the target really ranks
python3 -m tools.bench --verify   # replay loop still matches the evaluator
```

## Where we are

| | score | hit@10 | MRR | MTTC | mean turns to first hit |
|---|---|---|---|---|---|
| **overall** | **0.7474** | 0.865 | 0.551 | 3.52 | **2.35** |
| buying (80) | 0.7413 | 0.863 | 0.509 | 3.14 | 1.88 |
| browsing (80) | 0.7478 | 0.863 | 0.550 | 3.42 | 2.22 |
| intent_override (30) | 0.7428 | 0.867 | 0.607 | 4.63 | 3.65 |
| boundary (10) | 0.8080 | 0.900 | 0.727 | 4.00 | 3.22 |

First-hit turns: 22 sessions on turn 1, 87 on turn 2, 45 on turn 3, 19 on turn
4, 27 never. Rank 1 on 87 of the 173 hits.

Baseline on entry was 0.7483. The 0.0009 difference is one deliberate trade,
explained under *Repeats* below; hit rate and MTTC are unchanged.

`intent_override` MTTC cannot go below ~3.5 by construction: the evaluator only
counts a hit once `override_applied` is true, which happens on turn 3 or 4, so
turns 1–2 of those sessions cannot register a hit however good the ranking is.
Read that row against a floor of 3.5, not 1.

## The question-picking rule

Each turn, every askable slot is scored

```
value(slot) = min(entropy(table.distribution(slot, candidates)), 4.0) * table.coverage(slot)
              + PRIOR[slot]
```

— how evenly an answer would split the remaining pile, discounted by how often
the catalog can answer that slot at all, plus a measured prior. A slot is
*askable* only if it is not in `state.unanswerable` (the customer declined it),
not in `state.asked` (we already spent a turn on it), and not already filled.

The entropy term is capped at 4 bits. Uncapped, `brand` has 19,749 distinct
values and scores ~8.7 against `use_case`'s ~1.9, so the picker asks about
brands every single turn on cardinality alone.

The highest scorer becomes `ask_attribute`; the next two go into the prose. The
API scores one attribute, but the organizers' worked example asks three things
in one breath and the customer answers all three, so bundling is free upside and
turn count is 20% of the score.

**The wildcard.** `other` is an action, not a slot, and competes on the same
scale. It yields whatever the customer still has; a concrete slot yields only if
they happen to hold a preference of that type. It dominates on this simulator:
pricing it at 0 — i.e. always picking the highest-entropy concrete slot —
scores 0.6900 against 0.7474. Almost all of that is speed, not reach: hit rate
only moves 0.865 → 0.840, but MTTC blows out from 3.52 to 6.11. Information you
reliably GET beats information that would be worth more if you got it.

It is now priced by what it has actually returned *this session* rather than by
a fixed constant:

```
shrunk = (sum(observed yields) + 2 * PRIOR_YIELD) / (n + 2)
value  = OTHER_BASELINE * shrunk / PRIOR_YIELD
```

`OTHER_BASELINE` is the prior, not the verdict. This matters because the
constant is fitted to a simulator that answers "anything else?" with two
verbatim catalog strings, and a real customer will not. When the wildcard stops
paying, the estimate falls and concrete slots take the turn — with nobody
re-tuning anything. On the public set the change is score-neutral (every turn
after exhaustion is dead anyway), so the robustness is free.

Sweeping the prior on the public set:

| `TJ_OTHER_BASELINE` | 0 | 1 | 3 | 8 | 20 |
|---|---|---|---|---|---|
| score | 0.6900 | 0.7156 | 0.7330 | 0.7473 | 0.7474 |
| MTTC | 6.11 | 5.71 | 4.50 | 3.52 | 3.52 |

It saturates because `classify_constraint` in the evaluator only ever routes
answers to budget, material, color, size, style, use_case and feature — asking
`brand` or `category` here is a guaranteed dead turn. **Re-measure this the
moment customer messages become natural language.**

**Two refusals, not one, stand the wildcard down.** A boundary customer refuses
whatever they are asked first and then answers normally. Giving up after a
single refusal was implemented, measured, and reverted: boundary MTTC 4.00 →
4.90. `TJ_DECLINE_PATIENCE` controls it.

## Overrides and boundaries

*Override.* On an override cue, held values are retracted **only where the new
message contradicts them**. "Actually, what I need is leather" when leather is
already held is the customer stressing a priority, not changing one; retracting
there throws away a correct constraint. Retraction flips `polarity`, it never
deletes — `state.excluded()` feeds a negative rerank signal, and nothing in the
pipeline hard-filters, so a retraction that turns out to be wrong is recoverable.

*Boundary.* "No preference" adds the slot to `state.unanswerable`, and the
scorer never offers it again. When the customer does not name the slot, it
resolves to whatever we asked last turn.

*Repeats.* A customer who says "leather" on turn 2 and again on turn 4 is one
constraint, not two, but the state used to hold two `SlotValue`s and the
reranker scores one point per live value — so saying a thing twice outranked a
better overall match. Repeats now fold, and the emphasis is recorded as
`confidence`.

That fold is the entire 0.0009 regression, and it is worth understanding: the
duplicate happened to double-weight exactly the constraint an intent_override
re-asserts, which is the one the customer just said matters most. The accident
was doing real work (override MRR 0.626 → 0.607). **The principled recovery is
for `rerank` to weight by `SlotValue.confidence`, which it currently ignores** —
see the handoff below.

## Two findings that need another lane

**1. Six dead turns. Worth +0.079, needs three lines in `starter/agent.py`.**

No session on the public set has *ever* hit after turn 4. The simulator holds at
most four constraint strings and discloses two per answer, so the state is
complete by turn 3; after that the ranking is frozen and we re-send an identical
top-10 six times.

Meanwhile **all 27 misses are retrieved** — none is missing from the index, they
just sit below rank 10 (8 at rank 11–20, 11 at 21–50, 7 at 51–100, 1 deeper).
Paging one window deeper per silent turn, **measured end to end** with the
wiring below applied and then reverted:

| | score | hit@10 | MRR | MTTC |
|---|---|---|---|---|
| today | 0.7474 | 0.865 | 0.551 | 3.52 |
| with paging | **0.8370** | 0.985 | 0.611 | 2.93 |

Every scenario improves: buying 0.863 → 0.975, browsing 0.863 → 1.000,
intent_override 0.867 → 0.967, boundary 0.900 → 1.000.

Two guards in `recommendation_window` are load-bearing, both found by measuring
rather than reasoning. Paging must not start before turn `TJ_MIN_PAGE_TURN`
(default 4), and an override must not read as a silent turn. Without the second
guard, `intent_override` **collapses to hit 0.467 / MTTC 7.77** — those sessions
do not count a hit until the override lands on turn 3 or 4, and naive paging has
already scrolled past the target by then. Start-turn sweep: 3 → 0.8147,
4 → 0.8370, 6 → 0.7908, 8 → 0.7675.

The policy side is written and tested: `question.recommendation_window(state)`
returns the rank offset and holds at 0 until the customer has been silent twice,
so a slow discloser is never scrolled past. Wiring it is a slice in
`starter/agent.py`, which Lane C does not own:

```python
offset = recommendation_window(state, top_k)
ranked = [c.parent_asin for c in candidates[offset:offset + top_k]]
```

No Lane B change is needed — `search()` already returns 300 candidates and
`rerank()` keeps them all, so there is a deep list to page through today. This
is purely the `agent.py` slice, which one person lands on main.

**2. `rerank` should weight by `SlotValue.confidence`.** It currently adds a
flat 1.0 per matched live value, so the state cannot express that one constraint
matters more than another — which is exactly what an override is telling us.
Worth ~0.02 MRR on the override sessions.

## `tools/bench.py`

Everyone's measurement loop.

```bash
python3 -m tools.bench --failures 5              # failed transcripts, full dialogue
python3 -m tools.bench --only public_0003        # one session
python3 -m tools.bench --scenario boundary --transcript 3
python3 -m tools.bench --depth                   # is the target retrieved at all?
python3 -m tools.bench --compare base.json new.json
python3 -m tools.bench --sweep TJ_SLOT_WEIGHT=1,3,5 TJ_OTHER_BASELINE=8,20
```

`--compare` names the sessions whose outcome moved, which is usually more
informative than the delta. `--sweep` takes a grid and runs one subprocess per
cell — module constants read their environment once at import, so an in-process
sweep would silently measure the first cell repeatedly.

Transcript and depth modes need the dialogue, which the evaluator does not
expose, so `replay()` mirrors its session loop. A mirror can rot: `--verify`
asserts replay and evaluator still agree (they match to 1e-9). Run it after any
evaluator change.

## Reply text

`compose_message` does not affect the score — only `ask_attribute` and
`recommendations` do — but it is what judges read. Three failure modes it
avoids: reading the accumulated pile back every turn ("looking for something
leather, leather"), repeating one sentence for seven turns as the frozen state
produces nothing new, and promising to drop a topic while asking it again in the
same breath.

An optional local-LLM rewrite sits behind `TJ_LLM_MESSAGE`, wrapped in
try/except with the template as fallback. It is off by default and nothing else
in the pipeline may call a model: the evaluator scores a raised exception as a
MISS, and 800 sessions × 10 turns of model calls will blow any latency budget.

See `docs/lane_c_demo_transcript.md` for two annotated sessions.
