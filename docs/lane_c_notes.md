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
| **overall** | **0.8586** | 0.985 | 0.666 | 2.68 | **2.55** |
| buying (80) | 0.8473 | 0.975 | 0.624 | 2.36 | 2.14 |
| browsing (80) | 0.8742 | 1.000 | 0.672 | 2.38 | 2.38 |
| intent_override (30) | 0.8451 | 0.967 | 0.753 | 4.20 | 3.97 |
| boundary (10) | 0.8638 | 1.000 | 0.686 | 3.10 | 3.10 |

Baseline on entry was 0.7483. Only 3 sessions now miss. Recommendation selection
uses every continued turn as implicit negative feedback: products already shown
under the current intent are filtered out before the next top ten is selected.
An explicit intent override clears that history because pre-override products
were evaluated against a different need.

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
The same transition clears `state.shown_recommendations`, starting a fresh
recommendation epoch for the new intent.

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

## Findings

**1. Current-intent recommendation exclusions. Landed: +0.0042 over fixed
paging.**

Once a scored session continues, every product in the preceding top ten is
known not to be the hidden target. Fixed rank windows avoided repetition only
after silent turns, and a changed ranking could make adjacent windows overlap.
The agent now records every returned ASIN and selects the best currently ranked
candidates not yet shown under this intent.

| | score | hit@10 | MRR | MTTC |
|---|---|---|---|---|
| fixed silent-turn paging | 0.8544 | 0.980 | **0.672** | 2.865 |
| literal session-wide exclusion | 0.7452 | 0.860 | 0.557 | 3.600 |
| **current-intent exclusion** | **0.8586** | **0.985** | 0.666 | **2.680** |

A literal session-wide rule is wrong. Intent-override recommendations cannot
convert before the new intent is sent, so the eventual target may already have
appeared without ending the session. Keeping those exclusions collapsed override
hit rate to 0.133. Clearing them in `_apply_override` preserves override hit rate
at 0.967 while ordinary turns remain repetition-free.

```python
ranked = [
    c.parent_asin for c in candidates
    if c.parent_asin not in state.shown_recommendations
][:top_k]
```

**`starter/agent.py` is the shared wiring file and Lane C does not normally
touch it.** This was raised first and landed only on explicit instruction; treat
it as a merge point when branches come together. No Lane B change was needed —
`search()` already returns 300 candidates and `rerank()` keeps them all.

**2. `rerank` should weight by `SlotValue.confidence` (Lane B, open).** It adds
a flat 1.0 per matched live value, so the state cannot express that one
constraint matters more than another — which is exactly what an override is
telling us. Worth ~0.02 MRR on the override sessions.

**3. Ranking is now the whole game (Lane B, open).** In **200 of 200** sessions
the target is retrieved inside the top 300 — recall is perfect and nothing is
missing from the index. If the target ranked 1st whenever it is retrieved the
score would be **0.9881**. With hit rate at 0.985, essentially all remaining
headroom is MRR: only 96 of 197 hits land at rank 1.

**4. The extractor understands about half of what customers say (Lane A,
open).** `extract_slots` produces no slot value at all for 378 of 800 constraint
strings in the evaluator's intent cards (52.8% coverage). `rayon` and `fabric`
are in the evaluator's own `MATERIALS` list but missing from our material
lexicon, and the evaluator inserts a matched material as the *first hard
constraint*. Closures have no slot at all (`Pull On` 27, `Zipper` 16, `Button`
11, `Drawstring` 6, `Buckle` 4, `Snap` 4), nor do care instructions (`Hand Wash
Only` 18, `Machine Wash` 15). `Imported` (95) is genuinely meaningless and
should stay unextracted.

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
