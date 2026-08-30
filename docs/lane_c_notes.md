# Lane C — dialogue policy

Owns `src/policy/{state,question,message}.py`, `tools/bench.py`,
`tests/test_lane_c.py`. Reproduce everything below with:

```bash
python3 -m tools.build_index      # one-time local artifacts; see lane_b_notes
python3 -m tools.bench            # metrics, per scenario, first-hit histogram
python3 -m tools.bench --depth    # where the target really ranks
python3 -m tools.bench --verify   # replay loop still matches the evaluator
```

## Where we are

| | score | hit@10 | MRR | MTTC | mean turns to first hit |
|---|---|---|---|---|---|
| **overall** | **0.8601** | 0.990 | 0.662 | 2.67 | **2.59** |
| buying (80) | 0.8690 | 0.988 | 0.667 | 2.24 | 2.13 |
| browsing (80) | 0.8607 | 0.988 | 0.652 | 2.44 | 2.33 |
| intent_override (30) | 0.8314 | 1.000 | 0.649 | 4.17 | 4.17 |
| boundary (10) | 0.8712 | 1.000 | 0.738 | 3.50 | 3.50 |

Baseline on entry was 0.7483. Only 2 sessions now miss. Recommendation selection
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

**The open-ended question.** `other` is the evaluator's action name for asking
“anything else that matters?”. It is not a wildcard slot. It can yield a
preference of any type, while a concrete action yields only if the customer has
a preference for that topic.

The old policy started `other` at 20 while concrete questions were capped near
5. It therefore kept winning after many unproductive natural-language turns.
That value came from a public-simulator sweep:

| old open-question baseline | 0 | 1 | 3 | 8 | 20 |
|---|---|---|---|---|---|
| score | 0.6900 | 0.7156 | 0.7330 | 0.7473 | 0.7474 |
| MTTC | 6.11 | 5.71 | 4.50 | 3.52 | 3.52 |

The simulator reliably returns up to two catalog-like constraints for `other`,
so 8 and 20 score almost identically. Natural customers do not behave that way,
and the extra 12 points only make the question dominate the conversation.

The current policy puts `other` on the same scale as concrete questions and
applies explicit diminishing returns:

```
score = OPEN_QUESTION_BASELINE * OPEN_QUESTION_DECAY ** answered_open_questions
score *= 1.25 when the latest answer yields at least two new facts
score *= 0.90 when it yields one new fact
score *= 0.35 when it yields none
```

The defaults are a baseline of 8 and decay of 0.8. Hard guardrails prevent a
benchmark-tuned score from creating a visible loop:

- At most two consecutive open-ended questions.
- A zero-yield reply forces a concrete question next.
- Two zero-yield replies or explicit refusals retire `other` for this intent.
- An intent override resets the derived counts.
- Concrete topics bundled into the prose are marked as asked, so they do not
  reappear later.

The defaults were selected from a 4×3 public-set sweep on the latest merged
retrieval and orchestration code. Against untouched main on the same artifacts,
the new policy improves MRR and boundary handling, while losing one browsing
hit and paying 0.055 turns of MTTC for the conversational guardrails:

| policy | score | hit@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| untouched main | 0.8623 | 0.995 | 0.6569 | 2.615 |
| baseline 8, decay 0.8 | 0.8601 | 0.990 | 0.6618 | 2.670 |

The controls are `TJ_OPEN_QUESTION_BASELINE`, `TJ_OPEN_QUESTION_DECAY`,
`TJ_OPEN_QUESTION_EXPECTED_YIELD`, `TJ_OPEN_QUESTION_MAX_CONSECUTIVE`,
`TJ_OPEN_QUESTION_ZERO_YIELD_PATIENCE`, and
`TJ_OPEN_QUESTION_DECLINE_PATIENCE`. Once the open question and every concrete
slot are exhausted, `ask_attribute` becomes `null` instead of forcing another
known-dead question.

## Overrides and boundaries

*Override.* A local correction such as "actually, leather instead" retracts
contradictory values in the named slot. The broader "ignore my earlier
preference" retires the replaceable preference from the opener even when the new
value belongs to another slot; it preserves the product category and constraints
learned on intervening turns. A value explicitly reasserted in the override
remains live. Retraction flips `polarity`, it never deletes —
`state.excluded()` feeds a negative rerank signal, and nothing hard-filters the
candidate set. The same transition clears `state.shown_recommendations`,
starting a fresh recommendation epoch for the new intent.

*Boundary.* "No preference" adds the slot to `state.unanswerable`, and the
scorer never offers it again. When the customer does not name the slot, it
resolves to whatever we asked last turn.

*Repeats.* A customer who says "leather" on turn 2 and again on turn 4 is one
constraint, not two, but the state used to hold two `SlotValue`s and the
reranker scores one point per live value — so saying a thing twice outranked a
better overall match. Repeats now fold, and the emphasis is recorded as
`confidence`.

The reranker now weights the folded value by its customer confidence and the
confidence of the catalog source, preserving emphasis without counting one fact
twice.

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
| current-intent exclusion at landing | **0.8586** | **0.985** | 0.666 | **2.680** |
| + semantically correct broad overrides | 0.8541 | **0.985** | 0.651 | **2.680** |

A literal session-wide rule is wrong. Intent-override recommendations cannot
convert before the new intent is sent, so the eventual target may already have
appeared without ending the session. Keeping those exclusions collapsed override
hit rate to 0.133. Clearing them in `_apply_override` preserves override hit rate
at 0.967 while ordinary turns remain repetition-free.

The merge follow-up applies the unseen filter before question scoring and
message composition too. Entropy is therefore measured over products that can
actually be returned, and the explanation always describes the first eligible
recommendation rather than a filtered-out former top candidate.

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

**2. Confidence-weighted reranking (landed).** Slot contributions multiply
customer confidence by catalog-source confidence, so a structured/title match
can move farther than a description-only inference and repeated emphasis is
represented without duplicate values.

**3. Ranking is now the main headroom (Lane B, open).** Hit rate is 0.985 and
105 of 197 hits land at rank 1. Most remaining technical-score headroom is MRR,
not additional retrieval reach.

**4. Natural-language coverage remains the main understanding risk (Lane A,
open).** The extractor now covers rayon, broader materials, closure types, and
care features, but it remains deterministic phrase matching. Unseen
paraphrases, compound colors, brands, fashion subcultures, and non-US sizing can
still miss canonical slots even though their raw text remains available to
retrieval.

## `tools/bench.py`

Everyone's measurement loop.

```bash
python3 -m tools.bench --failures 5              # failed transcripts, full dialogue
python3 -m tools.bench --only public_0003        # one session
python3 -m tools.bench --scenario boundary --transcript 3
python3 -m tools.bench --depth                   # is the target retrieved at all?
python3 -m tools.bench --compare base.json new.json
python3 -m tools.bench --sweep TJ_OPEN_QUESTION_BASELINE=3,4,5 TJ_OPEN_QUESTION_DECAY=0.5,0.65,0.8
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

See `docs/lane_c_demo_transcript.md` for two annotated sessions.
