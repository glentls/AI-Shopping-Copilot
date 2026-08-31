# Evaluator verification notes

Verified against `evaluator/local_evaluator.py`, `docs/evaluation_config.json`,
and `docs/agent_api_contract.json`. No files under `evaluator/` or `data/` were
modified to produce these findings.

## 1. Can a turn return both a non-null `ask_attribute` and non-empty `recommendations`, with no penalty?

**Yes, confirmed, no penalty.**

- `docs/agent_api_contract.json:35-68` — `turn_response` schema lists `message`,
  `ask_attribute`, and `recommendations` as independent, co-required fields.
  Nothing in the schema makes them mutually exclusive or conditions one on the
  other being null/empty.
- `evaluator/local_evaluator.py:251` — `ranked = normalize_recommendations(response.get("recommendations"), ...)`
  runs unconditionally every turn, regardless of what `ask_attribute` was set to.
- `evaluator/local_evaluator.py:252-255` — the hit check (`if override_applied and target in ranked`)
  only inspects `recommendations`; `ask_attribute` never gates it.
- `evaluator/local_evaluator.py:266-268` — `ask_attribute` is only ever read by
  `customer_reply(...)` to decide what the simulated user says *next turn*. It
  has no effect on scoring for the *current* turn.
- No code path anywhere in `evaluate()` reduces `hit`, `reciprocal_rank`, or
  token accounting based on `ask_attribute` being present.

**Conclusion:** asking a clarifying question is free. Every turn should carry a
full Top-10, whether or not the agent also asks something.

## 2. MTTC: is it the first-hit turn index, and is it 0- or 1-indexed?

**Yes — mean of the first-hit turn, 1-indexed; misses count as turn 11.**

- `evaluator/local_evaluator.py:238` — `for turn in range(1, MAX_TURNS + 1):`
  with `MAX_TURNS = 10` (line 15). Turn numbering is 1-indexed (1..10),
  matching `turn_request.turn` in the contract (`minimum: 1, maximum: 10`,
  `docs/agent_api_contract.json:30`).
- `evaluator/local_evaluator.py:253-255` — on the first turn where the target
  ASIN appears in the normalized recommendations, `hit_turn = turn` is set and
  the loop `break`s immediately. Later turns in the same session are never
  reached, so `hit_turn` really is the *first* hit, not best/last.
- `evaluator/local_evaluator.py:193-195` —
  `mttc = fmean(item["first_hit_turn"] if not None else MAX_TURNS + 1 for ...)`.
  `MAX_TURNS + 1 = 11`, matching `docs/evaluation_config.json:5`
  (`"miss_turn_value": 11`). So MTTC is a plain mean across sessions of
  (1-indexed hit turn, or 11 on miss).

**Conclusion:** the CLAUDE.md formula `MTTC = mean first-hit turn; MISS -> 11`
is exactly what's implemented.

## 3. Is MRR computed from the hit-turn list only, or aggregated across turns?

**Hit-turn only — and by construction there is only one rank per session.**

- `evaluator/local_evaluator.py:252-255` — as soon as a hit is found, `best_rank
  = ranked.index(target) + 1` is computed from *that turn's* ranked list, and
  the loop breaks. Non-hit turns before the hit never contribute a rank
  (their `ranked` lists are simply discarded once the loop moves on), and no
  turn after the hit is ever evaluated.
- `evaluator/local_evaluator.py:275` — per-session
  `reciprocal_rank = 0.0 if best_rank is None else 1.0 / best_rank`. Misses get
  `0.0`, not skipped.
- `evaluator/local_evaluator.py:192` — overall
  `mrr = statistics.fmean(item["reciprocal_rank"] for item in sessions)` is a
  session-level mean, not a turn-level mean. Since each session contributes
  exactly one reciprocal-rank value (0.0 for a miss, `1/rank` for the first
  hit), "aggregated across turns" doesn't apply — there is nothing to
  aggregate within a session.

**Conclusion:** MRR rewards ranking the target as high as possible *the first
time it's returned*. Recommendations on turns after the hit are never scored
(the session already ended), and recommendations on turns before the hit that
don't contain the target contribute nothing — not even a zero — to the rank
term (only to `hit`/`mrr` being 0 if no turn ever contains it).

## Net effect on CLAUDE.md's derived strategy

All three checks support the "always return a full Top-10 every turn, question
or not" strategy in CLAUDE.md:

- Asking costs nothing (Q1).
- Efficiency only cares about *first*-hit turn (Q2), so front-loading full
  recommendation lists from turn 1 is never penalized relative to holding
  back.
- MRR only cares about the rank at that first-hit turn (Q3), so there's no
  advantage to "saving" a better-ranked list for a later turn — the earliest
  turn the target appears in the list is the one that counts, and getting it
  in at all beats a marginally better rank on a later turn (since a miss
  forces MTTC=11 and MRR contribution=0).

One nuance not spelled out in CLAUDE.md: because the evaluator `break`s on
first hit, a strategy that could plausibly rank the target #1 on turn 3 but
only #8 on turn 1 should still surface it on turn 1 — turn 1's hit locks in
`reciprocal_rank = 1/8` and `first_hit_turn = 1`, and turn 3 is never reached
to offer the better rank. There is no way to "wait for a better rank" once the
target is retrievable; once it's in the Top-10, ship it immediately.
