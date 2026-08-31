# Evaluator verification notes

Verified against `evaluator/local_evaluator.py`, `docs/evaluation_config.json`,
`docs/agent_api_contract.json`, and `docs/competition_specification.md` (plus
`docs/final_evaluation_faq.md` for anything the local files left ambiguous).
No files under `evaluator/` or `data/` were modified to produce these findings.

## 1. Can a turn return both a non-null `ask_attribute` and non-empty `recommendations`, with no penalty?

**Yes, confirmed, no penalty.**

- `docs/agent_api_contract.json:35-68` — `turn_response` schema lists `message`,
  `ask_attribute`, and `recommendations` as independent, co-required fields.
  Nothing makes them mutually exclusive or conditions one on the other being
  null/empty.
- `evaluator/local_evaluator.py:251` — `ranked = normalize_recommendations(response.get("recommendations"), ...)`
  runs unconditionally every turn, regardless of `ask_attribute`.
- `evaluator/local_evaluator.py:252-255` — the hit check
  (`if override_applied and target in ranked`) only inspects `recommendations`;
  `ask_attribute` never gates it.
- `evaluator/local_evaluator.py:266-268` — `ask_attribute` is only ever read by
  `customer_reply(...)` to decide the *next* simulated user message. It has no
  effect on scoring for the current turn.
- `docs/final_evaluation_faq.md:97-98` — "An Agent may ask a clarification
  question and return recommendations in the same turn," stated explicitly for
  the final evaluator too, not just the local one.

**Conclusion:** asking is free. Every turn should carry a full Top-10.

## 2. MTTC: first-hit turn index, 0- or 1-indexed? Miss = 11?

**Confirmed — mean of first-hit turn, 1-indexed; a miss counts as turn 11.**

- `evaluator/local_evaluator.py:238` — `for turn in range(1, MAX_TURNS + 1):`
  with `MAX_TURNS = 10` (line 15) → turns run 1..10, 1-indexed, matching
  `turn_request.turn` (`minimum: 1, maximum: 10`,
  `docs/agent_api_contract.json:30`).
- `evaluator/local_evaluator.py:253-255` — on the first turn the target
  appears in the normalized recommendations, `hit_turn = turn` is set and the
  loop `break`s immediately, so `hit_turn` is genuinely the *first* hit.
- `evaluator/local_evaluator.py:193-195` —
  `mttc = fmean(item["first_hit_turn"] if not None else MAX_TURNS + 1 for ...)`,
  i.e. `10 + 1 = 11` on miss, matching `docs/evaluation_config.json:5`
  (`"miss_turn_value": 11`) and `docs/competition_specification.md:72`
  (`MTTC = sum(first_hit_turn, with misses assigned 11) / N`).

## 3. Is MRR from the ranked list at the hit turn only, or aggregated across turns?

**Hit-turn only — and there is exactly one rank contributed per session, ever.**

- `evaluator/local_evaluator.py:252-255` — `best_rank = ranked.index(target) + 1`
  is computed from *that turn's* list at the moment of the first hit, then the
  loop breaks. No turn before the hit contributes a rank (its list is simply
  discarded), and no turn after the hit is ever evaluated because the session
  is already over.
- `evaluator/local_evaluator.py:275` — per-session
  `reciprocal_rank = 0.0 if best_rank is None else 1.0 / best_rank` (miss → 0.0).
- `evaluator/local_evaluator.py:192` — overall `mrr = fmean(reciprocal_rank for
  session in sessions)`, a session-level mean. Matches
  `docs/competition_specification.md:71`
  (`MRR = sum(1 / target_rank, with misses equal to 0) / N`).

## 4. Does the session terminate the instant the target appears in the Top-10? Is every item shown at turn t on a still-alive session provably not the target?

**Yes, on both counts — this is the load-bearing mechanic.**

- `evaluator/local_evaluator.py:252-255`:
  ```python
  if override_applied and target in ranked:
      best_rank = ranked.index(target) + 1
      hit_turn = turn
      break
  ```
  The `break` is unconditional on a hit (once `override_applied` is true — see
  the Intent Override caveat below). There is no "continue and see if a later
  turn does better" path.
- Consequence: if a session is still running at turn t+1, by construction turn
  t's `ranked` list did **not** contain the target (otherwise the loop would
  have broken at turn t). So yes — for any session still alive at turn t+1,
  every `parent_asin` shown at turn t is a confirmed non-target. This is
  exactly the "proven negative" CLAUDE.md's Phase-3a plan wants to exploit.
- Caveat: on an Intent Override sample, `override_applied` starts `False`
  (`evaluator/local_evaluator.py:234`) and a hit before the override turn is
  not scored as a hit at all (the `if override_applied and target in ranked`
  guard fails) — see Q5 for when it flips. `docs/final_evaluation_faq.md:108`
  confirms: "An Intent Override session cannot record a hit before the
  changed intent is revealed." So for Intent Override sessions specifically,
  pre-override turns don't prove the shown items are non-targets in the
  scored sense — they just don't count, hit or not. For Buying/Browsing/
  Boundary sessions `override_applied` is `True` from turn 1
  (`evaluator/local_evaluator.py:234`), so the proven-negative property holds
  from turn 1 onward.

## 5. How does the simulator generate user replies — scripted from the file, or reactive to `ask_attribute`? Does asking change the next message? Does asking nothing still advance the turn?

**Reactive to `ask_attribute`, deterministically generated (not hand-scripted per session file), and the turn always advances regardless of what — or whether — the agent asks.**

- Checked `data/public_set.jsonl` directly: each sample has only
  `category_bucket, difficulty_bucket, ground_truth, sample_id, scenario_type,
  user_profile` — **no `intent_card` or `behavior` field on disk.** Nothing is
  pre-scripted in the dataset file itself.
- `evaluator/local_evaluator.py:204-213` (`materialize_hidden_fields`) — since
  the public samples never carry `intent_card`/`behavior`, this always falls
  to the synthesis branch: `card = intent_card(product)` built from the
  **target product's own catalog record** (title/features/details/price,
  regex-extracted material and color — lines 52-71), and
  `behavior = behavior_for(scenario_type, card, rng)` with
  `rng = random.Random(f"{sample_id}\0{scenario_type}")` (line 211) — a fixed
  seed derived from the sample, so generation is deterministic and repeatable
  across evaluator runs, but it is generated, not authored per-session.
- `evaluator/local_evaluator.py:166-185` (`customer_reply`) — this is the
  reactive part. It reads `ask_attribute` from the agent's response, maps it
  to `ALLOWED_ATTRIBUTES` (falling back to `"other"` if invalid), and returns
  constraint values from the (pre-generated) intent card whose
  `classify_constraint(value)` matches the asked attribute — but only values
  not yet in `disclosed`. **If `ask_attribute` is `None`,** the reply is the
  generic nudge `"Those options are not quite right yet. Ask me about one
  specific attribute."` (line 171) — no new information is revealed. So yes:
  asking a specific attribute changes what's said next; asking nothing gets a
  generic non-answer instead of a disclosure.
- `evaluator/local_evaluator.py:238,256-268` — the `for turn in range(1, 11)`
  loop always proceeds to the next iteration (or ends at `MAX_TURNS`)
  regardless of whether a hit, an ask, or nothing happened — asking nothing
  does not skip or repeat a turn, it just yields a less useful reply.
- One more reactive wrinkle: `boundary` scenario sessions get a one-time
  "I don't have a preference for X; use your judgment" reply the *first* time
  any attribute is asked (`evaluator/local_evaluator.py:168-169`,
  `boundary_used` flag), independent of which attribute was asked.
- The Intent Override turn is scripted-but-scheduled, not reactive: at
  `evaluator/local_evaluator.py:258-264`, once `turn + 1` equals the
  (RNG-chosen, seeded) override turn, the customer's message is forcibly
  replaced with the override text regardless of what the agent asked that
  turn.
- `docs/final_evaluation_faq.md:100-101` confirms this is intentional protocol,
  not a local-evaluator quirk: "The simulator responds according to structured
  `ask_attribute`; it does not infer the requested attribute from the
  natural-language `message`."

## 6. Are per-scenario metrics (Buying/Browsing/Intent Override/Boundary) broken out in `results.json`?

**Yes.**

- `evaluator/local_evaluator.py:281-283` groups sessions by
  `session["scenario_type"]` into `grouped`.
- `evaluator/local_evaluator.py:293` — the returned dict includes
  `"scenario_metrics": {name: metric_summary(grouped[name]) for name in
  sorted(grouped)}`, i.e. `hit_rate_at_10`/`mrr`/`mttc` computed separately per
  scenario, alongside the overall numbers. `docs/evaluation_config.json:8`
  lists the four scenario names (`buying, browsing, intent_override,
  boundary`) matching `docs/competition_specification.md:23-28`.

## 7. Is there any cap on token usage, latency, or wall-clock time that affects scoring?

**No — not in the local evaluator, and not guaranteed in final evaluation either.**

- `evaluator/local_evaluator.py:239-250` — the only guard around
  `agent.respond(...)` is a bare `try/except Exception` that substitutes an
  empty response on error (line 241-244); there is no timeout wrapper, no
  `time.time()` call anywhere in the file, and no latency measurement at all.
- Token usage is accumulated (`total_prompt_tokens`/`total_completion_tokens`,
  lines 245-250) purely for the `reported_token_usage` field
  (line 288-292) — it is never read by `metric_summary` or folded into
  `technical_score` (line 280: `0.50 * hit_rate + 0.30 * mrr + 0.20 *
  efficiency` only).
- `docs/evaluation_config.json:10-14` — `recommended_composite` only lists
  `hit_rate_at_10`, `mrr`, `efficiency`; token/latency aren't part of it.
- `docs/competition_specification.md:79` — "Reported token use and latency are
  feasibility measures and do not change the core score."
- `docs/final_evaluation_faq.md:58` — "The current evaluator does not impose a
  separate explicit per-response timeout," and `:49-53` — no organizer-standard
  hardware/timeout exists because teams run final evaluation in their own
  environment. `:65` in `competition_specification.md` does note "Exceptions,
  invalid output, and timeouts **may** count as a miss" — so a hang is a
  self-inflicted miss via non-response, but there's no explicit clock in the
  evaluator itself docking points beyond that natural consequence.
- **Practical implication for us:** nothing stops an agent from being slow
  locally, but an unbounded call (e.g. a hung LLM request with no timeout) has
  no evaluator-side cutoff to save it — it will just never return a turn, and
  the session presumably stalls/errors out at that turn. This is exactly why
  CLAUDE.md's "wrap every external call in timeout and retry with graceful
  degradation" (Phase 5) matters even though nothing in the scoring formula
  directly punishes latency — self-imposed timeouts are the only thing
  preventing a real hang from becoming an effective miss.

## Net effect on CLAUDE.md's assumptions

**Survived, fully confirmed:**

- Asking is free, no penalty for co-returning `ask_attribute` + `recommendations` (Q1).
- `MTTC`/`Efficiency`/`TechnicalScore` formulas in CLAUDE.md match the code and
  spec exactly (Q2, Q7).
- MRR is a first-hit-turn-only, session-level quantity — no benefit to holding
  back a better rank for later; the earliest turn the target is retrievable is
  the one that counts (Q3).
- The stopping-on-first-hit rule holds, which validates the entire
  "proven-negative" plan for Phase 3a — for Buying/Browsing/Boundary sessions,
  anything shown on a still-running session is a confirmed non-target from
  turn 1 onward (Q4).
- Per-scenario breakdown is available for free from the evaluator, no need to
  build our own scenario-splitting logic for the ablation table (Q6).

**Refined / new information not in CLAUDE.md's original framing:**

- The customer simulator is not literally "scripted from the session file" —
  intent cards and rejection-disclosure content are synthesized
  deterministically from the **target product's own catalog record** at
  evaluation time, seeded by `sample_id + scenario_type` (Q5). This matters
  for Phase 3a's "downweight attribute values that dominated a rejected batch"
  idea: the belief update target is a real but synthetic intent card, entirely
  derivable in principle from catalog metadata patterns, not private
  information.
- Intent Override sessions are the one case where the "proven negative"
  property doesn't cleanly apply pre-override: a hit before the override turn
  is never scored as a hit regardless of whether the shown list actually
  contained the target, because `override_applied` is `False` until the
  scheduled override turn (Q4/Q5). Phase 3a/3d logic should special-case this
  scenario rather than assuming uniform proven-negative semantics across all
  four scenario types.
- Asking nothing (`ask_attribute: null`) doesn't just forgo new information —
  it gets an explicit generic non-answer reply, which is a strictly worse
  outcome than asking any valid attribute even if that attribute turns out to
  have nothing left to disclose (Q5). There is no scenario where asking
  `null` is reactively better than asking something.
- Latency/timeouts are not scored directly, but there is also no evaluator
  safety net for a hang — the burden of defensive timeouts is entirely on us,
  not mitigated by the scoring function the way token cost is (Q7).
