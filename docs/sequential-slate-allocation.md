# Sequential slate allocation: exact objective and safe optimization boundary

## Status and purpose

This is a decision-support artifact for ShopLens, not a new runtime planner and
not evidence that any candidate improves the Agent. It derives the exact
TechnicalScore contribution of each recommendation opportunity, proves the
fixed-belief open-loop optimum, and identifies what would be needed before that
model could safely become an adaptive policy.

The companion implementation is
`scripts/sequential_slot_oracle.py`. It uses only the Python standard library;
`tests/test_sequential_slot_oracle.py` checks the score identity, the allocation
theorem against brute force, the non-chronological counterexample, aggregate
headroom, validation, and the command-line interface.

## 1. Start from what the evaluator actually scores

For each session, the evaluator examines at most ten unique, catalog-valid
products per turn. The session stops at the first turn containing the hidden
target. A hit records its turn $t\in\{1,\ldots,10\}$ and rank
$r\in\{1,\ldots,10\}$; a miss is assigned turn 11 and reciprocal rank zero.

The aggregate objective over $N$ sessions is

$$
\operatorname{Score}
=0.50\operatorname{HR@10}
+0.30\operatorname{MRR}
+0.20\frac{11-\operatorname{MTTC}}{10}.
$$

The clipping in the specification is inactive for valid turns 1--10 and the
miss value 11. Because every term is an average, the objective is also the mean
of an exact per-session utility:

$$
u(t,r)=0.50+\frac{0.30}{r}+0.02(11-t)
\quad\text{for a hit},
\qquad
u(\text{miss})=0.
$$

This identity is important: it lets us attribute a score change at session
level without inventing a proxy objective. The implementation rounds HR, MRR,
and MTTC to six decimals before combining them and rounds the final score once
more. Thus the identity is exact for the underlying unrounded metrics; a score
reconstructed from session utilities can differ from the published value only
by that sub-millionth-scale rounding.

Some useful values are:

| Opportunity | Exact utility | Interpretation |
|---|---:|---|
| turn 1, rank 1 | 1.00 | Best possible session outcome |
| turn 1, rank 10 | 0.73 | Early but poorly ranked hit |
| turn 2, rank 1 | 0.98 | Later by one turn, ranked first |
| turn 10, rank 1 | 0.82 | Last-turn hit ranked first |
| turn 10, rank 10 | 0.55 | Worst valid hit; still much better than a miss |
| miss | 0.00 | No target in any scored slate |

Consequences follow directly:

The changes below are per-session utilities; in an (N)-session aggregate,
each contributes its value divided by (N).

- Moving a fixed hit one turn earlier adds exactly `0.02` to that session.
- Moving a fixed hit from rank 10 to rank 1 on the same turn adds `0.27`.
- Recovering a miss is worth between `0.55` and `1.00`, depending on turn and
  rank.
- For every $t=1,\ldots,9$,
  $u(t+1,1)-u(t,10)=0.25$. A first-ranked opportunity on the next turn is
  therefore worth much more than a tenth-ranked opportunity now.

## 2. Separate the three decisions

“Better recommendations” hides three different optimization levers:

| Lever | Decision | Primary metric effect | ShopLens example |
|---|---|---|---|
| Per-turn membership | Which products enter the Top 10? | HitRate@10, and possibly the other metrics | J lets disclosure-derived reranking select from a wider pool |
| Within-turn ranking | In what order are those ten shown? | MRR | Phrase, popularity, and profile rerankers |
| Cross-turn allocation | Which products are saved for or removed from later turns? | MTTC and multi-turn hit coverage, with possible MRR effects | K withholds already-scored products after a non-hit |

They interact. A change can recover targets while lowering their ranks, or
increase cross-turn breadth while changing clarification and future retrieval.
No single one of HR@10, MRR, or MTTC is a sufficient optimization objective.

## 3. Fixed-belief open-loop model

Suppose candidate product $i$ has a fixed belief $p_i\geq0$ of being the
target. For a literal expected-score interpretation, the products are mutually
exclusive target hypotheses and $\sum_i p_i\leq1$; any residual probability
means the target is outside the scheduled candidates. The allocation ordering
also works with unnormalized non-negative relevance weights, but its value is
then an objective index rather than an expected TechnicalScore.

Let $S=\{(t,r):t=1,\ldots,T;\ r=1,\ldots,R\}$ be the available slots. Under
the assumptions below, assigning each product to at most one slot gives

$$
\max_x \sum_i\sum_{s\in S}p_i u_s x_{is}
$$

subject to each candidate and each slot being used at most once. Unassigned
candidates contribute zero.

### Theorem: sort globally, not turn by turn

Sort candidate beliefs in descending order and sort **all** `(turn, rank)`
slots in descending utility. Pair the two lists. If there are more candidates
than slots, omit the lowest beliefs; if there are fewer, use only the highest
utility slots.

The exchange proof is short. Consider beliefs $p_i\geq p_j$ and utilities
$u_a\geq u_b$. Aligning high with high instead of crossing them changes the
objective by

$$
(p_i u_a+p_j u_b)-(p_i u_b+p_j u_a)
=(p_i-p_j)(u_a-u_b)\geq0.
$$

Repeatedly uncrossing inversions yields the sorted assignment. This is the
rearrangement inequality; no external solver is required.

### Why chronological greedy is wrong

Take four target probabilities `[0.4, 0.3, 0.2, 0.1]` and two ranks over two
turns. Slot utilities in globally descending order are

```text
(turn 1, rank 1): 1.00
(turn 2, rank 1): 0.98
(turn 1, rank 2): 0.85
(turn 2, rank 2): 0.83
```

The global assignment has expected score

```text
0.4(1.00) + 0.3(0.98) + 0.2(0.85) + 0.1(0.83) = 0.947.
```

Filling all of turn 1 before turn 2 instead gives

```text
0.4(1.00) + 0.3(0.85) + 0.2(0.98) + 0.1(0.83) = 0.934.
```

The `0.013` gap is not a numerical accident. Rank rewards dominate a one-turn
delay, so a high-belief candidate can optimally be saved for rank 1 next turn
while a lower-belief candidate occupies a weaker rank now.

### Assumptions of the theorem

The result is exact only for this deliberately narrow model:

1. Candidate beliefs do not change across turns.
2. A product appears at most once in the schedule.
3. A no-hit observation merely removes shown products; it does not alter the
   relative beliefs among the rest.
4. Customer replies, overrides, clarification questions, retrieval failures,
   and fallback behavior do not change future candidates or beliefs.
5. The objective is the published TechnicalScore, not qualitative judging
   criteria such as explanation quality.

These assumptions make the oracle useful as a ceiling, a diagnostic, and a
unit-testable reference. They are not an adequate description of the live
ShopLens conversation.

## 4. Why this should not become a runtime optimizer before freeze

The real Agent observes a customer reply after every unsuccessful slate. A
question can disclose constraints, an override can invalidate earlier intent,
and retrieval can make an entirely different pool available. A simplified
adaptive formulation is therefore a belief-state sequential decision problem.
For ordered slate $a=(a_1,\ldots,a_k)$, question $q$, and belief
$b_i=\Pr(i\text{ is the target})$:

$$
V_t(b)=\max_{a,q}\left\{
\sum_{r=1}^{k} b_{a_r}u(t,r)
+\left(1-\sum_{r=1}^{k}b_{a_r}\right)
\mathbb E_{o\mid\text{no hit},b,a,q}
\left[V_{t+1}(\tau(b,a,q,o))\right]
\right\},
\qquad V_{11}=0.
$$

Here $o$ is the customer reply observed only after a non-hit, and $\tau$
conditions on both the eliminated slate and that reply. A complete state would
also include disclosed constraints, override/hit-eligibility status, and the
retrieval pool. ShopLens does not currently have either a calibrated target
posterior or a validated response kernel, so putting this Bellman model into
production would add unearned complexity.

Questions also do **not** consume a separate evaluator action: the Agent
returns a slate and `ask_attribute` together. That removes a simple mechanical
"question versus recommendation" cost, but it does not prove that asking every
turn is causally beneficial. In particular, `ask_attribute=None` reveals no
new constraint under the local simulator, while `other` can reveal up to two
remaining constraints; neither fact proves that `other` is maximally
informative on the hidden evaluation distribution.

The safe use now is offline: use the exact utility to understand failures and
to reject changes whose apparent recall gain is bought with a larger ranking
or timing loss.

## 5. What J, K, and Y mean in this framework

### J: evidence-scoped per-turn membership

J gives disclosure-derived reranking access to a Top-50 window, then freezes
Top-10 membership before the later popularity and profile stages run.
It is a heuristic for **per-turn membership and ranking**, not the global
allocation theorem above. It has useful behavior tests but no canonical dev or
holdout row yet, so its benefit remains a hypothesis until it passes the frozen
evaluation gate. It can still displace a target if phrase evidence is noisy or
reflects a preference that later changes.

### K: no-repeat exploration on top of T

K enables `exclude_shown` on T. Reaching a later ordinary turn proves that none
of the previously scored products was the target, so removing them is valid
negative evidence. Before an Intent Override conversion is allowed, however,
that inference is invalid; the implementation clears shown-product memory on
override so those products become eligible again.

K is a heuristic for **cross-turn breadth**, not a probability optimizer. It
may still change more than breadth: filtering occurs before the clarification
pool is measured, so `over_general`, the next question, later disclosures, and
rank order can change. A finite retrieval pool can also be exhausted, in which
case the Agent deliberately falls back instead of returning no recommendations.
Those indirect effects are why K requires measured dev and holdout gates even
though the no-hit exclusion itself is logically sound.

### Y: an objective-misalignment warning

Y widened the rerank window for every reranker, including population priors.
The diagnostic recorded in J's commit message says Y recovered two Buying
conversions on dev but lost `0.108854` Buying MRR and finished `0.006243` below
T overall. Those numbers are commit-level diagnostics, not canonical
`results.jsonl` evidence, but the direction is instructive: more hits do not
guarantee a higher joint objective when the new or existing hits are ordered
poorly. J narrows the mechanism in direct response; measurement must determine
whether that is sufficient.

## 6. Exact T headroom, not a promised gain

If the set of hit sessions is held fixed and every existing hit is moved to
turn 1, rank 1, then

$$
\text{ranking headroom}=0.30(\operatorname{HR}-\operatorname{MRR}),
$$

$$
\text{timing headroom}=0.20(\operatorname{HR}-\operatorname{Efficiency}).
$$

The resulting fixed-membership oracle score is exactly HR: every hit is worth
one and every miss remains zero. Recovering every remaining miss at turn 1,
rank 1 adds the separate membership ceiling $1-\operatorname{HR}$.

Using the canonical T aggregates (with the holdout still labelled
exploratory):

| Split | Score | HR | Ranking headroom | Timing headroom | Fixed-membership oracle | Membership ceiling |
|---|---:|---:|---:|---:|---:|---:|
| Dev | 0.866774 | 0.941667 | 0.043726 | 0.031167 | 0.941667 | 0.058333 |
| Holdout (exploratory) | 0.891630 | 0.975000 | 0.051620 | 0.031750 | 0.975000 | 0.025000 |

These are mathematical ceilings conditional on heroic assumptions, not
forecasts for J or K. They do show why preserving rank quality matters: on T
dev, idealized ranking and timing headroom among **already-hit** sessions totals
`0.074893`, exceeding the `0.058333` perfect-membership ceiling.

## 7. How to use the oracle

Score one opportunity:

```powershell
python scripts/sequential_slot_oracle.py utility --turn 2 --rank 1
python scripts/sequential_slot_oracle.py utility --miss
```

Compare the exact global order with chronological fill:

```powershell
python scripts/sequential_slot_oracle.py allocate `
  --beliefs 0.4,0.3,0.2,0.1 --turns 2 --ranks 2
```

JSON output is available with `--format json`. Beliefs that sum above one are
accepted as unnormalized ranking weights, but the CLI explicitly marks that
their objective is not interpretable as an expected TechnicalScore.

Reproduce T dev headroom:

```powershell
python scripts/sequential_slot_oracle.py headroom `
  --hit-rate 0.941667 --mrr 0.795913 --mttc 3.141667
```

Run the focused tests:

```powershell
python -m pytest -q tests/test_sequential_slot_oracle.py
```

## 8. Evidence protocol for candidate attribution

For paired session rows, define

$$
\Delta_i=u_i(\text{candidate})-u_i(T).
$$

Then $N^{-1}\sum_i\Delta_i$ equals the candidate's TechnicalScore change
apart from published metric rounding. Classify each non-zero delta before
telling a story:

- T miss, candidate hit: membership recovery;
- T hit, candidate miss: membership loss;
- both hit on the same turn, different rank: ranking change;
- both hit at the same rank, different turn: timing change;
- both rank and turn differ: mixed change.

The reportable runner deliberately removes `sessions` from canonical
`results.jsonl`, so this attribution cannot be reconstructed from the aggregate
log. If the team captures session rows for diagnosis, that artifact should stay
outside the canonical results path and be labelled diagnostic. The promotion
decision must still use the preregistered reportable aggregate gates; session
inspection is for explanation, not post-hoc tuning.

## 9. Decision boundary

This artifact contributes three things without increasing submission risk:

1. an exact common unit for HR, rank, and timing changes;
2. a proved oracle against which sequential heuristics can be checked; and
3. a clear boundary between what is mathematically established and what still
   requires calibrated beliefs or reportable evidence.

It should influence whether J or K is retained and how the team explains the
result. It should not, by itself, change the submission Agent.
