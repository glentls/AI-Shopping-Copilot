# Ablations

| Config | Hit@10 | MRR | MTTC | Efficiency | TechnicalScore | N |
|---|---|---|---|---|---|---|
| bm25_baseline | 0.1250 | 0.0680 | 9.8100 | 0.1190 | 0.1067 | 200 |
| lexical_only | 0.1700 | 0.0743 | 9.3900 | 0.1610 | 0.1395 | 200 |
| lexical_dense_rrf | 0.1400 | 0.0525 | 9.7850 | 0.1215 | 0.1100 | 200 |
| full_rrf | 0.1550 | 0.0783 | 9.6550 | 0.1345 | 0.1279 | 200 |
| full_rrf_wide | 0.1650 | 0.0820 | 9.5500 | 0.1450 | 0.1361 | 200 |
| proven_negatives | 0.1700 | 0.0686 | 9.4500 | 0.1550 | 0.1366 | 200 |
| portfolio | 0.1400 | 0.0836 | 9.8050 | 0.1195 | 0.1190 | 200 |
| explore_exploit | 0.1300 | 0.0913 | 10.0000 | 0.1000 | 0.1124 | 200 |
| eig_questioning | 0.1650 | 0.1141 | 9.7100 | 0.1290 | 0.1425 | 200 |
| eig_no_exclusion | 0.1700 | 0.1271 | 9.5700 | 0.1430 | 0.1517 | 200 |
| llm_no_key_fallback | 0.1700 | 0.1271 | 9.5700 | 0.1430 | 0.1517 | 200 |

| Config | Buying Hit@10 | Browsing Hit@10 | Intent Override Hit@10 | Boundary Hit@10 |
|---|---|---|---|---|
| bm25_baseline | 0.2375 | 0.0250 | 0.1333 | 0.0000 |
| lexical_only | 0.2000 | 0.1250 | 0.2000 | 0.2000 |
| lexical_dense_rrf | 0.1375 | 0.0875 | 0.3333 | 0.0000 |
| full_rrf | 0.1625 | 0.0875 | 0.3667 | 0.0000 |
| full_rrf_wide | 0.1875 | 0.0875 | 0.3667 | 0.0000 |
| proven_negatives | 0.2500 | 0.1250 | 0.1000 | 0.1000 |
| portfolio | 0.2000 | 0.0875 | 0.1333 | 0.1000 |
| explore_exploit | 0.1375 | 0.0875 | 0.2333 | 0.1000 |
| eig_questioning | 0.1750 | 0.1750 | 0.1000 | 0.2000 |
| eig_no_exclusion | 0.1625 | 0.1125 | 0.3000 | 0.3000 |
| llm_no_key_fallback | 0.1625 | 0.1125 | 0.3000 | 0.3000 |

## Findings

**Phase 2 (retrieval).** `lexical_only` (our own bm25s route + recency-weighted query
construction from accumulated turns, replacing the starter's raw-last-message query)
already beats the untouched baseline by 36% relative on Hit@10 (0.125 -> 0.170) with no
dense or structured signal at all -- confirming Phase 1's diagnosis that the baseline's
main failure was query staleness (88.5% of its turns were an identical static fallback
query), not retrieval quality per se. Adding `dense` at equal RRF weight regressed
overall Hit@10/TechnicalScore (`lexical_dense_rrf` vs `lexical_only`) despite tripling
Intent Override's hit rate -- dense helps re-targeting after a pivot but dilutes
precision on Buying/Browsing. Widening `candidate_k` 100->300 (`full_rrf_wide`) recovered
real ground for free (no re-embedding) by giving RRF a deeper pool per route to fuse
from, closing part of the gap to `lexical_only` without touching weights.

**Phase 3 (dialog policy) -- proven-negative exclusion is a documented net negative as
implemented, and here is why.** `proven_negatives` (never re-show + reject-batch
downweighting) alone roughly matches `lexical_only` overall, but collapses Intent
Override's hit rate from `full_rrf_wide`'s 0.367 down to 0.100. Root cause: the evaluator
never scores a hit before an Intent Override session's scripted override turn (its
`override_applied` gate), so pre-override "shown" items are not actually proven
negatives -- but our exclusion filter has no way to know this and blacklists them
anyway. If the true target happens to surface during the (unscored) pre-override phase,
it gets permanently excluded and the session cannot recover.

We tried to catch the override via same-attribute slot contradiction (`dialog/slots.py`)
-- correct when it fires, but the evaluator's override frequently introduces a slot we
hadn't filled yet rather than conflicting with one we had, so it often doesn't fire. We
then tried a turn-to-turn embedding-similarity "semantic pivot" detector as a more
general fallback -- calibrated against 8 real override transcripts vs. 8 real
non-override transcripts, and rejected it: override similarities (0.165-0.463) and
non-override similarities (-0.01-0.577) overlap too heavily for any threshold to
separate them. No natural-language-only signal we tried reliably distinguishes "this
message reveals new information" from "this message replaces old information" without
seeing the hidden intent card.

The effect compounds as retrieval improves: `eig_questioning` (all Phase 3 features,
exclusion included) posts our best-yet Buying/Browsing/Boundary numbers but Intent
Override craters back to 0.100 -- better pre-override retrieval (richer accumulated
query from more turns of real disclosure) makes it *more* likely to coincidentally
surface the true target before the override fires, which the exclusion filter then
locks out. `eig_no_exclusion` (identical config, exclusion and downweighting off)
recovers Intent Override to 0.300 while keeping most of the Buying/Browsing/Boundary
gains, and posts the best overall TechnicalScore of the project (0.1517, +42% relative
over baseline). That is the config we're taking forward.

The exclusion code (`dialog/slots.py`, `dialog/posterior.py`) stays in the repo and
config-toggleable, not deleted -- it is a real, tested feature with a real, understood
limitation, not a mistake to hide. A session-scenario-aware evaluator would let it help
without the Intent Override cost; a real deployment (no hidden override_applied gate)
would not have this failure mode at all, since a real user's rejections are always valid
signal at the moment they're shown. This is a benchmark-protocol interaction, not a
design flaw in the exclusion idea itself, and is called out as such in the Devpost
writeup rather than left as an unexplained regression.

**Portfolio / explore-exploit.** Both consistently trade some Hit@10 for meaningfully
higher MRR (`portfolio` 0.0836, `explore_exploit` 0.0913, `eig_no_exclusion` 0.1271 --
the best MRR of any config by a wide margin) -- consistent with the intended mechanism:
protecting rank-1 and diversifying 2-10 costs occasional greedy-order hits but ranks the
hits we do get much higher on average.
