# Lane B: retrieval and ranking

## Design

The retriever uses three independent routes and reciprocal-rank fusion (RRF):

- SQLite FTS5 BM25 over title, features, description, categories, store, and details. The index is built once on disk; `Agent.__init__` opens it read-only.
- `sentence-transformers/all-MiniLM-L6-v2` at revision `5641a7880f40ebf4035d05e60c5f9b7a9c272c84`, using its 384-dimensional ONNX export. A small local WordPiece implementation avoids `transformers`, `sentence-transformers`, and PyTorch.
- Confidence-weighted slot matches and soft penalties for negative polarity and over-budget products. A sweep after Lane C landed selected `TJ_SLOT_WEIGHT=12`: it produces the best MRR/TechnicalScore in the 10–15 neighborhood while preserving the complete 300-candidate list.

BM25, cosine, and exact-phrase raw scores are never added together. BM25 and dense ranks are fused with RRF; the exact-phrase route is a 0.35-weight hedge for the public simulator's copied catalog clauses. Dense has weight 0.10 when an exact clause or the generic “still exploring” opener supplies stronger lexical evidence, and weight 1.0 for natural non-generic turns where semantic-only candidates need to surface. `TJ_DENSE_WEIGHT` can override the confidence-aware value for ablation runs.

No constraint hard-filters candidates. A retracted or negative slot can move a product down, but cannot remove it from the candidate set. Query construction drops no-preference/filler replies. BM25 retains all informative lexical evidence, matching Lane C's folded state behavior; semantic retrieval and exact-phrase matching retire stale pre-override intent. Popularity (`rating_number`) and aggregate profile preference tags are final tie-breakers. Every candidate records scorer components and a short `why` clause.

## Build and artifacts

Run from the repository root:

```bash
python3 -m tools.build_index
```

On the first run, the command downloads the pinned MiniLM ONNX model and vocabulary from Hugging Face. If ONNX Runtime is not installed, it installs version 1.22.1 and NumPy into `artifacts/_vendor`; it does not modify the frozen environment files. This is the only one-off network step. Subsequent builds reuse the model and local runtime.

NumPy is loaded lazily only when dense retrieval is constructed. Importing the starter, constructing an `Agent`, and running `tools.build_index --skip-dense` all work without NumPy or site packages. When dense dependencies or artifacts are absent, `Retriever` automatically keeps BM25 plus the exact-phrase route instead of failing the agent import.

Measured on the 50,000-product catalog on an Apple ARM development machine:

| Artifact | Logical size |
|---|---:|
| `attributes.json` | 4.6 MiB |
| `bm25.sqlite3` | 97.1 MiB |
| MiniLM `model.onnx` + vocabulary | 86.4 MiB |
| `dense_vectors.npy` (float16, 50,000 x 384) | 36.6 MiB |
| `dense_asins.npy` | 3.1 MiB |
| Local ONNX Runtime environment (only when needed) | about 220 MiB |

The complete first build took 560 seconds: about 30 seconds for attributes and BM25, then 530 seconds for CPU embedding with four build threads. Dense batches stream into a temporary NumPy file and replace the final artifact atomically only after success. Artifact output is gitignored.

At runtime the float16 matrix is converted once to float32 because CPU BLAS is substantially faster for float32 dot products. Full `Agent` initialization was 0.66 seconds, well below the 10-second limit.

## Latency

Fifty warm end-to-end `search` + `rerank` calls on the full catalog. BM25 and
dense retrieval execute concurrently before deterministic RRF:

| Query shape | Median | p95-like | Maximum |
|---|---:|---:|---:|
| Natural semantic query | 27.5 ms | 27.8 ms | 28.5 ms |
| Exact catalog-style clause | 34.5 ms | 41.4 ms | 49.8 ms |

The isolated dense query is about 5.5 ms. Dynamic ONNX padding is important: a one-query batch is padded only to its actual token length rather than the 128-token catalog maximum. Profile-tag FTS ranks are cached by the normalized tag set.

## Lane C integration and ranking results

Lane C added deep recommendation paging after this branch's first benchmark. The current comparison therefore uses the new `origin/main` score of 0.8370, not the earlier 0.7483 wiring baseline.

| Configuration | TechnicalScore | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Lane C `origin/main` | 0.8370 | 0.9850 | 0.6105 | 2.930 |
| Rebased Lane B before fixes | 0.8163 | 0.9600 | 0.5905 | 3.045 |
| Final fused ranking | **0.8445** | **0.9850** | **0.6261** | **2.790** |

Public-set scorer ablation with the final confidence-aware soft reranker:

| Retrieval mode | TechnicalScore | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| BM25 only | 0.8322 | 0.9850 | 0.5922 | 2.900 |
| Dense only | 0.5219 | 0.6600 | 0.3150 | 6.130 |
| Fused BM25 + dense + exact | **0.8445** | **0.9850** | **0.6261** | **2.790** |

The final per-scenario result is:

| Scenario | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|
| Buying | 0.975 | 0.602 | 2.40 |
| Browsing | 1.000 | 0.629 | 2.46 |
| Intent override | 0.967 | 0.629 | 4.53 |
| Boundary | 1.000 | 0.783 | 3.30 |

The final rank-at-first-hit profile has 96 of 197 hits at rank 1. `--depth` confirms all 200 targets occur within the 300-candidate ranking: the three scored misses finish at depths 11–20 (one) and 51–100 (two). No candidate list is truncated for the policy pager.

Slot-weight sweeps with the final fusion settings:

| `TJ_SLOT_WEIGHT` | TechnicalScore | HitRate@10 | MRR | MTTC |
|---:|---:|---:|---:|---:|
| 1 | 0.8400 | 0.985 | 0.609 | 2.75 |
| 3 | 0.8410 | 0.985 | 0.611 | 2.75 |
| 5 | 0.8430 | 0.985 | 0.618 | 2.74 |
| 11 | 0.8436 | 0.985 | 0.623 | 2.79 |
| **12** | **0.8445** | **0.985** | **0.626** | **2.79** |
| 13 | 0.8420 | 0.985 | 0.617 | 2.79 |

`python3 -m tools.bench --verify` reports exact agreement between replay and the untouched evaluator for hit rate, MRR, MTTC, and TechnicalScore.
