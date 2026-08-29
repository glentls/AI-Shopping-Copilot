# Lane B: retrieval and ranking

## Design

The retriever uses three independent routes and reciprocal-rank fusion (RRF):

- SQLite FTS5 BM25 over title, features, description, categories, store, and details. The index is built once on disk; `Agent.__init__` opens it read-only.
- `sentence-transformers/all-MiniLM-L6-v2` at revision `5641a7880f40ebf4035d05e60c5f9b7a9c272c84`, using its 384-dimensional ONNX export. A small local WordPiece implementation avoids `transformers`, `sentence-transformers`, and PyTorch.
- Confidence-weighted slot matches and soft penalties for negative polarity and over-budget products. Customer confidence is multiplied by Lane A's catalog-source confidence, so a description-only match cannot move a candidate as far as a structured/title match. Post-Lane-A sweeps select `TJ_SLOT_WEIGHT=2` and `TJ_BUDGET_WEIGHT=0.5`.

BM25, cosine, and exact-phrase raw scores are never added together. BM25 and dense ranks are fused with RRF; the exact-phrase route is a 0.275-weight hedge for the public simulator's copied catalog clauses. Dense contributes zero when an exact catalog clause is present, 0.10 for generic browsing, and 1.0 for natural non-generic turns where semantic-only candidates need to surface. `TJ_DENSE_WEIGHT` can override the route-aware value for ablation runs.

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
| `attributes.json` (Lane A artifact v4) | 9.4 MiB |
| `bm25.sqlite3` | 97.1 MiB |
| MiniLM `model.onnx` + vocabulary | 86.4 MiB |
| `dense_vectors.npy` (float16, 50,000 x 384) | 36.6 MiB |
| `dense_asins.npy` | 3.1 MiB |
| Local ONNX Runtime environment (only when needed) | about 220 MiB |

After the Lane A merge, rebuilding attributes and BM25 with `--skip-dense` took 119.2 seconds: 117.1 seconds for the broader provenance-aware attribute pass and 2.1 seconds for BM25. The previously measured CPU dense phase took about 530 seconds with four build threads. Dense batches stream into a temporary NumPy file and replace the final artifact atomically only after success. Artifact output is gitignored.

At runtime the float16 matrix is converted once to float32 because CPU BLAS is substantially faster for float32 dot products. Full `Agent` initialization was 0.66 seconds, well below the 10-second limit.

## Latency

Fifty warm end-to-end `search` + `rerank` calls on the full catalog. BM25 and
dense retrieval execute concurrently before deterministic RRF:

| Query shape | Median | p95-like | Maximum |
|---|---:|---:|---:|
| Natural semantic query | 27.5 ms | 27.8 ms | 28.5 ms |
| Exact catalog-style clause | 34.5 ms | 41.4 ms | 49.8 ms |

The isolated dense query is about 5.5 ms. Dynamic ONNX padding is important: a one-query batch is padded only to its actual token length rather than the 128-token catalog maximum. Profile-tag FTS ranks are cached by the normalized tag set.

## Lane A/C integration and ranking results

Lane A artifact v4 expands slot coverage and adds catalog-source confidence. The unchanged Lane B settings over-weighted those broader matches, so the table includes the post-rebase result before and after retuning.

| Configuration | TechnicalScore | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Lane A/C `origin/main` | 0.8382 | 0.9800 | 0.6277 | 3.005 |
| Rebased Lane B, old weights | 0.8400 | 0.9850 | 0.6087 | 2.755 |
| Retuned fused ranking | **0.8542** | **0.9800** | **0.6718** | **2.865** |

Public-set scorer ablation with the final confidence-aware soft reranker:

| Retrieval mode | TechnicalScore | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| BM25 only | 0.8384 | 0.9800 | 0.6271 | 2.985 |
| Dense only | 0.5377 | 0.7550 | 0.2355 | 6.520 |
| Fused BM25 + dense + exact | **0.8542** | **0.9800** | **0.6718** | **2.865** |

The final per-scenario result is:

| Scenario | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|
| Buying | 0.963 | 0.620 | 2.54 |
| Browsing | 1.000 | 0.666 | 2.56 |
| Intent override | 0.967 | 0.741 | 4.30 |
| Boundary | 1.000 | 0.925 | 3.60 |

The final rank-at-first-hit profile has 110 of 196 hits at rank 1, up from 93 of 197 immediately after the rebase. `--depth` confirms all 200 targets occur within the 300-candidate ranking: two scored misses finish at depths 51–100 and two at 101–300. No candidate list is truncated for the policy pager.

The first post-Lane-A slot sweep exposed the old weight-12 regression:

| `TJ_SLOT_WEIGHT` | TechnicalScore | HitRate@10 | MRR | MTTC |
|---:|---:|---:|---:|---:|
| 1 | 0.8390 | 0.980 | 0.616 | 2.79 |
| 3 | 0.8431 | 0.980 | 0.630 | 2.79 |
| 5 | 0.8389 | 0.980 | 0.614 | 2.77 |
| 7 | 0.8430 | 0.985 | 0.620 | 2.77 |
| 9 | 0.8425 | 0.985 | 0.618 | 2.77 |
| 12 | 0.8400 | 0.985 | 0.609 | 2.75 |
| 15 | 0.8344 | 0.985 | 0.590 | 2.75 |

With source confidence, slot weight 2, budget weight 0.5, and route-aware dense fusion fixed, the exact-phrase sweep selected the final hedge:

| `TJ_EXACT_WEIGHT` | TechnicalScore | HitRate@10 | MRR | MTTC |
|---:|---:|---:|---:|---:|
| 0.200 | 0.8494 | 0.980 | 0.655 | 2.85 |
| 0.250 | 0.8519 | 0.980 | 0.664 | 2.87 |
| **0.275** | **0.8542** | **0.980** | **0.672** | **2.87** |
| 0.300 | 0.8520 | 0.980 | 0.663 | 2.84 |
| 0.325 | 0.8486 | 0.980 | 0.650 | 2.82 |

`python3 -m tools.bench --verify` reports exact agreement between replay and the untouched evaluator for hit rate, MRR, MTTC, and TechnicalScore.
