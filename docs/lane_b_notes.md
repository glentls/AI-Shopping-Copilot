# Lane B: retrieval and ranking

## Design

The retriever uses three independent routes and reciprocal-rank fusion (RRF):

- SQLite FTS5 BM25 over title, features, description, categories, store, and details. The index is built once on disk; `Agent.__init__` opens it read-only.
- `sentence-transformers/all-MiniLM-L6-v2` at revision `5641a7880f40ebf4035d05e60c5f9b7a9c272c84`, using its 384-dimensional ONNX export. A small local WordPiece implementation avoids `transformers`, `sentence-transformers`, and PyTorch.
- Confidence-weighted slot matches and soft penalties for negative polarity and over-budget products. `TJ_SLOT_WEIGHT` defaults to 3.0 rank places.

BM25, cosine, and exact-phrase raw scores are never added together. BM25 and dense ranks are fused with RRF; the exact-phrase route is a 0.35-weight hedge for the public simulator's copied catalog clauses. Dense has weight 0.10 when an exact clause or the generic “still exploring” opener supplies stronger lexical evidence, and weight 1.0 for natural non-generic turns where semantic-only candidates need to surface. `TJ_DENSE_WEIGHT` can override the confidence-aware value for ablation runs.

No constraint hard-filters candidates. A retracted or negative slot can move a product down, but cannot remove it from the candidate set. Query construction drops no-preference/filler replies and retires raw text and exact phrases from before the latest intent override. Popularity (`rating_number`) and aggregate profile preference tags are final tie-breakers. Every candidate records scorer components and a short `why` clause.

## Build and artifacts

Run from the repository root:

```bash
python3 -m tools.build_index
```

On the first run, the command downloads the pinned MiniLM ONNX model and vocabulary from Hugging Face. If ONNX Runtime is not installed, it installs version 1.22.1 into `artifacts/_vendor`; it does not modify the frozen environment files. This is the only one-off network step. Subsequent builds reuse the model and local runtime.

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

## Public-set ablation

All rows use the same soft slot/budget reranker and popularity/profile tie-breaks. “Fused” additionally enables the exact-phrase hedge. Results are from `data/public_set.jsonl` after rebasing this branch onto `origin/main`.

| Retrieval mode | TechnicalScore | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| BM25 only | 0.7054 | 0.8200 | 0.5088 | 3.860 |
| Dense only | 0.3710 | 0.4400 | 0.2486 | 7.180 |
| Fused BM25 + dense + exact | **0.7529** | **0.8700** | **0.5535** | **3.410** |

The untouched post-rebase wiring baseline reproduced at 0.7483 (HitRate@10 0.8650, MRR 0.5541, MTTC 3.520). The dense-only public result is expected to understate its private-set value: public customer messages copy listing text, while dense retrieval is aimed at natural paraphrases such as “for a trip” and “comfortable” versus vacation/walking and cushioning/arch-support language.
