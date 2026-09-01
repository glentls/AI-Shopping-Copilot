"""Offline precompute: embed the catalog once with the dense route's model and cache the
result to disk (retrieval/dense.py only ever loads this cache; it never re-embeds the
catalog at runtime). Re-run this whenever data/catalog.jsonl changes.

Usage:
    python -m scripts.build_dense_index

Writes data/dense_index/{ids.json,embeddings.npy} (gitignored, regenerable -- see
final_evaluation_faq.md #4: "Precomputed local artifacts do not need to be rebuilt in
memory at startup... supplied through documented and reproducible download/build
instructions rather than committed directly to the repository").
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from retrieval.catalog import dense_text, load_catalog
from retrieval.dense import DEFAULT_MODEL_CACHE_DIR, DEFAULT_MODEL_NAME, DEFAULT_THREADS


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute and cache dense embeddings for the catalog")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output-dir", default="data/dense_index")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-cache-dir", default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_THREADS,
        help="onnxruntime intra-op threads (default 4, empirically best on dev hardware -- "
        "see retrieval/dense.py docstring). Deliberately NOT using fastembed's `parallel=` "
        "multiprocessing: worker-process spawn overhead measured slower than single-process "
        "on this machine.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=2000, help="print a flushed progress line every N embedded rows"
    )
    args = parser.parse_args()

    from fastembed import TextEmbedding

    products = load_catalog(args.catalog)
    ids = list(products.keys())
    texts = [dense_text(products[asin]) for asin in ids]

    print(f"Embedding {len(ids)} products with {args.model_name} (threads={args.threads}) ...", flush=True)
    model = TextEmbedding(model_name=args.model_name, cache_dir=args.model_cache_dir, threads=args.threads)
    start = time.time()
    vectors: list[np.ndarray] = []
    done = 0
    for vector in model.embed(texts, batch_size=args.batch_size):
        vectors.append(vector)
        done += 1
        if done % args.progress_every == 0:
            elapsed_so_far = time.time() - start
            rate = done / elapsed_so_far if elapsed_so_far > 0 else 0.0
            eta = (len(ids) - done) / rate if rate > 0 else float("inf")
            print(f"  {done}/{len(ids)} embedded ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)
    embeddings = np.vstack(vectors).astype(np.float32)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s -> {embeddings.shape}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    (output_dir / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"Wrote {output_dir}/embeddings.npy and {output_dir}/ids.json")


if __name__ == "__main__":
    main()
