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
from retrieval.dense import DEFAULT_MODEL_CACHE_DIR, DEFAULT_MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute and cache dense embeddings for the catalog")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output-dir", default="data/dense_index")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-cache-dir", default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    from fastembed import TextEmbedding

    products = load_catalog(args.catalog)
    ids = list(products.keys())
    texts = [dense_text(products[asin]) for asin in ids]

    print(f"Embedding {len(ids)} products with {args.model_name} ...")
    model = TextEmbedding(model_name=args.model_name, cache_dir=args.model_cache_dir)
    start = time.time()
    embeddings = np.vstack(list(model.embed(texts, batch_size=args.batch_size))).astype(np.float32)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s -> {embeddings.shape}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    (output_dir / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"Wrote {output_dir}/embeddings.npy and {output_dir}/ids.json")


if __name__ == "__main__":
    main()
