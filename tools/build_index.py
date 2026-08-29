"""Build every local artifact the agent needs from a clean checkout.

The default command builds the attribute table, persistent FTS5 index, and
MiniLM ONNX embeddings. It downloads the pinned model and, when necessary,
installs ONNX Runtime into the gitignored artifact directory.

    python3 -m tools.build_index
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.attributes import build_attribute_table
from src.retrieval.bm25 import build_bm25_index
from src.retrieval.dense import build_dense_index, ensure_model_files, ensure_onnxruntime


def _size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build agent artifacts")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="developer-only: build FTS/attributes without downloading or embedding MiniLM",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    table = build_attribute_table(args.catalog, args.artifacts)
    elapsed = time.perf_counter() - started

    print(f"attribute table built in {elapsed:.1f}s -> {Path(args.artifacts) / 'attributes.json'}")
    for slot in table.slots():
        print(f"  {slot:10} coverage {table.coverage(slot):6.1%}")

    stage = time.perf_counter()
    bm25_path = build_bm25_index(args.catalog, args.artifacts)
    print(
        f"BM25 index built in {time.perf_counter() - stage:.1f}s -> "
        f"{bm25_path} ({_size(bm25_path)})"
    )

    if args.skip_dense:
        print("dense index skipped by request")
    else:
        stage = time.perf_counter()
        print("preparing pinned all-MiniLM-L6-v2 ONNX model and runtime...")
        ensure_onnxruntime(args.artifacts, install=True)
        model_path, _ = ensure_model_files(args.artifacts)
        print(f"model ready -> {model_path} ({_size(model_path)})")
        vector_path, asin_path = build_dense_index(
            args.catalog,
            args.artifacts,
            batch_size=max(1, args.batch_size),
            install_runtime=False,
        )
        print(
            f"dense index built in {time.perf_counter() - stage:.1f}s -> "
            f"{vector_path} ({_size(vector_path)}), {asin_path} ({_size(asin_path)})"
        )

    print(f"all requested artifacts built in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
