"""Dense embedding index over the frozen catalog.

Phase 3 scope: build the document text per product, encode all 50k with a pluggable encoder,
L2-normalise at build time (so cosine similarity in Phase 4 is a single matmul), and cache the
matrix to disk with a key that can never silently serve stale vectors. No retrieval here --
`dense_search()` lands in Phase 4 (`src/retrieval/dense.py`), consuming the `EmbeddingIndex`
this module produces.

Cache key = sha256(model name + doc-template version + catalog SHA-256)[:16]. All three inputs
are baked into the filename *and* re-verified against the sidecar meta on load; any mismatch
rebuilds (or hard-fails if `allow_rebuild=False`) rather than proceeding with wrong vectors --
that silent-stale-cache failure is the one this whole team is most exposed to.

Cache files, per key:
    <cache_dir>/<key>.npy         float32 (n, dim), row i  <->  parent_asins[i], L2-normalised
    <cache_dir>/<key>.asins.txt   the parallel parent_asin index, one id per line, catalog order
    <cache_dir>/<key>.meta.json   provenance: model, template version, catalog sha, dim, build time

Nothing in this module imports another component. `torch` / `sentence_transformers` /
`transformers` are imported lazily inside the encoders so the module (and the doc-template
helpers, which the unit tests exercise) load with numpy alone.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import load_config

_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Document text template
# --------------------------------------------------------------------------- #


def _clean(value: str, limit: int | None = None) -> str:
    text = _WS_RE.sub(" ", str(value)).strip()
    return text[:limit].rstrip() if limit else text


def build_doc_text(product: dict, template: dict) -> str:
    """Render one catalog row to the string the encoder sees.

    template keys (config.yaml `retrieval.embedding.doc_template`):
      version           str  -- bump on ANY change here; part of the cache key
      field_sep         str  -- separator between sections
      category_slice    [lo, hi] -- categories[lo:hi]; drops the useless root (0) and, by
                                    default, the hyper-specific leaf
      max_features      int  -- first N feature strings
      include_details   bool -- the build plan's spec omitted details; we include it and
                                measure the delta (see docs/r1_log.md)
      details_keys      list -- which details keys contribute when include_details is true
      description_chars int  -- truncate the joined description to this many chars
    """
    sep = template.get("field_sep", " | ")
    parts: list[str] = []

    title = _clean(product.get("title") or "")
    if title:
        parts.append(title)

    categories = [str(c) for c in (product.get("categories") or [])]
    lo, hi = template.get("category_slice", [2, 5])
    levels = [c for c in categories[lo:hi] if c]
    if levels:
        parts.append(" ".join(levels))

    store = product.get("store")
    if store not in (None, ""):
        parts.append(_clean(store))

    features = [_clean(f) for f in (product.get("features") or [])]
    features = [f for f in features if f][: template.get("max_features", 8)]
    if features:
        parts.append(" ".join(features))

    if template.get("include_details", True):
        details = product.get("details")
        if isinstance(details, dict):
            wanted = {k.lower() for k in template.get("details_keys", [])}
            attrs = [
                f"{key}: {_clean(value)}"
                for key, value in details.items()
                if key.lower() in wanted and value not in (None, "", [])
            ]
            if attrs:
                parts.append(" ".join(attrs))

    description = _clean(" ".join(str(d) for d in (product.get("description") or [])),
                         template.get("description_chars", 300))
    if description:
        parts.append(description)

    return sep.join(parts)


# --------------------------------------------------------------------------- #
# Encoders -- pluggable behind one interface
# --------------------------------------------------------------------------- #


class Encoder:
    """Interface: `.name` (goes into the cache key), `.dim`, and `.encode(texts) -> (n, dim)
    float32, L2-normalised`. Subclasses implement `_load()` (once, lazily) and
    `_encode_batch()`. Batching is length-sorted and runs under `torch.no_grad()` in the
    subclasses that use torch."""

    name: str
    dim: int

    def __init__(self, batch_size: int = 128) -> None:
        self.batch_size = batch_size
        self._loaded = False

    def _load(self) -> None:  # pragma: no cover - exercised only in full builds
        raise NotImplementedError

    def _encode_batch(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._loaded:
            self._load()
            self._loaded = True
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            vecs = np.asarray(self._encode_batch([texts[i] for i in idx]), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            np.clip(norms, 1e-12, None, out=norms)
            out_slice = vecs / norms
            for row, original in zip(out_slice, idx):
                out[original] = row
        return out


class SentenceTransformerEncoder(Encoder):
    """Any sentence-transformers model. Default: BAAI/bge-small-en-v1.5 (384-dim)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 128,
                 max_seq_length: int = 256) -> None:
        super().__init__(batch_size)
        self.name = model_name
        self.max_seq_length = max_seq_length
        self._model = None
        self.dim = 384  # corrected on _load

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.name)
        try:
            self._model.max_seq_length = self.max_seq_length
        except Exception:
            pass
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, batch_size=len(texts), convert_to_numpy=True,
            normalize_embeddings=False, show_progress_bar=False,
        )


class BlairEncoder(Encoder):
    """hyp1231/blair-roberta-base -- roberta-base continually pretrained on (item metadata,
    review text) pairs from THIS dataset. Not a sentence-transformers model: encode with
    AutoModel, pool last_hidden_state[:, 0], then L2-normalise (done in Encoder.encode)."""

    def __init__(self, model_name: str = "hyp1231/blair-roberta-base", batch_size: int = 64,
                 max_seq_length: int = 256) -> None:
        super().__init__(batch_size)
        self.name = model_name
        self.max_seq_length = max_seq_length
        self._tok = None
        self._model = None
        self.dim = 768

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.name)
        self._model = AutoModel.from_pretrained(self.name)
        self._model.eval()
        self.dim = int(self._model.config.hidden_size)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            batch = self._tok(
                texts, padding=True, truncation=True,
                max_length=self.max_seq_length, return_tensors="pt",
            )
            cls = self._model(**batch).last_hidden_state[:, 0]
        return cls.cpu().numpy()


_ENCODERS = {
    "BAAI/bge-small-en-v1.5": SentenceTransformerEncoder,
    "hyp1231/blair-roberta-base": BlairEncoder,
}


def make_encoder(model_name: str, batch_size: int, max_seq_length: int) -> Encoder:
    factory = _ENCODERS.get(model_name)
    if factory is None:
        # Unknown name -> assume it is a sentence-transformers model.
        factory = SentenceTransformerEncoder
    return factory(model_name, batch_size=batch_size, max_seq_length=max_seq_length)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


@dataclass
class EmbeddingIndex:
    vectors: np.ndarray        # (n, dim) float32, L2-normalised; row i <-> parent_asins[i]
    parent_asins: list[str]
    model_name: str
    template_version: str
    catalog_sha: str
    dim: int
    build_seconds: float = 0.0


def catalog_sha256(catalog_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(catalog_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(model_name: str, template_version: str, catalog_sha: str) -> str:
    raw = f"{model_name}\0{template_version}\0{catalog_sha}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _iter_catalog(catalog_path: str | Path):
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_or_load(
    catalog_path: str | Path,
    config: dict | None = None,
    encoder: Encoder | None = None,
    *,
    rebuild: bool = False,
    allow_rebuild: bool = True,
) -> EmbeddingIndex:
    config = config or load_config()
    emb_cfg = config["retrieval"]["embedding"]
    template = emb_cfg["doc_template"]
    template_version = str(template["version"])
    model_name = encoder.name if encoder is not None else emb_cfg["model"]

    catalog_sha = catalog_sha256(catalog_path)
    key = cache_key(model_name, template_version, catalog_sha)
    cache_dir = Path(config["retrieval"]["embedding"].get("cache_dir", ".cache/embeddings"))
    if not cache_dir.is_absolute():
        cache_dir = Path(catalog_path).resolve().parent.parent / cache_dir
    npy_path = cache_dir / f"{key}.npy"
    asins_path = cache_dir / f"{key}.asins.txt"
    meta_path = cache_dir / f"{key}.meta.json"

    if not rebuild and npy_path.exists() and asins_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mismatch = [
            f"{field}: cached {meta.get(field)!r} != expected {expected!r}"
            for field, expected in (
                ("model_name", model_name),
                ("template_version", template_version),
                ("catalog_sha", catalog_sha),
            )
            if str(meta.get(field)) != str(expected)
        ]
        if mismatch:  # key is derived from these three, so this should be impossible -- defend anyway
            raise RuntimeError(
                f"embedding cache {npy_path.name} is internally inconsistent: {'; '.join(mismatch)}"
            )
        vectors = np.load(npy_path).astype(np.float32, copy=False)
        parent_asins = asins_path.read_text(encoding="utf-8").splitlines()
        if vectors.shape[0] != len(parent_asins):
            raise RuntimeError(
                f"embedding cache {npy_path.name}: {vectors.shape[0]} vectors vs "
                f"{len(parent_asins)} parent_asins"
            )
        return EmbeddingIndex(
            vectors=vectors, parent_asins=parent_asins, model_name=model_name,
            template_version=template_version, catalog_sha=catalog_sha, dim=vectors.shape[1],
            build_seconds=float(meta.get("build_seconds", 0.0)),
        )

    if not allow_rebuild:
        raise RuntimeError(
            f"no valid embedding cache for key {key} "
            f"(model={model_name}, template={template_version}, catalog_sha={catalog_sha[:12]}) "
            f"and allow_rebuild=False"
        )

    # Warn loudly if a cache for the same model exists under a different key (stale after a
    # template bump or a catalog swap) -- we do not touch it, but the operator should know.
    for other in cache_dir.glob("*.meta.json"):
        try:
            om = json.loads(other.read_text(encoding="utf-8"))
        except Exception:
            continue
        if om.get("model_name") == model_name and other.name != meta_path.name:
            print(
                f"[embed_index] NOTE stale cache {other.name} for {model_name} "
                f"(template {om.get('template_version')}, catalog {str(om.get('catalog_sha'))[:12]}) "
                f"-- building fresh {key}, old file left in place"
            )

    threads = emb_cfg.get("torch_num_threads")
    if threads:
        try:
            import torch

            torch.set_num_threads(int(threads))
        except Exception:
            pass

    if encoder is None:
        encoder = make_encoder(
            model_name,
            batch_size=int(emb_cfg.get("batch_size", 128)),
            max_seq_length=int(emb_cfg.get("max_seq_length", 256)),
        )

    products = list(_iter_catalog(catalog_path))
    parent_asins = [str(p["parent_asin"]) for p in products]
    texts = [build_doc_text(p, template) for p in products]
    print(
        f"[embed_index] building {len(texts)} docs with {model_name} "
        f"(template {template_version}); sample doc:\n  {texts[0][:240]!r}"
    )

    started = time.perf_counter()
    vectors = encoder.encode(texts).astype(np.float32, copy=False)
    build_seconds = round(time.perf_counter() - started, 2)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, vectors)
    asins_path.write_text("\n".join(parent_asins), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "template_version": template_version,
                "catalog_sha": catalog_sha,
                "dim": int(vectors.shape[1]),
                "count": int(vectors.shape[0]),
                "build_seconds": build_seconds,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[embed_index] wrote {npy_path.name} {vectors.shape} float32 "
        f"({npy_path.stat().st_size / 1e6:.1f} MB) in {build_seconds}s"
    )
    return EmbeddingIndex(
        vectors=vectors, parent_asins=parent_asins, model_name=model_name,
        template_version=template_version, catalog_sha=catalog_sha, dim=int(vectors.shape[1]),
        build_seconds=build_seconds,
    )


if __name__ == "__main__":  # manual build entry point
    import argparse

    parser = argparse.ArgumentParser(description="Build/refresh a dense embedding cache")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--model", default=None, help="override config retrieval.embedding.model")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    if args.model:
        cfg = json.loads(json.dumps(cfg))  # cheap deep copy of the plain dict
        cfg["retrieval"]["embedding"]["model"] = args.model
    index = build_or_load(args.catalog, cfg, rebuild=args.rebuild)
    print(
        f"OK  model={index.model_name}  dim={index.dim}  n={len(index.parent_asins)}  "
        f"template={index.template_version}  catalog_sha={index.catalog_sha[:12]}  "
        f"build={index.build_seconds}s"
    )
