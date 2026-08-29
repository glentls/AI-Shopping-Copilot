"""Small ONNX MiniLM encoder and NumPy dense retrieval index."""

from __future__ import annotations

import importlib
import json
import os
import ssl
import subprocess
import sys
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "5641a7880f40ebf4035d05e60c5f9b7a9c272c84"
MODEL_DIR = "all-MiniLM-L6-v2"
MODEL_FILE = "model.onnx"
VOCAB_FILE = "vocab.txt"
VECTORS_FILE = "dense_vectors.npy"
ASINS_FILE = "dense_asins.npy"
MANIFEST_FILE = "dense_manifest.json"
ONNXRUNTIME_VERSION = "1.22.1"
EMBEDDING_DIMENSION = 384
MAX_SEQUENCE_LENGTH = 128

_BASE_URL = f"https://huggingface.co/{MODEL_NAME}/resolve/{MODEL_REVISION}"
MODEL_URL = f"{_BASE_URL}/onnx/model.onnx?download=true"
VOCAB_URL = f"{_BASE_URL}/vocab.txt?download=true"


@dataclass(frozen=True)
class DenseHit:
    parent_asin: str
    score: float


def _vendor_dir(artifacts_dir: str | Path) -> Path:
    return Path(artifacts_dir) / "_vendor"


def import_onnxruntime(artifacts_dir: str | Path):
    """Import the system package or the build command's local artifact copy."""
    try:
        return importlib.import_module("onnxruntime")
    except ImportError:
        vendor = str(_vendor_dir(artifacts_dir).resolve())
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        try:
            return importlib.import_module("onnxruntime")
        except ImportError:
            return None


def ensure_onnxruntime(artifacts_dir: str | Path, install: bool = False):
    runtime = import_onnxruntime(artifacts_dir)
    if runtime is not None or not install:
        return runtime
    vendor = _vendor_dir(artifacts_dir)
    vendor.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--target",
            str(vendor),
            f"onnxruntime=={ONNXRUNTIME_VERSION}",
        ],
        check=True,
    )
    importlib.invalidate_caches()
    runtime = import_onnxruntime(artifacts_dir)
    if runtime is None:
        raise RuntimeError("onnxruntime installation completed but the module cannot be imported")
    return runtime


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "techjam-retrieval-builder/1.0"})
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urllib.request.urlopen(
        request, timeout=120, context=context
    ) as response, temporary.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
    os.replace(temporary, destination)


def ensure_model_files(artifacts_dir: str | Path) -> tuple[Path, Path]:
    model_dir = Path(artifacts_dir) / MODEL_DIR
    model_path = model_dir / MODEL_FILE
    vocab_path = model_dir / VOCAB_FILE
    _download(MODEL_URL, model_path)
    _download(VOCAB_URL, vocab_path)
    return model_path, vocab_path


def _is_control(character: str) -> bool:
    return character not in "\t\n\r" and unicodedata.category(character).startswith("C")


def _is_punctuation(character: str) -> bool:
    codepoint = ord(character)
    return (33 <= codepoint <= 47) or (58 <= codepoint <= 64) or (
        91 <= codepoint <= 96
    ) or (123 <= codepoint <= 126) or unicodedata.category(character).startswith("P")


def _is_chinese(character: str) -> bool:
    codepoint = ord(character)
    return any(
        start <= codepoint <= end
        for start, end in (
            (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF),
            (0xF900, 0xFAFF), (0x2F800, 0x2FA1F),
        )
    )


class WordPieceTokenizer:
    """The BERT uncased tokenizer needed by MiniLM, without Transformers."""

    def __init__(self, vocab_path: str | Path) -> None:
        tokens = Path(vocab_path).read_text(encoding="utf-8").splitlines()
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.unk = self.vocab["[UNK]"]
        self.cls = self.vocab["[CLS]"]
        self.sep = self.vocab["[SEP]"]
        self.pad = self.vocab["[PAD]"]

    @staticmethod
    def _basic(text: str) -> list[str]:
        spaced: list[str] = []
        for character in text:
            if _is_control(character):
                continue
            if character.isspace():
                spaced.append(" ")
            elif _is_chinese(character) or _is_punctuation(character):
                spaced.extend((" ", character, " "))
            else:
                spaced.append(character)
        lowered = "".join(spaced).lower()
        stripped = "".join(
            character
            for character in unicodedata.normalize("NFD", lowered)
            if unicodedata.category(character) != "Mn"
        )
        return stripped.split()

    def _wordpieces(self, token: str) -> list[int]:
        if len(token) > 100:
            return [self.unk]
        pieces: list[int] = []
        start = 0
        while start < len(token):
            end = len(token)
            found: int | None = None
            while start < end:
                piece = token[start:end]
                if start:
                    piece = "##" + piece
                if piece in self.vocab:
                    found = self.vocab[piece]
                    break
                end -= 1
            if found is None:
                return [self.unk]
            pieces.append(found)
            start = end
        return pieces

    def encode_batch(
        self, texts: Sequence[str], max_length: int = MAX_SEQUENCE_LENGTH
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        encoded: list[list[int]] = []
        for text in texts:
            ids = [self.cls]
            for token in self._basic(str(text)):
                ids.extend(self._wordpieces(token))
                if len(ids) >= max_length - 1:
                    break
            ids = ids[: max_length - 1] + [self.sep]
            encoded.append(ids)
        # ONNX accepts dynamic sequence lengths. Query batches usually need
        # 10-40 tokens, so padding every request to 128 wastes most inference.
        width = max((len(ids) for ids in encoded), default=2)
        input_ids = np.full((len(texts), width), self.pad, dtype=np.int64)
        attention = np.zeros((len(texts), width), dtype=np.int64)
        token_types = np.zeros((len(texts), width), dtype=np.int64)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = ids
            attention[row, : len(ids)] = 1
        return input_ids, attention, token_types


class OnnxSentenceEncoder:
    def __init__(
        self,
        model_path: str | Path,
        vocab_path: str | Path,
        runtime,
        threads: int | None = None,
    ) -> None:
        options = runtime.SessionOptions()
        requested_threads = (
            threads if threads is not None else int(os.environ.get("TJ_ONNX_THREADS", "1"))
        )
        options.intra_op_num_threads = max(1, requested_threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = runtime.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = WordPieceTokenizer(vocab_path)
        self.input_names = {item.name for item in self.session.get_inputs()}
        outputs = self.session.get_outputs()
        sentence_outputs = [item.name for item in outputs if "sentence_embedding" in item.name]
        self.sentence_output = sentence_outputs[0] if sentence_outputs else None
        self.fallback_output = outputs[0].name

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        input_ids, attention, token_types = self.tokenizer.encode_batch(texts)
        possible = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "token_type_ids": token_types,
        }
        feeds = {name: possible[name] for name in self.input_names if name in possible}
        output_name = self.sentence_output or self.fallback_output
        vectors = self.session.run([output_name], feeds)[0]
        if vectors.ndim == 3:
            mask = attention[:, :, None].astype(np.float32)
            vectors = (vectors * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
        if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(f"unexpected MiniLM output shape: {vectors.shape}")
        return self._normalize(vectors)


def _product_dense_text(product: dict) -> str:
    title = str(product.get("title") or "")
    categories = " ".join(str(value) for value in (product.get("categories") or []))
    features = " ".join(str(value) for value in (product.get("features") or []))
    description = " ".join(str(value) for value in (product.get("description") or []))
    store = str(product.get("store") or "")
    # The first 128 wordpieces should contain product identity and benefits,
    # rather than dates and package dimensions from details.
    return (
        f"product: {title}. category: {categories}. brand: {store}. "
        f"features: {features}. description: {description}"
    )[:3000]


def build_dense_index(
    catalog_path: str | Path,
    artifacts_dir: str | Path,
    batch_size: int = 64,
    install_runtime: bool = True,
) -> tuple[Path, Path]:
    """Download MiniLM if needed and embed the catalog into NumPy artifacts."""
    runtime = ensure_onnxruntime(artifacts_dir, install=install_runtime)
    if runtime is None:
        raise RuntimeError(
            "onnxruntime is required to build dense artifacts; run python3 -m tools.build_index"
        )
    model_path, vocab_path = ensure_model_files(artifacts_dir)
    default_build_threads = min(4, os.cpu_count() or 1)
    build_threads = int(os.environ.get("TJ_ONNX_BUILD_THREADS", str(default_build_threads)))
    encoder = OnnxSentenceEncoder(model_path, vocab_path, runtime, threads=build_threads)

    catalog = Path(catalog_path)
    with catalog.open(encoding="utf-8") as handle:
        total = sum(1 for line in handle if line.strip())

    directory = Path(artifacts_dir)
    vector_path = directory / VECTORS_FILE
    asin_path = directory / ASINS_FILE
    vector_temporary = directory / f"{VECTORS_FILE}.tmp"
    asin_temporary = directory / f"{ASINS_FILE}.tmp"
    for temporary in (vector_temporary, asin_temporary):
        if temporary.exists():
            temporary.unlink()
    vectors = np.lib.format.open_memmap(
        vector_temporary,
        mode="w+",
        dtype=np.float16,
        shape=(total, EMBEDDING_DIMENSION),
    )

    asins: list[str] = []
    texts: list[str] = []
    offset = 0
    with catalog.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            asins.append(str(product["parent_asin"]))
            texts.append(_product_dense_text(product))
            if len(texts) == batch_size:
                encoded = encoder.encode(texts).astype(np.float16)
                vectors[offset : offset + len(encoded)] = encoded
                offset += len(encoded)
                texts.clear()
                if offset % (batch_size * max(1, 5000 // batch_size)) == 0:
                    print(f"  embedded {offset:,}/{total:,} products", flush=True)
    if texts:
        encoded = encoder.encode(texts).astype(np.float16)
        vectors[offset : offset + len(encoded)] = encoded
        offset += len(encoded)
    vectors.flush()
    del vectors
    with asin_temporary.open("wb") as handle:
        np.save(handle, np.asarray(asins, dtype="U16"), allow_pickle=False)
    os.replace(vector_temporary, vector_path)
    os.replace(asin_temporary, asin_path)
    manifest = {
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "dimension": EMBEDDING_DIMENSION,
        "count": len(asins),
        "dtype": "float16",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return vector_path, asin_path


class DenseIndex:
    """In-memory float32 matrix for a fast exact cosine scan."""

    def __init__(self, artifacts_dir: str | Path) -> None:
        directory = Path(artifacts_dir)
        runtime = ensure_onnxruntime(directory, install=False)
        required = (
            directory / VECTORS_FILE,
            directory / ASINS_FILE,
            directory / MODEL_DIR / MODEL_FILE,
            directory / MODEL_DIR / VOCAB_FILE,
        )
        if runtime is None or not all(path.exists() for path in required):
            raise FileNotFoundError("dense artifacts or onnxruntime are unavailable")
        self.asins = np.load(required[1], allow_pickle=False)
        # float32 BLAS is substantially faster than float16 on common CPUs.
        self.vectors = np.asarray(np.load(required[0], allow_pickle=False), dtype=np.float32)
        if len(self.asins) != len(self.vectors):
            raise ValueError("dense ASIN and vector artifacts have different lengths")
        self.encoder = OnnxSentenceEncoder(required[2], required[3], runtime)

    def search(self, query: str, limit: int) -> list[DenseHit]:
        if not query.strip() or limit <= 0 or not len(self.asins):
            return []
        query_vector = self.encoder.encode([query])[0]
        scores = self.vectors @ query_vector
        count = min(int(limit), len(scores))
        if count == len(scores):
            selected = np.arange(len(scores))
        else:
            selected = np.argpartition(scores, -count)[-count:]
        selected = selected[np.argsort(scores[selected], kind="stable")[::-1]]
        return [DenseHit(str(self.asins[index]), float(scores[index])) for index in selected]
