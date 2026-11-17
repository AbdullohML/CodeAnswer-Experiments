"""Isolated-process resource profiles for the existing PCA and FAISS paths."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

import faiss
import joblib
import numpy as np
import psutil
import pyarrow.dataset as pads
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from codeanswer.indexing import open_embeddings
from codeanswer.optimization import normalize_rows
from codeanswer.pca_sweep import project_prefix
from codeanswer.provenance import sha256_file
from codeanswer.rag import clean_text

SYSTEMS = {
    "flat384": (384, "flat"),
    "flat128": (128, "flat"),
    "hnsw128": (128, "hnsw"),
}


class MemoryPeak:
    """Sample this process's RSS while a profiling phase runs."""

    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self.initial = self.process.memory_info().rss
        self.peak = self.initial
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> MemoryPeak:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)

    def summary(self) -> dict[str, float]:
        unit = 1024**2
        # Linux reports ru_maxrss in KiB and retains earlier process peaks.
        high_water_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        peak = max(self.peak, high_water_bytes)
        return {
            "rss_start_mib": round(self.initial / unit, 3),
            "rss_peak_mib": round(peak / unit, 3),
            "rss_delta_mib": round((peak - self.initial) / unit, 3),
        }


def percentiles(milliseconds: list[float]) -> dict[str, float]:
    if not milliseconds or not all(np.isfinite(milliseconds)):
        raise ValueError("Need finite nonempty latency samples")
    return {
        "p50_ms": round(float(np.percentile(milliseconds, 50)), 3),
        "p95_ms": round(float(np.percentile(milliseconds, 95)), 3),
    }


def _gpu_state() -> dict:
    try:
        run = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": type(exc).__name__}
    if run.returncode:
        return {
            "available": False,
            "reason": run.stderr.strip()[:250] or f"nvidia-smi exit {run.returncode}",
        }
    return {"available": True, "device": run.stdout.strip()}


def _hardware() -> dict:
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "ram_total_gib": round(psutil.virtual_memory().total / 1024**3, 3),
        "gpu": _gpu_state(),
        "packages": {
            name: version(name)
            for name in (
                "faiss-cpu",
                "numpy",
                "scikit-learn",
                "pyarrow",
                "psutil",
                "sentence-transformers",
                "torch",
            )
        },
    }


def _paths(root: Path, system: str) -> tuple[Path, Path, int, str]:
    dimension, kind = SYSTEMS[system]
    embedding = (
        root / "artifacts/embeddings/question_body_minilm_384.f32"
        if dimension == 384
        else root / "artifacts/b1/embeddings_128.f32"
    )
    index = (
        root / "artifacts/indexes/flat_ip_384.faiss"
        if system == "flat384"
        else root / "artifacts/b1/flat_ip_128.faiss"
        if system == "flat128"
        else root / "artifacts/b2/hnsw_128.faiss"
    )
    return embedding, index, dimension, kind


def _pca_profile(root: Path, settings: dict, scope: str) -> dict:
    source = root / "artifacts/embeddings/question_body_minilm_384.f32"
    rows = settings["full_rows"] if scope == "full" else settings["subset_rows"]
    sample_size = (
        settings["pca_sample_full"]
        if scope == "full"
        else settings["pca_sample_subset"]
    )
    embeddings = open_embeddings(source, objects=settings["full_rows"], dimension=384)
    vectors = embeddings[:rows]
    sampled_ids = np.random.default_rng(settings["seed"]).choice(
        rows, size=sample_size, replace=False
    )
    sampled = np.asarray(vectors[sampled_ids], dtype=np.float32)
    pca = PCA(
        n_components=settings["pca_components"],
        svd_solver="randomized",
        random_state=settings["seed"],
    )
    with MemoryPeak() as memory:
        t = time.perf_counter()
        pca.fit(sampled)
        fit_seconds = time.perf_counter() - t
        transforms = {}
        for dimension in (256, 128):
            digest = hashlib.sha256()
            t = time.perf_counter()
            for start in range(0, rows, settings["batch_size"]):
                batch = vectors[start : start + settings["batch_size"]]
                if dimension == 256:
                    reduced = normalize_rows(pca.transform(batch))
                else:
                    reduced = project_prefix(pca, batch, dimension)
                digest.update(np.ascontiguousarray(reduced).tobytes())
            transforms[str(dimension)] = {
                "seconds": round(time.perf_counter() - t, 3),
                "output_bytes": rows * dimension * 4,
                "stream_sha256": digest.hexdigest(),
            }
    return {
        "scope": scope,
        "rows": rows,
        "fit_sample_rows": sample_size,
        "fit_seconds": round(fit_seconds, 3),
        "explained_variance_256": round(float(pca.explained_variance_ratio_.sum()), 6),
        "transforms": transforms,
        "memory": memory.summary(),
        "source_sha256": sha256_file(source),
        "method": "randomized PCA fit; full streaming transform, no output file write",
    }


def _build_profile(root: Path, settings: dict, scope: str, system: str) -> dict:
    source, existing, dimension, kind = _paths(root, system)
    rows = settings["full_rows"] if scope == "full" else settings["subset_rows"]
    vectors = open_embeddings(
        source, objects=settings["full_rows"], dimension=dimension
    )
    index = (
        faiss.IndexFlatIP(dimension)
        if kind == "flat"
        else faiss.IndexHNSWFlat(
            dimension, settings["hnsw_m"], faiss.METRIC_INNER_PRODUCT
        )
    )
    if kind == "hnsw":
        index.hnsw.efConstruction = settings["hnsw_ef_construction"]
    with MemoryPeak() as memory:
        t = time.perf_counter()
        for start in range(0, rows, settings["batch_size"]):
            batch = np.ascontiguousarray(
                vectors[start : start + settings["batch_size"]], dtype=np.float32
            )
            index.add(batch)
        build_seconds = time.perf_counter() - t
        if index.ntotal != rows:
            raise RuntimeError("Index build count differs from requested rows")
        if scope == "subset":
            # Measure a real subset artifact without replacing the full indexes.
            output = root / "artifacts/b4" / f"{system}_subset20k.faiss"
            output.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(output))
            file_bytes = output.stat().st_size
        else:
            output = existing
            file_bytes = existing.stat().st_size
    return {
        "scope": scope,
        "system": system,
        "rows": rows,
        "dimension": dimension,
        "build_seconds_add_only": round(build_seconds, 3),
        "file_bytes": file_bytes,
        "file_source": "new subset artifact"
        if scope == "subset"
        else "existing full artifact",
        "memory": memory.summary(),
        "embedding_sha256": sha256_file(source),
        "existing_index_sha256": sha256_file(existing),
        "index_type": type(index).__name__,
        "note": (
            "Full build adds vectors in memory; existing artifact size is "
            "reported. File serialization is excluded from build seconds."
        ),
    }


def fetch_candidates(dataset: pads.Dataset, ids: list[int]) -> list[int]:
    """Match the production Parquet filter, rank-order restore and cleaning."""
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be nonempty and unique")
    table = dataset.to_table(filter=pads.field("doc_id").isin(ids))
    frame = table.to_pandas().set_index("doc_id").loc[ids].reset_index()
    for column in ("title", "question_body", "answer"):
        frame[column] = frame[column].map(clean_text)
    return frame["doc_id"].astype(int).tolist()


def _runtime_profile(root: Path, settings: dict, system: str) -> dict:
    from codeanswer.embeddings import SentenceTransformerEncoder

    source, index_path, dimension, kind = _paths(root, system)
    queries = (
        __import__("pandas")
        .read_csv(root / "artifacts/evaluation/frozen_v1/test.csv")["query"]
        .iloc[: settings["query_count"]]
        .tolist()
    )
    if len(queries) != settings["query_count"]:
        raise ValueError("Frozen test lacks requested query count")
    encoder = SentenceTransformerEncoder(
        "sentence-transformers/all-MiniLM-L6-v2",
        max_sequence_length=256,
        device="cpu",
    )
    pca = (
        joblib.load(root / "artifacts/b1/pca_384_to_256.joblib")
        if dimension == 128
        else None
    )
    dataset = pads.dataset(
        root / "artifacts/corpus/qa_corpus.parquet", format="parquet"
    )
    with MemoryPeak() as memory:
        load_times = []
        for _ in range(2):
            t = time.perf_counter()
            index = faiss.read_index(str(index_path))
            load_times.append(round(time.perf_counter() - t, 3))
            if index.ntotal != settings["full_rows"] or index.d != dimension:
                raise ValueError("Loaded index shape differs")
            if len(load_times) == 1:
                del index
                gc.collect()
        if kind == "hnsw":
            index.hnsw.efSearch = settings["hnsw_ef_search"]

        def query_once(query: str) -> dict[str, float]:
            t = time.perf_counter()
            raw = encoder.encode([query], batch_size=1)
            encoded = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            vector = project_prefix(pca, raw, dimension) if pca else raw
            projected = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            _, matches = index.search(np.ascontiguousarray(vector), settings["top_k"])
            searched = (time.perf_counter() - t) * 1000
            ids = [int(x) for x in matches[0] if x >= 0]
            t = time.perf_counter()
            found = fetch_candidates(dataset, ids)
            fetched = (time.perf_counter() - t) * 1000
            if found != ids:
                raise ValueError("Corpus fetch returned different or out-of-order IDs")
            return {
                "encode": encoded,
                "pca": projected,
                "search": searched,
                "fetch40": fetched,
                "total_to_fetch": encoded + projected + searched + fetched,
            }

        first = query_once(queries[0])
        for query in queries[: settings["warmup_queries"]]:
            query_once(query)
        rounds = []
        for _ in range(settings["repeats"]):
            samples = {key: [] for key in first}
            for query in queries:
                stages = query_once(query)
                for key, value in stages.items():
                    samples[key].append(value)
            rounds.append({key: percentiles(values) for key, values in samples.items()})
    return {
        "scope": "full",
        "system": system,
        "rows": settings["full_rows"],
        "query_ids_source": "first 100 frozen test queries in file order",
        "first_load_seconds": load_times[0],
        "second_load_seconds": load_times[1],
        "first_query_ms": {key: round(value, 3) for key, value in first.items()},
        "warm_rounds": rounds,
        "memory": memory.summary(),
        "index_bytes": index_path.stat().st_size,
        "index_sha256": sha256_file(index_path),
        "embedding_sha256": sha256_file(source),
        "frozen_test_sha256": sha256_file(
            root / "artifacts/evaluation/frozen_v1/test.csv"
        ),
        "corpus_sha256": sha256_file(root / "artifacts/corpus/qa_corpus.parquet"),
        "encoder_revision": encoder.model_revision,
        "timing_boundary": (
            "CPU encode, optional PCA, FAISS top-40 search, Parquet fetch "
            "and cleaning; excludes rerank and generation"
        ),
        "load_boundary": (
            "fresh process, first and second FAISS read_index; "
            "OS page cache not cleared"
        ),
    }


def run(
    root: Path, output_dir: Path, *, phase: str, scope: str, system: str | None
) -> dict:
    settings = json.loads((root / "configs/resource_profile.json").read_text())
    if phase not in ("pca", "build", "runtime") or scope not in ("full", "subset"):
        raise ValueError("Unsupported profile phase/scope")
    if (phase == "pca") != (system is None) or (system and system not in SYSTEMS):
        raise ValueError("Choose a system for build/runtime only")
    if phase == "runtime" and scope != "full":
        raise ValueError("Runtime uses existing full indexes")
    faiss.omp_set_num_threads(settings["threads"])
    with threadpool_limits(limits=settings["threads"]):
        if phase == "pca":
            details = _pca_profile(root, settings, scope)
        elif phase == "build":
            details = _build_profile(root, settings, scope, system)
        else:
            details = _runtime_profile(root, settings, system)
    # All three profiled paths are CPU paths. With no usable driver, GPU VRAM is
    # explicitly unavailable rather than inferred from index bytes.
    hardware = _hardware()
    details["gpu_peak_vram_mib"] = None
    details["gpu_note"] = (
        "CPU-only phase; GPU driver unavailable"
        if not hardware["gpu"]["available"]
        else "CPU-only phase; no CUDA allocations expected"
    )
    filename = f"{phase}_{scope}_{system or 'pca'}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    payload = {
        "schema_version": 1,
        "phase": phase,
        "scope": scope,
        "system": system,
        "settings": settings,
        "hardware": hardware,
        "command": " ".join(sys.argv),
        "details": details,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return {"output": str(path), **payload}
