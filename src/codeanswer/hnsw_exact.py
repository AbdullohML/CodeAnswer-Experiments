"""B2 same-space exact versus HNSW retrieval with paired outputs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from codeanswer.embeddings import SentenceTransformerEncoder
from codeanswer.evaluation import evaluate_known_item, known_item_ranks
from codeanswer.evaluation_protocol import load_frozen_split
from codeanswer.indexing import load_index
from codeanswer.optimization import build_hnsw_index
from codeanswer.pca_sweep import project_prefix
from codeanswer.provenance import sha256_file

SETTINGS = {
    "schema_version": 1,
    "dimensions": [128, 384],
    "m": 16,
    "ef_construction": 100,
    "ef_search": 128,
    "top_k": 40,
    "latency_queries": 100,
    "warmup_queries": 3,
    "threads": 4,
}


def paired_overlap(exact: np.ndarray, approximate: np.ndarray, k: int) -> np.ndarray:
    """Fraction of exact top-k IDs recovered by ANN, query by query."""
    if (
        exact.shape != approximate.shape
        or exact.ndim != 2
        or k <= 0
        or k > exact.shape[1]
    ):
        raise ValueError("Paired ranking shapes or cutoff differ")
    return np.asarray(
        [
            len(set(a[:k]) & set(b[:k])) / k
            for a, b in zip(exact, approximate, strict=True)
        ],
        dtype=float,
    )


def search_and_time(
    index: object, vectors: np.ndarray, settings: dict
) -> tuple[np.ndarray, dict]:
    count = len(vectors)
    ids = np.empty((count, settings["top_k"]), dtype=np.int64)
    started = time.perf_counter()
    for start in range(0, count, 32):
        end = min(start + 32, count)
        _, ids[start:end] = index.search(
            np.ascontiguousarray(vectors[start:end]), settings["top_k"]
        )
    batch_seconds = time.perf_counter() - started
    n = min(settings["latency_queries"], count)
    for i in range(min(settings["warmup_queries"], n)):
        index.search(vectors[i : i + 1], settings["top_k"])
    times = []
    for i in range(n):
        t = time.perf_counter()
        index.search(vectors[i : i + 1], settings["top_k"])
        times.append((time.perf_counter() - t) * 1000)
    return ids, {
        "queries": n,
        "batch_seconds": round(batch_seconds, 3),
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
    }


def run(
    root: Path,
    output: Path,
    smoke_limit: int | None = None,
    only_dimension: int | None = None,
) -> dict:
    settings_path = root / "configs/hnsw_exact.json"
    settings = json.loads(settings_path.read_text())
    if settings != SETTINGS or (
        only_dimension is not None and only_dimension not in (128, 384)
    ):
        raise ValueError("Unsupported B2 settings")
    corpus = root / "artifacts/corpus/qa_corpus.parquet"
    frozen_path = root / "artifacts/evaluation/frozen_v1/manifest.json"
    queries, frozen = load_frozen_split(frozen_path, corpus, split="test")
    if smoke_limit is not None:
        if smoke_limit <= 0 or smoke_limit >= len(queries):
            raise ValueError("Smoke limit must be below full test")
        queries = queries.iloc[:smoke_limit].reset_index(drop=True)
    b1_dir = root / "results/evaluation/b1"
    b1 = json.loads((b1_dir / "results.json").read_text())
    if (
        b1["smoke_limit"] is not None
        or b1["query_count"] != 1200
        or b1["provenance"]["frozen_manifest_sha256"] != sha256_file(frozen_path)
        or b1["provenance"]["corpus_sha256"] != frozen["corpus"]["sha256"]
        or sha256_file(b1_dir / "per_query.csv.gz") != b1["files"]["per_query.csv.gz"]
    ):
        raise ValueError("B1 reference provenance differs")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch

    torch.set_num_threads(settings["threads"])
    faiss.omp_set_num_threads(settings["threads"])
    dimensions = [only_dimension] if only_dimension else settings["dimensions"]
    with threadpool_limits(limits=settings["threads"]):
        encoder = SentenceTransformerEncoder(
            "sentence-transformers/all-MiniLM-L6-v2",
            max_sequence_length=256,
            device="cpu",
        )
        if encoder.model_revision != b1["provenance"]["encoder_revision"]:
            raise ValueError("Query encoder revision differs from B1")
        raw = encoder.encode(queries["query"].tolist(), batch_size=64)
        if (
            smoke_limit is None
            and hashlib.sha256(raw.tobytes()).hexdigest()
            != b1["provenance"]["query_vectors_sha256"]
        ):
            raise ValueError("Query vectors differ from B1")
        pca = joblib.load(root / "artifacts/b1/pca_384_to_256.joblib")
        if (
            sha256_file(root / "artifacts/b1/pca_384_to_256.joblib")
            != b1["provenance"]["pca_model_sha256"]
        ):
            raise ValueError("PCA fit differs from B1")
        saved = pd.read_csv(b1_dir / "per_query.csv.gz")
        output.mkdir(parents=True, exist_ok=True)
        records, summaries = [], []
        for dimension in dimensions:
            reference = json.loads((b1_dir / f"dim_{dimension}.json").read_text())
            embedding = (
                root / "artifacts/b1" / f"embeddings_{dimension}.f32"
                if dimension == 128
                else root / "artifacts/embeddings/question_body_minilm_384.f32"
            )
            flat_path = (
                root / "artifacts/b1/flat_ip_128.faiss"
                if dimension == 128
                else root / "artifacts/indexes/flat_ip_384.faiss"
            )
            if (
                sha256_file(embedding) != reference["embedding_sha256"]
                or sha256_file(flat_path) != reference["index_sha256"]
            ):
                raise ValueError("Same-space exact artifacts differ from B1")
            vectors = project_prefix(pca, raw, 128) if dimension == 128 else raw
            hnsw_path = root / f"artifacts/b2/hnsw_{dimension}.faiss"
            hnsw_path.parent.mkdir(parents=True, exist_ok=True)
            if not hnsw_path.exists():
                if Path(f"{hnsw_path}.manifest.json").exists():
                    raise ValueError("Incomplete HNSW artifact")
                build_hnsw_index(
                    embedding,
                    hnsw_path,
                    dimension=dimension,
                    m=settings["m"],
                    ef_construction=settings["ef_construction"],
                )
            hnsw_info = json.loads(Path(f"{hnsw_path}.manifest.json").read_text())
            if (
                hnsw_info["dimension"] != dimension
                or hnsw_info["objects"] != 520000
                or hnsw_info["m"] != settings["m"]
                or hnsw_info["ef_construction"] != settings["ef_construction"]
                or Path(hnsw_info["embedding_path"]) != embedding.resolve()
            ):
                raise ValueError("HNSW manifest differs from B2 settings")
            flat = load_index(
                flat_path, expected_objects=520000, expected_dimension=dimension
            )
            hnsw = load_index(
                hnsw_path, expected_objects=520000, expected_dimension=dimension
            )
            if (
                type(flat) is not faiss.IndexFlatIP
                or type(hnsw) is not faiss.IndexHNSWFlat
            ):
                raise ValueError("Expected exact and HNSW index types")
            hnsw.hnsw.efSearch = settings["ef_search"]
            exact_ids, exact_timing = search_and_time(flat, vectors, settings)
            hnsw_ids, hnsw_timing = search_and_time(hnsw, vectors, settings)
            del flat, hnsw
            saved_dim = saved[saved.dimension == dimension].iloc[: len(queries)]
            expected_ids = np.asarray(
                [json.loads(x) for x in saved_dim["retrieved_doc_ids@40"]]
            )
            if (
                len(saved_dim) != len(queries)
                or not np.array_equal(exact_ids, expected_ids)
                or saved_dim.query_id.tolist() != queries.query_id.tolist()
            ):
                raise ValueError("Live exact ranking differs from B1 reference")
            labels = queries["relevant_doc_id"].to_numpy()
            exact_quality, _ = evaluate_known_item(
                exact_ids, labels, recall_ks=(10, 40), metric_cutoff=10
            )
            hnsw_quality, _ = evaluate_known_item(
                hnsw_ids, labels, recall_ks=(10, 40), metric_cutoff=10
            )
            if smoke_limit is None and exact_quality != reference["quality"]:
                raise ValueError("Exact quality differs from B1")
            overlap10 = paired_overlap(exact_ids, hnsw_ids, 10)
            overlap40 = paired_overlap(exact_ids, hnsw_ids, 40)
            eranks = known_item_ranks(exact_ids, labels)
            aranks = known_item_ranks(hnsw_ids, labels)
            summary = {
                "dimension": dimension,
                "queries": len(queries),
                "exact_quality": exact_quality,
                "hnsw_quality": hnsw_quality,
                "overlap@10": round(float(overlap10.mean()), 6),
                "overlap@40": round(float(overlap40.mean()), 6),
                "recall@10_delta": round(
                    hnsw_quality["recall@10"] - exact_quality["recall@10"], 6
                ),
                "recall@40_delta": round(
                    hnsw_quality["recall@40"] - exact_quality["recall@40"], 6
                ),
                "exact_timing": exact_timing,
                "hnsw_timing": hnsw_timing,
                "exact_index_bytes": flat_path.stat().st_size,
                "hnsw_index_bytes": hnsw_path.stat().st_size,
                "exact_index_sha256": reference["index_sha256"],
                "hnsw_index_sha256": sha256_file(hnsw_path),
                "embedding_sha256": reference["embedding_sha256"],
                "hnsw_build_seconds": hnsw_info["build_seconds"],
            }
            summaries.append(summary)
            for i, q in enumerate(queries.itertuples(index=False)):
                records.append(
                    {
                        "dimension": dimension,
                        "query_index": i,
                        "query_id": int(q.query_id),
                        "family_id": int(q.family_id),
                        "relevant_doc_id": int(q.relevant_doc_id),
                        "exact_rank": int(eranks[i]),
                        "hnsw_rank": int(aranks[i]),
                        "overlap@10": overlap10[i],
                        "overlap@40": overlap40[i],
                        "exact_ids@40": json.dumps(exact_ids[i].tolist()),
                        "hnsw_ids@40": json.dumps(hnsw_ids[i].tolist()),
                    }
                )
            print(
                f"B2 {dimension}: exact R@10={exact_quality['recall@10']}, "
                f"HNSW R@10={hnsw_quality['recall@10']}",
                flush=True,
            )
        path = output / "per_query.csv.gz"
        pd.DataFrame(records).to_csv(
            path, index=False, compression={"method": "gzip", "mtime": 0}
        )
        table = pd.DataFrame(
            [
                {
                    "dimension": s["dimension"],
                    "exact_recall@10": s["exact_quality"]["recall@10"],
                    "hnsw_recall@10": s["hnsw_quality"]["recall@10"],
                    "exact_recall@40": s["exact_quality"]["recall@40"],
                    "hnsw_recall@40": s["hnsw_quality"]["recall@40"],
                    "overlap@10": s["overlap@10"],
                    "overlap@40": s["overlap@40"],
                    "exact_p50_ms": s["exact_timing"]["p50_ms"],
                    "exact_p95_ms": s["exact_timing"]["p95_ms"],
                    "hnsw_p50_ms": s["hnsw_timing"]["p50_ms"],
                    "hnsw_p95_ms": s["hnsw_timing"]["p95_ms"],
                    "exact_index_bytes": s["exact_index_bytes"],
                    "hnsw_index_bytes": s["hnsw_index_bytes"],
                }
                for s in summaries
            ]
        )
        table.to_csv(output / "comparison.csv", index=False)
        result = {
            "task": "B2",
            "scope": "single-positive title-to-own-body known-item",
            "query_count": len(queries),
            "smoke_limit": smoke_limit,
            "settings": settings,
            "settings_sha256": sha256_file(settings_path),
            "command": sys.argv,
            "provenance": {
                "frozen_manifest_sha256": sha256_file(frozen_path),
                "b1_results_sha256": sha256_file(b1_dir / "results.json"),
                "pca_model_sha256": b1["provenance"]["pca_model_sha256"],
                "query_encoder_revision": encoder.model_revision,
            },
            "summaries": summaries,
            "files": {
                "per_query.csv.gz": sha256_file(path),
                "comparison.csv": sha256_file(output / "comparison.csv"),
            },
            "environment": {
                "platform": platform.platform(),
                "cpu_logical_count": os.cpu_count(),
                "faiss_threads": faiss.omp_get_max_threads(),
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "packages": {
                    name: version(name)
                    for name in ("numpy", "faiss-cpu", "sentence-transformers", "torch")
                },
            },
        }
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
