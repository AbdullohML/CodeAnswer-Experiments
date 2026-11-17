"""B3 frozen efSearch sweep on one fixed HNSW-128 index."""

from __future__ import annotations

import hashlib
import json
import os
import platform
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
from codeanswer.hnsw_exact import paired_overlap
from codeanswer.indexing import load_index
from codeanswer.pca_sweep import project_prefix
from codeanswer.provenance import sha256_file

SETTINGS = {
    "schema_version": 1,
    "dimension": 128,
    "ef_search_values": [16, 32, 64, 128, 256],
    "top_k": 40,
    "latency_queries": 100,
    "warmup_queries": 3,
    "threads": 4,
    "selected_ef_search": 128,
}


def search_and_time(
    index: object, vectors: np.ndarray, settings: dict
) -> tuple[np.ndarray, dict]:
    """Time the same sequential prefix used for quality search and QPS."""
    ids = np.empty((len(vectors), settings["top_k"]), dtype=np.int64)
    for start in range(0, len(vectors), 32):
        end = min(start + 32, len(vectors))
        _, ids[start:end] = index.search(
            np.ascontiguousarray(vectors[start:end]), settings["top_k"]
        )
    n = min(settings["latency_queries"], len(vectors))
    for i in range(min(settings["warmup_queries"], n)):
        index.search(vectors[i : i + 1], settings["top_k"])
    times = []
    for i in range(n):
        started = time.perf_counter()
        index.search(vectors[i : i + 1], settings["top_k"])
        times.append((time.perf_counter() - started) * 1000)
    return ids, {
        "queries": n,
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "sequential_qps": round(n * 1000 / sum(times), 2),
    }


def run(
    root: Path,
    output: Path,
    *,
    limit: int | None = None,
    settings_path: Path | None = None,
) -> dict:
    settings_path = settings_path or root / "configs/ef_sweep.json"
    settings = json.loads(settings_path.read_text())
    if settings != SETTINGS:
        raise ValueError("Unsupported B3 settings")
    corpus = root / "artifacts/corpus/qa_corpus.parquet"
    frozen_path = root / "artifacts/evaluation/frozen_v1/manifest.json"
    queries, frozen = load_frozen_split(frozen_path, corpus, split="test")
    if limit is not None:
        if limit <= 0 or limit >= len(queries):
            raise ValueError("Smoke limit must be below full test")
        queries = queries.iloc[:limit].reset_index(drop=True)
    b1 = json.loads((root / "results/evaluation/b1/results.json").read_text())
    b2_path = root / "results/evaluation/b2/results.json"
    b2 = json.loads(b2_path.read_text())
    reference = next(s for s in b2["summaries"] if s["dimension"] == 128)
    if (
        b2["smoke_limit"] is not None
        or b2["query_count"] != 1200
        or b2["provenance"]["frozen_manifest_sha256"] != sha256_file(frozen_path)
        or b2["provenance"]["b1_results_sha256"]
        != sha256_file(root / "results/evaluation/b1/results.json")
        or sha256_file(root / "results/evaluation/b2/per_query.csv.gz")
        != b2["files"]["per_query.csv.gz"]
        or frozen["corpus"]["sha256"] != b1["provenance"]["corpus_sha256"]
    ):
        raise ValueError("B1/B2 reference provenance differs")
    index_path = root / "artifacts/b2/hnsw_128.faiss"
    if sha256_file(index_path) != reference["hnsw_index_sha256"]:
        raise ValueError("HNSW index differs from B2")
    pca_path = root / "artifacts/b1/pca_384_to_256.joblib"
    if sha256_file(pca_path) != b1["provenance"]["pca_model_sha256"]:
        raise ValueError("PCA model differs from B1")
    b2_rows = pd.read_csv(root / "results/evaluation/b2/per_query.csv.gz")
    exact_rows = b2_rows[b2_rows.dimension == 128].iloc[: len(queries)]
    exact = np.asarray(
        [json.loads(value) for value in exact_rows["exact_ids@40"]], dtype=np.int64
    )
    if (
        len(exact_rows) != len(queries)
        or exact_rows.query_id.tolist() != queries.query_id.tolist()
        or exact_rows.relevant_doc_id.tolist() != queries.relevant_doc_id.tolist()
    ):
        raise ValueError("B2 exact rows differ from frozen queries")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch

    torch.set_num_threads(settings["threads"])
    faiss.omp_set_num_threads(settings["threads"])
    with threadpool_limits(limits=settings["threads"]):
        encoder = SentenceTransformerEncoder(
            "sentence-transformers/all-MiniLM-L6-v2",
            max_sequence_length=256,
            device="cpu",
        )
        if encoder.model_revision != b1["provenance"]["encoder_revision"]:
            raise ValueError("Query encoder differs from B1")
        raw = encoder.encode(queries["query"].tolist(), batch_size=64)
        if (
            limit is None
            and hashlib.sha256(raw.tobytes()).hexdigest()
            != b1["provenance"]["query_vectors_sha256"]
        ):
            raise ValueError("Query vectors differ from B1")
        vectors = project_prefix(joblib.load(pca_path), raw, 128)
        index = load_index(index_path, expected_objects=520000, expected_dimension=128)
        if type(index) is not faiss.IndexHNSWFlat:
            raise ValueError("B3 requires the B2 HNSW index")
        labels = queries.relevant_doc_id.to_numpy()
        exact_quality, _ = evaluate_known_item(
            exact, labels, recall_ks=(10, 40), metric_cutoff=10
        )
        rows, summaries = [], []
        for ef in settings["ef_search_values"]:
            index.hnsw.efSearch = ef
            found, timing = search_and_time(index, vectors, settings)
            quality, _ = evaluate_known_item(
                found, labels, recall_ks=(10, 40), metric_cutoff=10
            )
            overlap10 = paired_overlap(exact, found, 10)
            overlap40 = paired_overlap(exact, found, 40)
            ranks = known_item_ranks(found, labels)
            summary = {
                "ef_search": ef,
                "quality": quality,
                "overlap@10": round(float(overlap10.mean()), 6),
                "overlap@40": round(float(overlap40.mean()), 6),
                "timing": timing,
                "sequential_qps": timing["sequential_qps"],
                "recall@10_delta_vs_exact": round(
                    quality["recall@10"] - exact_quality["recall@10"], 6
                ),
                "recall@40_delta_vs_exact": round(
                    quality["recall@40"] - exact_quality["recall@40"], 6
                ),
            }
            summaries.append(summary)
            for i, q in enumerate(queries.itertuples(index=False)):
                rows.append(
                    {
                        "ef_search": ef,
                        "query_index": i,
                        "query_id": int(q.query_id),
                        "family_id": int(q.family_id),
                        "relevant_doc_id": int(q.relevant_doc_id),
                        "rank": int(ranks[i]),
                        "overlap@10": overlap10[i],
                        "overlap@40": overlap40[i],
                        "retrieved_doc_ids@40": json.dumps(found[i].tolist()),
                    }
                )
            print(
                f"B3 efSearch={ef}: R@10={quality['recall@10']}, "
                f"R@40={quality['recall@40']}",
                flush=True,
            )
        output.mkdir(parents=True, exist_ok=True)
        row_path = output / "per_query.csv.gz"
        pd.DataFrame(rows).to_csv(
            row_path, index=False, compression={"method": "gzip", "mtime": 0}
        )
        table = pd.DataFrame(
            [
                {
                    "ef_search": s["ef_search"],
                    "recall@10": s["quality"]["recall@10"],
                    "recall@40": s["quality"]["recall@40"],
                    "overlap@10": s["overlap@10"],
                    "overlap@40": s["overlap@40"],
                    "search_p50_ms": s["timing"]["p50_ms"],
                    "search_p95_ms": s["timing"]["p95_ms"],
                    "sequential_qps": s["sequential_qps"],
                }
                for s in summaries
            ]
        )
        table_path = output / "comparison.csv"
        table.to_csv(table_path, index=False)
        if limit is None:
            os.environ.setdefault("MPLCONFIGDIR", str(root / "artifacts/b2/mplcache"))
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
            axes[0].plot(
                table.ef_search, table["recall@10"], marker="o", label="Recall@10"
            )
            axes[0].plot(
                table.ef_search, table["recall@40"], marker="o", label="Recall@40"
            )
            axes[0].plot(
                table.ef_search, table["overlap@40"], marker="o", label="ANN overlap@40"
            )
            axes[0].set(xlabel="efSearch", ylabel="Known-item recall / ANN overlap")
            axes[0].legend()
            axes[0].grid(alpha=0.25)
            axes[1].plot(table.ef_search, table.search_p95_ms, marker="o")
            axes[1].set(xlabel="efSearch", ylabel="Search p95 (ms)")
            axes[1].grid(alpha=0.25)
            fig.tight_layout()
            plot = output / "tradeoff.svg"
            fig.savefig(plot)
            plt.close(fig)
            plot.write_text(
                "\n".join(line.rstrip() for line in plot.read_text().splitlines())
                + "\n"
            )
        files = {
            p.name: sha256_file(p)
            for p in output.iterdir()
            if p.is_file() and p.name != "results.json"
        }
        result = {
            "task": "B3",
            "scope": "single-positive title-to-own-body known-item",
            "query_count": len(queries),
            "smoke_limit": limit,
            "settings": settings,
            "settings_sha256": sha256_file(settings_path),
            "command": sys.argv,
            "reference": {
                "b2_results_sha256": sha256_file(b2_path),
                "exact_quality": exact_quality,
                "hnsw_index_sha256": reference["hnsw_index_sha256"],
                "pca_model_sha256": b1["provenance"]["pca_model_sha256"],
                "frozen_manifest_sha256": sha256_file(frozen_path),
                "query_encoder_revision": encoder.model_revision,
            },
            "selected_ef_search": settings["selected_ef_search"],
            "selection_reason": (
                "retain previously configured efSearch=128; test sweep is diagnostic"
            ),
            "summaries": summaries,
            "files": files,
            "environment": {
                "platform": platform.platform(),
                "logical_cpus": os.cpu_count(),
                "faiss_threads": faiss.omp_get_max_threads(),
                "packages": {
                    name: version(name)
                    for name in ("numpy", "faiss-cpu", "sentence-transformers", "torch")
                },
            },
        }
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
