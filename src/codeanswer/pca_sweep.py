"""B1 nested PCA dimensions with exact, normalized inner-product search."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import shutil
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
from codeanswer.indexing import build_flat_index, load_index, open_embeddings
from codeanswer.optimization import fit_pca_embeddings, normalize_rows
from codeanswer.provenance import sha256_file

EXPECTED = {
    "schema_version": 1,
    "dimensions": [384, 256, 128, 64],
    "fit_components": 256,
    "sample_size": 100000,
    "seed": 20260714,
    "transform_batch_size": 10000,
    "search_batch_size": 32,
    "latency_queries": 100,
    "warmup_queries": 3,
    "faiss_threads": 4,
}


def load_settings(path: Path) -> dict:
    settings = json.loads(path.read_text())
    if settings != EXPECTED:
        raise ValueError("Unsupported B1 settings")
    return settings


def sample_ids(rows: int, count: int, seed: int) -> np.ndarray:
    """Return the exact doc ID order used by the existing PCA fitter."""
    return np.random.default_rng(seed).choice(rows, size=count, replace=False)


def sample_bytes(ids: np.ndarray) -> bytes:
    content = "doc_id\n" + "".join(f"{int(i)}\n" for i in ids)
    return gzip.compress(content.encode(), compresslevel=6, mtime=0)


def project_prefix(pca: object, vectors: np.ndarray, dimension: int) -> np.ndarray:
    """Project on the first components of the shared fit and L2 normalize."""
    if dimension <= 0 or dimension > pca.components_.shape[0]:
        raise ValueError("Invalid PCA prefix dimension")
    batch = np.asarray(vectors, dtype=np.float32)
    if batch.ndim != 2 or batch.shape[1] != pca.components_.shape[1]:
        raise ValueError("Input vector shape differs from PCA fit")
    components = np.ascontiguousarray(pca.components_[:dimension], dtype=np.float32)
    projected = (batch - pca.mean_) @ components.T
    return normalize_rows(projected)


def ensure_fit(
    artifacts: Path, source: Path, source_sha: str, settings: dict, rows: int
) -> tuple[object, dict]:
    """Fit once on fixed IDs or verify a previously completed fit."""
    artifacts.mkdir(parents=True, exist_ok=True)
    ids = sample_ids(rows, settings["sample_size"], settings["seed"])
    ids_path = artifacts / "fit_sample_ids.csv.gz"
    expected_bytes = sample_bytes(ids)
    if ids_path.exists():
        if ids_path.read_bytes() != expected_bytes:
            raise ValueError("Cached PCA fit sample IDs differ")
    else:
        ids_path.write_bytes(expected_bytes)
    reduced = artifacts / "embeddings_256.f32"
    model = artifacts / "pca_384_to_256.joblib"
    identity_path = artifacts / "fit_identity.json"
    if not identity_path.exists():
        if reduced.exists() or model.exists():
            raise ValueError("Incomplete B1 PCA fit artifacts")
        fit = fit_pca_embeddings(
            source,
            reduced,
            model,
            input_dimension=384,
            output_dimension=256,
            sample_size=settings["sample_size"],
            transform_batch_size=settings["transform_batch_size"],
            seed=settings["seed"],
        )
        identity = {
            "source_sha256": source_sha,
            "sample_ids_sha256": sha256_file(ids_path),
            "model_sha256": sha256_file(model),
            "embeddings_256_sha256": sha256_file(reduced),
            "fit": fit,
        }
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    identity = json.loads(identity_path.read_text())
    if (
        identity["source_sha256"] != source_sha
        or identity["sample_ids_sha256"] != sha256_file(ids_path)
        or identity["model_sha256"] != sha256_file(model)
        or identity["embeddings_256_sha256"] != sha256_file(reduced)
        or identity["fit"]["sample_size"] != settings["sample_size"]
        or identity["fit"]["seed"] != settings["seed"]
    ):
        raise ValueError("Cached PCA fit identity differs")
    pca = joblib.load(model)
    if pca.components_.shape != (256, 384):
        raise ValueError("Cached PCA model has unexpected shape")
    return pca, identity


def ensure_embedding(
    artifacts: Path,
    source: Path,
    pca: object,
    dimension: int,
    rows: int,
    fit: dict,
    settings: dict,
) -> tuple[Path, dict]:
    """Write one normalized PCA prefix and verify its artifact identity."""
    output = artifacts / f"embeddings_{dimension}.f32"
    manifest = artifacts / f"embeddings_{dimension}.b1.json"
    if not manifest.exists():
        if dimension == 256:
            if not output.exists():
                raise ValueError("Fitted 256-dimensional embeddings are missing")
            seconds = fit["fit"]["transform_seconds"]
        else:
            if output.exists():
                raise ValueError("Reduced embeddings exist without B1 manifest")
            source_vectors = open_embeddings(source, objects=rows, dimension=384)
            temporary = output.with_name(f".{output.name}.temporary")
            if temporary.exists():
                raise ValueError("Incomplete reduced embedding output")
            target = np.memmap(
                temporary, dtype=np.float32, mode="w+", shape=(rows, dimension)
            )
            started = time.perf_counter()
            for start in range(0, rows, settings["transform_batch_size"]):
                end = min(start + settings["transform_batch_size"], rows)
                target[start:end] = project_prefix(
                    pca, source_vectors[start:end], dimension
                )
            target.flush()
            del target
            temporary.replace(output)
            seconds = round(time.perf_counter() - started, 3)
        info = {
            "dimension": dimension,
            "rows": rows,
            "source_sha256": fit["source_sha256"],
            "pca_model_sha256": fit["model_sha256"],
            "sample_ids_sha256": fit["sample_ids_sha256"],
            "embedding_sha256": sha256_file(output),
            "transform_seconds": seconds,
            "normalization": "row-wise L2 after centered PCA prefix",
        }
        manifest.write_text(json.dumps(info, indent=2) + "\n")
    info = json.loads(manifest.read_text())
    if (
        info["dimension"] != dimension
        or info["rows"] != rows
        or info["source_sha256"] != fit["source_sha256"]
        or info["pca_model_sha256"] != fit["model_sha256"]
        or info["sample_ids_sha256"] != fit["sample_ids_sha256"]
        or info["embedding_sha256"] != sha256_file(output)
        or output.stat().st_size != rows * dimension * 4
    ):
        raise ValueError(f"Cached {dimension}-dimensional embeddings differ")
    return output, info


def ensure_index(
    artifacts: Path, embeddings: Path, dimension: int, rows: int
) -> tuple[Path, dict]:
    index_path = artifacts / f"flat_ip_{dimension}.faiss"
    manifest_path = Path(f"{index_path}.manifest.json")
    if not index_path.exists() and not manifest_path.exists():
        build_flat_index(embeddings, index_path, dimension=dimension)
    if not index_path.exists() or not manifest_path.exists():
        raise ValueError(f"Incomplete {dimension}-dimensional index")
    info = json.loads(manifest_path.read_text())
    if (
        info["index_type"] != "IndexFlatIP"
        or info["metric"] != "inner_product"
        or info["objects"] != rows
        or info["dimension"] != dimension
        or Path(info["embedding_path"]) != embeddings.resolve()
    ):
        raise ValueError(f"Cached {dimension}-dimensional index differs")
    return index_path, info


def evaluate_dimension(
    index: object,
    vectors: np.ndarray,
    queries: pd.DataFrame,
    settings: dict,
    pca: object | None,
    raw_vectors: np.ndarray,
) -> tuple[np.ndarray, dict, pd.DataFrame, dict]:
    """Search all paired queries, then time identical single-query prefixes."""
    count = len(queries)
    rankings = np.empty((count, 40), dtype=np.int64)
    started = time.perf_counter()
    for start in range(0, count, settings["search_batch_size"]):
        end = min(start + settings["search_batch_size"], count)
        _, rankings[start:end] = index.search(
            np.ascontiguousarray(vectors[start:end], dtype=np.float32), 40
        )
    batch_seconds = time.perf_counter() - started
    quality, per_query = evaluate_known_item(
        rankings,
        queries["relevant_doc_id"].to_numpy(),
        recall_ks=(10, 40),
        metric_cutoff=10,
    )
    latency_count = min(settings["latency_queries"], count)
    for i in range(min(settings["warmup_queries"], latency_count)):
        index.search(vectors[i : i + 1], 40)
        if pca is not None:
            project_prefix(pca, raw_vectors[i : i + 1], index.d)
    search_ms, transform_ms = [], []
    for i in range(latency_count):
        if pca is not None:
            t = time.perf_counter()
            project_prefix(pca, raw_vectors[i : i + 1], index.d)
            transform_ms.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        index.search(vectors[i : i + 1], 40)
        search_ms.append((time.perf_counter() - t) * 1000)
    timing = {
        "queries": latency_count,
        "batch_search_seconds": round(batch_seconds, 3),
        "transform_p95_ms": round(float(np.percentile(transform_ms, 95)), 3)
        if transform_ms
        else 0.0,
        "search_p50_ms": round(float(np.percentile(search_ms, 50)), 3),
        "search_p95_ms": round(float(np.percentile(search_ms, 95)), 3),
    }
    return rankings, quality, per_query, timing


def run(
    root: Path,
    settings_path: Path,
    output: Path,
    smoke_limit: int | None = None,
    only_dimension: int | None = None,
) -> dict:
    settings = load_settings(settings_path)
    if only_dimension is not None and only_dimension not in (64, 128, 256):
        raise ValueError("Only a PCA dimension may be selected")
    corpus = root / "artifacts/corpus/qa_corpus.parquet"
    embedding = root / "artifacts/embeddings/question_body_minilm_384.f32"
    baseline_index = root / "artifacts/indexes/flat_ip_384.faiss"
    frozen_path = root / "artifacts/evaluation/frozen_v1/manifest.json"
    queries, frozen = load_frozen_split(frozen_path, corpus, split="test")
    if smoke_limit is not None:
        if smoke_limit <= 0 or smoke_limit >= len(queries):
            raise ValueError("Smoke limit must be below the full test size")
        queries = queries.iloc[:smoke_limit].reset_index(drop=True)
    c2 = json.loads(
        (root / "results/evaluation/c2_frozen_baseline_summary.json").read_text()
    )
    identity = c2["artifact_identity"]
    if (
        identity["corpus_sha256"] != frozen["corpus"]["sha256"]
        or identity["embedding_sha256"] != sha256_file(embedding)
        or identity["index_sha256"] != sha256_file(baseline_index)
        or c2["split_manifest_sha256"] != sha256_file(frozen_path)
    ):
        raise ValueError("C2 baseline artifacts differ from frozen evaluation")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch

    torch.set_num_threads(settings["faiss_threads"])
    faiss.omp_set_num_threads(settings["faiss_threads"])
    with threadpool_limits(limits=settings["faiss_threads"]):
        encoder = SentenceTransformerEncoder(
            c2["embedding_model"],
            max_sequence_length=c2["settings"]["max_sequence_length"],
            device="cpu",
        )
        if (
            encoder.dimension != 384
            or encoder.model_revision != identity["model_revision"]
        ):
            raise ValueError("Query encoder differs from corpus encoder")
        t = time.perf_counter()
        raw_queries = encoder.encode(queries["query"].tolist(), batch_size=64)
        query_encode_seconds = round(time.perf_counter() - t, 3)
        query_vectors_sha = hashlib.sha256(raw_queries.tobytes()).hexdigest()
        artifacts = root / "artifacts/b1"
        pca, fit = ensure_fit(
            artifacts,
            embedding,
            identity["embedding_sha256"],
            settings,
            frozen["corpus"]["rows"],
        )
        dimensions = [384, only_dimension] if only_dimension else settings["dimensions"]
        all_rows = []
        summaries = []
        baseline_ranks = None
        output.mkdir(parents=True, exist_ok=True)
        sample_copy = output / "fit_sample_ids.csv.gz"
        shutil.copyfile(artifacts / "fit_sample_ids.csv.gz", sample_copy)
        for dimension in dimensions:
            if dimension == 384:
                index_path = baseline_index
                index_info = json.loads(Path(f"{index_path}.manifest.json").read_text())
                vectors = raw_queries
                embedded_info = {
                    "dimension": 384,
                    "embedding_sha256": identity["embedding_sha256"],
                    "transform_seconds": 0.0,
                }
            else:
                embedded_path, embedded_info = ensure_embedding(
                    artifacts,
                    embedding,
                    pca,
                    dimension,
                    frozen["corpus"]["rows"],
                    fit,
                    settings,
                )
                index_path, index_info = ensure_index(
                    artifacts, embedded_path, dimension, frozen["corpus"]["rows"]
                )
                vectors = project_prefix(pca, raw_queries, dimension)
            index = load_index(
                index_path,
                expected_objects=frozen["corpus"]["rows"],
                expected_dimension=dimension,
            )
            if type(index) is not faiss.IndexFlatIP:
                raise ValueError("B1 requires exact IndexFlatIP")
            if dimension != 384:
                stored = open_embeddings(
                    embedded_path, objects=index.ntotal, dimension=dimension
                )
                for i in np.linspace(0, index.ntotal - 1, 100, dtype=int):
                    if not np.array_equal(index.reconstruct(int(i)), stored[i]):
                        raise ValueError(
                            "Exact index vectors differ from PCA embeddings"
                        )
            rankings, quality, per_query, timing = evaluate_dimension(
                index,
                vectors,
                queries,
                settings,
                None if dimension == 384 else pca,
                raw_queries,
            )
            del index
            ranks = known_item_ranks(rankings, queries["relevant_doc_id"].to_numpy())
            if dimension == 384:
                baseline_ranks = ranks
                if smoke_limit is None:
                    recorded = json.loads(
                        (root / "results/evaluation/a3_test_results.json").read_text()
                    )["quality"]["dense"]
                    if any(quality[key] != recorded[key] for key in quality):
                        raise ValueError(
                            "Live Flat-384 baseline differs from A3 quality"
                        )
            gained = int(
                (
                    ((baseline_ranks == 0) | (baseline_ranks > 10))
                    & ((ranks > 0) & (ranks <= 10))
                ).sum()
            )
            lost = int(
                (
                    ((baseline_ranks > 0) & (baseline_ranks <= 10))
                    & ((ranks == 0) | (ranks > 10))
                ).sum()
            )
            summary = {
                "dimension": dimension,
                "explained_variance": 1.0
                if dimension == 384
                else round(float(pca.explained_variance_ratio_[:dimension].sum()), 6),
                "index_bytes": index_path.stat().st_size,
                "embedding_bytes": embedding.stat().st_size
                if dimension == 384
                else embedded_path.stat().st_size,
                "quality": quality,
                "timing": timing,
                "gained@10_vs_384": gained,
                "lost@10_vs_384": lost,
                "retrieval_delta_recall@10_vs_384": round(
                    quality["recall@10"] - summaries[0]["quality"]["recall@10"], 6
                )
                if summaries
                else 0.0,
                "ann_loss": 0.0,
                "ann_loss_reason": "both systems use exact IndexFlatIP",
                "index_sha256": sha256_file(index_path),
                "embedding_sha256": embedded_info["embedding_sha256"],
                "corpus_transform_seconds": embedded_info["transform_seconds"],
                "index_build_seconds": index_info["build_seconds"],
            }
            summaries.append(summary)
            (output / f"dim_{dimension}.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            for i, row in enumerate(queries.itertuples(index=False)):
                all_rows.append(
                    {
                        "dimension": dimension,
                        "query_index": i,
                        "query_id": int(row.query_id),
                        "family_id": int(row.family_id),
                        "relevant_doc_id": int(row.relevant_doc_id),
                        "rank": int(per_query.iloc[i]["rank"]),
                        "reciprocal_rank@10": float(
                            per_query.iloc[i]["reciprocal_rank@10"]
                        ),
                        "ndcg@10": float(per_query.iloc[i]["ndcg@10"]),
                        "retrieved_doc_ids@40": json.dumps(rankings[i].tolist()),
                    }
                )
            print(
                f"B1 dimension {dimension}: Recall@10={quality['recall@10']}, "
                f"Recall@40={quality['recall@40']}",
                flush=True,
            )
        rows_path = output / "per_query.csv.gz"
        pd.DataFrame(all_rows).to_csv(
            rows_path, index=False, compression={"method": "gzip", "mtime": 0}
        )
        table = pd.DataFrame(
            [
                {
                    "dimension": s["dimension"],
                    "explained_variance": s["explained_variance"],
                    "recall@10": s["quality"]["recall@10"],
                    "recall@40": s["quality"]["recall@40"],
                    "mrr@10": s["quality"]["mrr@10"],
                    "index_bytes": s["index_bytes"],
                    "transform_p95_ms": s["timing"]["transform_p95_ms"],
                    "search_p95_ms": s["timing"]["search_p95_ms"],
                    "delta_recall@10_vs_384": s["retrieval_delta_recall@10_vs_384"],
                    "gained@10": s["gained@10_vs_384"],
                    "lost@10": s["lost@10_vs_384"],
                }
                for s in summaries
            ]
        )
        table_path = output / "comparison.csv"
        table.to_csv(table_path, index=False)
        if smoke_limit is None and only_dimension is None:
            os.environ.setdefault("MPLCONFIGDIR", str(artifacts / "mplcache"))
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot = table.sort_values("dimension")
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
            for metric in ("recall@10", "recall@40"):
                axes[0].plot(plot.dimension, plot[metric], marker="o", label=metric)
            axes[0].set(xlabel="Vector dimensions", ylabel="Known-item recall")
            axes[0].legend()
            axes[0].grid(alpha=0.25)
            axes[1].plot(plot.dimension, plot.index_bytes / 1024**2, marker="o")
            axes[1].set(xlabel="Vector dimensions", ylabel="Exact index size (MiB)")
            axes[1].grid(alpha=0.25)
            fig.tight_layout()
            plot_path = output / "tradeoff.svg"
            fig.savefig(plot_path)
            plt.close(fig)
            plot_path.write_text(
                "\n".join(line.rstrip() for line in plot_path.read_text().splitlines())
                + "\n"
            )
        files = {
            path.name: sha256_file(path)
            for path in output.iterdir()
            if path.is_file() and path.name != "results.json"
        }
        result = {
            "task": "B1",
            "scope": "single-positive in-corpus title-to-body known-item",
            "smoke_limit": smoke_limit,
            "dimensions": dimensions,
            "query_count": len(queries),
            "settings": settings,
            "settings_sha256": sha256_file(settings_path),
            "command": sys.argv,
            "provenance": {
                "frozen_manifest_sha256": sha256_file(frozen_path),
                "corpus_sha256": frozen["corpus"]["sha256"],
                "source_embedding_sha256": identity["embedding_sha256"],
                "source_index_sha256": identity["index_sha256"],
                "sample_ids_sha256": sha256_file(sample_copy),
                "pca_model_sha256": fit["model_sha256"],
                "query_vectors_sha256": query_vectors_sha,
                "encoder_revision": encoder.model_revision,
            },
            "fit": fit["fit"],
            "query_encode_seconds": query_encode_seconds,
            "summaries": summaries,
            "files": files,
            "environment": {
                "platform": platform.platform(),
                "logical_cpus": os.cpu_count(),
                "device": encoder.device,
                "faiss_threads": faiss.omp_get_max_threads(),
                "packages": {
                    name: version(name)
                    for name in (
                        "numpy",
                        "scikit-learn",
                        "faiss-cpu",
                        "sentence-transformers",
                        "torch",
                    )
                },
            },
        }
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
