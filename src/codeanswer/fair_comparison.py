"""Paired frozen BM25, dense, and hybrid retrieval comparison."""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from codeanswer.bm25 import open_index as open_bm25_index
from codeanswer.bm25 import search as search_bm25
from codeanswer.config import ProjectConfig
from codeanswer.embeddings import SentenceTransformerEncoder
from codeanswer.evaluation import _latency_summary, evaluate_known_item
from codeanswer.evaluation_protocol import load_frozen_split, verify_flat_artifacts
from codeanswer.fusion import fuse_rrf
from codeanswer.indexing import load_index, search_index
from codeanswer.provenance import sha256_file

EXACT_TERM_PATTERNS = {
    "underscore_identifier": re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b"),
    "version_number": re.compile(r"\b\d+\.\d+(?:\.\d+)?\b"),
    "error_word": re.compile(r"\b(?:error|exception|traceback)\b", re.I),
}
SYSTEMS = ("bm25", "dense", "hybrid")


def load_settings(path: str | Path) -> dict[str, Any]:
    """Validate the fixed A3 cutoffs and latency protocol."""
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "retrieval_depth": 40,
        "recall_cutoffs": [1, 5, 10, 40],
        "rank_metric_cutoff": 10,
    }
    if not isinstance(settings, dict) or any(
        settings.get(k) != v for k, v in required.items()
    ):
        raise ValueError("Unsupported fair-comparison configuration")
    for name, minimum in (
        ("latency_queries", 1),
        ("warmup_queries", 0),
        ("exact_term_queries_per_category", 1),
    ):
        value = settings.get(name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid fair-comparison setting: {name}")
    if set(settings) != set(required) | {
        "latency_queries",
        "warmup_queries",
        "exact_term_queries_per_category",
    }:
        raise ValueError("Unknown fair-comparison setting")
    return settings


def _ranking_matrix(
    frame: pd.DataFrame, queries: pd.DataFrame, rows: int
) -> np.ndarray:
    """Require exact C2 row order, labels, and valid padded top-40 IDs."""
    required = {
        "query_index",
        "query_id",
        "family_id",
        "relevant_doc_id",
        "rank",
        "retrieved_doc_ids@40",
    }
    if not required.issubset(frame.columns) or len(frame) != len(queries):
        raise ValueError("Saved ranking schema or query count differs from C2")
    ids = np.full((len(frame), 40), -1, dtype=np.int64)
    for i, (row, query) in enumerate(
        zip(frame.to_dict("records"), queries.to_dict("records"), strict=True)
    ):
        if (
            int(row["query_index"]) != i
            or int(row["query_id"]) != int(query["query_id"])
            or int(row["family_id"]) != int(query["family_id"])
            or int(row["relevant_doc_id"]) != int(query["relevant_doc_id"])
        ):
            raise ValueError(
                "Saved rankings have different query IDs or relevance labels"
            )
        ranking = json.loads(row["retrieved_doc_ids@40"])
        if (
            not isinstance(ranking, list)
            or len(ranking) != 40
            or any(
                type(doc_id) is not int or doc_id < -1 or doc_id >= rows
                for doc_id in ranking
            )
        ):
            raise ValueError("Saved ranking contains invalid document IDs")
        real = [doc_id for doc_id in ranking if doc_id >= 0]
        if ranking != real + [-1] * (40 - len(real)) or len(real) != len(set(real)):
            raise ValueError("Saved ranking has duplicate IDs or invalid padding")
        rank = next(
            (
                j + 1
                for j, doc_id in enumerate(ranking)
                if doc_id == int(query["relevant_doc_id"])
            ),
            0,
        )
        if int(row["rank"]) != rank:
            raise ValueError("Saved relevant rank differs from ranking IDs")
        ids[i] = ranking
    return ids


def load_paired_sources(
    results_dir: Path,
    split: str,
    queries: pd.DataFrame,
    *,
    evaluation_manifest_sha256: str,
    corpus_sha256: str,
    bm25_index_sha256: str,
    dense_index_sha256: str,
    bm25_config_sha256: str,
    hybrid_config_sha256: str,
    rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Verify A1/A2 provenance, rankings, and their recorded macro metrics."""
    a1_result_path = results_dir / f"a1_bm25_{split}_results.json"
    a1_rank_path = results_dir / f"a1_bm25_{split}_per_query.csv"
    a2_result_path = results_dir / f"a2_{split}_results.json"
    a1 = json.loads(a1_result_path.read_text(encoding="utf-8"))
    a2 = json.loads(a2_result_path.read_text(encoding="utf-8"))
    a1_result_hash = sha256_file(a1_result_path)
    a1_rank_hash = sha256_file(a1_rank_path)
    if (
        a1.get("split") != split
        or a2.get("split") != split
        or a1.get("smoke_limit") is not None
        or a2.get("smoke_limit") is not None
        or a1.get("evaluation_manifest_sha256") != evaluation_manifest_sha256
        or a2.get("evaluation_manifest_sha256") != evaluation_manifest_sha256
        or a1.get("index", {}).get("corpus_sha256") != corpus_sha256
        or a2.get("corpus_sha256") != corpus_sha256
        or a1.get("index", {}).get("index_sha256") != bm25_index_sha256
        or a2.get("bm25_index_sha256") != bm25_index_sha256
        or a2.get("dense_artifact_identity", {}).get("index_sha256")
        != dense_index_sha256
        or a1.get("config_sha256") != bm25_config_sha256
        or a2.get("hybrid_config_sha256") != hybrid_config_sha256
        or a1.get("per_query_sha256") != a1_rank_hash
        or a2.get("a1_results", {}).get("results_sha256") != a1_result_hash
        or a2.get("a1_results", {}).get("rankings_sha256") != a1_rank_hash
        or a2.get("settings", {}).get("test_candidate_depth") != 40
    ):
        raise ValueError("A1/A2 results do not share the frozen inputs and artifacts")
    paths = {
        "bm25": a1_rank_path,
        "dense": results_dir / a2["files"]["dense_top40"]["file"],
        "hybrid": results_dir / a2["files"]["hybrid_depth40"]["file"],
    }
    expected_hashes = {
        "bm25": a1_rank_hash,
        "dense": a2["files"]["dense_top40"]["sha256"],
        "hybrid": a2["files"]["hybrid_depth40"]["sha256"],
    }
    matrices: dict[str, np.ndarray] = {}
    for system in SYSTEMS:
        if sha256_file(paths[system]) != expected_hashes[system]:
            raise ValueError(f"Saved {system} ranking checksum differs")
        frame = pd.read_csv(paths[system])
        matrices[system] = _ranking_matrix(frame, queries, rows)
        quality, _ = evaluate_known_item(
            matrices[system],
            queries["relevant_doc_id"].to_numpy(),
            recall_ks=(1, 5, 10, 40),
            metric_cutoff=10,
        )
        recorded = (
            a1["quality"]
            if system == "bm25"
            else a2["dense"]["quality"]
            if system == "dense"
            else a2["hybrid"]["40"]["quality"]
        )
        if quality != recorded:
            raise ValueError(f"Saved {system} macro metrics differ from per-query IDs")
    provenance = {
        "a1_results_sha256": a1_result_hash,
        "a2_results_sha256": sha256_file(a2_result_path),
        "rankings_sha256": expected_hashes,
    }
    return matrices, provenance


def exact_term_slice(
    queries: pd.DataFrame, per_category: int
) -> tuple[list[int], list[str]]:
    """Choose a bounded, outcome-blind slice by frozen query order."""
    tags = [[] for _ in range(len(queries))]
    for name, pattern in EXACT_TERM_PATTERNS.items():
        selected = 0
        for i, title in enumerate(queries["query"]):
            if pattern.search(title):
                tags[i].append(name)
                selected += 1
                if selected == per_category:
                    break
    indices = [i for i, names in enumerate(tags) if names]
    return indices, ["|".join(names) for names in tags]


def build_paired_table(
    queries: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    *,
    slice_tags: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    """Build per-query ranks and identical-cutoff metrics for all systems."""
    if set(matrices) != set(SYSTEMS) or len(slice_tags) != len(queries):
        raise ValueError("Paired comparison inputs do not align")
    paired = (
        queries[["query_id", "family_id", "relevant_doc_id"]]
        .reset_index(drop=True)
        .copy()
    )
    paired.insert(0, "query_index", np.arange(len(queries)))
    paired["exact_term_tags"] = slice_tags
    paired["exact_term_slice"] = [bool(tags) for tags in slice_tags]
    quality: dict[str, dict[str, float | int]] = {}
    labels = paired["relevant_doc_id"].to_numpy()
    for system in SYSTEMS:
        metrics, per_query = evaluate_known_item(
            matrices[system], labels, recall_ks=(1, 5, 10, 40), metric_cutoff=10
        )
        quality[system] = metrics
        for column in (
            "rank",
            "reciprocal_rank@10",
            "ndcg@10",
            "recall@1",
            "recall@5",
            "recall@10",
            "recall@40",
        ):
            paired[f"{system}_{column}"] = per_query[column].to_numpy()
    return paired, quality


def benchmark_method(
    run_one: Callable[[int], list[int]],
    expected: np.ndarray,
    *,
    latency_queries: int,
    warmup_queries: int,
) -> dict[str, Any]:
    """Time one raw-title retrieval method on the same query prefix."""
    count = min(len(expected), latency_queries)
    for i in range(min(len(expected), warmup_queries)):
        run_one(i)
    measurements = []
    for i in range(count):
        started = time.perf_counter()
        observed = run_one(i)
        elapsed = time.perf_counter() - started
        if observed != expected[i].tolist():
            raise ValueError(f"Live ranking differs from saved ranking at query {i}")
        measurements.append(elapsed)
    total = sum(measurements)
    return {
        "queries": count,
        "search_ms": _latency_summary([value * 1000 for value in measurements]),
        "sequential_qps": round(count / total, 2),
        "total_seconds": round(total, 6),
    }


def run_frozen_comparison(args: Any, project: ProjectConfig) -> None:
    """Replay full paired evidence, verify live rankings, and benchmark raw titles."""
    settings = load_settings(args.comparison_config)
    queries, frozen = load_frozen_split(
        args.evaluation_manifest,
        project.paths.artifacts_dir / "corpus" / "qa_corpus.parquet",
        split=args.split,
    )
    corpus = project.paths.artifacts_dir / "corpus" / "qa_corpus.parquet"
    dense_path = project.paths.artifacts_dir / "indexes" / "flat_ip_384.faiss"
    bm25_path = project.paths.artifacts_dir / "indexes" / "bm25_fts5.sqlite"
    connection, bm25_manifest = open_bm25_index(bm25_path, corpus, args.bm25_config)
    try:
        dense_index = load_index(
            dense_path,
            expected_objects=frozen["corpus"]["rows"],
            expected_dimension=project.embedding.dimension,
        )
        dense_identity = verify_flat_artifacts(
            corpus,
            dense_path,
            dense_index,
            expected_model=project.embedding.model_name,
            expected_sequence_length=project.embedding.max_sequence_length,
        )
        if (
            bm25_manifest["corpus_sha256"] != frozen["corpus"]["sha256"]
            or dense_identity["corpus_sha256"] != frozen["corpus"]["sha256"]
        ):
            raise ValueError("Retriever corpus hashes differ from frozen corpus")
        matrices, source_provenance = load_paired_sources(
            Path(args.source_results_dir),
            args.split,
            queries,
            evaluation_manifest_sha256=sha256_file(args.evaluation_manifest),
            corpus_sha256=frozen["corpus"]["sha256"],
            bm25_index_sha256=bm25_manifest["index_sha256"],
            dense_index_sha256=dense_identity["index_sha256"],
            bm25_config_sha256=sha256_file(args.bm25_config),
            hybrid_config_sha256=sha256_file(args.hybrid_config),
            rows=frozen["corpus"]["rows"],
        )
        encoder = SentenceTransformerEncoder(
            project.embedding.model_name,
            max_sequence_length=project.embedding.max_sequence_length,
            device=args.device,
        )
        if (
            dense_identity["model_revision"] == "unknown"
            or encoder.model_revision != dense_identity["model_revision"]
        ):
            raise ValueError("Query encoder revision differs from dense index")
        # Keep the same full-split encoder batching as the saved A2 rankings,
        # including in smoke runs that inspect only an initial query prefix.
        vectors = encoder.encode(
            queries["query"].tolist(), batch_size=project.embedding.batch_size
        )
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            queries = queries.iloc[: args.limit].copy()
            matrices = {name: matrix[: args.limit] for name, matrix in matrices.items()}
            vectors = vectors[: args.limit]
        titles = queries["query"].tolist()
        _, live_dense = search_index(dense_index, vectors, top_k=40)
        if not np.array_equal(live_dense, matrices["dense"]):
            raise ValueError("Full live dense rankings differ from saved A2 rankings")
        lexical_started = time.perf_counter()
        live_bm25 = np.full((len(titles), 40), -1, dtype=np.int64)
        for i, title in enumerate(titles):
            hits = [doc_id for doc_id, _ in search_bm25(connection, title, 40)]
            live_bm25[i, : len(hits)] = hits
            if (i + 1) % 200 == 0:
                print(
                    f"Verified {i + 1:,}/{len(titles):,} live BM25 rankings", flush=True
                )
        lexical_seconds = time.perf_counter() - lexical_started
        if not np.array_equal(live_bm25, matrices["bm25"]):
            raise ValueError("Full live BM25 rankings differ from saved A1 rankings")
        rrf_constant = json.loads(Path(args.hybrid_config).read_text())["rrf_constant"]
        live_hybrid = np.asarray(
            [
                fuse_rrf(
                    dense_row,
                    bm25_row,
                    candidate_depth=40,
                    output_depth=40,
                    rrf_constant=rrf_constant,
                )[0]
                for dense_row, bm25_row in zip(live_dense, live_bm25, strict=True)
            ],
            dtype=np.int64,
        )
        if not np.array_equal(live_hybrid, matrices["hybrid"]):
            raise ValueError("Full live hybrid rankings differ from saved A2 rankings")
        slice_indices, tags = exact_term_slice(
            queries, settings["exact_term_queries_per_category"]
        )
        if not slice_indices:
            raise ValueError("No queries matched the predefined exact-term slice")
        paired, quality = build_paired_table(queries, matrices, slice_tags=tags)
        slice_quality = {
            name: evaluate_known_item(
                matrix[slice_indices],
                queries["relevant_doc_id"].to_numpy()[slice_indices],
                recall_ks=(1, 5, 10, 40),
                metric_cutoff=10,
            )[0]
            for name, matrix in matrices.items()
        }

        def live_bm25_one(i: int) -> list[int]:
            hits = [doc_id for doc_id, _ in search_bm25(connection, titles[i], 40)]
            return hits + [-1] * (40 - len(hits))

        def live_dense_one(i: int) -> list[int]:
            vector = encoder.encode([titles[i]], batch_size=1)
            return search_index(dense_index, vector, top_k=40)[1][0].tolist()

        def live_hybrid_one(i: int) -> list[int]:
            lexical = live_bm25_one(i)
            dense = live_dense_one(i)
            return fuse_rrf(
                dense,
                lexical,
                candidate_depth=40,
                output_depth=40,
                rrf_constant=rrf_constant,
            )[0]

        funcs = {
            "bm25": live_bm25_one,
            "dense": live_dense_one,
            "hybrid": live_hybrid_one,
        }
        latency = {
            name: benchmark_method(
                funcs[name],
                matrices[name],
                latency_queries=settings["latency_queries"],
                warmup_queries=settings["warmup_queries"],
            )
            for name in SYSTEMS
        }
        output = (
            (
                Path(args.output_dir)
                if args.output_dir
                else project.paths.artifacts_dir
                / "evaluation"
                / "frozen_v1"
                / f"fair_{args.split}"
            )
            .expanduser()
            .resolve()
        )
        output.mkdir(parents=True, exist_ok=True)
        paired_path = output / "paired_per_query.csv"
        slice_path = output / "exact_term_slice.csv"
        table_path = output / "system_table.csv"
        paired.to_csv(paired_path, index=False)
        paired.iloc[slice_indices].to_csv(slice_path, index=False)
        table = pd.DataFrame(
            [
                {
                    "system": name,
                    **quality[name],
                    "search_p50_ms": latency[name]["search_ms"]["p50"],
                    "search_p95_ms": latency[name]["search_ms"]["p95"],
                    "sequential_qps": latency[name]["sequential_qps"],
                }
                for name in SYSTEMS
            ]
        )
        table.to_csv(table_path, index=False)
        result = {
            "task": "A3",
            "split": args.split,
            "smoke_limit": args.limit,
            "scope": "single-positive in-corpus title-to-body known-item retrieval",
            "settings": settings,
            "comparison_config_sha256": sha256_file(args.comparison_config),
            "project_config_sha256": sha256_file(args.config),
            "bm25_config_sha256": sha256_file(args.bm25_config),
            "hybrid_config_sha256": sha256_file(args.hybrid_config),
            "evaluation_manifest_sha256": sha256_file(args.evaluation_manifest),
            "corpus_sha256": frozen["corpus"]["sha256"],
            "dense_artifact_identity": dense_identity,
            "bm25_index_sha256": bm25_manifest["index_sha256"],
            "source_provenance": source_provenance,
            "quality": quality,
            "exact_term_slice": {
                "selection": (
                    "first 10 frozen-order matches per underscore identifier, "
                    "version number, error word; deduplicated"
                ),
                "queries": len(slice_indices),
                "query_ids": paired.iloc[slice_indices]["query_id"]
                .astype(int)
                .tolist(),
                "quality": slice_quality,
            },
            "latency": latency,
            "full_live_verification": {
                "queries": len(queries),
                "dense": "all top-40 IDs identical to A2",
                "bm25": "all top-40 IDs identical to A1",
                "hybrid": "all top-40 IDs identical to A2",
                "bm25_full_search_seconds": round(lexical_seconds, 6),
            },
            "latency_scope": (
                "same raw titles, 100 sequential single-query calls after 3 "
                "warmups; BM25 tokenization and SQLite; MiniLM encoding and "
                "FAISS; both plus RRF"
            ),
            "files": {
                "paired": {
                    "file": paired_path.name,
                    "sha256": sha256_file(paired_path),
                },
                "exact_term_slice": {
                    "file": slice_path.name,
                    "sha256": sha256_file(slice_path),
                },
                "system_table": {
                    "file": table_path.name,
                    "sha256": sha256_file(table_path),
                },
            },
            "environment": {
                "command": sys.argv,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_logical_count": os.cpu_count(),
                "faiss_threads": faiss.omp_get_max_threads(),
                "encoder_device": encoder.device,
                "encoder_revision": encoder.model_revision,
                "packages": {
                    name: version(name)
                    for name in ("numpy", "faiss-cpu", "sentence-transformers", "torch")
                },
            },
        }
        (output / "results.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(table.to_string(index=False))
        print(f"Exact-term slice: {len(slice_indices)} queries; output: {output}")
    finally:
        connection.close()
