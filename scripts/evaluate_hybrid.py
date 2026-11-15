"""Evaluate A2 dense/BM25 reciprocal rank fusion on frozen C2 queries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from codeanswer.bm25 import open_index as open_bm25_index
from codeanswer.bm25 import search as search_bm25
from codeanswer.config import load_config
from codeanswer.embeddings import SentenceTransformerEncoder
from codeanswer.evaluation import _latency_summary, evaluate_known_item
from codeanswer.evaluation_protocol import load_frozen_split, verify_flat_artifacts
from codeanswer.fusion import fuse_rrf, union_contribution
from codeanswer.indexing import load_index, search_index
from codeanswer.provenance import sha256_file


def load_hybrid_settings(path: str | Path) -> dict[str, Any]:
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or settings.get("schema_version") != 1:
        raise ValueError("Unsupported hybrid configuration")
    if settings.get("method") != "reciprocal_rank_fusion":
        raise ValueError("Hybrid method must be reciprocal_rank_fusion")
    depths = settings.get("candidate_depths")
    if (
        not isinstance(depths, list)
        or not depths
        or any(type(depth) is not int or depth <= 0 or depth > 40 for depth in depths)
        or sorted(set(depths)) != depths
        or settings.get("test_candidate_depth") not in depths
        or settings.get("output_depth") != 40
    ):
        raise ValueError("Invalid hybrid candidate or output depths")
    for key, minimum in (
        ("rrf_constant", 1),
        ("latency_queries", 1),
        ("warmup_queries", 0),
    ):
        value = settings.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid hybrid setting: {key}")
    expected = {
        "schema_version",
        "method",
        "rrf_constant",
        "candidate_depths",
        "test_candidate_depth",
        "output_depth",
        "latency_queries",
        "warmup_queries",
    }
    if set(settings) != expected:
        raise ValueError("Unknown hybrid configuration setting")
    return settings


def load_saved_bm25(
    split: str,
    queries: pd.DataFrame,
    *,
    results_dir: Path,
    evaluation_manifest_sha256: str,
    index_sha256: str,
    config_sha256: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
    """Check every A1 result row against the frozen IDs and labels."""
    result_path = results_dir / f"a1_bm25_{split}_results.json"
    ranking_path = results_dir / f"a1_bm25_{split}_per_query.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("split") != split
        or result.get("smoke_limit") is not None
        or result.get("evaluation_manifest_sha256") != evaluation_manifest_sha256
        or result.get("per_query_sha256") != sha256_file(ranking_path)
        or result.get("config_sha256") != config_sha256
        or result.get("index", {}).get("index_sha256") != index_sha256
        or result.get("quality", {}).get("queries") != len(queries)
    ):
        raise ValueError("Saved A1 BM25 results do not match the frozen run")
    rankings = np.full((len(queries), 40), -1, dtype=np.int64)
    with ranking_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {
            "query_id",
            "family_id",
            "relevant_doc_id",
            "retrieved_doc_ids@40",
        }.issubset(reader.fieldnames or []):
            raise ValueError("Saved BM25 ranking columns are missing")
        count = 0
        for count, row in enumerate(reader, start=1):
            if count > len(queries):
                raise ValueError("Saved BM25 rankings contain extra rows")
            expected = queries.iloc[count - 1]
            if (
                int(row["query_id"]) != int(expected["query_id"])
                or int(row["family_id"]) != int(expected["family_id"])
                or int(row["relevant_doc_id"]) != int(expected["relevant_doc_id"])
            ):
                raise ValueError("Saved BM25 query IDs or labels differ from C2")
            ids = json.loads(row["retrieved_doc_ids@40"])
            if (
                not isinstance(ids, list)
                or len(ids) != 40
                or any(type(doc_id) is not int for doc_id in ids)
            ):
                raise ValueError("Saved BM25 ranking is not 40 integer IDs")
            if any(doc_id >= result["index"]["corpus_rows"] for doc_id in ids):
                raise ValueError("Saved BM25 ranking has out-of-range document ID")
            rankings[count - 1] = ids
        if count != len(queries):
            raise ValueError("Saved BM25 rankings are incomplete")
    return (
        rankings,
        result,
        {
            "results_sha256": sha256_file(result_path),
            "rankings_sha256": sha256_file(ranking_path),
        },
    )


def benchmark_hybrid(
    connection: Any,
    index: Any,
    query_texts: list[str],
    vectors: np.ndarray,
    *,
    candidate_depth: int,
    output_depth: int,
    rrf_constant: int,
    latency_queries: int,
    warmup_queries: int,
) -> dict[str, Any]:
    """Measure sequential BM25 + FAISS + fusion, excluding encoding."""
    count = min(len(query_texts), latency_queries)

    def run_one(i: int) -> None:
        lexical = [
            doc_id
            for doc_id, _ in search_bm25(connection, query_texts[i], candidate_depth)
        ]
        _, dense = search_index(index, vectors[i], top_k=candidate_depth)
        fuse_rrf(
            dense[0].tolist(),
            lexical + [-1] * (candidate_depth - len(lexical)),
            candidate_depth=candidate_depth,
            output_depth=output_depth,
            rrf_constant=rrf_constant,
        )

    for i in range(min(len(query_texts), warmup_queries)):
        run_one(i)
    latency_ms = []
    for i in range(count):
        started = time.perf_counter()
        run_one(i)
        latency_ms.append((time.perf_counter() - started) * 1000)
    return {"latency_queries": count, "search_ms": _latency_summary(latency_ms)}


def _add_identity(frame: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    frame.insert(1, "query_id", queries["query_id"].to_numpy())
    frame.insert(2, "family_id", queries["family_id"].to_numpy())
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", default="configs/default.yaml")
    parser.add_argument("--hybrid-config", default="configs/hybrid.json")
    parser.add_argument("--bm25-config", default="configs/bm25.json")
    parser.add_argument("--corpus", default="artifacts/corpus/qa_corpus.parquet")
    parser.add_argument("--dense-index", default="artifacts/indexes/flat_ip_384.faiss")
    parser.add_argument("--bm25-index", default="artifacts/indexes/bm25_fts5.sqlite")
    parser.add_argument(
        "--evaluation-manifest", default="artifacts/evaluation/frozen_v1/manifest.json"
    )
    parser.add_argument("--a1-results-dir", default="results/evaluation")
    parser.add_argument(
        "--split", choices=["development", "test"], default="development"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke run only")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    settings = load_hybrid_settings(args.hybrid_config)
    project = load_config(args.project_config)
    queries, frozen = load_frozen_split(
        args.evaluation_manifest, args.corpus, split=args.split
    )
    connection, bm25_manifest = open_bm25_index(
        args.bm25_index, args.corpus, args.bm25_config
    )
    try:
        if bm25_manifest["corpus_rows"] != frozen["corpus"]["rows"]:
            raise ValueError("BM25 and frozen corpus sizes differ")
        lexical, a1_result, a1_hashes = load_saved_bm25(
            args.split,
            queries,
            results_dir=Path(args.a1_results_dir),
            evaluation_manifest_sha256=sha256_file(args.evaluation_manifest),
            index_sha256=bm25_manifest["index_sha256"],
            config_sha256=sha256_file(args.bm25_config),
        )
        if args.limit is not None:
            queries = queries.iloc[: args.limit].copy()
            lexical = lexical[: args.limit]
        index = load_index(
            args.dense_index,
            expected_objects=frozen["corpus"]["rows"],
            expected_dimension=project.embedding.dimension,
        )
        dense_identity = verify_flat_artifacts(
            args.corpus,
            args.dense_index,
            index,
            expected_model=project.embedding.model_name,
            expected_sequence_length=project.embedding.max_sequence_length,
        )
        if dense_identity["corpus_sha256"] != frozen["corpus"]["sha256"]:
            raise ValueError("Dense and frozen corpus hashes differ")
        encoder = SentenceTransformerEncoder(
            project.embedding.model_name,
            max_sequence_length=project.embedding.max_sequence_length,
            device=args.device,
        )
        if (
            dense_identity["model_revision"] == "unknown"
            or encoder.model_revision != dense_identity["model_revision"]
        ):
            raise ValueError("Query encoder revision differs from dense embeddings")
        query_texts = queries["query"].tolist()
        vectors = encoder.encode(query_texts, batch_size=project.embedding.batch_size)
        dense_started = time.perf_counter()
        dense_scores, dense_ids = search_index(index, vectors, top_k=40)
        dense_seconds = time.perf_counter() - dense_started
        labels = queries["relevant_doc_id"].to_numpy(dtype=np.int64)
        dense_quality, dense_per_query = evaluate_known_item(
            dense_ids, labels, recall_ks=(1, 5, 10, 40), metric_cutoff=10
        )
        output = (
            (
                Path(args.output_dir)
                if args.output_dir
                else Path(args.evaluation_manifest).parent / f"hybrid_{args.split}"
            )
            .expanduser()
            .resolve()
        )
        output.mkdir(parents=True, exist_ok=True)
        _add_identity(dense_per_query, queries)
        dense_per_query["retrieved_doc_ids@40"] = [
            json.dumps(row.tolist()) for row in dense_ids
        ]
        dense_per_query["cosine_scores@40"] = [
            json.dumps(row.tolist()) for row in dense_scores
        ]
        dense_path = output / "dense_top40.csv"
        dense_per_query.to_csv(dense_path, index=False)
        depths = (
            settings["candidate_depths"]
            if args.split == "development"
            else [settings["test_candidate_depth"]]
        )
        hybrid_results: dict[str, Any] = {}
        files = {
            "dense_top40": {"file": dense_path.name, "sha256": sha256_file(dense_path)}
        }
        for depth in depths:
            fused_rows = [
                fuse_rrf(
                    dense_row,
                    bm25_row,
                    candidate_depth=depth,
                    output_depth=settings["output_depth"],
                    rrf_constant=settings["rrf_constant"],
                )
                for dense_row, bm25_row in zip(dense_ids, lexical, strict=True)
            ]
            fused_ids = np.asarray([row[0] for row in fused_rows], dtype=np.int64)
            quality, per_query = evaluate_known_item(
                fused_ids, labels, recall_ks=(1, 5, 10, 40), metric_cutoff=10
            )
            union, cases = union_contribution(
                dense_ids, lexical, labels, candidate_depth=depth
            )
            _add_identity(per_query, queries)
            per_query["union_case"] = cases
            per_query["retrieved_doc_ids@40"] = [
                json.dumps(row[0]) for row in fused_rows
            ]
            per_query["rrf_scores@40"] = [json.dumps(row[1]) for row in fused_rows]
            path = output / f"hybrid_depth{depth}.csv"
            per_query.to_csv(path, index=False)
            performance = benchmark_hybrid(
                connection,
                index,
                query_texts,
                vectors,
                candidate_depth=depth,
                output_depth=settings["output_depth"],
                rrf_constant=settings["rrf_constant"],
                latency_queries=settings["latency_queries"],
                warmup_queries=settings["warmup_queries"],
            )
            hybrid_results[str(depth)] = {
                "quality": quality,
                "union_contribution": union,
                "performance": performance,
            }
            files[f"hybrid_depth{depth}"] = {
                "file": path.name,
                "sha256": sha256_file(path),
            }
        results = {
            "system": "Flat-384 + SQLite-FTS5-BM25 RRF",
            "split": args.split,
            "smoke_limit": args.limit,
            "settings": settings,
            "hybrid_config_sha256": sha256_file(args.hybrid_config),
            "project_config_sha256": sha256_file(args.project_config),
            "evaluation_manifest_sha256": sha256_file(args.evaluation_manifest),
            "corpus_sha256": frozen["corpus"]["sha256"],
            "dense_artifact_identity": dense_identity,
            "bm25_index_sha256": bm25_manifest["index_sha256"],
            "a1_results": {"quality": a1_result["quality"], **a1_hashes},
            "query_encoder_revision": encoder.model_revision,
            "dense": {
                "quality": dense_quality,
                "batch_search_seconds@40": round(dense_seconds, 6),
            },
            "hybrid": hybrid_results,
            "files": files,
            "latency_scope": (
                "precomputed query vectors; BM25 tokenization + SQLite search + "
                "FAISS search + RRF; single query"
            ),
            "environment": {
                "command": sys.argv,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_logical_count": os.cpu_count(),
                "faiss_threads": faiss.omp_get_max_threads(),
                "encoder_device": encoder.device,
            },
        }
        (output / "results.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "dense": dense_quality,
                    "hybrid": hybrid_results,
                    "output": str(output),
                },
                indent=2,
            )
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
