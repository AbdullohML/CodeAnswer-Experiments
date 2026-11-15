"""Evaluate A1 BM25 on the frozen C2 known-item queries."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sqlite3
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from codeanswer.bm25 import load_settings, open_index, search
from codeanswer.evaluation import _latency_summary, evaluate_known_item
from codeanswer.evaluation_protocol import load_frozen_split
from codeanswer.provenance import sha256_file

EXACT_TERM_PATTERNS = {
    "underscore_identifier": re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b"),
    "version_number": re.compile(r"\b\d+\.\d+(?:\.\d+)?\b"),
    "error_word": re.compile(r"\b(?:error|exception|traceback)\b", re.I),
}


def benchmark(
    connection: sqlite3.Connection,
    queries: list[str],
    *,
    top_k: int,
    latency_queries: int,
    warmup_queries: int,
) -> dict[str, Any]:
    """Measure search-only single-query latency after a fixed warmup."""
    for query in queries[:warmup_queries]:
        search(connection, query, top_k)
    elapsed_ms = []
    for query in queries[: min(len(queries), latency_queries)]:
        started = time.perf_counter()
        search(connection, query, top_k)
        elapsed_ms.append((time.perf_counter() - started) * 1000)
    return {
        "latency_queries": len(elapsed_ms),
        "search_ms": _latency_summary(elapsed_ms),
    }


def inspect_exact_terms(queries: Any, ranks: np.ndarray) -> dict[str, Any]:
    """Summarize lexical case slices on development queries only."""
    inspection: dict[str, Any] = {}
    for name, pattern in EXACT_TERM_PATTERNS.items():
        indices = [
            i for i, query in enumerate(queries["query"]) if pattern.search(query)
        ]
        hits = sum(0 < int(ranks[i]) <= 40 for i in indices)
        inspection[name] = {
            "queries": len(indices),
            "recall@40": round(hits / len(indices), 6) if indices else None,
            "examples": [
                {
                    "query_id": int(queries.iloc[i]["query_id"]),
                    "relevant_doc_id": int(queries.iloc[i]["relevant_doc_id"]),
                    "rank@40": int(ranks[i]),
                }
                for i in indices[:5]
            ],
        }
    return inspection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="artifacts/corpus/qa_corpus.parquet")
    parser.add_argument("--index", default="artifacts/indexes/bm25_fts5.sqlite")
    parser.add_argument("--config", default="configs/bm25.json")
    parser.add_argument(
        "--evaluation-manifest", default="artifacts/evaluation/frozen_v1/manifest.json"
    )
    parser.add_argument(
        "--split", choices=["development", "calibration", "test"], default="development"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test only; do not use for final results",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    settings = load_settings(args.config)
    queries, frozen = load_frozen_split(
        args.evaluation_manifest, args.corpus, split=args.split
    )
    if args.limit is not None:
        queries = queries.iloc[: args.limit].copy()
    connection, index_manifest = open_index(args.index, args.corpus, args.config)
    if index_manifest["corpus_rows"] != frozen["corpus"]["rows"]:
        raise ValueError("BM25 index row count differs from frozen corpus")
    output = (
        (
            Path(args.output_dir)
            if args.output_dir
            else Path(args.evaluation_manifest).parent / f"bm25_{args.split}"
        )
        .expanduser()
        .resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    query_texts = queries["query"].tolist()
    top_k = max(settings["top_ks"])
    started = time.perf_counter()
    try:
        rankings = [search(connection, query, top_k) for query in query_texts]
        batch_seconds = time.perf_counter() - started
        latency = {
            str(k): benchmark(
                connection,
                query_texts,
                top_k=k,
                latency_queries=settings["latency_queries"],
                warmup_queries=settings["warmup_queries"],
            )
            for k in settings["top_ks"]
        }
    finally:
        connection.close()
    ids = np.full((len(rankings), top_k), -1, dtype=np.int64)
    scores: list[str] = []
    for i, ranking in enumerate(rankings):
        ids[i, : len(ranking)] = [doc_id for doc_id, _ in ranking]
        scores.append(json.dumps([score for _, score in ranking]))
    relevant = queries["relevant_doc_id"].to_numpy(dtype=np.int64)
    quality, per_query = evaluate_known_item(
        ids, relevant, recall_ks=(1, 5, 10, 40), metric_cutoff=10
    )
    per_query.insert(1, "query_id", queries["query_id"].to_numpy())
    per_query.insert(2, "family_id", queries["family_id"].to_numpy())
    per_query["retrieved_doc_ids@40"] = [json.dumps(row.tolist()) for row in ids]
    per_query["bm25_scores@40"] = scores
    per_query_path = output / "per_query_rankings.csv"
    per_query.to_csv(per_query_path, index=False)
    result = {
        "system": "SQLite-FTS5-BM25",
        "evaluation_type": frozen["evaluation_type"],
        "split": args.split,
        "smoke_limit": args.limit,
        "quality": quality,
        "performance": {
            "batch_queries": len(query_texts),
            "batch_search_seconds@40": round(batch_seconds, 6),
            "search_qps@40": round(len(query_texts) / batch_seconds, 2),
            "single_query": latency,
        },
        "index": index_manifest,
        "config_sha256": sha256_file(args.config),
        "evaluation_manifest_sha256": sha256_file(args.evaluation_manifest),
        "per_query_sha256": sha256_file(per_query_path),
        "exact_term_inspection": (
            inspect_exact_terms(queries, per_query["rank"].to_numpy())
            if args.split == "development"
            else None
        ),
        "environment": {
            "command": sys.argv,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "pyarrow": version("pyarrow"),
            "cpu_logical_count": os.cpu_count(),
        },
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "quality": result["quality"],
                "performance": result["performance"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
