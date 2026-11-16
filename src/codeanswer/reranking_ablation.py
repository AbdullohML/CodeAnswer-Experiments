"""A4 paired cross-encoder reranking on saved frozen top-40 candidates."""

from __future__ import annotations

import gzip
import json
import os
import platform
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from codeanswer.evaluation import evaluate_known_item, known_item_ranks
from codeanswer.evaluation_protocol import load_frozen_split
from codeanswer.fair_comparison import SYSTEMS, _ranking_matrix
from codeanswer.provenance import sha256_file
from codeanswer.reranking import CrossEncoderScorer


def load_inputs(
    root: Path, split: str, limit: int | None
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    """Verify frozen labels and A3 source rankings before any model inference."""
    evaluation = root / "artifacts/evaluation/frozen_v1/manifest.json"
    corpus = root / "artifacts/corpus/qa_corpus.parquet"
    queries, frozen = load_frozen_split(evaluation, corpus, split=split)
    a3_path = root / f"results/evaluation/a3_{split}_results.json"
    a3 = json.loads(a3_path.read_text())
    if (
        a3["split"] != split
        or a3["smoke_limit"] is not None
        or a3["evaluation_manifest_sha256"] != sha256_file(evaluation)
        or a3["corpus_sha256"] != frozen["corpus"]["sha256"]
    ):
        raise ValueError("A3 provenance does not match frozen evaluation")
    source = root / "results/evaluation"
    paths = {
        "bm25": source / f"a1_bm25_{split}_per_query.csv",
        "dense": source / f"a2_{split}_dense_top40.csv",
        "hybrid": source / f"a2_{split}_hybrid_depth40.csv",
    }
    matrices = {}
    for system, path in paths.items():
        if sha256_file(path) != a3["source_provenance"]["rankings_sha256"][system]:
            raise ValueError(f"{system} rankings differ from A3")
        matrix = _ranking_matrix(pd.read_csv(path), queries, frozen["corpus"]["rows"])
        quality, _ = evaluate_known_item(
            matrix,
            queries["relevant_doc_id"].to_numpy(),
            recall_ks=(1, 5, 10, 40),
            metric_cutoff=10,
        )
        if quality != a3["quality"][system]:
            raise ValueError(f"{system} baseline metrics differ from A3")
        matrices[system] = matrix
    if limit is not None:
        if limit <= 0 or limit >= len(queries):
            raise ValueError("Smoke limit must be positive and below full split size")
        queries = queries.iloc[:limit].reset_index(drop=True)
        matrices = {name: value[:limit] for name, value in matrices.items()}
    return (
        queries,
        matrices,
        {
            "frozen": frozen,
            "a3_sha256": sha256_file(a3_path),
            "ranking_sha256": a3["source_provenance"]["rankings_sha256"],
        },
    )


def fetch_candidates(
    corpus: Path, matrices: dict[str, np.ndarray]
) -> dict[int, tuple[str, str, int]]:
    """Read only candidate document text and vote scores from the corpus."""
    needed = set(
        int(doc) for matrix in matrices.values() for doc in matrix.flat if doc >= 0
    )
    found = {}
    for batch in pq.ParquetFile(corpus).iter_batches(
        batch_size=4096, columns=["doc_id", "title", "question_body", "answer_score"]
    ):
        for row in batch.to_pylist():
            doc = int(row["doc_id"])
            if doc in needed:
                found[doc] = (
                    row["title"] or "",
                    row["question_body"] or "",
                    int(row["answer_score"] or 0),
                )
    if len(found) != len(needed):
        raise ValueError("Some saved candidate IDs are absent from the corpus")
    return found


def order_candidates(
    ids: list[int], scores: np.ndarray, votes: list[int], weight: float
) -> tuple[list[int], list[float], list[float]]:
    """Sort by raw cross-encoder score plus optional vote bonus; preserve input ties."""
    if (
        len(ids) != len(scores)
        or len(ids) != len(votes)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Candidate scores or vote inputs are invalid")
    bonus = weight * np.log1p(np.maximum(votes, 0))
    final = np.asarray(scores, dtype=np.float64) + bonus
    positions = sorted(range(len(ids)), key=lambda i: (-final[i], i))
    return (
        [ids[i] for i in positions],
        [float(scores[i]) for i in positions],
        [float(bonus[i]) for i in positions],
    )


def score_query(
    query: str,
    candidate_ids: dict[str, list[int]],
    corpus: dict,
    scorer: CrossEncoderScorer,
    batch_size: int,
    vote_weight: float,
) -> tuple[list[dict], dict]:
    """Score the union once per input mode, then apply identical scoring rules."""
    union = list(
        dict.fromkeys(
            doc for system in SYSTEMS for doc in candidate_ids[system] if doc >= 0
        )
    )
    records = []
    elapsed = {}
    for mode in ("title_body", "body_only"):
        pairs = [
            (
                query,
                f"{corpus[doc][0]}\n{corpus[doc][1]}"
                if mode == "title_body"
                else corpus[doc][1],
            )
            for doc in union
        ]
        start = time.perf_counter()
        values = scorer.predict(pairs, batch_size=batch_size)
        elapsed[mode] = (time.perf_counter() - start) * 1000
        if len(values) != len(union) or not np.isfinite(values).all():
            raise ValueError("Cross-encoder returned invalid scores")
        scores = dict(zip(union, values, strict=True))
        for system in SYSTEMS:
            ids = [doc for doc in candidate_ids[system] if doc >= 0]
            raw = np.asarray([scores[doc] for doc in ids], dtype=np.float32)
            votes = [corpus[doc][2] for doc in ids]
            for enabled in (False, True):
                ordered, raw_ordered, bonuses = order_candidates(
                    ids, raw, votes, vote_weight if enabled else 0.0
                )
                records.append(
                    {
                        "system": system,
                        "input_mode": mode,
                        "vote_bonus": enabled,
                        "ranked_doc_ids@40": ordered,
                        "raw_scores@40": raw_ordered,
                        "vote_bonuses@40": bonuses,
                    }
                )
    return records, elapsed


def summarize(
    records: pd.DataFrame,
    queries: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    timings: dict[str, list[float]],
) -> tuple[pd.DataFrame, list[dict]]:
    """Calculate paired before/after metrics and per-query gains and losses."""
    labels = queries["relevant_doc_id"].to_numpy()
    rows = []
    for system in SYSTEMS:
        before, _ = evaluate_known_item(
            matrices[system], labels, recall_ks=(3, 10, 40), metric_cutoff=10
        )
        for mode in ("title_body", "body_only"):
            for enabled in (False, True):
                selected = records[
                    (records.system == system)
                    & (records.input_mode == mode)
                    & (records.vote_bonus == enabled)
                ].sort_values("query_index")
                if len(selected) != len(queries):
                    raise ValueError("Reranked output is missing paired queries")
                matrix = np.asarray(
                    selected["ranked_doc_ids@40"].tolist(), dtype=np.int64
                )
                after, _ = evaluate_known_item(
                    matrix, labels, recall_ks=(3, 10, 40), metric_cutoff=10
                )
                if after["recall@40"] != before["recall@40"]:
                    raise ValueError("Reranking changed the candidate set")
                old_rank = known_item_ranks(matrices[system], labels)
                new_rank = known_item_ranks(matrix, labels)
                old_hit = (old_rank > 0) & (old_rank <= 10)
                new_hit = (new_rank > 0) & (new_rank <= 10)
                times = timings[mode]
                rows.append(
                    {
                        "system": system,
                        "input_mode": mode,
                        "vote_bonus": enabled,
                        "candidate_recall@40": before["recall@40"],
                        **{
                            f"before_{key}": before[key]
                            for key in ("recall@3", "recall@10", "mrr@10", "ndcg@10")
                        },
                        **{
                            f"after_{key}": after[key]
                            for key in ("recall@3", "recall@10", "mrr@10", "ndcg@10")
                        },
                        "gained@10": int((~old_hit & new_hit).sum()),
                        "lost@10": int((old_hit & ~new_hit).sum()),
                        "rank_improved": int(
                            (
                                (new_rank > 0)
                                & ((old_rank == 0) | (new_rank < old_rank))
                            ).sum()
                        ),
                        "rank_worsened": int(
                            (
                                (old_rank > 0)
                                & ((new_rank == 0) | (new_rank > old_rank))
                            ).sum()
                        ),
                        "rerank_p50_ms": round(float(np.percentile(times, 50)), 3),
                        "rerank_p95_ms": round(float(np.percentile(times, 95)), 3),
                    }
                )
    return pd.DataFrame(rows), rows


def run(
    root: Path,
    split: str,
    output: Path,
    model_path: Path,
    settings_path: Path,
    limit: int | None = None,
    device: str = "cpu",
) -> dict:
    settings = json.loads(settings_path.read_text())
    if settings != {
        "schema_version": 1,
        "candidate_depth": 40,
        "batch_size": 32,
        "max_length": 384,
        "vote_weight": 0.03,
        "input_modes": ["title_body", "body_only"],
        "seed": 20260714,
    }:
        raise ValueError("Unsupported A4 settings")
    queries, matrices, provenance = load_inputs(root, split, limit)
    corpus = fetch_candidates(root / "artifacts/corpus/qa_corpus.parquet", matrices)
    import torch

    torch.set_num_threads(4)
    if device not in {"cpu", "cuda"} or (
        device == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError(f"Requested device is unavailable: {device}")
    scorer = CrossEncoderScorer(str(model_path), device=device, max_length=384)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    records = []
    timings = {"title_body": [], "body_only": []}
    checkpoint = (
        root
        / "artifacts/evaluation/frozen_v1"
        / (f"a4_{split}_{limit or 'full'}_{device}_progress.jsonl")
    )
    signature = {
        "a3_sha256": provenance["a3_sha256"],
        "settings_sha256": sha256_file(settings_path),
        "weights_sha256": sha256_file(model_path / "model.safetensors"),
        "split": split,
        "limit": limit,
        "device": device,
    }
    completed = 0
    if checkpoint.exists():
        with checkpoint.open() as handle:
            header = json.loads(next(handle))
            if header != {"signature": signature}:
                raise ValueError("A4 progress belongs to different inputs")
            for line in handle:
                item = json.loads(line)
                index = item["query_index"]
                if (
                    index != completed
                    or len(item["records"]) != 12
                    or item["records"][0]["query_id"]
                    != int(queries.iloc[index]["query_id"])
                ):
                    raise ValueError("A4 progress has invalid paired query order")
                for record in item["records"]:
                    record["rerank_ms"] = item["timing_ms"][record["input_mode"]]
                records.extend(item["records"])
                for mode in timings:
                    timings[mode].append(item["timing_ms"][mode])
                completed += 1
    else:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"signature": signature}) + "\n")
    started = time.perf_counter()
    for i in range(completed, len(queries)):
        query = queries.iloc[i]
        candidates = {name: matrix[i].tolist() for name, matrix in matrices.items()}
        scored, times = score_query(query.query, candidates, corpus, scorer, 32, 0.03)
        for record in scored:
            record["rerank_ms"] = times[record["input_mode"]]
            record.update(
                {
                    "query_index": i,
                    "query_id": int(query.query_id),
                    "family_id": int(query.family_id),
                    "relevant_doc_id": int(query.relevant_doc_id),
                    "before_rank": int(
                        np.where(
                            matrices[record["system"]][i] == query.relevant_doc_id
                        )[0][0]
                        + 1
                    )
                    if query.relevant_doc_id in matrices[record["system"]][i]
                    else 0,
                    "after_rank": record["ranked_doc_ids@40"].index(
                        query.relevant_doc_id
                    )
                    + 1
                    if query.relevant_doc_id in record["ranked_doc_ids@40"]
                    else 0,
                }
            )
        records.extend(scored)
        for mode in timings:
            timings[mode].append(times[mode])
        with checkpoint.open("a") as handle:
            handle.write(
                json.dumps({"query_index": i, "records": scored, "timing_ms": times})
                + "\n"
            )
        if (i + 1) % 100 == 0:
            print(f"scored {i + 1}/{len(queries)} queries", flush=True)
    frame = pd.DataFrame(records)
    table, summary = summarize(frame, queries, matrices, timings)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "per_query.jsonl.gz"
    with rows_path.open("wb") as raw_handle:
        with gzip.GzipFile(
            fileobj=raw_handle, mode="wb", filename="", mtime=0
        ) as handle:
            for record in records:
                handle.write((json.dumps(record) + "\n").encode())
    table_path = output / "results_table.csv"
    table.to_csv(table_path, index=False)
    result = {
        "task": "A4",
        "split": split,
        "smoke_limit": limit,
        "scope": "single-positive in-corpus title-to-body known-item retrieval",
        "query_count": len(queries),
        "settings": settings,
        "settings_sha256": sha256_file(settings_path),
        "provenance": provenance,
        "model": {
            "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "revision": (
                model_path / ".cache/huggingface/download/config.json.metadata"
            )
            .read_text()
            .splitlines()[0],
            "config_sha256": sha256_file(model_path / "config.json"),
            "weights_sha256": sha256_file(model_path / "model.safetensors"),
            "tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
            "tokenizer_config_sha256": sha256_file(
                model_path / "tokenizer_config.json"
            ),
        },
        "hardware": {
            "device": scorer.device,
            "cuda_available": torch.cuda.is_available(),
            "vram_peak_bytes": (
                torch.cuda.max_memory_allocated() if device == "cuda" else None
            ),
            "vram_reserved_peak_bytes": (
                torch.cuda.max_memory_reserved() if device == "cuda" else None
            ),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
        "command": sys.argv,
        "packages": {
            name: version(name)
            for name in ("torch", "sentence-transformers", "numpy", "pandas")
        },
        "resumed_queries": completed,
        "invocation_elapsed_seconds": round(time.perf_counter() - started, 3),
        "total_scoring_seconds": round(
            sum(sum(values) for values in timings.values()) / 1000, 3
        ),
        "scoring_scope": (
            "per-query union across three top-40 sets; one cross-encoder call "
            "per input mode; excludes retrieval and corpus loading"
        ),
        "timing_ms": {
            mode: {
                "p50": round(float(np.percentile(values, 50)), 3),
                "p95": round(float(np.percentile(values, 95)), 3),
                "total": round(sum(values), 3),
            }
            for mode, values in timings.items()
        },
        "summary": summary,
        "files": {
            "per_query": {"name": rows_path.name, "sha256": sha256_file(rows_path)},
            "results_table": {
                "name": table_path.name,
                "sha256": sha256_file(table_path),
            },
        },
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
