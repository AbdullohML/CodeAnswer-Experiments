"""Paired uncertainty and slice analysis for the frozen StackSample test."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from codeanswer.evaluation_protocol import load_frozen_split
from codeanswer.fair_comparison import EXACT_TERM_PATTERNS
from codeanswer.provenance import sha256_file

SYSTEMS = ("bm25", "dense", "hybrid")
METRICS = ("recall@10", "recall@40", "mrr@10")


def relevance_at(rank: np.ndarray, cutoff: int) -> np.ndarray:
    """One-positive hit indicator, with zero rank meaning not retrieved."""
    rank = np.asarray(rank, dtype=int)
    if cutoff < 1 or rank.ndim != 1 or np.any(rank < 0):
        raise ValueError("Invalid ranks or cutoff")
    return ((rank > 0) & (rank <= cutoff)).astype(float)


def reciprocal_rank(rank: np.ndarray, cutoff: int = 10) -> np.ndarray:
    """One-positive reciprocal rank through a fixed cutoff."""
    hits = relevance_at(rank, cutoff)
    return np.divide(hits, rank, out=np.zeros_like(hits), where=rank > 0)


def paired_bootstrap(
    differences: dict[str, np.ndarray], *, seed: int, replicates: int, level: float
) -> dict[str, dict[str, float]]:
    """Resample query IDs together; percentile interval for each mean delta."""
    if not differences or replicates < 100 or not 0 < level < 1:
        raise ValueError("Invalid paired bootstrap settings")
    arrays = {key: np.asarray(value, dtype=float) for key, value in differences.items()}
    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
        raise ValueError("Paired differences must have equal nontrivial length")
    if any(
        array.ndim != 1 or not np.isfinite(array).all() for array in arrays.values()
    ):
        raise ValueError("Paired differences must be finite vectors")
    count = next(iter(lengths))
    rng = np.random.default_rng(seed)
    samples = {key: np.empty(replicates, dtype=float) for key in arrays}
    for start in range(0, replicates, 500):
        end = min(start + 500, replicates)
        indices = rng.integers(0, count, size=(end - start, count))
        for key, array in arrays.items():
            samples[key][start:end] = array[indices].mean(axis=1)
    alpha = (1 - level) / 2
    return {
        key: {
            "mean_delta": round(float(array.mean()), 6),
            "ci_low": round(float(np.quantile(samples[key], alpha)), 6),
            "ci_high": round(float(np.quantile(samples[key], 1 - alpha)), 6),
        }
        for key, array in arrays.items()
    }


def _load_inputs(root: Path, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    results_dir = root / "results/evaluation"
    a3_path = results_dir / "a3_test_results.json"
    a3 = json.loads(a3_path.read_text())
    paired_path = results_dir / a3["files"]["paired"]["file"]
    frozen_path = root / "artifacts/evaluation/frozen_v1/manifest.json"
    corpus_path = root / "artifacts/corpus/qa_corpus.parquet"
    if (
        a3.get("task") != "A3"
        or a3.get("split") != settings["split"]
        or a3.get("smoke_limit") is not None
        or a3.get("evaluation_manifest_sha256") != sha256_file(frozen_path)
        or a3.get("corpus_sha256") != sha256_file(corpus_path)
        or a3["files"]["paired"]["sha256"] != sha256_file(paired_path)
    ):
        raise ValueError("A3 inputs are not the frozen full test")
    queries, _ = load_frozen_split(frozen_path, corpus_path, split=settings["split"])
    paired = pd.read_csv(paired_path)
    if len(queries) != settings["query_count"] or len(paired) != len(queries):
        raise ValueError("Frozen query count differs from C3 configuration")
    for name in ("query_id", "family_id", "relevant_doc_id"):
        if not np.array_equal(paired[name].to_numpy(), queries[name].to_numpy()):
            raise ValueError(f"A3 {name} differs from frozen test")
    if not np.array_equal(paired.query_index.to_numpy(), np.arange(len(paired))):
        raise ValueError("A3 query order differs from frozen test")
    for system in SYSTEMS:
        rank = paired[f"{system}_rank"].to_numpy(dtype=int)
        if np.any((rank < 0) | (rank > 40)):
            raise ValueError(f"Invalid {system} rank")
        for cutoff in (10, 40):
            expected = relevance_at(rank, cutoff)
            saved = paired[f"{system}_recall@{cutoff}"].to_numpy(dtype=float)
            if not np.array_equal(expected, saved):
                raise ValueError(f"A3 {system} recall@{cutoff} differs from rank")
            if (
                round(float(expected.mean()), 6)
                != a3["quality"][system][f"recall@{cutoff}"]
            ):
                raise ValueError(f"A3 {system} aggregate recall differs")
        rr = reciprocal_rank(rank)
        if not np.allclose(rr, paired[f"{system}_reciprocal_rank@10"]):
            raise ValueError(f"A3 {system} reciprocal rank differs")
        if round(float(rr.mean()), 6) != a3["quality"][system]["mrr@10"]:
            raise ValueError(f"A3 {system} MRR differs")
    return queries, paired, a3


def _groups(
    queries: pd.DataFrame, settings: dict
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    titles = queries["query"].fillna("").astype(str)
    token_count = titles.map(lambda title: len(title.split())).to_numpy(dtype=int)
    flags = {
        name: titles.map(lambda title, pat=pattern: bool(pat.search(title))).to_numpy(
            dtype=bool
        )
        for name, pattern in EXACT_TERM_PATTERNS.items()
    }
    any_pattern = np.logical_or.reduce(list(flags.values()))
    groups = {
        "all": np.ones(len(titles), dtype=bool),
        **flags,
        "any_exact_term_pattern": any_pattern,
        "no_exact_term_pattern": ~any_pattern,
        "short_1_to_5_tokens": token_count <= settings["short_max_tokens"],
        "medium_6_to_10_tokens": (
            (token_count > settings["short_max_tokens"])
            & (token_count <= settings["medium_max_tokens"])
        ),
        "long_11_plus_tokens": token_count > settings["medium_max_tokens"],
    }
    annotations = pd.DataFrame({"title_token_count": token_count, **flags})
    return groups, annotations


def _group_table(paired: pd.DataFrame, groups: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, mask in groups.items():
        count = int(mask.sum())
        if count == 0:
            continue
        row = {"group": name, "queries": count}
        for system in SYSTEMS:
            rank = paired[f"{system}_rank"].to_numpy(dtype=int)[mask]
            row[f"{system}_recall@10"] = round(float(relevance_at(rank, 10).mean()), 6)
            row[f"{system}_recall@40"] = round(float(relevance_at(rank, 40).mean()), 6)
            row[f"{system}_mrr@10"] = round(float(reciprocal_rank(rank).mean()), 6)
        for cutoff in (10, 40):
            dense = relevance_at(paired.dense_rank.to_numpy(dtype=int)[mask], cutoff)
            hybrid = relevance_at(paired.hybrid_rank.to_numpy(dtype=int)[mask], cutoff)
            row[f"hybrid_minus_dense_recall@{cutoff}"] = round(
                float((hybrid - dense).mean()), 6
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _examples(per_query: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select deterministic extreme paired changes, without query text."""
    gains = (
        per_query[per_query["delta_recall@10"] > 0]
        .sort_values(["delta_mrr@10", "query_index"], ascending=[False, True])
        .head(count)
    )
    losses = (
        per_query[per_query["delta_recall@10"] < 0]
        .sort_values(["delta_mrr@10", "query_index"], ascending=[True, True])
        .head(count)
    )
    selected = pd.concat(
        [gains.assign(direction="gain"), losses.assign(direction="loss")]
    )
    return selected[
        [
            "direction",
            "query_index",
            "query_id",
            "relevant_doc_id",
            "title_token_count",
            *EXACT_TERM_PATTERNS,
            "bm25_rank",
            "dense_rank",
            "hybrid_rank",
            "delta_recall@10",
            "delta_mrr@10",
        ]
    ]


def run(root: Path, output: Path) -> dict:
    started = time.perf_counter()
    config_path = root / "configs/error_analysis.json"
    settings = json.loads(config_path.read_text())
    if (
        settings.get("schema_version") != 1
        or settings.get("split") != "test"
        or settings.get("query_count") != 1200
        or settings.get("short_max_tokens") != 5
        or settings.get("medium_max_tokens") != 10
        or settings.get("examples_per_direction") != 6
    ):
        raise ValueError("Unsupported C3 configuration")
    queries, paired, a3 = _load_inputs(root, settings)
    groups, annotations = _groups(queries, settings)
    per_query = pd.concat(
        [
            paired[
                [
                    "query_index",
                    "query_id",
                    "family_id",
                    "relevant_doc_id",
                    "bm25_rank",
                    "dense_rank",
                    "hybrid_rank",
                ]
            ],
            annotations,
        ],
        axis=1,
    )
    differences = {}
    for metric in METRICS:
        if metric == "mrr@10":
            dense = reciprocal_rank(paired.dense_rank.to_numpy())
            hybrid = reciprocal_rank(paired.hybrid_rank.to_numpy())
        else:
            cutoff = int(metric.split("@")[1])
            dense = relevance_at(paired.dense_rank.to_numpy(), cutoff)
            hybrid = relevance_at(paired.hybrid_rank.to_numpy(), cutoff)
        delta = hybrid - dense
        differences[metric] = delta
        per_query[f"delta_{metric}"] = delta
    uncertainty = paired_bootstrap(
        differences,
        seed=settings["bootstrap_seed"],
        replicates=settings["bootstrap_replicates"],
        level=settings["confidence_level"],
    )
    group_table = _group_table(paired, groups)
    examples = _examples(per_query, settings["examples_per_direction"])
    gains = per_query[per_query["delta_recall@10"] > 0]
    losses = per_query[per_query["delta_recall@10"] < 0]
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_query": output / "per_query.csv",
        "groups": output / "group_table.csv",
        "examples": output / "illustrative_examples.csv",
    }
    per_query.to_csv(paths["per_query"], index=False, lineterminator="\n")
    group_table.to_csv(paths["groups"], index=False, lineterminator="\n")
    examples.to_csv(paths["examples"], index=False, lineterminator="\n")
    result = {
        "task": "C3",
        "scope": "frozen in-corpus StackSample title-to-own-body known-item test",
        "queries": len(queries),
        "settings": settings,
        "uncertainty_method": (
            "query-level paired percentile bootstrap; resample the same IDs "
            "for both systems; no multiple-comparison correction for slices"
        ),
        "hybrid_minus_dense": uncertainty,
        "paired_top10_gain_count": int((differences["recall@10"] > 0).sum()),
        "paired_top10_loss_count": int((differences["recall@10"] < 0).sum()),
        "paired_top40_gain_count": int((differences["recall@40"] > 0).sum()),
        "paired_top40_loss_count": int((differences["recall@40"] < 0).sum()),
        "top10_mechanisms": {
            "gains_with_bm25_gold_in_top40": int((gains.bm25_rank > 0).sum()),
            "gains_with_dense_gold_absent_top40": int((gains.dense_rank == 0).sum()),
            "losses_with_bm25_gold_absent_top40": int((losses.bm25_rank == 0).sum()),
            "both_sources_gold_absent_top40": int(
                ((paired.bm25_rank == 0) & (paired.dense_rank == 0)).sum()
            ),
        },
        "group_policy": (
            "Regex categories use A3's patterns on all 1200 titles and may "
            "overlap. Token bins use whitespace-separated title tokens. "
            "All groups are descriptive; no subgroup significance claim."
        ),
        "example_policy": (
            "Six largest top-10 reciprocal-rank gains and losses, with ties "
            "broken by frozen query order; titles not committed"
        ),
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "a3_results_sha256": sha256_file(
                root / "results/evaluation/a3_test_results.json"
            ),
            "a3_paired_sha256": a3["files"]["paired"]["sha256"],
            "frozen_manifest_sha256": a3["evaluation_manifest_sha256"],
            "corpus_sha256": a3["corpus_sha256"],
        },
        "files": {
            name: {"file": path.name, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "command": " ".join(sys.argv),
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
