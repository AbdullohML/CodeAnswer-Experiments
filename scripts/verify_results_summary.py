"""Verify E2 headline results against committed per-query evidence and configs."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from codeanswer.error_analysis import paired_bootstrap
from codeanswer.rag_comparison import strategy_summary
from codeanswer.rag_evaluation import summarize

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/evaluation"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def same(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{name}: recomputed {actual!r}, saved {expected!r}")


def close(
    name: str, actual: float, expected: float, tolerance: float = 0.000001
) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=tolerance):
        raise ValueError(f"{name}: recomputed {actual}, saved {expected}")


def verify_config(path: str, expected_sha256: str) -> None:
    actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    same(path, actual, expected_sha256)


def recall(ranks: pd.Series, cutoff: int) -> float:
    return float(ranks.between(1, cutoff).mean())


def verify_a3_a4() -> None:
    a3 = read_json(RESULTS / "a3_test_results.json")
    verify_config("configs/fair_comparison.json", a3["comparison_config_sha256"])
    verify_config("configs/hybrid.json", a3["hybrid_config_sha256"])
    verify_config("configs/bm25.json", a3["bm25_config_sha256"])
    verify_config("configs/default.yaml", a3["project_config_sha256"])
    paired = pd.read_csv(RESULTS / "a3_test_paired_per_query.csv")
    table = pd.read_csv(RESULTS / "a3_test_system_table.csv")
    same("A3 query count", len(paired), 1200)
    for system in ("bm25", "dense", "hybrid"):
        row = table.set_index("system").loc[system]
        ranks = paired[f"{system}_rank"]
        same(f"A3 {system} count", int(row["queries"]), len(ranks))
        for cutoff in (10, 40):
            close(
                f"A3 {system} Recall@{cutoff}",
                recall(ranks, cutoff),
                row[f"recall@{cutoff}"],
            )
        rr = np.where(ranks.between(1, 10), 1 / ranks.replace(0, np.nan), 0)
        close(f"A3 {system} MRR@10", float(np.mean(rr)), row["mrr@10"])

    a4 = read_json(RESULTS / "a4_test/results.json")
    verify_config("configs/reranking_ablation.json", a4["settings_sha256"])
    a4_table = pd.read_csv(RESULTS / "a4_test/results_table.csv")
    a4_rows = read_jsonl(RESULTS / "a4_test/per_query.jsonl.gz")
    for system in ("bm25", "dense", "hybrid"):
        for mode in ("title_body", "body_only"):
            rows = [
                row
                for row in a4_rows
                if row["system"] == system
                and row["input_mode"] == mode
                and not row["vote_bonus"]
            ]
            same(f"A4 {system}/{mode} count", len(rows), 1200)
            row = a4_table.loc[
                (a4_table.system == system)
                & (a4_table.input_mode == mode)
                & (~a4_table.vote_bonus)
            ].iloc[0]
            value = sum(1 <= item["after_rank"] <= 10 for item in rows) / len(rows)
            close(f"A4 {system}/{mode} Recall@10", value, row["after_recall@10"])


def verify_b() -> None:
    for task, config in (
        ("b1", "pca_sweep.json"),
        ("b2", "hnsw_exact.json"),
        ("b3", "ef_sweep.json"),
    ):
        result = read_json(RESULTS / task / "results.json")
        verify_config(f"configs/{config}", result["settings_sha256"])

    b1 = pd.read_csv(RESULTS / "b1/per_query.csv.gz")
    b1_table = pd.read_csv(RESULTS / "b1/comparison.csv").set_index("dimension")
    for dimension, group in b1.groupby("dimension"):
        same(f"B1 {dimension} count", len(group), 1200)
        close(
            f"B1 {dimension} Recall@10",
            recall(group["rank"], 10),
            b1_table.loc[dimension, "recall@10"],
        )

    b2 = pd.read_csv(RESULTS / "b2/per_query.csv.gz")
    b2_table = pd.read_csv(RESULTS / "b2/comparison.csv").set_index("dimension")
    for dimension, group in b2.groupby("dimension"):
        same(f"B2 {dimension} count", len(group), 1200)
        for method in ("exact", "hnsw"):
            close(
                f"B2 {dimension} {method} Recall@10",
                recall(group[f"{method}_rank"], 10),
                b2_table.loc[dimension, f"{method}_recall@10"],
            )

    b3 = pd.read_csv(RESULTS / "b3/per_query.csv.gz")
    b3_table = pd.read_csv(RESULTS / "b3/comparison.csv").set_index("ef_search")
    for ef_search, group in b3.groupby("ef_search"):
        same(f"B3 ef={ef_search} count", len(group), 1200)
        close(
            f"B3 ef={ef_search} Recall@10",
            recall(group["rank"], 10),
            b3_table.loc[ef_search, "recall@10"],
        )

    b4_config = read_json(ROOT / "configs/resource_profile.json")
    same("B4 full rows", b4_config["full_rows"], 520000)
    same("B4 timed queries", b4_config["query_count"], 100)
    b4 = pd.read_csv(RESULTS / "b4/resource_table.csv")
    for system in ("flat384", "flat128", "hnsw128"):
        build = read_json(RESULTS / "b4" / f"build_full_{system}.json")["details"]
        runtime = read_json(RESULTS / "b4" / f"runtime_full_{system}.json")["details"]
        build_row = b4.loc[
            (b4.phase == "build") & (b4.scope == "full") & (b4.system == system)
        ].iloc[0]
        runtime_row = b4.loc[
            (b4.phase == "runtime") & (b4.scope == "full") & (b4.system == system)
        ].iloc[0]
        close(
            f"B4 {system} build",
            build["build_seconds_add_only"],
            build_row["build_or_fit_s"],
        )
        close(
            f"B4 {system} build RSS",
            build["memory"]["rss_peak_mib"],
            build_row["peak_rss_mib"],
        )
        close(
            f"B4 {system} runtime RSS",
            runtime["memory"]["rss_peak_mib"],
            runtime_row["peak_rss_mib"],
        )
        close(
            f"B4 {system} search p95",
            runtime["warm_rounds"][0]["search"]["p95_ms"],
            runtime_row["search_p95_ms"],
        )
        close(
            f"B4 {system} fetch p95",
            runtime["warm_rounds"][0]["fetch40"]["p95_ms"],
            runtime_row["fetch_p95_ms"],
        )


def verify_c3_d() -> None:
    c3 = read_json(RESULTS / "c3/results.json")
    verify_config("configs/error_analysis.json", c3["provenance"]["config_sha256"])
    rows = pd.read_csv(RESULTS / "c3/per_query.csv")
    same("C3 count", len(rows), 1200)
    differences = {
        metric: rows[f"delta_{metric}"].to_numpy()
        for metric in ("recall@10", "recall@40", "mrr@10")
    }
    interval = paired_bootstrap(
        differences,
        seed=c3["settings"]["bootstrap_seed"],
        replicates=c3["settings"]["bootstrap_replicates"],
        level=c3["settings"]["confidence_level"],
    )
    same("C3 paired bootstrap", interval, c3["hybrid_minus_dense"])
    same(
        "C3 top-10 gains",
        int((rows["delta_recall@10"] > 0).sum()),
        c3["paired_top10_gain_count"],
    )
    same(
        "C3 top-10 losses",
        int((rows["delta_recall@10"] < 0).sum()),
        c3["paired_top10_loss_count"],
    )

    d1 = read_json(RESULTS / "d1/fixture_audit.json")
    cases = d1["cases"]
    same("D1 fixture count", len(cases), d1["fixture_counts"]["total"])
    same(
        "D1 fallback count",
        sum(row.get("used_fallback", False) for row in cases),
        d1["fixture_counts"]["fallback"],
    )
    same(
        "D1 generator errors",
        sum("error_type" in row for row in cases),
        d1["fixture_counts"]["generator_errors"],
    )

    d2_dir = RESULTS / "d2/run_local_b1b2"
    d2 = read_json(d2_dir / "summary.json")
    d2_rows = read_jsonl(d2_dir / "test.jsonl")
    same("D2 test summary", summarize(d2_rows), d2["test"])
    threshold = read_json(d2_dir / "selected_threshold.json")["threshold"]
    same(
        "D2 calibration threshold",
        threshold,
        read_json(d2_dir / "run_manifest.json")["selected_threshold"],
    )
    verify_config(
        "configs/default.yaml", read_json(d2_dir / "run_manifest.json")["config_sha256"]
    )

    d3_dir = RESULTS / "d3/run_local_b1b2"
    d3 = read_json(d3_dir / "summary.json")["main"]
    verify_config(
        "configs/bm25.json",
        read_json(d3_dir / "run_manifest.json")["bm25_config_sha256"],
    )
    identities = []
    for strategy in ("dense", "bm25", "hybrid"):
        records = read_jsonl(d3_dir / f"{strategy}.jsonl")
        same(f"D3 {strategy} summary", strategy_summary(records), d3[strategy])
        identities.append([(row["id"], row["expected_behavior"]) for row in records])
    same("D3 paired IDs and labels", identities[1:], identities[:1] * 2)


def main() -> None:
    verify_a3_a4()
    verify_b()
    verify_c3_d()
    print("Verified A3/A4, B1–B4, C3, D1–D3 against saved rows and configs.")


if __name__ == "__main__":
    main()
