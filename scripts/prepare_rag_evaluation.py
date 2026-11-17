"""Create a fixed, labeled StackSample RAG behavior evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from codeanswer.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/evaluation/frozen_v1"
OUT = ROOT / "results/evaluation/d2"

# Handwritten negative controls, not another dataset. Each split is disjoint.
OUT_OF_DOMAIN = {
    "development": [
        "How long does it take to bake a sourdough loaf?",
        "What causes the tides in the ocean?",
        "Which flowers bloom in early spring?",
        "How do I tune an acoustic guitar?",
    ],
    "calibration": [
        "What ingredients make a classic tomato soup?",
        "How do migrating birds navigate?",
        "What are the main rules of volleyball?",
        "How do I grow basil on a windowsill?",
        "What is the capital of Iceland?",
        "How do I paint a watercolor landscape?",
        "Why do some stars appear red?",
        "How do I repair a bicycle tire?",
        "What is the history of the tango?",
        "How do I make a paper airplane?",
    ],
    "test": [
        "How do I knead bread dough?",
        "What causes a solar eclipse?",
        "Which instrument has six strings in standard tuning?",
        "How do I care for an orchid?",
        "What is the distance between Earth and the Moon?",
        "How do I cook lentils?",
        "Why do leaves change color in autumn?",
        "How do I serve a tennis ball?",
    ],
}
IN_DOMAIN_COUNTS = {"development": 8, "calibration": 10, "test": 20}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    source_hashes = {}
    counts = {}
    for split, count in IN_DOMAIN_COUNTS.items():
        source = SOURCE / f"{split}.csv"
        source_hashes[split] = sha256_file(source)
        with source.open(newline="", encoding="utf-8") as handle:
            selected = list(csv.DictReader(handle))[:count]
        if len(selected) != count:
            raise ValueError(f"Not enough C2 {split} queries")
        for item in selected:
            rows.append(
                {
                    "id": f"so-{item['query_id']}",
                    "split": split,
                    "query": item["query"],
                    "expected_behavior": "answer",
                    "origin": "StackSample C2 title",
                    "c2_query_id": int(item["query_id"]),
                    "relevant_doc_id": int(item["relevant_doc_id"]),
                    "family_id": int(item["family_id"]),
                }
            )
        for number, query in enumerate(OUT_OF_DOMAIN[split], start=1):
            rows.append(
                {
                    "id": f"ood-{split}-{number:02d}",
                    "split": split,
                    "query": query,
                    "expected_behavior": "abstain",
                    "origin": "handwritten nonprogramming control",
                    "c2_query_id": None,
                    "relevant_doc_id": None,
                    "family_id": None,
                }
            )
        counts[split] = {"answer": count, "abstain": len(OUT_OF_DOMAIN[split])}
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
    labels_path = OUT / "labels.jsonl"
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "purpose": (
            "behavior answer/abstain only; no correctness or source-support labels"
        ),
        "in_domain_selection": "first N rows of each disjoint C2 frozen split",
        "in_domain_boundary": (
            "indexed StackSample question titles; may be easy known-item queries"
        ),
        "out_of_domain_selection": (
            "handwritten nonprogramming controls listed in preparation script"
        ),
        "threshold_selection_split": "calibration",
        "test_threshold_policy": "fixed before reading test scores",
        "labels_file": "labels.jsonl",
        "labels_sha256": sha256_file(labels_path),
        "counts": counts,
        "source_c2_csv_sha256": source_hashes,
        "source_c2_manifest_sha256": sha256_file(SOURCE / "manifest.json"),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(rows)} labeled queries: {labels_path}")


if __name__ == "__main__":
    main()
