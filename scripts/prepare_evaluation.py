"""Freeze deterministic known-item query splits from a corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codeanswer.config import load_config
from codeanswer.evaluation_protocol import freeze_known_item_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--development-count", type=int)
    parser.add_argument("--calibration-count", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    count = (
        args.query_count
        if args.query_count is not None
        else config.evaluation.retrieval_queries
    )
    development = (
        args.development_count if args.development_count is not None else count // 5
    )
    calibration = (
        args.calibration_count if args.calibration_count is not None else count // 5
    )
    corpus = args.corpus or config.paths.artifacts_dir / "corpus" / "qa_corpus.parquet"
    output = args.output_dir or config.paths.results_dir / "evaluation" / "frozen_v1"
    manifest = freeze_known_item_splits(
        corpus,
        output,
        query_count=count,
        seed=config.dataset.random_seed,
        development_count=development,
        calibration_count=calibration,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
