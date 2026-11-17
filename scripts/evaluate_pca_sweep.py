"""Run B1 PCA dimension sweep on the C2 frozen test queries."""

from __future__ import annotations

import argparse
from pathlib import Path

from codeanswer.pca_sweep import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--only-dimension", type=int, choices=(64, 128, 256))
    parser.add_argument("--settings", type=Path, default=Path("configs/pca_sweep.json"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/evaluation/b1")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run(
        root, args.settings, args.output_dir, args.smoke_limit, args.only_dimension
    )
    print(f"B1 {result['query_count']} queries; results: {args.output_dir}")


if __name__ == "__main__":
    main()
