"""Run B2 paired HNSW versus exact search on frozen C2 queries."""

from __future__ import annotations

import argparse
from pathlib import Path

from codeanswer.hnsw_exact import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--only-dimension", type=int, choices=(128, 384))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/evaluation/b2")
    )
    args = parser.parse_args()
    result = run(
        Path(__file__).resolve().parents[1],
        args.output_dir,
        args.smoke_limit,
        args.only_dimension,
    )
    print(f"B2 {result['query_count']} queries: {args.output_dir}")


if __name__ == "__main__":
    main()
