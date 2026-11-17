"""Run C6 paired error analysis on A3's frozen StackSample rankings."""

from __future__ import annotations

import argparse
from pathlib import Path

from codeanswer.error_analysis import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/evaluation/c6")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run(root, args.output_dir)
    print(f"C6 analyzed {result['queries']} frozen test queries: {args.output_dir}")


if __name__ == "__main__":
    main()
