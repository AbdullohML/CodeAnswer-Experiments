"""Build the A1 SQLite FTS5 BM25 index."""

from __future__ import annotations

import argparse
import json

from codeanswer.bm25 import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="artifacts/corpus/qa_corpus.parquet")
    parser.add_argument("--index", default="artifacts/indexes/bm25_fts5.sqlite")
    parser.add_argument("--config", default="configs/bm25.json")
    arguments = parser.parse_args()
    print(
        json.dumps(
            build_index(arguments.corpus, arguments.index, arguments.config), indent=2
        )
    )


if __name__ == "__main__":
    main()
