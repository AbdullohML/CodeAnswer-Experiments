"""Reproducible lexical retrieval over question bodies using SQLite FTS5."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from codeanswer.provenance import sha256_file

TOKENIZER = "unicode61 remove_diacritics 1 tokenchars '_'"
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:_[^\W_]+)*", re.UNICODE)
REQUIRED_SETTINGS = {
    "schema_version": 1,
    "backend": "sqlite_fts5",
    "indexed_field": "question_body",
    "tokenizer": TOKENIZER,
    "query_operator": "OR",
    "bm25_k1": 1.2,
    "bm25_b": 0.75,
    "top_ks": [10, 40],
}


def load_settings(path: str | Path) -> dict[str, Any]:
    """Load the pinned BM25 protocol, rejecting unsupported changes."""
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("BM25 configuration must be an object")
    for key, value in REQUIRED_SETTINGS.items():
        if settings.get(key) != value:
            raise ValueError(f"Unsupported BM25 setting: {key}")
    if (
        not isinstance(settings.get("latency_queries"), int)
        or settings["latency_queries"] <= 0
    ):
        raise ValueError("latency_queries must be a positive integer")
    if (
        not isinstance(settings.get("warmup_queries"), int)
        or settings["warmup_queries"] < 0
    ):
        raise ValueError("warmup_queries must be a nonnegative integer")
    if set(settings) != set(REQUIRED_SETTINGS) | {"latency_queries", "warmup_queries"}:
        raise ValueError("Unknown BM25 configuration setting")
    return settings


def tokenize_query(query: str) -> list[str]:
    """Extract unique terms, preserving underscores in code identifiers."""
    return list(
        dict.fromkeys(token.casefold() for token in TOKEN_PATTERN.findall(query))
    )


def match_expression(query: str) -> str | None:
    """Build an escaped FTS5 OR query, or None when no terms exist."""
    terms = tokenize_query(query)
    if not terms:
        return None
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def build_index(
    corpus_path: str | Path,
    index_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Build a contentless FTS5 index with rowid = doc_id + 1."""
    settings = load_settings(config_path)
    corpus = Path(corpus_path).expanduser().resolve()
    output = Path(index_path).expanduser().resolve()
    manifest_path = Path(f"{output}.manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"BM25 index or manifest already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(corpus)
    if not {"doc_id", "question_body"}.issubset(parquet.schema_arrow.names):
        raise ValueError("Corpus must contain doc_id and question_body")

    descriptor, temp_name = tempfile.mkstemp(
        prefix=".bm25-", suffix=".sqlite", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temp_name)
    started = time.perf_counter()
    rows = 0
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(temporary)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE VIRTUAL TABLE documents USING "
            "fts5(body, content='', "
            "tokenize=\"unicode61 remove_diacritics 1 tokenchars '_'\")"
        )
        for batch in parquet.iter_batches(
            batch_size=4096, columns=["doc_id", "question_body"]
        ):
            doc_ids = batch.column("doc_id").to_pylist()
            bodies = batch.column("question_body").to_pylist()
            expected = list(range(rows, rows + len(doc_ids)))
            if doc_ids != expected:
                raise ValueError(
                    "Corpus doc_id must be contiguous and ordered from zero"
                )
            connection.executemany(
                "INSERT INTO documents(rowid, body) VALUES (?, ?)",
                (
                    (doc_id + 1, body or "")
                    for doc_id, body in zip(doc_ids, bodies, strict=True)
                ),
            )
            rows += len(doc_ids)
            connection.commit()
            if rows % 50_000 < len(doc_ids):
                print(
                    f"Indexed {rows:,}/{parquet.metadata.num_rows:,} documents",
                    flush=True,
                )
        if rows != parquet.metadata.num_rows:
            raise ValueError("Parquet row count changed during BM25 build")
        connection.execute("INSERT INTO documents(documents) VALUES('optimize')")
        connection.execute("INSERT INTO documents(documents) VALUES('integrity-check')")
        connection.commit()
        connection.close()
        connection = None
        os.replace(temporary, output)
        manifest = {
            "schema_version": 1,
            "backend": "sqlite_fts5",
            "index_file": output.name,
            "index_sha256": sha256_file(output),
            "index_size_bytes": output.stat().st_size,
            "corpus_sha256": sha256_file(corpus),
            "corpus_rows": rows,
            "doc_id_mapping": "fts5_rowid = doc_id + 1",
            "sqlite_version": sqlite3.sqlite_version,
            "config_sha256": sha256_file(config_path),
            "settings": settings,
            "build_seconds": round(time.perf_counter() - started, 3),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
    finally:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)


def open_index(
    index_path: str | Path,
    corpus_path: str | Path,
    config_path: str | Path,
) -> tuple[sqlite3.Connection, dict[str, Any]]:
    """Verify index bytes, corpus identity, settings, and SQLite version."""
    index = Path(index_path).expanduser().resolve()
    manifest = json.loads(Path(f"{index}.manifest.json").read_text(encoding="utf-8"))
    settings = load_settings(config_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("backend") != "sqlite_fts5"
        or manifest.get("index_file") != index.name
        or manifest.get("index_sha256") != sha256_file(index)
        or manifest.get("index_size_bytes") != index.stat().st_size
        or manifest.get("corpus_sha256") != sha256_file(corpus_path)
        or manifest.get("config_sha256") != sha256_file(config_path)
        or manifest.get("settings") != settings
        or manifest.get("sqlite_version") != sqlite3.sqlite_version
    ):
        raise ValueError(
            "BM25 index, corpus, config, or SQLite version differs from build manifest"
        )
    connection = sqlite3.connect(f"{index.as_uri()}?mode=ro", uri=True)
    try:
        count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if count != manifest["corpus_rows"]:
            raise ValueError("BM25 index row count differs from build manifest")
    except BaseException:
        connection.close()
        raise
    return connection, manifest


def search(
    connection: sqlite3.Connection, query: str, top_k: int
) -> list[tuple[int, float]]:
    """Return stable (doc_id, FTS5 BM25 rank) pairs, best rank first."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    expression = match_expression(query)
    if expression is None:
        return []
    rows = connection.execute(
        "SELECT rowid - 1, rank FROM documents "
        "WHERE documents MATCH ? ORDER BY rank ASC, rowid ASC LIMIT ?",
        (expression, top_k),
    ).fetchall()
    return [(int(doc_id), float(rank)) for doc_id, rank in rows]
