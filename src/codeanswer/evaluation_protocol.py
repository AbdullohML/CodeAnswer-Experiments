"""Frozen, single-positive known-item evaluation inputs and provenance."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from codeanswer.provenance import sha256_file

SCHEMA_VERSION = 1
SPLITS = ("development", "calibration", "test")
FIELDS = (
    "query_id",
    "relevant_doc_id",
    "query",
    "family_id",
    "title_hash",
    "body_hash",
)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _text_hash(text: str) -> str:
    return hashlib.sha256(_normalized(text).encode("utf-8")).hexdigest()


def _selection_key(seed: int, question_id: int) -> int:
    payload = f"{seed}:{question_id}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _sample_corpus(
    corpus_path: Path, *, count: int, seed: int
) -> tuple[list[dict[str, Any]], int]:
    parquet = pq.ParquetFile(corpus_path)
    required = {"doc_id", "question_id", "title", "question_body"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Corpus is missing columns: {', '.join(sorted(missing))}")

    seen_question_ids: set[int] = set()
    heap: list[tuple[int, int, dict[str, Any]]] = []
    row_number = 0
    for batch in parquet.iter_batches(batch_size=4096, columns=sorted(required)):
        for row in batch.to_pylist():
            doc_id = int(row["doc_id"])
            question_id = int(row["question_id"])
            if doc_id != row_number:
                raise ValueError(
                    "Corpus doc_id must be contiguous and ordered from zero"
                )
            if question_id in seen_question_ids:
                raise ValueError(f"Duplicate question_id in corpus: {question_id}")
            seen_question_ids.add(question_id)
            row_number += 1
            title = str(row["title"] or "")
            if not _normalized(title):
                continue
            item = {
                "query_id": question_id,
                "relevant_doc_id": doc_id,
                "query": title,
                "title_hash": _text_hash(title),
                "body_hash": _text_hash(str(row["question_body"] or "")),
            }
            key = _selection_key(seed, question_id)
            entry = (-key, -question_id, item)
            if len(heap) < count:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)

    if len(heap) < count:
        raise ValueError(f"Only {len(heap)} nonempty title queries; requested {count}")
    selected = [item for _, _, item in sorted(heap, reverse=True)]
    return selected, row_number


def _group_families(
    rows: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], dict[str, int]]:
    """Keep exact normalized title/body duplicates in one split."""
    parent = list(range(len(rows)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    first_title: dict[str, int] = {}
    first_body: dict[str, int] = {}
    title_duplicates = 0
    body_duplicates = 0
    empty_body_hash = _text_hash("")
    for index, row in enumerate(rows):
        title_hash = row["title_hash"]
        body_hash = row["body_hash"]
        if title_hash in first_title:
            parent[root(index)] = root(first_title[title_hash])
            title_duplicates += 1
        else:
            first_title[title_hash] = index
        if body_hash != empty_body_hash:
            if body_hash in first_body:
                parent[root(index)] = root(first_body[body_hash])
                body_duplicates += 1
            else:
                first_body[body_hash] = index

    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(root(index), []).append(index)
    ordered = [
        [rows[index] for index in indices]
        for indices in sorted(groups.values(), key=lambda indices: indices[0])
    ]
    for group in ordered:
        family_id = min(row["query_id"] for row in group)
        for row in group:
            row["family_id"] = family_id
    return ordered, {
        "exact_duplicate_title_rows": title_duplicates,
        "exact_duplicate_body_rows": body_duplicates,
        "families": len(ordered),
    }


def freeze_known_item_splits(
    corpus_path: str | Path,
    output_dir: str | Path,
    *,
    query_count: int,
    seed: int,
    development_count: int,
    calibration_count: int,
) -> dict[str, Any]:
    """Write deterministic development, calibration and test query files."""
    if query_count < 3 or development_count < 1 or calibration_count < 1:
        raise ValueError("All three splits require at least one query")
    if development_count + calibration_count >= query_count:
        raise ValueError("Development and calibration counts leave no test queries")
    corpus = Path(corpus_path).expanduser().resolve()
    if not corpus.is_file():
        raise FileNotFoundError(f"Corpus not found: {corpus}")
    output = Path(output_dir).expanduser().resolve()
    paths = [output / f"{name}.csv" for name in SPLITS] + [output / "manifest.json"]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Frozen evaluation files already exist in {output}")

    selected, corpus_rows = _sample_corpus(corpus, count=query_count, seed=seed)
    groups, duplicate_counts = _group_families(selected)
    if len(groups) < 3:
        raise ValueError("Fewer than three distinct query families")
    assigned: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    current_split = 0
    targets = (development_count, calibration_count)
    for index, group in enumerate(groups):
        remaining_groups = len(groups) - index
        if current_split < 2 and (
            len(assigned[SPLITS[current_split]]) >= targets[current_split]
            or remaining_groups <= len(SPLITS) - current_split - 1
        ):
            current_split += 1
        assigned[SPLITS[current_split]].extend(group)
    if any(not assigned[name] for name in SPLITS):
        raise ValueError("Could not create three nonempty family-disjoint splits")

    output.mkdir(parents=True, exist_ok=True)
    split_info: dict[str, Any] = {}
    for name in SPLITS:
        path = output / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(assigned[name])
        split_info[name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "queries": len(assigned[name]),
            "families": len({row["family_id"] for row in assigned[name]}),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_type": "known_item_title_to_body_single_positive",
        "selection": "lowest_blake2b64(seed:question_id)",
        "family_rule": "exact normalized title or nonempty body",
        "seed": seed,
        "requested_queries": query_count,
        "requested_development": development_count,
        "requested_calibration": calibration_count,
        "corpus": {
            "file": corpus.name,
            "sha256": sha256_file(corpus),
            "rows": corpus_rows,
        },
        "duplicates_in_selected_queries": duplicate_counts,
        "query_corpus_overlap": query_count,
        "splits": split_info,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_frozen_split(
    manifest_path: str | Path, corpus_path: str | Path, *, split: str = "development"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify all split files and corpus bytes before loading one split."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported evaluation manifest version")
    corpus = Path(corpus_path).expanduser().resolve()
    if sha256_file(corpus) != manifest["corpus"]["sha256"]:
        raise ValueError("Corpus checksum does not match evaluation manifest")
    if pq.ParquetFile(corpus).metadata.num_rows != manifest["corpus"]["rows"]:
        raise ValueError("Corpus row count does not match evaluation manifest")

    all_ids: set[int] = set()
    all_families: set[int] = set()
    all_labels: set[int] = set()
    frames: dict[str, pd.DataFrame] = {}
    for name in SPLITS:
        info = manifest["splits"][name]
        if info["file"] != f"{name}.csv":
            raise ValueError(f"Invalid {name} split path")
        split_path = path.parent / info["file"]
        if sha256_file(split_path) != info["sha256"]:
            raise ValueError(f"Checksum mismatch for {name} split")
        frame = pd.read_csv(
            split_path,
            keep_default_na=False,
            dtype={
                "query_id": "int64",
                "relevant_doc_id": "int64",
                "family_id": "int64",
                "query": "string",
                "title_hash": "string",
                "body_hash": "string",
            },
        )
        if (
            set(frame.columns) != set(FIELDS)
            or len(frame) != info["queries"]
            or frame.empty
        ):
            raise ValueError(f"Invalid {name} split schema or count")
        ids = set(frame["query_id"].astype(int))
        families = set(frame["family_id"].astype(int))
        if len(ids) != len(frame) or ids & all_ids or families & all_families:
            raise ValueError(
                "Duplicate or overlapping query IDs/families across splits"
            )
        if frame["query"].isna().any() or (frame["query"].str.strip() == "").any():
            raise ValueError("Empty frozen query")
        labels = frame["relevant_doc_id"].astype(int)
        if ((labels < 0) | (labels >= manifest["corpus"]["rows"])).any():
            raise ValueError("Relevant doc_id outside frozen corpus")
        label_ids = set(labels)
        if len(label_ids) != len(frame) or label_ids & all_labels:
            raise ValueError("Duplicate relevant doc_id across frozen queries")
        if len(families) != info["families"]:
            raise ValueError("Family count mismatch")
        all_ids.update(ids)
        all_families.update(families)
        all_labels.update(label_ids)
        frames[name] = frame
    if len(all_ids) != manifest["requested_queries"]:
        raise ValueError("Total query count mismatch")
    return frames[split], manifest


def verify_flat_artifacts(
    corpus_path: str | Path,
    index_path: str | Path,
    index: Any,
    *,
    expected_model: str,
    expected_sequence_length: int,
) -> dict[str, Any]:
    """Check legacy manifests and all exact index vectors against embeddings."""
    corpus = Path(corpus_path).expanduser().resolve()
    index_file = Path(index_path).expanduser().resolve()
    index_manifest_path = Path(f"{index_file}.manifest.json")
    if not index_manifest_path.is_file():
        raise FileNotFoundError(f"Index manifest is required: {index_manifest_path}")
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    if (
        index_manifest.get("index_type") != "IndexFlatIP"
        or index_manifest.get("metric") != "inner_product"
        or index_manifest.get("objects") != index.ntotal
        or index_manifest.get("dimension") != index.d
        or Path(index_manifest.get("index_path", "")).resolve() != index_file
    ):
        raise ValueError("Index manifest does not match the loaded exact index")
    embedding = Path(index_manifest["embedding_path"]).expanduser().resolve()
    embedding_manifest_path = Path(f"{embedding}.manifest.json")
    if not embedding_manifest_path.is_file():
        raise FileNotFoundError(
            f"Embedding manifest is required: {embedding_manifest_path}"
        )
    embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    if (
        Path(embedding_manifest.get("corpus_path", "")).resolve() != corpus
        or Path(embedding_manifest.get("output_path", "")).resolve() != embedding
        or embedding_manifest.get("objects") != index.ntotal
        or embedding_manifest.get("dimension") != index.d
        or embedding_manifest.get("model_name") != expected_model
        or embedding_manifest.get("max_sequence_length") != expected_sequence_length
    ):
        raise ValueError("Embedding manifest does not match corpus, index, or encoder")
    corpus_hash = sha256_file(corpus)
    recorded_hash = embedding_manifest.get("corpus_sha256")
    if recorded_hash is not None and recorded_hash != corpus_hash:
        raise ValueError("Embedding corpus checksum does not match current corpus")
    expected_bytes = index.ntotal * index.d * np.dtype(np.float32).itemsize
    if embedding.stat().st_size != expected_bytes:
        raise ValueError("Embedding byte size does not match exact index")
    vectors = np.memmap(
        embedding, dtype=np.float32, mode="r", shape=(index.ntotal, index.d)
    )
    for start in range(0, index.ntotal, 10_000):
        end = min(start + 10_000, index.ntotal)
        if not np.array_equal(
            index.reconstruct_n(start, end - start), vectors[start:end]
        ):
            raise ValueError(f"Index vectors differ from embeddings at row {start}")
    del vectors
    return {
        "index_sha256": sha256_file(index_file),
        "embedding_sha256": sha256_file(embedding),
        "corpus_sha256": corpus_hash,
        "index_embedding_match": "all_vectors",
        "embedding_corpus_match": "sha256" if recorded_hash else "legacy_path_only",
        "model_revision": embedding_manifest.get("model_revision", "unknown"),
    }
