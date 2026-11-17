# B4 — Build and runtime resource costs

**Status: DONE.** This profile measures work outside the search-only boundary
used in B1–B3. It reuses their full 520,000-vector artifacts and profiles a
separate, named **first-20,000-vector subset**. The full systems are exact
Flat-384, exact Flat-128 from B1's PCA basis, and the selected HNSW-128 from
B2/B3 (`M=16`, `efConstruction=100`, `efSearch=128`). No new relevance scores
are claimed. The [configuration](../configs/resource_profile.json),
[resource table](../results/evaluation/b4/resource_table.csv),
[run manifest](../results/evaluation/b4/run_manifest.json), and phase JSON files
under `results/evaluation/b4/` retain the settings, commands, hashes,
hardware, timings, and resource measurements.

## Measurement boundaries

- Each PCA, build, and runtime profile ran in a fresh Python process, in
  sequence, with four BLAS/FAISS threads. Peak process RSS is the sampled peak
  or Linux `ru_maxrss`, whichever is higher. It includes Python, libraries,
  source mapping, temporary arrays, and the index where applicable.
- PCA fit used randomized SVD with seed `20260714`, 256 components, 100,000
  sampled rows for full or 10,000 for subset. Transform times stream all rows
  through 10,000-row batches and hash the output, without writing it. This is
  an independent refit for resource measurement, not a claim to reproduce
  B1's saved PCA bytes.
- Index build times cover adding all vectors to an in-memory FAISS index. Full
  file sizes are from existing verified artifacts. The subset indexes were
  newly serialized to ignored `artifacts/b4/`. Serialization is **outside**
  the reported add/build seconds.
- Runtime first and second loads are two `faiss.read_index` calls in one fresh
  process, with the first index object released before the second. The first
  call is **process-first**, not guaranteed cold disk I/O: OS page cache was
  not cleared. A first query precedes three warmup queries; then the same
  first 100 frozen test titles run twice, sequentially. Each timed query does
  CPU MiniLM encoding, optional PCA, FAISS top-40 search, then the production
  Parquet filter, rank-order restore, and text cleaning for 40 corpus rows.
  Reranking, answerability, and generation are excluded.

## PCA and index construction

| Scope | Phase | Rows | Seconds | Peak RSS MiB | Artifact MiB |
|---|---|---:|---:|---:|---:|
| 20k subset | PCA fit 256 | 20,000 | 0.245 | 340.16 | — |
| 20k subset | PCA transform 256 / 128 | 20,000 | 0.046 / 0.030 | same process | no file |
| Full | PCA fit 256 | 520,000 | 3.321 | 1,684.31 | — |
| Full | PCA transform 256 / 128 | 520,000 | 1.088 / 0.700 | same process | no file |
| 20k subset | Flat-384 add | 20,000 | 0.022 | 276.18 | 29.30 |
| 20k subset | Flat-128 add | 20,000 | 0.010 | 237.00 | 9.77 |
| 20k subset | HNSW-128 add | 20,000 | 0.672 | 242.04 | 12.52 |
| Full | Flat-384 add | 520,000 | 0.827 | 1,740.87 | 761.72 |
| Full | Flat-128 add | 520,000 | 0.242 | 724.99 | 253.91 |
| Full | HNSW-128 add | 520,000 | 64.437 | 819.80 | 325.45 |

The 20k subset is an infrastructure smoke measurement, not a retrieval
benchmark or a scaled estimate of the full build. In particular, HNSW graph
construction does not scale linearly. The full HNSW add was much slower than
Flat-128 add and its index file was larger because graph links consume space.
The previous B2 HNSW build time was 67.111 seconds; the new 64.437-second
measurement is a separate run, not an exact replay.

## Full-artifact loading and warmed queries

| Full system | First / second load s | Peak runtime RSS MiB | Search p50 / p95 ms | Fetch 40 p50 / p95 ms | To fetched rows p50 / p95 ms |
|---|---:|---:|---:|---:|---:|
| Flat-384 | 0.434 / 0.427 | 4,314.00 | 53.762 / 62.643 | 522.803 / 596.498 | 584.628 / 654.697 |
| Flat-128 | 0.235 / 0.193 | 3,951.97 | 15.432 / 17.284 | 513.323 / 561.447 | 538.992 / 586.654 |
| HNSW-128 | 0.342 / 0.259 | 3,926.91 | 0.702 / 0.933 | 520.210 / 574.776 | 530.963 / 586.751 |

These latency columns are from the **first warmed round**. The second round
gave search p95 63.587 / 17.451 / 0.950 ms and fetch p95
596.693 / 570.999 / 573.713 ms for Flat-384 / Flat-128 / HNSW-128.
The per-phase JSON files retain both rounds and the first-query values.
The first HNSW query took 1.647 ms search and 678.058 ms fetch; its initial
encoding took 27.194 ms. Warm HNSW encoding p95 was 10.865 ms in round one.
The p95 of a total is computed per query, rather than by adding stage p95s.

On this machine, fetching 40 rows dominates the measured path, so replacing
exact search with HNSW has a much smaller effect on **time to fetched rows**
than on index search alone. This is a measurement of the existing Parquet
access pattern, not the complete RAG response latency. It points to corpus
access as a separate optimization target. The roughly 3.93 GiB HNSW runtime
peak includes the encoder, FAISS index, corpus access, and process allocations;
its 325.45 MiB index file alone does not predict peak process memory.

## Hardware and GPU boundary

The phase manifests record Linux 7.0.0, Intel Core i5-11400H, 12 logical
CPUs, 14.901 GiB RAM, Python 3.13.9, FAISS 1.14.3, NumPy 2.3.5,
scikit-learn 1.8.0, PyArrow 25.0.0, Sentence Transformers 5.6.0, and
PyTorch 2.11.0. The NVIDIA driver was unavailable (`nvidia-smi` exit 9).
All profiled phases used CPU, and GPU peak VRAM is recorded as **unavailable**
(`null`), not zero. Machine load, OS cache state, and per-query corpus layout
can change latency; these are one-run resource observations, not capacity
guarantees.

## Reproduce

With the full corpus, embeddings, B1 PCA, and B2 indexes present locally:

```bash
python scripts/run_b4_profile.py
```

This runs eleven fresh processes sequentially and writes the phase files and
summary table. To measure one phase, use for example:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/profile_resources.py runtime --system hnsw128
```
