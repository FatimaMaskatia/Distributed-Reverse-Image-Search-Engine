# Milestone 2 - Distributed LSH Index Construction

This module implements Arham's Milestone 2 deliverable:
- LSH index with multiple hash tables using random projections.
- Distributed partitioning across worker nodes (via Raviha's consistent hashing).
- Parallel index construction with thread pools.
- Approximate nearest neighbor (ANN) query interface.
- Integration with M1 deep feature outputs.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/run_index.py \
  --features "../arham_m1/outputs/features.npy" \
  --metadata "../arham_m1/outputs/metadata.csv" \
  --output-dir "outputs" \
  --num-tables 10 \
  --hash-width 32 \
  --num-nodes 4 \
  --num-threads 4
```

## Test

```bash
python tests/test_lsh.py
```

## Pipeline Flow

```text
M1 Features (features.npy + metadata.csv)
        |
        v
Load features and metadata
        |
        v
Initialize LSH tables (k tables, each with b-bit projections)
        |
        v
Distribute images across nodes (Raviha's consistent hashing)
        |
        v
Parallel insertion into LSH tables (thread pool)
        |
        v
Distributed LSH Index (in-memory, queryable per node)
        |
        v
Query interface for ANN search
```

## Key Parameters

- `--num-tables`: Number of independent hash tables (higher = better recall, more space).
- `--hash-width`: Bits per table (higher = finer granularity).
- `--num-nodes`: Number of worker nodes (from Raviha's sharding).
- `--num-threads`: Threads for parallel insertion.

## Output

- `outputs/index_summary.txt`: Metadata about the constructed index (partition counts, table info).
- In-memory LSH structure with query capability.

