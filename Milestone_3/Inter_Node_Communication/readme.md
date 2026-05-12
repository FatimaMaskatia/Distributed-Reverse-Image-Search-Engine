# Milestone 3: Inter-Node Communication Optimization

---

## Overview

When a query comes in, the system sends the query's feature vector to all 4 nodes so each node can search its partition. Each vector is 1184 numbers stored as float32 — that's 4736 bytes per query dispatch.

The solution is scalar quantization: converting each number from float32 (4 bytes) to uint8 (1 byte), making the vector **4x smaller** before sending. The accuracy loss is only 0.05%, which is negligible for similarity search.

We also profiled the full serialization pipeline to identify where the real bottleneck is — it turns out to be payload size, not serialization format — and measured how batching multiple queries together improves throughput.

---

## Files

| File | What it does |
|---|---|
| `compressor.py` | `ScalarQuantizer` class — compress and decompress feature vectors |
| `profiler.py` | `CommunicationProfiler` class — measures serialization time and payload size |
| `main.py` | Runs everything end to end and saves `final_report.txt` |

**Also required in the same folder:**

M2 files: `consistent_hashing.py`, `sharding.py`, `lsh_index.py`, `query_processor.py`

M1 data files: `fused_features.npy`, `master_alignment_map.csv`

---

## How to Run

```bash
python main.py
```

This will:
1. Load the real feature vectors from M1
2. Show compression stats
3. Build the full M2 distributed index (~30 seconds)
4. Run a sample query
5. Run all three profiling analyses
6. Save results to `final_report.txt`

To quickly verify the compressor works without any other files:

```bash
python compressor.py
```

---

## Results (real data — 7128 images, 1184-dim vectors)

| Metric | Value |
|---|---|
| Vector size before compression | 4736 bytes (float32) |
| Vector size after compression | 1184 bytes (uint8) |
| Compression ratio | **4x smaller** |
| Accuracy loss | **0.05%** (negligible) |
| Transfer time improvement | **~3.6x faster** (simulated 100 Mbps network) |
| Sample query latency | 5.073 ms (single query, 7128 images, 4 nodes) |

The ~3.6x speedup matches the 4x size reduction closely, confirming that payload size is the bottleneck, not CPU serialization time.

> **Note on network simulation:** We are running on a single machine, so there is no real network between nodes. The profiler simulates transfer time based on payload size and a 100 Mbps link speed. The compression ratio (4x) and accuracy loss (0.05%) are real measurements from real data. On an actual cluster, the speedup would match the size reduction directly.

---

## Understanding the Output

- `image_id=0` gets similarity `1.0000` — it is querying itself, which is correct behaviour.
- Other results have similarity ~0.80–0.82 — these are the genuine nearest neighbours.
- Node load is slightly uneven (27.3%, 23.8%, 24.5%, 24.5%) — this is expected with consistent hashing, not a bug.

---

## Note on `hash_width`

The LSH index uses `hash_width=16` instead of 32. With only 7128 images, `hash_width=32` creates 2³² possible buckets — almost every image ends up alone in its own bucket, which means queries return almost no candidates. `hash_width=16` gives ~500–950 non-empty buckets per table with ~7–14 images each, which is the correct tradeoff for this dataset size. This is set correctly in `main.py` and does not need to be changed.

---

## Integration with Benchmarking (M3)

### Using the compressor

```python
from compressor import ScalarQuantizer
import numpy as np

features = np.load("fused_features.npy", allow_pickle=False).astype(np.float32)

# Fit once on the full dataset
quantizer = ScalarQuantizer()
quantizer.fit(features)

# Compress before sending to a node, decompress on arrival
compressed = quantizer.compress(features[0])   # float32 -> uint8
restored   = quantizer.decompress(compressed)  # uint8 -> float32

# For batches
compressed_batch = quantizer.compress_batch(features[:50])
restored_batch   = quantizer.decompress_batch(compressed_batch)
```

### Using the profiler

```python
from profiler import CommunicationProfiler

profiler = CommunicationProfiler(compressor=quantizer)

profiler.profile_serialization(features[:50], num_trials=100)
profiler.profile_batch_throughput(features[:100], batch_sizes=[1, 5, 10, 20, 50])
profiler.profile_pickle_vs_numpy(features[:50])
profiler.save_report("my_report.txt")
```

Run the benchmark twice — once without compression and once with — and report both sets of latency numbers to show the improvement.