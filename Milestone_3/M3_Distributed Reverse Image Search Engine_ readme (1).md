# Distributed Reverse Image Search Engine

A distributed reverse image search engine that retrieves visually similar images from a large-scale dataset using parallel feature extraction, distributed LSH indexing, and approximate nearest neighbour search.

**Team:**
- Fatima Faisal — Traditional Feature Descriptors, Distributed Query Processing, Incremental Index Updates
- Raviha Ahmed — Image Preprocessing & Hashing, Index Sharding & Load Balancing, Inter-Node Communication Optimization
- Arham Altaf — Deep Learning Feature Extraction, Distributed LSH Index Construction, Parallel Re-ranking
- M. Zain Chinoy — Feature Fusion & Baseline Evaluation, Caching & Node Infrastructure, Benchmarking & Scaling Analysis

**GitHub:** https://github.com/FatimaMaskatia/Distributed-Reverse-Image-Search-Engine

---

## Dataset

A sample dataset is included in the repository containing a small number of images from each weather category for quick testing.

The full implementation was built and tested on the complete Weather Image Dataset (6,378 images across 11 categories: dew, fog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, snow) available on Kaggle:
https://www.kaggle.com/datasets/jehanbhathena/weather-dataset

---

## System Requirements

- Python 3.8 or higher
- RAM: 8 GB recommended
- GPU: Optional (CUDA used automatically if available for deep feature extraction)
- OS: Windows, Linux, or macOS

---

## End-to-End Pipeline

```
Query Image
    
Feature Extraction (Milestone 1)
  - Perceptual hashing (aHash, dHash, pHash)
  - ResNet-50 deep features (512-dim)
  - SIFT + ORB + colour histograms (1184-dim)
  - Fused vector: 1696-dim per image
    
Distributed LSH Index (Milestone 2)
  - Consistent hashing -> 4 nodes
  - 10 LSH hash tables per node
  - Parallel query fanout
  - LRU caching for hot queries
    
Performance Optimization (Milestone 3)
  - Incremental inserts (no rebuild)
  - Per-partition RW locking
  - Parallel re-ranking
  - Scalar quantization (4x compression)
    
Top-K Results
```

---


## Milestone 3 - Incremental Updates and Performance Optimization

**Goal:** Support real-time index updates without rebuilding, parallel re-ranking, inter-node communication compression, and large-scale benchmarking.

### Prerequisites

```
pip install numpy readerwriterlock scipy
```

---

### Module 1 — Incremental Index Updates & Concurrency 

Adds `insert_image()` to the live LSH index with per-partition read-write locking. New images can be inserted while queries run simultaneously — no index rebuild required.

**Install RW lock package:**
```
pip install readerwriterlock
```

**Run unit tests:**
```
cd Incremental_Index_Updates
python test_incremental.py
```

All 10 tests should pass, covering:
- Basic insert correctness
- Partition size tracking
- Error handling (wrong dimension, duplicate ID, insert before build)
- 20 concurrent threads inserting simultaneously
- Queries running correctly during live inserts
- Inserted image findable via query
- Partition lock isolation
- Latency tracking

**Run stress test (30 seconds, 10 readers + 3 writers):**

First open `stress_test.py` and update `DATA_PATH` to point to your `fused_features.npy`:
```python
DATA_PATH = r"path/to/fused_features.npy"
```

Then run:
```
python stress_test.py
```

This runs 10 reader threads and 3 writer threads simultaneously for 30 seconds against a live index of real data. Expected output: PASSED with 150 inserts and ~4999 queries completed with no errors.

**Stress Test Results:**
- Total queries: 4,999 | Total inserts: 150
- Query throughput: 166 queries/sec
- Insert throughput: 5 inserts/sec
- Index grew: 6,377 -> 6,527 images
- P50: 56ms | P95: 82ms | P99: 89ms
- Result: PASSED

---

### Module 2 - Parallel Re-ranking 

Re-ranks LSH candidate sets using exact cosine similarity computed in parallel via a thread pool. Includes automatic tuning to find the optimal candidate set size.

**Install:**
```
pip install numpy scipy
```

**Run unit tests:**
```
cd Parallel_Reranking/tests
python test_reranker.py
```

All 5 tests should pass.

**Run candidate set size tuning:**
```
cd Parallel_Reranking/src
python run_tuning.py \
  --features "path/to/fused_features.npy" \
  --num-queries 100 \
  --num-candidates 500 \
  --num-threads 4 \
  --metric cosine \
  --output-dir results
```

**Output:** `results/tuning_results.json` - latency per candidate set size

**Tuning Results:**

| Candidate Set Size | Per-Query Latency |
|---|---|
| 10 | 0.08 ms |
| 50 | 0.10 ms <- recommended |
| 100 | 0.14 ms |
| 500 | 0.86 ms |

Recommended setting: k=50 (good recall, 0.10ms latency).

---

### Module 3 - Inter-Node Communication Optimization

Implements scalar quantization to compress float32 feature vectors to uint8, reducing inter-node payload size by 4x with only 0.05% accuracy loss.

**Run compressor self-test:**
```
cd Inter_Node_Communication
python compressor.py
```

Expected output:
```
Original size:   4736 bytes (float32)
Compressed size: 1184 bytes (uint8)
Ratio:           4x smaller
Cosine sim:      0.9995 (1.0 = perfect)
```

**Run full profiling pipeline:**

Place these files in the `Inter_Node_Communication/` folder:
- `fused_features.npy` — from M1
- `master_alignment_map.csv` — from M1
- `consistent_hashing.py` — from M2
- `sharding.py` — from M2
- `lsh_index.py` — from M2
- `query_processor.py` — from M2

Then run:
```
python main.py
```

This runs all three profiling analyses and saves results to `final_report.txt`.

**Communication Optimization Results:**

| Metric | Value |
|---|---|
| Original payload | 4,736 bytes (float32) |
| Compressed payload | 1,184 bytes (uint8) |
| Compression ratio | 4x smaller |
| Transfer speedup | 3.61x faster (100 Mbps simulated) |
| Accuracy loss | 0.05% |
| Bottleneck identified | Payload size, not serialization format |

---

### Module 4 — Benchmarking & Scaling Analysis 

Benchmarks query latency (P50/P95/P99) at 100K, 1M, and 10M image scales using synthetic feature vectors, and produces strong/weak scaling plots.

### Run:
```
python generate_scaling_plots.py
```

Output:
```
Scaling plots generated and saved as PNG files!
```

This generates:

- Strong scaling plots
- Weak scaling plots
- Benchmark visualization graphs

All plots are saved as PNG images in the current folder.

## 2. Generate Synthetic Datasets

### Run:
```
python generate_synthetic_data.py
```

This generates large synthetic feature-vector datasets for benchmarking distributed indexing performance without requiring expensive CNN feature extraction.

Generated datasets:

- synthetic_100k.npy
- synthetic_1m.npy

Output example:
```
Successfully saved synthetic_100k.npy
Successfully saved synthetic_1m.npy
```

These datasets are used for:

- Scaling analysis
- Query latency benchmarking
- Throughput measurements
- Large-scale distributed indexing experiments

## 3. Run Pareto Frontier Evaluation

### Run:
```
python run_pareto_frontier.py
```

Important

If you get:

ModuleNotFoundError: No module named 'lsh_index'

copy lsh_index.py into this folder OR add the Milestone 2 folder to your Python path.

This script evaluates:

- Precision vs latency trade-offs
- Recall vs latency trade-offs
- Different LSH parameter configurations
- Number of hash tables
- Hash width
- Candidate size

The goal is to identify the optimal operating point between retrieval accuracy and query speed.

## 4. Run Scaling Benchmarks

### Run:

```
python run_scaling_benchmarks.py
```

Important

If you get:

ModuleNotFoundError: No module named 'lsh_index'

copy lsh_index.py into this folder OR add the Milestone 2 folder to your Python path.

PowerShell Note


What the Scaling Benchmarks Measure:

- Strong Scaling
- Fixed dataset size
- Increasing number of distributed nodes

Measures how query latency improves as more nodes are added.

Weak Scaling
- Dataset size increases proportionally with node count

- Measures whether the system maintains stable performance as workload scales.

Benchmark Metrics

The benchmarking pipeline records:

1. Query throughput
2. Query latency
3. P50
4. P95
5. P99
6. Index build time
7. Scaling efficiency
8. Accuracy vs latency trade-offs



Notes
Synthetic datasets are used to simulate large-scale deployments efficiently.
The distributed system is implemented using multi-process node simulation on a single machine.
Benchmarking focuses on distributed indexing scalability and low-latency query processing.

---

## Running Everything End to End

To reproduce all results from scratch:

**Step 1 — Download dataset:**
```
https://www.kaggle.com/datasets/jehanbhathena/weather-dataset
```

**Step 2 — Run M1 feature extraction:**
```
# Hashing
cd Milestone_1/Image_Preprocessing_and_Hashing
python main.py

# Deep features
cd Milestone_1/Deep_Learning_Feature_Extraction/src
python run_extraction.py --image-dir "path/to/dataset" --output-dir "outputs"

# Traditional features
cd Milestone_1/Traditional_Features_Descriptors
python Traditional_Features_Descriptors.py  # option 1

# Feature fusion
cd Milestone_1/Feature_Fusion_and_Baseline_Evaluation
python feature_fusion_and_evaluation.py
```

**Step 3 — Run M2 distributed index:**
```
cd Milestone_2/Distributed_LSH_Index_Construction/src
python run_index.py --features "path/to/fused_features.npy" --metadata "path/to/metadata.csv" --num-tables 10 --hash-width 32 --num-nodes 4 --num-threads 4

cd Milestone_2/Distributed_Query_Processing_and_Aggregation/src
python run_query.py --features "path/to/fused_features.npy" --metadata "path/to/metadata.csv" --query-name "fogsmog/4099.jpg" --top-k 5
```

**Step 4 — Run M3 optimizations:**
```
cd Milestone_3/Incremental_Index_Updates
python test_incremental.py
python stress_test.py

cd Milestone_3/Parallel_Reranking/src
python run_tuning.py --features "path/to/fused_features.npy" --num-queries 100 --num-candidates 500 --num-threads 4 --metric cosine --output-dir results

cd Milestone_3/Inter_Node_Communication
python compressor.py
python main.py
```

cd Milestone_3/Benchmarking_and_Scaling_Analysis
python generate_scaling_plots.py
python generate_sythetic_data.py
python run_pareto_frontier.py
python run_scaling_benchmarks.py
---


## References

1. Weather Image Dataset — Kaggle: https://www.kaggle.com/datasets/jehanbhathena/weather-dataset
2. PyTorch and torchvision: https://pytorch.org/
3. OpenCV Documentation: https://docs.opencv.org/
4. Python concurrent.futures: https://docs.python.org/3/library/concurrent.futures.html
5. Python multiprocessing: https://docs.python.org/3/library/multiprocessing.html
6. NumPy Documentation: https://numpy.org/doc/
7. readerwriterlock Python library: https://pypi.org/project/readerwriterlock/
8. Pillow (PIL): https://python-pillow.org
