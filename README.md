
## Distributed Reverse Image Search Engine
This project implements a Reverse Image Search Engine that retrieves visually similar images from a large-scale dataset using parallel feature extraction, similarity matching and 
approximate nearest neighbour searchusing feature extraction.

### Dataset
A sample dataset is included in the repository containing a small number of images from each weather category for quick testing.
The full implementation was built and tested on the complete Weather Image Dataset (6,378 images) available on Kaggle: https://www.kaggle.com/datasets/jehanbhathena/weather-dataset

### System Requirements:
Python (3.8 or higher)
RAM (8 GB recommended)
GPU (Optional - CUDA for 3rd module)
OS (Windows, Linux or macOS)

---

## End-to-End Pipeline

```
Query Image
    
1. Feature Extraction (Milestone 1)
  - Perceptual hashing (aHash, dHash, pHash)
  - ResNet-50 deep features (512-dim)
  - SIFT + ORB + colour histograms (1184-dim)
  - Fused vector: 1696-dim per image
    
2. Distributed LSH Index (Milestone 2)
  - Consistent hashing -> 4 nodes
  - 10 LSH hash tables per node
  - Parallel query fanout
  - LRU caching for hot queries
    
3. Performance Optimization (Milestone 3)
  - Incremental inserts (no rebuild)
  - Per-partition RW locking
  - Parallel re-ranking
  - Scalar quantization (4x compression)
    
Top-K Results
```

---

### Milestone 1 - Parallel Feature Extraction Pipeline 
### Module 1 - Image Preprocessing & Hashing
Loads all images using a multi-threaded pipeline and computes three perceptual hashes per image.
#### Install
```bash
pip install pillow imagehash numpy opencv-python tqdm
```

### Run
```bash
python main.py
```
Output saved to: `hash_results.json`

---

### Module 2 - Deep Learning Feature Extraction
Extracts 2048-dimensional CNN feature vectors using a pre-trained ResNet-50 model.
### Install
pip install torch torchvision pillow numpy

Or

use the requirements file:
pip install -r requirements.txt

### Run
python src/run_extraction.py \
  --image-dir "dataset/Weather_Dataset" \
  --output-dir "outputs" \
  --batch-size 32 \
  --num-workers 2 \
  --model-name resnet50

### Output
outputs/features.npy — float32 matrix of shape [N, 2048] L2-normalized
outputs/metadata.csv — maps each row index to its image path

Notes:
GPU (CUDA) is used automatically if available
Falls back to CPU if no GPU is present
Supported formats: JPG, JPEG, PNG, BMP, WEBP

### Module 3 - Traditional Feature Descriptors
Extracts SIFT, ORB, and color histogram features in parallel using a thread pool.

### Install
pip install opencv-contrib-python numpy tqdm

- Must use opencv-contrib-python not opencv-python — SIFT requires the contrib package.

### Run
python Traditional_Features_Descriptors.py

Select option:
1 -> Build feature database (run this first)
2 -> Search similar images by entering a query image path

### Output
features_output/features.npy - float32 matrix of shape [N, 1184]
features_output/image_paths.npy - corresponding image paths

## Module 4 — Feature Fusion & Baseline Evaluation
Aligns outputs from all three modules, fuses feature vectors, and evaluates retrieval accuracy.

Requirements
Google Colab (recommended) or Jupyter Notebook
All output files from Modules 1, 2, and 3

### Setup
Upload these files to Google Drive under PDC_Project_M1/:
Files:
Source hash_results.json (Module 1 output)
features.npy + image_paths.npy (Module 2 output)
features.npy + metadata.csv (Module 3 output)

### Run
Open notebook in Google Colab
Mount Google Drive when prompted
Run all cells in order

### Output
fused_features.npy: final fused matrix of shape [N, 1184]
master_alignment_map.csv: maps each image to its index in all three feature arrays

If don't want to run on google colab, remove the Google Drive parts and use local paths instead.
### Remove these lines:
from google.colab import drive
drive.mount('/content/drive')
BASE_PATH = "/content/drive/MyDrive/PDC_Project_M1/"
### Replace with:
BASE_PATH = "./"  # or where your files are

### Install the required libraries:
pip install numpy pandas scikit-learn

---

### Milestone 2 - Distributed LSH Indexing and Search 

Milestone 2 builds on the parallel feature extraction pipeline from Milestone 1 and implements the distributed indexing and search infrastructure. The system distributes image feature vectors across 4 worker nodes using consistent hashing, builds a distributed LSH index for approximate nearest neighbour search, processes queries concurrently across all nodes, and caches hot query results to reduce redundant index lookups. The entire system runs as a multi-process simulation on a single machine.

---

### Prerequisites

- Python 3.8 or above
- Install the only required dependency:

```bash
pip install numpy
```


```bash
pip install pandas matplotlib
```

No GPU is required. OpenCV and PyTorch are only needed if re-running the Milestone 1 feature extraction pipeline.

## Running All Tests (No Dataset Required)

All tests except Zain's files generate their own synthetic feature vectors internally. No Milestone 1 dataset files are needed to run them.

```bash
# Module 1 - Index sharding and load balancing tests
cd Index_Sharding_and_Load_Balancing
python test_hashing.py
python test_sharding.py
python test_monitor.py
python test_rebalancer.py

# Module 2 - LSH index construction tests
cd Distributed_LSH_Index_Construction
python tests/test_lsh.py

# Module 3 — distributed query processing tests
cd Distributed_Query_Processing_and_Aggregation
python tests/test_query.py
```

All tests should pass with no failures.

---

## Running with Real Milestone 1 Data

To run the full pipeline with real data you need the Milestone 1 output files. These are produced by running the Milestone 1 feature extraction pipeline on the Weather Image Dataset from Kaggle (https://www.kaggle.com/datasets/jehanbhathena/weather-dataset).

The files you need are:

- `fused_features.npy` — produced by the Feature Fusion module (shape 6378 x 1184)
- `features.npy` — produced by the Deep Learning Feature Extraction module (shape 6378 x 512)
- `metadata.csv` — produced by the Deep Learning Feature Extraction module
- `image_paths.npy` — produced by the Traditional Features Descriptors module

Replace `<path>` in all commands below with the actual path to these files on your machine.

---
### 1. Index Sharding Evaluation

Open `evaluation.py` and update the `np.load()` path at the top of the file to point to your `fused_features.npy`. Then run:

```bash
cd Index_Sharding_and_Load_Balancing
python evaluation.py
```

---

### 2. Build the LSH Index

```bash
cd Distributed_LSH_Index_Construction
python src/run_index.py \
  --features "<path to features.npy from Deep Learning Feature Extraction>" \
  --metadata "<path to metadata.csv from Deep Learning Feature Extraction>" \
  --output-dir "outputs" \
  --num-tables 10 \
  --hash-width 32 \
  --num-nodes 4 \
  --num-threads 4
```

Output is saved to `Distributed_LSH_Index_Construction/outputs/index_summary.txt`.

---

### 3. Run Queries

**Single query** (query image at index 0, return top 10 results):

```bash
cd Distributed_Query_Processing_and_Aggregation
python src/run_query.py \
  --features "<path to features.npy>" \
  --metadata "<path to image_paths.npy>" \
  --query-id 0 \
  --top-k 10
```

**Random query** (useful for quick testing):

```bash
python src/run_query.py \
  --features "<path to features.npy>" \
  --metadata "<path to image_paths.npy>" \
  --random-query \
  --top-k 10
```

**Latency benchmark** (runs 50 random queries and reports P50/P95/P99):

```bash
python src/run_query.py \
  --features "<path to features.npy>" \
  --metadata "<path to image_paths.npy>" \
  --benchmark \
  --num-queries 50
```

Benchmark results are saved to `Distributed_Query_Processing_and_Aggregation/outputs/benchmark_results.txt`.

**Multiprocess mode** (spawns real worker processes with OS-level queue communication):

Add `--multiprocess` to any command above. Results are identical to default in-process mode; only the execution model differs.

```bash
python src/run_query.py \
  --features "<path to features.npy>" \
  --metadata "<path to image_paths.npy>" \
  --query-id 0 \
  --multiprocess
```

---

### 4. Caching and Evaluation

Both files require the Milestone 1 `fused_features.npy` file. Before running either file, open it and update the `BASE_PATH` variable at the top to point to your local Milestone 1 output folder.

**Run the cache demonstration** (shows cold start vs hot query latency):

```bash
cd Caching_and_Node_Infrastructure
python query_cache.py
```

**Run the accuracy vs latency evaluation** (benchmarks 4 LSH configurations across 50 queries and produces a trade-off graph):

```bash
python evaluate_lsh_tradeoff.py
```

The evaluation graph is saved as `lsh_tradeoff_plot.png` in your Milestone 1 output folder.

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

Generate Synthetic Datasets

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

Run Pareto Frontier Evaluation

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

Run Scaling Benchmarks

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

cd Milestone_3/Benchmarking_and_Scaling_Analysis
python generate_scaling_plots.py
python generate_sythetic_data.py
python run_pareto_frontier.py
python run_scaling_benchmarks.py
```
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


