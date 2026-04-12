
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
