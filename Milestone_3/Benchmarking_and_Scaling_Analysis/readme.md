### Milestone 3 — Benchmarking & Scaling Analysis 

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
