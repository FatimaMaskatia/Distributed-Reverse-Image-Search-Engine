## Milestone 3 — Parallel Re-ranking

This module implements parallel reranking Milestone 3 deliverable:
- Parallel re-ranking of LSH candidate sets using exact similarity computation.
- Thread-pool-based parallelization for high throughput.
- Vectorized numpy operations for optimal performance.
- Automatic tuning to find optimal candidate set size tradeoff.
- Latency and accuracy benchmarking across dataset scales.

## Setup

```bash
pip install -r requirements.txt
```

## Run Tuning Evaluation

```bash
python src/run_tuning.py \
  --features "../arham_m1/outputs/features.npy" \
  --num-queries 100 \
  --num-candidates 500 \
  --num-threads 4 \
  --metric cosine \
  --output-dir results
```

## Run Tests

```bash
python tests/test_reranker.py
```

## Pipeline Flow

```text
LSH Candidates (from Fatima's aggregation)
        |
        v
Feature Vectors (from M1)
        |
        v
Parallel Re-ranking
  - Thread pool dispatch
  - Exact similarity computation (cosine/L2)
  - Vectorized numpy batch operations
        |
        v
Re-ranked Results [(image_id, score), ...]
        |
        v
Candidate Set Size Tuning
  - Test k = 10, 25, 50, 100, 250, 500
  - Measure latency per query
  - Identify optimal accuracy-vs-latency point
        |
        v
results/tuning_results.json
```

## Key Classes

- `ParallelReRanker`: Main re-ranking engine with thread-pool parallelization.
- `CandidateSetTuner`: Automated tuning framework for candidate set sizes.

## Metrics & Output

- **Latency (total/per-query)**: Time to re-rank all candidates across all queries.
- **Recall@k**: Percentage of relevant images found in re-ranked results.
- **Accuracy-Latency Tradeoff**: JSON report with metrics per candidate set size.

## Notes

- Vectors are normalized if using cosine similarity (automatic).
- Thread pool size is configurable (default 4).
- Supports both synthetic (for scalability testing) and real candidate sets.
- Ready for integration with Fatima's query aggregation and Zain's benchmarking.
