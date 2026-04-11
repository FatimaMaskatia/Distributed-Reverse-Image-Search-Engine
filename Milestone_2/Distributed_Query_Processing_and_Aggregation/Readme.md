## Milestone 2 - Distributed Query Processing & Aggregation
This module implements the Milestone 2 distributed query processing deliverable:

* Distributed query fanout: a query image is hashed and dispatched to all relevant index partitions simultaneously using inter-process communication (multiprocessing queues) or in-process simulation.
* Result aggregation: candidate matches from all partitions are merged and de-duplicated.
* Exact re-ranking: candidates are re-ranked by exact cosine similarity using the full feature matrix from Milestone 1.
* Latency optimisation: synchronisation barriers are minimised; dispatch to all nodes happens before any result is awaited.

---

### Setup
bash

pip install numpy


No other dependencies required beyond what Milestone 1 already uses.

---

### How to Run

Run tests (no dataset needed)

bash

python tests/test_query.py


### Single query with real Milestone 1 data
bash

python src/run_query.py \
  --features "../PDC Project/Milestone_1/Traditional Features Descriptors/features_output/features.npy" \
  --metadata "../PDC Project/Milestone_1/Traditional Features Descriptors/features_output/image_paths.npy" \
  --query-id 0 \
  --top-k 10

  ### Random query (quick testing)
  python src/run_query.py \
  --features "../PDC Project/Milestone_1/Traditional Features Descriptors/features_output/features.npy" \
  --metadata "../PDC Project/Milestone_1/Traditional Features Descriptors/features_output/image_paths.npy" \
  --random-query \
  --top-k 10

  ### Latency benchmark (P50/P95/P99)
  python src/run_query.py \
  --features "../PDC Project/Milestone_1/Traditional Features Descriptors/features_output/features.npy" \
  --metadata "../PDC Project/Milestone_1/Traditional Features Descriptors/features_output/image_paths.npy" \
  --benchmark \
  --num-queries 50

---
  ### Integration
This module sits on top of the distributed LSH index and sharding modules from Milestone 2. It uses the LSH index for bucket lookup during fanout, the sharding module for authoritative node-to-image assignments, and the Milestone 1 feature matrix for exact cosine similarity re-ranking. When the sharding module is available it is used as the primary partition source; otherwise it falls back to the LSH index's internal node partitions.
