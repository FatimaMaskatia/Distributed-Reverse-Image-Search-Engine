## Milestone 3 — Incremental Updates & Concurrency Testing
### run
```
pip install readerwriterlock
```

- lsh_index.py is the file from milestone 2 with few modificaions: insert_image() added and locks upgraded

---

### run the files:

### Run:
```
python stress_test.py
```

In stress_test.py:
update DATA_PATH = r"C:\Users\maska\OneDrive\Desktop\PDC Project\..." to wherever your fused_features.npy is.

Stress test: concurrent queries (readers) + incremental inserts (writers)
running simultaneously against a live LSH index.

What this validates?

1. System remains stable and returns consistent results under concurrent load.
2. No crashes or deadlocks under sustained concurrent read/write load.
3. Every inserted image is present in the partition maps after the test.
4. Measures query throughput, insert throughput, and latency (P50/P95/P99).

---

### Run:

```
python test_incremental.py
```

This file is for unit tests for incremental index updates.
Test include:
1.  test_insert_image_basic           — inserted image appears in partition maps
2.  test_insert_updates_partition     — partition size increases by exactly 1
3.  test_insert_wrong_dim_raises      — wrong vector dim raises ValueError
4.  test_insert_duplicate_id_raises   — duplicate image_id raises ValueError
5.  test_insert_before_build_raises   — insert before build raises RuntimeError
6.  test_concurrent_insert_no_crash   — 20 threads inserting simultaneously
7.  test_query_during_insert          — queries correct while inserts run
8.  test_inserted_image_is_findable   — inserted image found with its own vector
9.  test_partition_lock_isolation     — inserts on different partitions don't block
10. test_latency_tracking             — query latencies recorded correctly
---



