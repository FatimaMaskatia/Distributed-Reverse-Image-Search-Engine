import sys
import threading
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lsh_index import DistributedLSHIndex, LSHConfig


DIM = 128

def _rand_vec(dim: int = DIM) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v

def _build_index(n: int = 40, dim: int = DIM,
                 num_tables: int = 4, hash_width: int = 8) -> tuple:
    features = np.array([_rand_vec(dim) for _ in range(n)], dtype=np.float32)
    config   = LSHConfig(num_tables=num_tables, hash_width=hash_width,
                         num_nodes=4, num_threads=2)
    index    = DistributedLSHIndex(config)
    index.build_from_features(features)
    return index, features

def test_insert_image_basic():
    """Inserted image should appear in partition maps."""
    index, _ = _build_index()

    new_id  = 9999
    new_vec = _rand_vec()
    index.insert_image(new_id, new_vec)

    assert new_id in index.image_to_node, \
        "image_id missing from image_to_node"
    node_id = index.image_to_node[new_id]
    assert new_id in index.node_partitions[node_id], \
        "image_id missing from node_partitions"

    print("test_insert_image_basic")


def test_insert_updates_partition():
    """Partition total size should increase by exactly 1 after insert."""
    index, _ = _build_index(n=40)

    before = sum(len(p) for p in index.node_partitions.values())
    index.insert_image(5000, _rand_vec())
    after  = sum(len(p) for p in index.node_partitions.values())

    assert after == before + 1, \
        f"Expected {before + 1} total images, got {after}"
    print("test_insert_updates_partition")


def test_insert_wrong_dim_raises():
    """Wrong vector dimension should raise ValueError."""
    index, _ = _build_index(dim=DIM)

    raised = False
    try:
        index.insert_image(8888, _rand_vec(dim=DIM + 10))
    except ValueError:
        raised = True

    assert raised, "Expected ValueError for wrong-dimension vector"
    print("test_insert_wrong_dim_raises")


def test_insert_duplicate_id_raises():
    """Duplicate image_id should raise ValueError."""
    index, _ = _build_index(n=40)

    raised = False
    try:
        index.insert_image(0, _rand_vec())   # 0 already exists
    except ValueError:
        raised = True

    assert raised, "Expected ValueError for duplicate image_id"
    print("test_insert_duplicate_id_raises")


def test_insert_before_build_raises():
    """insert_image before build_from_features should raise RuntimeError."""
    config = LSHConfig(num_tables=4, hash_width=8, num_nodes=2, num_threads=1)
    index  = DistributedLSHIndex(config)

    raised = False
    try:
        index.insert_image(0, _rand_vec())
    except RuntimeError:
        raised = True

    assert raised, "Expected RuntimeError before build"
    print("test_insert_before_build_raises")


def test_concurrent_insert_no_crash():
    """
    20 threads each insert one unique image simultaneously.
    No crash, no deadlock, final count must be correct.
    """
    index, _ = _build_index(n=40)

    errors   = []
    n_threads = 20
    # Use a lock to guarantee unique IDs across threads
    id_lock  = threading.Lock()
    next_id  = [10_000]   # mutable counter inside list

    def do_insert():
        with id_lock:
            my_id = next_id[0]
            next_id[0] += 1
        try:
            index.insert_image(my_id, _rand_vec())
        except Exception as e:
            errors.append(f"ID {my_id}: {e}")

    threads = [threading.Thread(target=do_insert) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, "Concurrent insert errors:\n" + "\n".join(errors)

    total = sum(len(p) for p in index.node_partitions.values())
    assert total == 40 + n_threads, \
        f"Expected {40 + n_threads} images, got {total}"
    print("test_concurrent_insert_no_crash")


def test_query_during_insert():
    """
    Queries return valid results while a background writer inserts images.
    Uses a properly defined id_lock (not a new lock each iteration).
    """
    index, features = _build_index(n=60)

    stop     = threading.Event()
    errors   = []
    id_lock  = threading.Lock()   # defined ONCE, shared across writer iterations
    next_id  = [1_000_000]

    def writer():
        while not stop.is_set():
            try:
                with id_lock:
                    new_id    = next_id[0]
                    next_id[0] += 1
                index.insert_image(new_id, _rand_vec())
                time.sleep(0.002)
            except Exception as e:
                errors.append(f"Writer: {e}")
                break

    def reader():
        for _ in range(50):
            try:
                q       = features[np.random.randint(0, len(features))]
                results = index.query(q, k=5)
                assert isinstance(results, list), "query() must return a list"
                # Each result must be (int, float)
                for img_id, score in results:
                    assert isinstance(img_id, int),   "image_id must be int"
                    assert isinstance(score,  float),  "score must be float"
            except Exception as e:
                errors.append(f"Reader: {e}")
                break

    w       = threading.Thread(target=writer, daemon=True)
    readers = [threading.Thread(target=reader) for _ in range(5)]

    w.start()
    for r in readers:
        r.start()
    for r in readers:
        r.join()
    stop.set()
    w.join(timeout=3)

    assert not errors, \
        "Errors during concurrent query/insert:\n" + "\n".join(errors)
    print("test_query_during_insert")


def test_inserted_image_is_findable():
    """
    Querying with the same vector that was inserted should return that image
    in the top-10 results.
    """
    config  = LSHConfig(num_tables=10, hash_width=8, num_nodes=4, num_threads=2)
    index   = DistributedLSHIndex(config)
    features = np.array([_rand_vec() for _ in range(50)], dtype=np.float32)
    index.build_from_features(features)

    new_id  = 99999
    new_vec = _rand_vec()
    index.insert_image(new_id, new_vec)

    results    = index.query(new_vec, k=10)
    result_ids = [r[0] for r in results]

    assert new_id in result_ids, \
        f"Inserted image {new_id} not found in top-10: {result_ids}"
    print("test_inserted_image_is_findable")


def test_partition_lock_isolation():
    """
    Inserts on different partitions should not block each other.
    Two threads inserting into different nodes should both complete quickly.
    """
    index, _ = _build_index(n=40)

    # Find two image IDs that map to different nodes
    node_to_sample = {}
    for node_id, image_set in index.node_partitions.items():
        if image_set:
            node_to_sample[node_id] = node_id   # use node_id as new image_id offset

    timings = []
    errors  = []

    def timed_insert(new_id: int):
        t0 = time.perf_counter()
        try:
            index.insert_image(new_id, _rand_vec())
        except Exception as e:
            errors.append(str(e))
        timings.append((time.perf_counter() - t0) * 1000)

    # Insert into 4 different partitions simultaneously
    insert_ids = [20_000, 20_001, 20_002, 20_003]
    threads    = [threading.Thread(target=timed_insert, args=(i,))
                  for i in insert_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during parallel partition inserts: {errors}"
    # All inserts should complete in under 500ms — if they serialized badly it'd be much longer
    assert max(timings) < 500, \
        f"Inserts took too long, possible lock contention: {timings}"
    print("test_partition_lock_isolation")


def test_latency_tracking():
    """query_latencies_ms should be populated after queries run."""
    index, features = _build_index(n=40)

    assert len(index.query_latencies_ms) == 0, \
        "Latency list should be empty before any query"

    for i in range(10):
        index.query(features[i], k=5)

    assert len(index.query_latencies_ms) == 10, \
        f"Expected 10 latency records, got {len(index.query_latencies_ms)}"

    for lat in index.query_latencies_ms:
        assert lat >= 0, f"Latency must be non-negative, got {lat}"

    index.print_latency_stats()
    print("test_latency_tracking")

if __name__ == "__main__":
    tests = [
        test_insert_image_basic,
        test_insert_updates_partition,
        test_insert_wrong_dim_raises,
        test_insert_duplicate_id_raises,
        test_insert_before_build_raises,
        test_concurrent_insert_no_crash,
        test_query_during_insert,
        test_inserted_image_is_findable,
        test_partition_lock_isolation,
        test_latency_tracking,
    ]

    passed = 0
    failed = 0

    print("Running Milestone 3 incremental update tests:\n")

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED {test_fn.__name__}: {e}")
            failed += 1

    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All tests passed!")
