import threading
import time
import random
import numpy as np

from lsh_index import DistributedLSHIndex, LSHConfig


DIM                = 1696
NUM_READER_THREADS = 10
NUM_WRITER_THREADS = 3
INSERTS_PER_WRITER = 50
TEST_DURATION_SEC  = 30

DATA_PATH = r"C:\Users\maska\OneDrive\Desktop\PDC Project\Milestone_1\Feature_Fusion_and_Baseline_Evaluation\data\fused_features.npy"

def make_random_vector(dim: int = DIM) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def build_initial_index() -> tuple:
    print("Loading real dataset...")
    features = np.load(DATA_PATH, allow_pickle=True).astype(np.float32)
    norms    = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms
    print(f"Loaded {features.shape[0]} real images, dim={features.shape[1]}")

    config = LSHConfig(num_tables=10, hash_width=16, num_nodes=4, num_threads=4)
    index  = DistributedLSHIndex(config)
    index.build_from_features(features)
    print("Initial index ready.\n")
    return index, features


class ReaderWorker(threading.Thread):
    """
    Continuously queries the index until stop_event is set.
    Records per-query latency for P50/P95/P99 reporting.
    """

    def __init__(self, index, features, stop_event, worker_id):
        super().__init__(daemon=True)
        self.index       = index
        self.features    = features
        self.stop_event  = stop_event
        self.worker_id   = worker_id
        self.query_count = 0
        self.latencies   = []
        self.error       = None

    def run(self):
        try:
            while not self.stop_event.is_set():
                idx   = random.randint(0, len(self.features) - 1)
                q_vec = self.features[idx]

                t0      = time.perf_counter()
                results = self.index.query(q_vec, k=5)
                elapsed = (time.perf_counter() - t0) * 1000

                assert isinstance(results, list), \
                    f"Reader {self.worker_id}: got {type(results)}, expected list"
                for img_id, score in results:
                    assert isinstance(img_id, int),   "image_id must be int"
                    assert isinstance(score,  float),  "score must be float"
                    assert -1.0 <= score <= 1.01, \
                        f"Cosine similarity out of range: {score}"

                self.latencies.append(elapsed)
                self.query_count += 1

        except Exception as e:
            self.error = e


class WriterWorker(threading.Thread):
    """
    Inserts a fixed number of new images into the live index.
    Uses a shared lock + counter to guarantee unique IDs.
    """

    def __init__(self, index, id_lock, next_id, num_inserts, worker_id, inserted_ids):
        super().__init__(daemon=True)
        self.index        = index
        self.id_lock      = id_lock       # shared — defined ONCE outside
        self.next_id      = next_id       # shared mutable counter
        self.num_inserts  = num_inserts
        self.worker_id    = worker_id
        self.inserted_ids = inserted_ids
        self.insert_count = 0
        self.error        = None

    def run(self):
        try:
            for _ in range(self.num_inserts):
                vec = make_random_vector()

                with self.id_lock:
                    new_id = self.next_id[0]
                    self.next_id[0] += 1
                    self.inserted_ids.append((new_id, vec))

                self.index.insert_image(new_id, vec)
                self.insert_count += 1
                time.sleep(random.uniform(0.001, 0.01))

        except Exception as e:
            self.error = e


def verify_inserted_images(index, inserted_ids) -> None:
    """
    Verify inserted images are in partition maps (hard guarantee)
    and findable via query (soft — LSH is approximate).
    """
    print("\nVerifying inserted images...")
    map_present = 0
    query_found = 0
    sample      = inserted_ids[:20]

    for image_id, vec in sample:
        if image_id in index.image_to_node:
            map_present += 1
        results    = index.query(vec, k=10)
        result_ids = [r[0] for r in results]
        if image_id in result_ids:
            query_found += 1

    print(f"  Checked  : {len(sample)} inserted images")
    print(f"  In maps  : {map_present} / {len(sample)}  (must be 100%)")
    print(f"  Via query: {query_found} / {len(sample)}  "
          f"(LSH approximate — some misses expected)")

    assert map_present == len(sample), \
        "Some inserted images missing from partition maps — BUG"


def print_latency_report(all_latencies: list) -> None:
    if not all_latencies:
        print("  No latency data recorded.")
        return
    arr = np.array(all_latencies)
    print(f"\n  Query Latency ({len(all_latencies)} queries across all reader threads):")
    print(f"    Mean : {arr.mean():.3f} ms")
    print(f"    P50  : {np.percentile(arr, 50):.3f} ms")
    print(f"    P95  : {np.percentile(arr, 95):.3f} ms")
    print(f"    P99  : {np.percentile(arr, 99):.3f} ms")
    print(f"    Min  : {arr.min():.3f} ms")
    print(f"    Max  : {arr.max():.3f} ms")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_stress_test():
    
    print("  Milestone 3 - Stress Test: Concurrent Reads + Writes")

    index, features = build_initial_index()
    dataset_size    = len(features)

    stop_event   = threading.Event()
    id_lock      = threading.Lock()        # ONE shared lock for unique IDs
    next_id      = [dataset_size]          # start IDs after real dataset
    inserted_ids = []

    readers = [
        ReaderWorker(index, features, stop_event, worker_id=i)
        for i in range(NUM_READER_THREADS)
    ]
    writers = [
        WriterWorker(
            index        = index,
            id_lock      = id_lock,
            next_id      = next_id,
            num_inserts  = INSERTS_PER_WRITER,
            worker_id    = w,
            inserted_ids = inserted_ids,
        )
        for w in range(NUM_WRITER_THREADS)
    ]

    print(f"Starting {NUM_READER_THREADS} reader threads "
          f"and {NUM_WRITER_THREADS} writer threads...")
    print(f"Readers run for {TEST_DURATION_SEC}s. "
          f"Writers insert {NUM_WRITER_THREADS * INSERTS_PER_WRITER} images total.\n")

    t_start = time.time()

    for w in writers:
        w.start()
    for r in readers:
        r.start()

    time.sleep(TEST_DURATION_SEC)
    stop_event.set()

    for r in readers:
        r.join(timeout=5)
    for w in writers:
        w.join(timeout=10)

    t_elapsed = time.time() - t_start

  
    print("  STRESS TEST RESULTS")
   

    reader_errors = [r.error for r in readers if r.error is not None]
    writer_errors = [w.error for w in writers if w.error is not None]

    if reader_errors:
        print(f"\n  READER ERRORS ({len(reader_errors)}):")
        for e in reader_errors:
            print(f"    {e}")
    else:
        print("\n  Reader errors : NONE")

    if writer_errors:
        print(f"\n  WRITER ERRORS ({len(writer_errors)}):")
        for e in writer_errors:
            print(f"    {e}")
    else:
        print("  Writer errors : NONE")

    total_queries = sum(r.query_count for r in readers)
    total_inserts = sum(w.insert_count for w in writers)
    actual_size   = sum(len(p) for p in index.node_partitions.values())

    print(f"\n  Duration          : {t_elapsed:.1f} sec")
    print(f"  Total queries     : {total_queries}")
    print(f"  Total inserts     : {total_inserts}")
    print(f"  Query throughput  : {total_queries / t_elapsed:.1f} queries/sec")
    print(f"  Insert throughput : {total_inserts / t_elapsed:.1f} inserts/sec")
    print(f"  Initial index size: {dataset_size} images")
    print(f"  Final index size  : {actual_size} images")

    all_latencies = []
    for r in readers:
        all_latencies.extend(r.latencies)
    print_latency_report(all_latencies)

    print("\n  Per-reader query counts:")
    for r in readers:
        avg = f"{sum(r.latencies)/len(r.latencies):.2f} ms" if r.latencies else "N/A"
        print(f"    Reader {r.worker_id}: {r.query_count} queries, avg {avg}")

    print("\n  Per-writer insert counts:")
    for w in writers:
        print(f"    Writer {w.worker_id}: {w.insert_count} inserts")

    verify_inserted_images(index, inserted_ids)
    index.print_latency_stats()

    passed = not reader_errors and not writer_errors and total_inserts > 0
    print("\n")
    if passed:
        print("  RESULT: PASSED — system stable under concurrent load")
    else:
        print("  RESULT: FAILED — check errors above")


    return passed


if __name__ == "__main__":
    run_stress_test()
