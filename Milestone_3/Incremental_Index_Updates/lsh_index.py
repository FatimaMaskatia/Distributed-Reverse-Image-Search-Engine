from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

import numpy as np

try:
    from readerwriterlock import rwlock
    _HAS_RWLOCK = True
except ImportError:
    _HAS_RWLOCK = False


@dataclass
class LSHConfig:
    """Configuration for LSH index construction."""
    num_tables:    int = 10
    hash_width:    int = 32
    num_nodes:     int = 4
    virtual_nodes: int = 150
    num_threads:   int = 4


class RandomProjection:
    """A single random projection for LSH hashing."""

    def __init__(self, dimension: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.projection = rng.randn(dimension)
        self.projection /= np.linalg.norm(self.projection)

    def hash(self, vector: np.ndarray) -> int:
        return 1 if np.dot(vector, self.projection) >= 0 else 0


class LSHHashTable:
    """
    Single LSH hash table with multiple random projections.
    Lock-free internally — locking is handled at the partition level
    by DistributedLSHIndex.
    """

    def __init__(self, num_projections: int, vector_dimension: int, table_id: int = 0):
        self.table_id        = table_id
        self.num_projections = num_projections
        self.projections     = [
            RandomProjection(vector_dimension, seed=table_id * 1000 + i)
            for i in range(num_projections)
        ]
        self.buckets: Dict[int, Set[int]] = {}

    def hash_vector(self, vector: np.ndarray) -> int:
        code = 0
        for proj in self.projections:
            code = (code << 1) | proj.hash(vector)
        return code

    def insert(self, image_id: int, vector: np.ndarray) -> None:
        """Insert image into bucket. Caller must hold the write lock."""
        bucket_id = self.hash_vector(vector)
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = set()
        self.buckets[bucket_id].add(image_id)

    def query(self, vector: np.ndarray) -> Set[int]:
        """Return candidates in matching bucket. Caller must hold the read lock."""
        bucket_id = self.hash_vector(vector)
        return set(self.buckets.get(bucket_id, set()))

    def get_bucket_count(self) -> int:
        return len(self.buckets)


def _make_rwlock():
    """Return (read_lock_factory, write_lock_factory) using best available lock."""
    if _HAS_RWLOCK:
        rw = rwlock.RWLockFair()
        return rw.gen_rlock, rw.gen_wlock
    else:
        # Fallback: plain threading.Lock (correct, just no read concurrency)
        lock = threading.Lock()
        return lambda: lock, lambda: lock


class DistributedLSHIndex:
    """
    LSH index distributed across multiple nodes using consistent hashing.

    M3 changes vs M2
    ----------------
    1. Hardcoded vector_dimension=512 REMOVED — tables built lazily in
       build_from_features() once real dimension is known.

    2. insert_image() added — single image inserted into all tables +
       partition maps updated, no rebuild needed.

    3. PER-PARTITION read-write locks (self._partition_locks).
       - One RWLock per node/partition.
       - Query on node N  → acquires read  lock for N only.
       - Insert on node N → acquires write lock for N only.
       - Nodes on different partitions never block each other.

    4. Thread-safe metadata (self._meta_lock).
       - image_to_node and node_partitions updates are protected by a
         single small threading.Lock so two concurrent inserts can never
         produce inconsistent state.

    5. Latency tracking — query() returns real cosine similarity scores
       and records per-query latency in self.query_latencies_ms.
    """

    def __init__(self, config: LSHConfig):
        self.config = config

        # Tables built in build_from_features once dim is known
        self.tables: List[LSHHashTable] = []

        # Partition maps
        self.node_partitions: Dict[int, Set[int]] = {
            i: set() for i in range(config.num_nodes)
        }
        self.image_to_node: Dict[int, int] = {}

        self._dim: int              = 0
        self._consistent_hash_ring  = None
        self._feature_store: Dict[int, np.ndarray] = {}  # image_id -> vector

        # ── Per-partition RW locks (one per node) ──────────────────────────
        self._partition_locks: Dict[int, Tuple] = {}
        for node_id in range(config.num_nodes):
            r, w = _make_rwlock()
            self._partition_locks[node_id] = (r, w)   # (read_factory, write_factory)

        # ── Metadata lock (protects image_to_node + node_partitions) ───────
        self._meta_lock = threading.Lock()

        # ── Latency tracking ────────────────────────────────────────────────
        self.query_latencies_ms: List[float] = []

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_from_features(
        self,
        feature_vectors: np.ndarray,
        metadata_csv: Path | None = None,
        consistent_hash_ring=None,
    ) -> None:
        """Build distributed LSH index from feature vectors."""

        self._dim                  = feature_vectors.shape[1]
        self._consistent_hash_ring = consistent_hash_ring

        # Build tables now that real dimension is known
        self.tables = [
            LSHHashTable(
                num_projections=self.config.hash_width,
                vector_dimension=self._dim,
                table_id=i,
            )
            for i in range(self.config.num_tables)
        ]

        num_images = feature_vectors.shape[0]
        print(f"Building distributed LSH index: {num_images} images, "
              f"{self.config.num_tables} tables, dim={self._dim}")

        # Assign images to nodes
        if consistent_hash_ring is None:
            print(f"  Using simple modulo partitioning across "
                  f"{self.config.num_nodes} nodes")
            for image_id in range(num_images):
                node_id = image_id % self.config.num_nodes
                self.image_to_node[image_id] = node_id
                self.node_partitions[node_id].add(image_id)
        else:
            print("  Using consistent hashing for partitioning")
            for image_id in range(num_images):
                node_id = consistent_hash_ring.get_node(image_id)
                self.image_to_node[image_id] = node_id
                self.node_partitions[node_id].add(image_id)

        # Store feature vectors for exact re-ranking
        for image_id in range(num_images):
            self._feature_store[image_id] = feature_vectors[image_id]

        self._insert_parallel(feature_vectors)
        self._print_summary()

    def _insert_parallel(self, feature_vectors: np.ndarray) -> None:
        """Insert all vectors into all tables using a thread pool."""
        num_images = feature_vectors.shape[0]

        def insert_one(image_id: int) -> None:
            node_id = self.image_to_node[image_id]
            vector  = feature_vectors[image_id]
            # Acquire write lock for this image's partition
            _, write_lock = self._partition_locks[node_id]
            with write_lock():
                for table in self.tables:
                    table.insert(image_id, vector)

        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            futures = [executor.submit(insert_one, i) for i in range(num_images)]
            for future in as_completed(futures):
                future.result()

        print(f"  Inserted {num_images} images into "
              f"{self.config.num_tables} hash tables")

    # ------------------------------------------------------------------
    # M3: Incremental insert
    # ------------------------------------------------------------------

    def insert_image(self, image_id: int, vector: np.ndarray) -> int:
        """
        Insert a single NEW image into the live index without rebuilding.

        Steps
        -----
        1. Validate inputs.
        2. Assign image to a node (consistent hash ring or modulo).
        3. Update metadata (image_to_node, node_partitions) under meta_lock
           so two concurrent inserts never corrupt the maps.
        4. Store vector for future exact re-ranking.
        5. Acquire the WRITE lock for that node's partition only,
           then insert into every hash table.
           → Other nodes' partitions remain fully readable during this insert.

        Parameters
        ----------
        image_id : unique integer ID for the new image
        vector   : 1-D float32, L2-normalised, length must equal self._dim

        Returns
        -------
        node_id the image was assigned to
        """
        if self._dim == 0:
            raise RuntimeError(
                "Index not built yet. Call build_from_features() first."
            )
        if len(vector) != self._dim:
            raise ValueError(
                f"Vector dim mismatch: expected {self._dim}, got {len(vector)}"
            )
        if self._consistent_hash_ring is not None:
            node_id = self._consistent_hash_ring.get_node(image_id)
        else:
            node_id = image_id % self.config.num_nodes

        with self._meta_lock:
            if image_id in self.image_to_node:
                raise ValueError(
                    f"image_id {image_id} already exists. Use a unique ID."
        )
            self.image_to_node[image_id] = node_id
            self.node_partitions[node_id].add(image_id)
            self._feature_store[image_id] = vector

        # 3. Acquire WRITE lock for this partition only, insert into all tables
        _, write_lock = self._partition_locks[node_id]
        with write_lock():
            for table in self.tables:
                table.insert(image_id, vector)

        return node_id

    # ------------------------------------------------------------------
    # Query — with real cosine similarity scores + latency tracking
    # ------------------------------------------------------------------

    def query(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Query the index and return top-k candidates with exact cosine similarity.

        For each node's partition:
          - Acquire READ lock (shared — multiple queries can run simultaneously).
          - Collect candidates from all hash tables for that partition.
          - Release read lock.
        Then compute exact cosine similarity on the merged candidate set
        and return top-k sorted by similarity (descending).

        Note: exact re-ranking across the full candidate set is Arham's M3
        parallel re-ranking task. This method returns real scores (not 0.0
        placeholders) so the pipeline is correct end-to-end.
        """
        t_start = time.perf_counter()

        # Collect candidates per partition using per-partition read locks
        all_candidates: Set[int] = set()
        for node_id in range(self.config.num_nodes):
            read_lock, _ = self._partition_locks[node_id]
            with read_lock():
                # Read node_images inside the lock to avoid race with insert_image
                node_images = self.node_partitions[node_id].copy()
                for table in self.tables:
                    bucket_candidates = table.query(query_vector)
                    # Keep only images that belong to this partition
                    all_candidates.update(bucket_candidates & node_images)

        if not all_candidates:
            self.query_latencies_ms.append(
                (time.perf_counter() - t_start) * 1000
            )
            return []

        # Exact cosine similarity (dot product on L2-normalised vectors)
        candidate_ids = list(all_candidates)
        with self._meta_lock:
            candidate_vecs = np.array(
                [self._feature_store[cid] for cid in candidate_ids],
                dtype=np.float32,
            )
        similarities = candidate_vecs @ query_vector   # (C,)

        # Sort descending and take top-k
        top_indices = np.argsort(similarities)[::-1][:k]
        results = [
            (candidate_ids[i], float(similarities[i]))
            for i in top_indices
        ]

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000
        self.query_latencies_ms.append(t_elapsed_ms)

        return results

    # ------------------------------------------------------------------
    # Latency report
    # ------------------------------------------------------------------

    def print_latency_stats(self) -> None:
        """Print P50 / P95 / P99 query latency from recorded queries."""
        if not self.query_latencies_ms:
            print("No queries recorded yet.")
            return

        arr  = sorted(self.query_latencies_ms)
        n    = len(arr)
        p50  = arr[int(n * 0.50)]
        p95  = arr[int(n * 0.95)]
        p99  = arr[min(int(n * 0.99), n - 1)]
        mean = sum(arr) / n

        print("\n=== Query Latency Stats ===")
        print(f"  Queries recorded : {n}")
        print(f"  Mean             : {mean:.3f} ms")
        print(f"  P50              : {p50:.3f} ms")
        print(f"  P95              : {p95:.3f} ms")
        print(f"  P99              : {p99:.3f} ms")
        print(f"  Min              : {arr[0]:.3f} ms")
        print(f"  Max              : {arr[-1]:.3f} ms")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        print("\n=== LSH Index Summary ===")
        print(f"Number of hash tables : {self.config.num_tables}")
        print(f"Hash width (bits)     : {self.config.hash_width}")
        print(f"Number of nodes       : {self.config.num_nodes}")
        print(f"Feature dimension     : {self._dim}")
        print(f"Locking strategy      : per-partition RW lock "
              f"({'readerwriterlock' if _HAS_RWLOCK else 'threading.Lock fallback'})")

        print("\nPartition Summary:")
        print(f"{'Node':<8} {'Images':<12} {'Percentage'}")
        print("-" * 35)
        total = sum(len(p) for p in self.node_partitions.values())
        for node_id in range(self.config.num_nodes):
            count   = len(self.node_partitions[node_id])
            percent = (count / total * 100) if total > 0 else 0
            print(f"Node {node_id:<4} {count:<12} {percent:.1f}%")

        print("\nHash Table Summary:")
        print(f"{'Table':<8} {'Buckets':<12} {'Avg Bucket Size'}")
        print("-" * 35)
        for i, table in enumerate(self.tables):
            nb  = table.get_bucket_count()
            avg = total / max(nb, 1)
            print(f"Table {i:<5} {nb:<12} {avg:.2f}")
