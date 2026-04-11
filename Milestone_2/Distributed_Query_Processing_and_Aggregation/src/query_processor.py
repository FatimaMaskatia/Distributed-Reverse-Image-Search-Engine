from __future__ import annotations
"""
Distributed Query Processing & Aggregation
Responsibilities
- Distributed query fanout  : dispatch a query image's feature vector to all
  relevant index partitions simultaneously using multiprocessing queues.
- Result aggregation        : merge candidate matches from every partition and
  re-rank them by exact cosine similarity.
- Latency optimisation      : minimise synchronisation barriers; overlap
  per-partition work as much as possible.

Integration
- DistributedLSHIndex  (LSH candidate retrieval per table)
- IndexSharding        (knows which feature vectors live on each node)
- M1   ->  fused_features.npy   (full feature matrix; needed for exact re-ranking)
"""

import time
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

@dataclass
class QueryResult:
    """A single retrieved image with its exact similarity score."""
    image_id: int
    score: float          # cosine similarity (higher = more similar)
    node_id: int          # which partition returned this candidate

    def __lt__(self, other: "QueryResult") -> bool:   # enables sorting
        return self.score > other.score               # descending by score


@dataclass
class QueryStats:
    """Timing breakdown for a single query, all times are in milliseconds."""
    total_ms: float = 0.0
    fanout_ms: float = 0.0
    aggregation_ms: float = 0.0
    reranking_ms: float = 0.0
    num_candidates: int = 0
    num_nodes_queried: int = 0

def _compute_bucket_id(projections: List[np.ndarray], vector: np.ndarray) -> int:
    """
    Recompute the LSH bucket ID for *vector* using serialised projection vectors.
    Mirrors Arham's LSHHashTable.hash_vector() exactly so results are consistent.
    """
    code = 0
    for proj in projections:
        bit = 1 if np.dot(vector, proj) >= 0 else 0
        code = (code << 1) | bit
    return code


def _node_worker(
    node_id: int,
    tables_data: List[Tuple[List[np.ndarray], Dict[int, List[int]]]],
    query_queue: "Queue[Optional[np.ndarray]]",
    result_queue: "Queue[Tuple[int, Set[int]]]",
) -> None:
    """
    Long-running worker process that serves one partition of the index.

    Protocol
    --------
    * Receives query vectors from *query_queue*.
    * For each LSH table, recomputes the bucket hash using the serialised
      projection vectors, then fetches only the matching bucket — true LSH
      selectivity, not a full scan.
    * Puts ``(node_id, candidate_set)`` into *result_queue*.
    * Exits when it receives ``None`` as a sentinel.

    Parameters

    tables_data : list of (projections, bucket_dict) per table
        projections - list of 1-D numpy arrays (the random projection vectors)
        bucket_dict - {bucket_id: [image_id, ...]} for this node's images only
    """
    while True:
        query_vector = query_queue.get()
        if query_vector is None:          # shutdown sentinel
            break

        candidates: Set[int] = set()
        for projections, bucket_dict in tables_data:
            # Recompute the exact same bucket hash Arham's index uses
            bucket_id = _compute_bucket_id(projections, query_vector)
            if bucket_id in bucket_dict:
                candidates.update(bucket_dict[bucket_id])

        result_queue.put((node_id, candidates))


class DistributedQueryProcessor:
    """
    Wraps DistributedLSHIndex and IndexSharding to provide
    parallel query fanout, candidate aggregation and exact re-ranking.

    Parameters

    lsh_index: Arham's DistributedLSHIndex (already built)
    feature_matrix: full (N, D) float32 array, needed for exact cosine similarity during re-ranking
    sharder: Raviha's IndexSharding used to resolve which features live on which node for smarter fanout
    num_nodes: number of simulated worker nodes (default: 4)
    use_multiprocess: if True, use real multiprocessing queues; if False, use in-process simulation (faster for testing)
    """

    def __init__(
        self,
        lsh_index,
        feature_matrix: np.ndarray,
        sharder=None,
        num_nodes: int = 4,
        use_multiprocess: bool = False,
    ) -> None:
        self.index = lsh_index
        self.feature_matrix = feature_matrix # shape (N, D)
        self.sharder = sharder
        self.num_nodes = num_nodes
        self.use_multiprocess = use_multiprocess

        # Pre-build per-node lookup: node_id -> set of image_ids on that node
        self._node_image_sets: Dict[int, Set[int]] = {
            i: set() for i in range(num_nodes)
        }
        self._build_node_sets()

        # Multiprocessing infrastructure (only when use_multiprocess=True)
        self._workers: List[Process] = []
        self._query_queues: List[Queue] = []
        self._result_queue: Optional[Queue] = None

        if use_multiprocess:
            self._start_workers()

    # Setup helpers

    def _build_node_sets(self) -> None:
        """Populate per-node image-id sets from the LSH index's partition map."""
        if self.sharder is not None:
            # Use Raviha's authoritative partition data
            for node_id in range(self.num_nodes):
                partition = self.sharder.get_partition(node_id)
                self._node_image_sets[node_id] = set(partition.keys())
        else:
            # Fall back to Arham's index's own partition tracking
            for node_id, image_set in self.index.node_partitions.items():
                self._node_image_sets[node_id] = set(image_set)

    def _start_workers(self) -> None:
        """Spawn one worker Process per node with its own queue pair."""
        self._result_queue = Queue()

        for node_id in range(self.num_nodes):
            q: Queue = Queue()
            self._query_queues.append(q)

            # Serialise bucket structure + projections for this node's images
            node_images = self._node_image_sets[node_id]
            tables_data = self._serialise_node_tables(node_images)

            p = Process(
                target=_node_worker,
                args=(node_id, tables_data, q, self._result_queue),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        print(f"[QueryProcessor] Started {self.num_nodes} worker processes")

    def _serialise_node_tables(
        self, node_images: Set[int]
    ) -> List[Tuple[List[np.ndarray], Dict[int, List[int]]]]:
        """
        Serialise  LSH tables for a worker process.

        For each table we export:
        - projections : the random projection vectors (needed to recompute
                        bucket hashes inside the worker process)
        - bucket_dict : {bucket_id: [image_id, ...]} filtered to this node's
                        images only

        This allows the worker to perform true LSH lookup (hash then fetch)
        rather than iterating all buckets.
        """
        result = []
        for table in self.index.tables:
            # Export projection vectors (plain numpy arrays, picklable)
            projections = [p.projection for p in table.projections]

            # Filter buckets to this node's images only
            filtered: Dict[int, List[int]] = {}
            for bucket_id, members in table.buckets.items():
                local = [m for m in members if m in node_images]
                if local:
                    filtered[bucket_id] = local

            result.append((projections, filtered))
        return result

    # Public API

    def query(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        collect_stats: bool = False,
    ) -> Tuple[List[QueryResult], Optional[QueryStats]]:
        """
        Run a distributed query and return the top-k results.

        Parameters
        ----------
        query_vector   : shape (D,) float32, L2-normalised
        k              : number of results to return
        collect_stats  : whether to return a QueryStats timing breakdown

        Returns
        -------
        results  : list of QueryResult sorted by cosine similarity (best first)
        stats    : QueryStats if collect_stats=True, else None
        """
        stats = QueryStats(num_nodes_queried=self.num_nodes) if collect_stats else None
        t_total_start = time.perf_counter()

        #1 Fanout 
        t_fanout_start = time.perf_counter()

        if self.use_multiprocess:
            raw_candidates = self._fanout_multiprocess(query_vector)
        else:
            raw_candidates = self._fanout_inprocess(query_vector)

        t_fanout_end = time.perf_counter()

        #2 Aggregate candidates 
        t_agg_start = time.perf_counter()
        merged = self._aggregate(raw_candidates)
        t_agg_end = time.perf_counter()

        #3 Exact re-ranking by cosine similarity
        t_rerank_start = time.perf_counter()
        results = self._rerank(query_vector, merged, k)
        t_rerank_end = time.perf_counter()

        t_total_end = time.perf_counter()

        if collect_stats:
            stats.total_ms       = (t_total_end   - t_total_start)   * 1000
            stats.fanout_ms      = (t_fanout_end  - t_fanout_start)  * 1000
            stats.aggregation_ms = (t_agg_end     - t_agg_start)     * 1000
            stats.reranking_ms   = (t_rerank_end  - t_rerank_start)  * 1000
            stats.num_candidates = len(merged)

        return results, stats

    def batch_query(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        collect_stats: bool = False,
    ) -> List[Tuple[List[QueryResult], Optional[QueryStats]]]:
        """
        Run multiple queries sequentially.  Individual per-query stats are
        returned so that P50/P95/P99 latencies can be computed externally.

        Parameters
        
        query_vectors : shape (Q, D) float32
        k             : top-k per query
        """
        return [
            self.query(query_vectors[i], k=k, collect_stats=collect_stats)
            for i in range(len(query_vectors))
        ]

    def shutdown(self) -> None:
        """Send shutdown sentinel to all worker processes."""
        if self.use_multiprocess:
            for q in self._query_queues:
                q.put(None)
            for p in self._workers:
                p.join(timeout=5)
            print("[QueryProcessor] Workers shut down")

    # Fanout strategies

    def _fanout_inprocess(
        self, query_vector: np.ndarray
    ) -> Dict[int, Set[int]]:
        """
        In-process fanout: query every node's LSH tables directly.
        Each node queries only the buckets that contain its own images,
        minimising cross-node candidate leakage.
        """
        per_node_candidates: Dict[int, Set[int]] = {}

        for node_id in range(self.num_nodes):
            node_images = self._node_image_sets[node_id]
            candidates: Set[int] = set()

            for table in self.index.tables:
                bucket_id = table.hash_vector(query_vector)
                bucket_members = table.buckets.get(bucket_id, set())
                # Keep only images that belong to this node's partition
                candidates.update(bucket_members & node_images)

            per_node_candidates[node_id] = candidates

        return per_node_candidates

    def _fanout_multiprocess(
        self, query_vector: np.ndarray
    ) -> Dict[int, Set[int]]:
        """
        Multiprocess fanout: push query to all worker queues simultaneously,
        then collect results.  Workers run the same bucket-lookup logic but
        inside their own process address space.
        """
        # Dispatch to all nodes simultaneously (no barrier between sends)
        for q in self._query_queues:
            q.put(query_vector)

        # Collect — blocks until all nodes have responded
        per_node_candidates: Dict[int, Set[int]] = {}
        for _ in range(self.num_nodes):
            node_id, candidates = self._result_queue.get()
            per_node_candidates[node_id] = candidates

        return per_node_candidates

    # Aggregation & re-ranking

    def _aggregate(
        self, per_node_candidates: Dict[int, Set[int]]
    ) -> Dict[int, int]:
        """
        Merge candidates from all partitions.

        Returns a dict of image_id -> node_id so we know the provenance
        of each candidate (useful for debugging and stats).
        Candidates that appear in multiple nodes (shouldn't happen with
        consistent hashing, but handled defensively) keep the first
        node assignment.
        """
        merged: Dict[int, int] = {}
        for node_id, candidates in per_node_candidates.items():
            for image_id in candidates:
                if image_id not in merged:
                    merged[image_id] = node_id
        return merged

    def _rerank(
        self,
        query_vector: np.ndarray,
        merged_candidates: Dict[int, int],
        k: int,
    ) -> List[QueryResult]:
        """
        Compute exact cosine similarity between the query and every candidate,
        then return the top-k sorted by similarity (descending).

        Because feature_matrix rows are L2-normalised, cosine similarity
        reduces to a dot product — very fast with NumPy.
        """
        if not merged_candidates:
            return []

        candidate_ids = list(merged_candidates.keys())
        candidate_vectors = self.feature_matrix[candidate_ids] # (C, D)

        # Dot product = cosine similarity for L2-normalised vectors
        similarities = candidate_vectors @ query_vector # (C,)

        # Pair up and sort (descending similarity)
        scored = sorted(
            zip(candidate_ids, similarities.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        results = [
            QueryResult(
                image_id=img_id,
                score=float(score),
                node_id=merged_candidates[img_id],
            )
            for img_id, score in scored[:k]
        ]
        return results

    # Diagnostics

    def print_stats(self, stats: QueryStats) -> None:
        """print a QueryStats object."""
        print("\nQuery Statistics")
        print(f"  Total latency     : {stats.total_ms:.3f} ms")
        print(f"  Fanout            : {stats.fanout_ms:.3f} ms")
        print(f"  Aggregation       : {stats.aggregation_ms:.3f} ms")
        print(f"  Re-ranking        : {stats.reranking_ms:.3f} ms")
        print(f"  Candidates found  : {stats.num_candidates}")
        print(f"  Nodes queried     : {stats.num_nodes_queried}")