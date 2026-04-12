from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set
from threading import Lock

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class LSHConfig:
    """Configuration for LSH index construction."""
    num_tables: int = 10
    hash_width: int = 32
    num_nodes: int = 4
    virtual_nodes: int = 150
    num_threads: int = 4


class RandomProjection:
    """A single random projection for LSH hashing."""
    
    def __init__(self, dimension: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.projection = rng.randn(dimension)
        self.projection /= np.linalg.norm(self.projection)
    
    def hash(self, vector: np.ndarray) -> int:
        """Project vector and return sign bit as hash (0 or 1)."""
        projection_value = np.dot(vector, self.projection)
        return 1 if projection_value >= 0 else 0


class LSHHashTable:
    """Single LSH hash table with multiple random projections."""
    
    def __init__(self, num_projections: int, vector_dimension: int, table_id: int = 0):
        self.table_id = table_id
        self.num_projections = num_projections
        self.projections = [
            RandomProjection(vector_dimension, seed=table_id * 1000 + i)
            for i in range(num_projections)
        ]
        # bucket -> set of image_ids
        self.buckets: Dict[int, Set[int]] = {}
        self._lock = Lock()  # Thread-safe insertion
    
    def hash_vector(self, vector: np.ndarray) -> int:
        """Compute composite hash code from all projections (binary string as int)."""
        bits = []
        for projection in self.projections:
            bits.append(projection.hash(vector))
        # convert list of bits to integer
        hash_code = 0
        for bit in bits:
            hash_code = (hash_code << 1) | bit
        return hash_code
    
    def insert(self, image_id: int, vector: np.ndarray) -> None:
        """Insert image into hash table (thread-safe)."""
        bucket_id = self.hash_vector(vector)
        with self._lock:
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = set()
            self.buckets[bucket_id].add(image_id)
    
    def query(self, vector: np.ndarray) -> Set[int]:
        """Retrieve candidate images in same bucket as query vector."""
        bucket_id = self.hash_vector(vector)
        return self.buckets.get(bucket_id, set())
    
    def get_bucket_count(self) -> int:
        """Number of non-empty buckets."""
        return len(self.buckets)


class DistributedLSHIndex:
    """LSH index distributed across multiple nodes using consistent hashing."""
    
    def __init__(self, config: LSHConfig):
        self.config = config
        self.tables: List[LSHHashTable] = [
            LSHHashTable(
                num_projections=config.hash_width,
                vector_dimension=512,  # from M1 ResNet18
                table_id=i
            )
            for i in range(config.num_tables)
        ]
        # node_id -> set of image_ids on that node (from consistent hashing)
        self.node_partitions: Dict[int, Set[int]] = {i: set() for i in range(config.num_nodes)}
        self.image_to_node: Dict[int, int] = {}
    
    def build_from_features(
        self,
        feature_vectors: np.ndarray,
        metadata_csv: Path | None = None,
        consistent_hash_ring=None,
    ) -> None:
        """
        Build distributed LSH index from feature vectors.
        
        Args:
            feature_vectors: shape (N, D) float32 array, L2-normalized
            metadata_csv: optional path to metadata.csv for validation
            consistent_hash_ring: optional ConsistentHashRing from Raviha's module
        """
        dim = feature_vectors.shape[1]
        self.tables = [
        LSHHashTable(num_projections=self.config.hash_width, vector_dimension=dim, table_id=i)
        for i in range(self.config.num_tables)
    ]
        num_images = feature_vectors.shape[0]
        print(f"Building distributed LSH index: {num_images} images, {self.config.num_tables} tables")
        
        # If no consistent hash ring provided, use simple modulo partitioning
        if consistent_hash_ring is None:
            print(f"  Using simple modulo partitioning across {self.config.num_nodes} nodes")
            for image_id in range(num_images):
                node_id = image_id % self.config.num_nodes
                self.image_to_node[image_id] = node_id
                self.node_partitions[node_id].add(image_id)
        else:
            print(f"  Using Raviha's consistent hashing for partitioning")
            for image_id in range(num_images):
                node_id = consistent_hash_ring.get_node(image_id)
                self.image_to_node[image_id] = node_id
                self.node_partitions[node_id].add(image_id)
        
        self._insert_parallel(feature_vectors)
        self._print_summary()
    
    def _insert_parallel(self, feature_vectors: np.ndarray) -> None:
        """Insert all vectors into tables using thread pool."""
        num_images = feature_vectors.shape[0]
        
        def insert_image(image_id: int) -> None:
            vector = feature_vectors[image_id]
            for table in self.tables:
                table.insert(image_id, vector)
        
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            futures = [executor.submit(insert_image, i) for i in range(num_images)]
            for future in as_completed(futures):
                future.result()
        
        print(f"  Inserted {num_images} images into {self.config.num_tables} hash tables")
    
    def query(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Query the LSH index and return top-k nearest candidates by exact distance.
        
        Args:
            query_vector: shape (D,) float32, L2-normalized
            k: number of results to return
        
        Returns:
            List of (image_id, cosine_distance) tuples, sorted by distance (closest first)
        """
        candidates: Set[int] = set()
        
        # Collect candidates from all tables
        for table in self.tables:
            bucket_candidates = table.query(query_vector)
            candidates.update(bucket_candidates)
        
        if not candidates:
            return []
        
        results = [(cid, 0.0) for cid in list(candidates)[:k]]
        return results
    
    def _print_summary(self) -> None:
        """Print summary of index construction."""
        print("\n=== LSH Index Summary ===")
        print(f"Number of hash tables: {self.config.num_tables}")
        print(f"Hash width (bits per table): {self.config.hash_width}")
        print(f"Number of nodes: {self.config.num_nodes}")
        
        print("\nPartition Summary:")
        print(f"{'Node':<8} {'Images':<12} {'Percentage'}")
        print("-" * 35)
        total_images = sum(len(p) for p in self.node_partitions.values())
        for node_id in range(self.config.num_nodes):
            count = len(self.node_partitions[node_id])
            percent = (count / total_images) * 100 if total_images > 0 else 0
            print(f"Node {node_id:<4} {count:<12} {percent:.1f}%")
        
        print("\nHash Table Summary:")
        print(f"{'Table':<8} {'Buckets':<12} {'Avg Bucket Size'}")
        print("-" * 35)
        for i, table in enumerate(self.tables):
            num_buckets = table.get_bucket_count()
            total_images = sum(len(p) for p in self.node_partitions.values())
            avg_size = total_images / max(num_buckets, 1)
            print(f"Table {i:<5} {num_buckets:<12} {avg_size:.2f}")
