import sys
import numpy as np
from pathlib import Path
from collections import OrderedDict

# Ensure Python finds your team's code
sys.path.insert(0, "/content/drive/MyDrive/PDC_Project_M2/src")

from lsh_index import DistributedLSHIndex, LSHConfig
from sharding import IndexSharding
from query_processor import DistributedQueryProcessor
from utils import load_features_and_metadata

# ==========================================
# 1. YOUR CACHE IMPLEMENTATION
# ==========================================
class DistributedQueryCache:
    """
    An LRU (Least Recently Used) Cache for the Distributed Search Engine.
    Stores the results of frequent/hot queries to prevent redundant LSH index lookups.
    """
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _generate_key(self, query_vector: np.ndarray) -> int:
        return hash(query_vector.round(4).tobytes())

    def get(self, query_vector: np.ndarray):
        key = self._generate_key(query_vector)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, query_vector: np.ndarray, results: list, stats):
        key = self._generate_key(query_vector)
        if stats:
            stats.total_ms = 0.0 
            stats.fanout_ms = 0.0
            
        self.cache[key] = (results, stats)
        self.cache.move_to_end(key)
        
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def print_cache_stats(self):
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        print("\n=== Cache Performance ===")
        print(f"Total Requests : {total_requests}")
        print(f"Cache Hits     : {self.hits}")
        print(f"Cache Misses   : {self.misses}")
        print(f"Hit Rate       : {hit_rate:.1f}%")
        print(f"Items in Cache : {len(self.cache)} / {self.capacity}")

# ==========================================
# 2. SYSTEM SETUP & PROOF TEST
# ==========================================
BASE_PATH = "/content/drive/MyDrive/PDC_Project_M1/"

print("Loading data and spinning up test index...")
features, _ = load_features_and_metadata(
    Path(f"{BASE_PATH}fused_features.npy"),
    Path(f"{BASE_PATH}metadata.csv")
)

# Build a quick index just for this test
config = LSHConfig(num_tables=5, hash_width=32, num_nodes=4, virtual_nodes=150)
sharder = IndexSharding(num_nodes=4, virtual_nodes=150)
sharder.distribute(features)
index = DistributedLSHIndex(config)
index.build_from_features(features, consistent_hash_ring=sharder.ring)
processor = DistributedQueryProcessor(lsh_index=index, feature_matrix=features, sharder=sharder)


# --- RUN THE TEST ---
print("\nInitializing Cache...")
cache = DistributedQueryCache(capacity=50)

# We will use the very first image in the database as our test query
test_vector = features[0]

print("\n--- First Query (Cold Start) ---")
cached_data = cache.get(test_vector)

if cached_data:
    print("✅ Cache HIT!")
    results, stats = cached_data
else:
    print("❌ Cache MISS! Routing to Distributed Index...")
    results, stats = processor.query(test_vector, k=5, collect_stats=True)

# PRINT FIRST, THEN PUT IN CACHE!
print(f"Time Taken: {stats.total_ms:.3f} ms")
cache.put(test_vector, results, stats)


print("\n--- Second Query (Hot Query) ---")
cached_data = cache.get(test_vector)

if cached_data:
    print("✅ Cache HIT! Bypassing Index...")
    results, stats = cached_data
else:
    print("❌ Cache MISS!")
    results, stats = processor.query(test_vector, k=5, collect_stats=True)

print(f"Time Taken: {stats.total_ms:.3f} ms")

cache.print_cache_stats()