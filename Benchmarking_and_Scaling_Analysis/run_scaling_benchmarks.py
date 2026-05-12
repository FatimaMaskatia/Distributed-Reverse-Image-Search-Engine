import numpy as np
import time
import os
from lsh_index import DistributedLSHIndex, LSHConfig
from reranker import ParallelReRanker, ReRankConfig
from compressor import ScalarQuantizer
from sharding import IndexSharding

DATA_DIR = '../data'
NUM_QUERIES = 50

def run_scaling_test(test_name, data_file, num_nodes_list, max_images=None):
    print(f"\n{'='*50}\n🚀 RUNNING {test_name.upper()}\n{'='*50}")
    
    # Load the memory-mapped data
    filepath = os.path.join(DATA_DIR, data_file)
    features = np.lib.format.open_memmap(filepath, mode='r', dtype=np.float32)
    
    # --- COLAB RAM FIX: Slice the dataset if it's too large ---
    if max_images is not None and max_images < features.shape[0]:
        features = features[:max_images]
        
    num_images = features.shape[0]
    
    # Prepare queries and compression
    np.random.seed(42)
    query_indices = np.random.choice(num_images, NUM_QUERIES, replace=False)
    queries = features[query_indices].copy()
    
    quantizer = ScalarQuantizer()
    quantizer.fit(features[:10000])
    
    rerank_config = ReRankConfig(num_threads=4, similarity_metric="cosine")
    reranker = ParallelReRanker(rerank_config, features)

    for nodes in num_nodes_list:
        print(f"\n--- Testing Configuration: {nodes} Nodes | {num_images:,} Images ---")
        
        # 1. Build Raviha's Consistent Hash Ring
        sharder = IndexSharding(num_nodes=nodes, virtual_nodes=150)
        
        # 2. Build Fatima's M3 Index
        config = LSHConfig(num_tables=10, hash_width=16, num_nodes=nodes, num_threads=4)
        index = DistributedLSHIndex(config)
        
        print("Building Distributed Index with Consistent Hashing...")
        start_build = time.time()
        
        index.build_from_features(features, consistent_hash_ring=sharder.ring)
        
        print(f"Index built in {time.time() - start_build:.2f} seconds.")
        
        print(f"Running {NUM_QUERIES} test queries...")
        latencies = []
        
        for q_vec in queries:
            start_q = time.time()
            
            comp_q = quantizer.compress(q_vec)
            decomp_q = quantizer.decompress(comp_q)
            
            candidates = index.query(decomp_q, k=100) 
            
            # Fatima's code returns a tuple (image_id, score), so we use c[0]
            cand_ids = [c[0] for c in candidates] if candidates else []
            
            if cand_ids:
                final_results = reranker.rerank_vectorized(decomp_q, cand_ids)
                
            latencies.append((time.time() - start_q) * 1000) 
            
        latencies = np.array(latencies)
        print(f"✅ {test_name} - {nodes} Nodes Results:")
        print(f"   P50 Latency : {np.percentile(latencies, 50):.2f} ms")
        print(f"   P95 Latency : {np.percentile(latencies, 95):.2f} ms")
        print(f"   P99 Latency : {np.percentile(latencies, 99):.2f} ms")

# --- EXECUTE THE BENCHMARKS ---

# 1. Strong Scaling: Fixed at 250K images (fits in Colab RAM), testing across 1, 2, and 4 nodes
run_scaling_test("Strong Scaling Analysis", "synthetic_1m.npy", num_nodes_list=[1, 2, 4], max_images=250_000)

# 2. Weak Scaling: Proportional load. 62.5k on 1 node -> 250k on 4 nodes.
run_scaling_test("Weak Scaling Baseline (1 Node)", "synthetic_100k.npy", num_nodes_list=[1], max_images=62_500)
run_scaling_test("Weak Scaling Load (4 Nodes)", "synthetic_1m.npy", num_nodes_list=[4], max_images=250_000)