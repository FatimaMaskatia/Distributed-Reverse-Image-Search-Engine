from __future__ import annotations
"""
Inter-Node Communication Optimization  —  Milestone 3 

Runs the full optimization pipeline on real M1 feature vectors:

  1. Load + normalize fused feature vectors from M1
  2. Fit the scalar quantizer and show compression stats
  3. Build the distributed LSH index using Arham's + Raviha's sharding
  4. Run the serialization profiler (time + payload size, with simulated network)
  5. Run batch throughput analysis
  6. Run bottleneck analysis (pickle vs numpy)
  7. Save final report to final_report.txt

Usage
    python main.py

Expected runtime: ~40 seconds (dominated by LSH index build).
All results are also printed to stdout as they run.
"""

import sys
from pathlib import Path

import numpy as np

# ── imports from M2 ─────────────────────────────────────────────────────────
# These files must be in the same directory or on PYTHONPATH:
#   consistent_hashing.py  (Raviha - M2)
#   sharding.py            (Raviha - M2)
#   lsh_index.py           (Arham  - M2)
#   query_processor.py     (Fatima - M2)
try:
    from consistent_hashing import ConsistentHashRing
    from sharding import IndexSharding
    from lsh_index import DistributedLSHIndex, LSHConfig
    from query_processor import DistributedQueryProcessor
    HAS_M2 = True
except ImportError as e:
    print(f"[Warning] Could not import M2 modules: {e}")
    print("  Compression and profiling will still run.")
    print("  Place M2 files in the same directory for full pipeline.\n")
    HAS_M2 = False

from compressor import ScalarQuantizer
from profiler import CommunicationProfiler


# helpers 

def load_features(features_path: str, metadata_path: str):
    """
    Load and normalize real M1 feature vectors.
    Falls back to synthetic data if files not found (valid per project spec).
    """
    try:
        print("Loading real feature vectors from M1...")
        features = np.load(features_path, allow_pickle=False).astype(np.float32)

        # L2-normalize (same as M2 pipeline)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        features = features / norms

        image_ids = []
        with open(metadata_path, "r") as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split(",")
                if parts:
                    image_ids.append(parts[0])

        print(f"  Loaded {len(features)} images, {features.shape[1]}-dim vectors")
        return features, image_ids

    except FileNotFoundError:
        print("Real data files not found — generating synthetic vectors.")
        print("(Valid per project spec for scalability experiments.)")
        rng = np.random.default_rng(42)
        features = rng.standard_normal((7128, 1184)).astype(np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        features = features / norms
        image_ids = [f"synthetic_{i}.jpg" for i in range(len(features))]
        print(f"  Generated {len(features)} synthetic images, {features.shape[1]}-dim vectors")
        return features, image_ids


def build_m2_pipeline(features: np.ndarray):
    """
    Build the distributed LSH index using M2 components.
    Returns (index, processor) or (None, None) if M2 not available.

    LSH config note:
        hash_width=16 is used (not 32) because 32-bit hashing on 7128 images
        creates too many unique buckets (sparse index, slow build).
        16-bit gives a good accuracy/speed tradeoff for this dataset size.
        num_threads=2 avoids thread contention on the build machine.
    """
    if not HAS_M2:
        return None, None

    print("\nBuilding distributed LSH index (M2 pipeline)...")

    sharder = IndexSharding(num_nodes=4, virtual_nodes=150)
    sharder.distribute(features)

    config = LSHConfig(num_tables=5, hash_width=16, num_nodes=4, num_threads=2)
    index = DistributedLSHIndex(config)
    index.build_from_features(features, consistent_hash_ring=sharder.ring)

    processor = DistributedQueryProcessor(
        lsh_index=index,
        feature_matrix=features,
        sharder=sharder,
        num_nodes=4,
        use_multiprocess=False,
    )

    return index, processor


def demo_query(processor, features: np.ndarray) -> None:
    """Run a sample query and print results."""
    print("\n--- Sample Query (image_id=0) ---")
    results, stats = processor.query(features[0], k=5, collect_stats=True)
    print(f"Top-5 results:")
    for r in results:
        print(f"  image_id={r.image_id:<6}  similarity={r.score:.4f}  node={r.node_id}")
    processor.print_stats(stats)


#main

def main() -> None:

    print("Inter-Node Communication Optimization  —  Milestone 3")

    # 1. Load data
    features, image_ids = load_features(
        features_path="fused_features.npy",
        metadata_path="master_alignment_map.csv",
    )

    # 2. Fit scalar quantizer
    print("\nFitting scalar quantizer...")
    quantizer = ScalarQuantizer()
    quantizer.fit(features)

    # Show compression stats on one real vector
    sample = features[0]
    compressed_sample = quantizer.compress(sample)
    print(f"\nCompression stats (single vector, dim={len(sample)}):")
    print(f"  Original size  : {sample.nbytes} bytes (float32)")
    print(f"  Compressed size: {compressed_sample.nbytes} bytes (uint8)")
    print(f"  Ratio          : {quantizer.compression_ratio(sample):.0f}x smaller")
    print(f"  Accuracy loss  : {(1 - quantizer.measure_accuracy_loss(sample)) * 100:.4f}%")

    # 3. Build M2 pipeline and run a demo query
    index, processor = build_m2_pipeline(features)
    if processor is not None:
        demo_query(processor, features)

    # 4. Run all profiling
    profiler = CommunicationProfiler(compressor=quantizer)

    profiler.profile_serialization(
        vectors=features[:50],
        num_trials=100,
        network_bandwidth_mbps=100.0,
    )

    profiler.profile_batch_throughput(
        all_vectors=features[:100],
        batch_sizes=[1, 5, 10, 20, 50, 100],
        num_trials=50,
    )

    profiler.profile_pickle_vs_numpy(
        vectors=features[:50],
        num_trials=100,
    )

    # 5. Save report
    profiler.save_report("final_report.txt")

    print("\n")
    print("Done. Files produced:")
    print("  compressor.py     — ScalarQuantizer class")
    print("  profiler.py       — CommunicationProfiler class")
    print("  main.py           — this script (runs everything)")
    print("  final_report.txt  — results written by profiler")


if __name__ == "__main__":
    main()