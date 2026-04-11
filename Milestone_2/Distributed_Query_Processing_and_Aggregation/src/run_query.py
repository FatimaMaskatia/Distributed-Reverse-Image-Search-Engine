from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent.parent / "Distributed_LSH_Index_Construction" / "src"))
sys.path.insert(0, str(_SRC.parent.parent / "Index_Sharding_and_Load_Balancing"))

from query_processor import DistributedQueryProcessor
from lsh_index import DistributedLSHIndex, LSHConfig
from utils import load_features_and_metadata, load_consistent_hash_ring

def build_index(features: np.ndarray, config: LSHConfig, hash_ring) -> DistributedLSHIndex:
    index = DistributedLSHIndex(config)
    index.build_from_features(features, consistent_hash_ring=hash_ring)
    return index

def load_sharder(features: np.ndarray, num_nodes: int):
    """Try to load Raviha's IndexSharding; return None on failure."""
    try:
        from sharding import IndexSharding
        sharder = IndexSharding(num_nodes=num_nodes, virtual_nodes=150)
        sharder.distribute(features)
        return sharder
    except (ImportError, ModuleNotFoundError):
        print("  Warning: Could not import  IndexSharding; using index partitions")
        return None


def percentile(values: list, p: float) -> float:
    arr = sorted(values)
    idx = int(len(arr) * p / 100)
    return arr[min(idx, len(arr) - 1)]

# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 2: Distributed Query Processing"
    )
    parser.add_argument("--features",    type=Path, required=True, help="Path to features.npy from M1")
    parser.add_argument("--metadata",    type=Path, required=True, help="Path to metadata.csv from M1")
    parser.add_argument("--output-dir",  type=Path, default=Path("outputs"))

    # Index params
    parser.add_argument("--num-tables",  type=int, default=10)
    parser.add_argument("--hash-width",  type=int, default=32)
    parser.add_argument("--num-nodes",   type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=4)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query-id",     type=int,  help="Use features[query_id] as the query vector")
    mode.add_argument("--random-query", action="store_true", help="Use a random normalised vector")
    mode.add_argument("--benchmark",    action="store_true", help="Run N random queries and report latencies")

    parser.add_argument("--num-queries", type=int, default=50,  help="Number of queries for --benchmark")
    parser.add_argument("--top-k",       type=int, default=10,  help="Number of results to return")
    parser.add_argument("--multiprocess", action="store_true",  help="Use real worker processes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    #Load data 
    print("Loading features and metadata...")
    import numpy as np
    from pathlib import Path

    features = np.load(args.features, allow_pickle=False).astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    features = features / norms

# Handle both .npy and .csv metadata formats
    if str(args.metadata).endswith('.npy'):
        image_paths = list(np.load(args.metadata, allow_pickle=True))
    else:
        image_paths = []
        with open(args.metadata, "r") as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    image_paths.append(parts[1])

    assert len(image_paths) == len(features), \
    f"Metadata/features count mismatch: {len(image_paths)} vs {len(features)}"

    print(f"  {len(features)} images, {features.shape[1]}-dim features")

    #Build index
    config = LSHConfig(
        num_tables=args.num_tables,
        hash_width=args.hash_width,
        num_nodes=args.num_nodes,
        num_threads=args.num_threads,
    )
    hash_ring = load_consistent_hash_ring()
    print("\nBuilding LSH index:")
    index = build_index(features, config, hash_ring)

    #Load sharder
    sharder = load_sharder(features, args.num_nodes)

    #Build query processor
    print("\nInitialising query processor:")
    processor = DistributedQueryProcessor(
        lsh_index=index,
        feature_matrix=features,
        sharder=sharder,
        num_nodes=args.num_nodes,
        use_multiprocess=args.multiprocess,
    )

    # Run query / benchmark
    if args.benchmark:
        _run_benchmark(processor, features, args)
    else:
        _run_single_query(processor, features, image_paths, args)

    processor.shutdown()


def _run_single_query(processor, features, image_paths, args) -> None:
    if args.query_id is not None:
        if args.query_id >= len(features):
            print(f"Error: query-id {args.query_id} out of range (max {len(features)-1})")
            sys.exit(1)
        q_vec = features[args.query_id]
        print(f"\nQuerying with image #{args.query_id}: {image_paths[args.query_id]}")
    else:
        rng = np.random.default_rng(42)
        q_vec = rng.standard_normal(features.shape[1]).astype(np.float32)
        q_vec /= np.linalg.norm(q_vec)
        print("\nQuerying with a random normalised vector")

    results, stats = processor.query(q_vec, k=args.top_k, collect_stats=True)

    print(f"\nTop-{args.top_k} results:")
    print(f"{'Rank':<6} {'ImageID':<10} {'Node':<6} {'CosineSim':<12} {'Path'}")
    print("-" * 70)
    for rank, r in enumerate(results, 1):
        path = image_paths[r.image_id] if r.image_id < len(image_paths) else "N/A"
        print(f"{rank:<6} {r.image_id:<10} {r.node_id:<6} {r.score:<12.6f} {path}")

    processor.print_stats(stats)


def _run_benchmark(processor, features, args) -> None:
    rng = np.random.default_rng(0)
    Q = args.num_queries
    print(f"\nBenchmark: {Q} random queries, top-{args.top_k}...")

    latencies = []
    for i in range(Q):
        q_vec = rng.standard_normal(features.shape[1]).astype(np.float32)
        q_vec /= np.linalg.norm(q_vec)
        _, stats = processor.query(q_vec, k=args.top_k, collect_stats=True)
        latencies.append(stats.total_ms)
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{Q} queries...")

    print(f"\n=== Benchmark Results ({Q} queries) ===")
    print(f"  P50  : {percentile(latencies, 50):.3f} ms")
    print(f"  P95  : {percentile(latencies, 95):.3f} ms")
    print(f"  P99  : {percentile(latencies, 99):.3f} ms")
    print(f"  Mean : {sum(latencies)/len(latencies):.3f} ms")
    print(f"  Min  : {min(latencies):.3f} ms")
    print(f"  Max  : {max(latencies):.3f} ms")

    # Save results
    out = args.output_dir / "benchmark_results.txt"
    with open(out, "w") as f:
        f.write(f"Benchmark Results\n")
        f.write(f"\n")
        f.write(f"Queries       : {Q}\n")
        f.write(f"Top-k         : {args.top_k}\n")
        f.write(f"Nodes         : {args.num_nodes}\n")
        f.write(f"Tables        : {args.num_tables}\n")
        f.write(f"P50  (ms)     : {percentile(latencies, 50):.3f}\n")
        f.write(f"P95  (ms)     : {percentile(latencies, 95):.3f}\n")
        f.write(f"P99  (ms)     : {percentile(latencies, 99):.3f}\n")
        f.write(f"Mean (ms)     : {sum(latencies)/len(latencies):.3f}\n")
    print(f"\nResults saved to: {out}")

if __name__ == "__main__":
    main()