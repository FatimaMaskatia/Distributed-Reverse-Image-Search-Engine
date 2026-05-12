from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import numpy as np

from reranker import ParallelReRanker, ReRankConfig, CandidateSetTuner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 3: Parallel Re-ranking Tuning"
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to features.npy from M1",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of random queries to generate",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=500,
        help="Number of candidates per query (before truncation)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for parallel re-ranking",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "l2"],
        help="Similarity metric",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print(f"Loading features...")
    features = np.load(args.features, allow_pickle=False).astype(np.float32)
    print(f"  Loaded {features.shape[0]} images, {features.shape[1]} features per image")
    
    config = ReRankConfig(
        num_threads=args.num_threads,
        similarity_metric=args.metric,
    )
    
    print(f"\nInitializing parallel re-ranker...")
    reranker = ParallelReRanker(config, features)
    
    # Generate random query vectors and candidates
    print(f"\nGenerating {args.num_queries} random queries...")
    query_indices = np.random.choice(features.shape[0], args.num_queries, replace=False)
    query_vectors = features[query_indices]
    
    # Generate random candidate sets
    print(f"Generating candidate sets ({args.num_candidates} per query)...")
    candidate_sets = []
    for _ in range(args.num_queries):
        candidates = list(np.random.choice(features.shape[0], args.num_candidates, replace=False))
        candidate_sets.append(candidates)
    
    # Tune candidate set sizes
    print(f"\nTuning candidate set sizes...")
    tuner = CandidateSetTuner(reranker, config)
    metrics = tuner.tune(query_vectors, candidate_sets)
    tuner.print_summary()
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_file = args.output_dir / "tuning_results.json"
    
    # Convert metrics to JSON-serializable format
    json_metrics = {}
    for k, v in metrics.items():
        json_metrics[str(k)] = {
            "latency_total_sec": float(v["latency_total_sec"]),
            "latency_per_query_ms": float(v["latency_per_query_ms"]),
            "num_queries": int(v["num_queries"]),
        }
    
    with open(results_file, "w") as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
