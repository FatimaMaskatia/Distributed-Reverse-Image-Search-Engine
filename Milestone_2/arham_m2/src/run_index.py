from __future__ import annotations

import argparse
from pathlib import Path

from lsh_index import DistributedLSHIndex, LSHConfig
from utils import load_features_and_metadata, load_consistent_hash_ring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 2: Distributed LSH Index Construction"
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to features.npy from M1",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to metadata.csv from M1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for index artifacts",
    )
    parser.add_argument(
        "--num-tables",
        type=int,
        default=10,
        help="Number of LSH hash tables",
    )
    parser.add_argument(
        "--hash-width",
        type=int,
        default=32,
        help="Number of bits per hash table",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=4,
        help="Number of worker nodes",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for parallel insertion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print(f"Loading features and metadata...")
    features, image_paths = load_features_and_metadata(args.features, args.metadata)
    print(f"  Loaded {len(features)} images, {features.shape[1]} features per image")
    
    config = LSHConfig(
        num_tables=args.num_tables,
        hash_width=args.hash_width,
        num_nodes=args.num_nodes,
        num_threads=args.num_threads,
    )
    
    print(f"\nBuilding distributed LSH index...")
    index = DistributedLSHIndex(config)
    
    # Try to load Raviha's consistent hash ring
    hash_ring = load_consistent_hash_ring()
    
    index.build_from_features(features, args.metadata, hash_ring)
    
    # Save index metadata (structure, not data, since index is in-memory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "index_summary.txt", "w") as f:
        f.write(f"LSH Index Summary\n")
        f.write(f"==================\n")
        f.write(f"Images indexed: {len(features)}\n")
        f.write(f"Feature dimension: {features.shape[1]}\n")
        f.write(f"Hash tables: {config.num_tables}\n")
        f.write(f"Hash width (bits): {config.hash_width}\n")
        f.write(f"Worker nodes: {config.num_nodes}\n")
    
    print(f"\nIndex construction complete!")
    print(f"Summary saved to: {args.output_dir / 'index_summary.txt'}")


if __name__ == "__main__":
    main()
