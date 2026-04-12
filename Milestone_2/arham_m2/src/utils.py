from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import sys


def load_features_and_metadata(
    features_path: Path,
    metadata_path: Path,
) -> Tuple[np.ndarray, list]:
    """
    Load feature vectors and metadata from M1 outputs.
    
    Returns:
        features: shape (N, D) float32 array
        image_paths: list of image paths
    """
    features = np.load(features_path, allow_pickle=False).astype(np.float32)
    
    image_paths = []
    with open(metadata_path, "r") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                image_paths.append(parts[1])
    
    assert len(image_paths) == len(features), \
        f"Metadata/features count mismatch: {len(image_paths)} vs {len(features)}"
    
    return features, image_paths


def load_consistent_hash_ring():
    """
    Attempt to load Raviha's ConsistentHashRing if available.
    Falls back to None if import fails.
    """
    try:
        # Adjust path to Raviha's module
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "milestone2_Raviha"))
        from consistent_hashing import ConsistentHashRing
        return ConsistentHashRing(num_nodes=4, virtual_nodes=150)
    except (ImportError, ModuleNotFoundError):
        print("  Warning: Could not import Raviha's ConsistentHashRing; using simple partitioning")
        return None
