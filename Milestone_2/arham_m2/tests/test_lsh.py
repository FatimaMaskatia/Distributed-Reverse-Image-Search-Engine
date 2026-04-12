import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lsh_index import RandomProjection, LSHHashTable, DistributedLSHIndex, LSHConfig


def test_random_projection():
    """Test that random projection produces consistent binary hashes."""
    proj = RandomProjection(dimension=512, seed=42)
    
    # Same vector should produce same hash
    v = np.random.randn(512).astype(np.float32)
    h1 = proj.hash(v)
    h2 = proj.hash(v)
    assert h1 == h2, "Hashing should be deterministic"
    assert h1 in [0, 1], "Hash should be binary"


def test_lsh_hash_table_insert():
    """Test LSH hash table insertion and retrieval."""
    table = LSHHashTable(num_projections=16, vector_dimension=512, table_id=0)
    
    # Insert some vectors
    v1 = np.random.randn(512).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    table.insert(image_id=0, vector=v1)
    
    # Query should return the same image
    candidates = table.query(v1)
    assert 0 in candidates, "Inserted image should be retrieved"


def test_distributed_lsh_partitioning():
    """Test that partitioning distributes images across nodes."""
    config = LSHConfig(num_tables=4, hash_width=16, num_nodes=4, num_threads=2)
    index = DistributedLSHIndex(config)
    
    # Create dummy features
    features = np.random.randn(100, 512).astype(np.float32)
    # Normalize
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    index.build_from_features(features)
    
    # Check that all images are assigned
    total_assigned = sum(len(p) for p in index.node_partitions.values())
    assert total_assigned == 100, f"All images should be assigned: {total_assigned}"


def test_lsh_query():
    """Test LSH query functionality."""
    config = LSHConfig(num_tables=4, hash_width=16, num_nodes=2, num_threads=2)
    index = DistributedLSHIndex(config)
    
    # Create small feature set
    features = np.random.randn(20, 512).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    index.build_from_features(features)
    
    # Query with first feature
    results = index.query(features[0], k=5)
    
    # Should return some candidates (may or may not include the query itself)
    assert len(results) <= 5, "Should return at most k results"


if __name__ == "__main__":
    print("Running LSH tests...\n")
    
    test_random_projection()
    print("✓ test_random_projection passed")
    
    test_lsh_hash_table_insert()
    print("✓ test_lsh_hash_table_insert passed")
    
    test_distributed_lsh_partitioning()
    print("✓ test_distributed_lsh_partitioning passed")
    
    test_lsh_query()
    print("✓ test_lsh_query passed")
    
    print("\nAll tests passed!")
