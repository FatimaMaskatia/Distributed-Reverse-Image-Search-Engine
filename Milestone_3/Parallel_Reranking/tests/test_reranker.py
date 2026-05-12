import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reranker import ParallelReRanker, ReRankConfig, CandidateSetTuner


def test_reranker_initialization():
    """Test that re-ranker initializes without errors."""
    features = np.random.randn(100, 512).astype(np.float32)
    config = ReRankConfig(num_threads=2)
    reranker = ParallelReRanker(config, features)
    assert reranker.feature_vectors.shape == (100, 512)


def test_rerank_single():
    """Test single re-ranking operation."""
    features = np.random.randn(100, 512).astype(np.float32)
    config = ReRankConfig(similarity_metric="cosine")
    reranker = ParallelReRanker(config, features)
    
    query_vec = features[0]
    candidates = list(range(10, 30))
    
    results = reranker.rerank_vectorized(query_vec, candidates)
    
    assert len(results) == 20
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    # Results should be sorted by score (descending)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_batch_parallel():
    """Test batch re-ranking with thread pool."""
    features = np.random.randn(100, 512).astype(np.float32)
    config = ReRankConfig(num_threads=4)
    reranker = ParallelReRanker(config, features)
    
    query_vectors = features[:5]
    candidate_sets = [[i + 10 for i in range(20)] for _ in range(5)]
    
    results = reranker.rerank_batch_parallel(candidate_sets, query_vectors)
    
    assert len(results) == 5
    for r in results:
        assert len(r) == 20


def test_tuner():
    """Test candidate set tuning."""
    features = np.random.randn(50, 512).astype(np.float32)
    config = ReRankConfig(
        num_threads=2,
        candidate_set_sizes=[5, 10, 15],
    )
    reranker = ParallelReRanker(config, features)
    tuner = CandidateSetTuner(reranker, config)
    
    query_vectors = features[:10]
    candidate_sets = [[i + 5 for i in range(20)] for _ in range(10)]
    
    metrics = tuner.tune(query_vectors, candidate_sets)
    
    assert len(metrics) == 3
    assert 5 in metrics and 10 in metrics and 15 in metrics
    assert "latency_total_sec" in metrics[5]
    assert "latency_per_query_ms" in metrics[5]


def test_l2_similarity():
    """Test L2 distance metric."""
    features = np.random.randn(50, 512).astype(np.float32)
    config = ReRankConfig(similarity_metric="l2")
    reranker = ParallelReRanker(config, features)
    
    query_vec = features[0]
    candidates = list(range(10, 20))
    
    results = reranker.rerank_vectorized(query_vec, candidates)
    
    assert len(results) == 10
    # For L2, lower distance = closer (higher score since we negate)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    print("Running re-ranker tests...\n")
    
    test_reranker_initialization()
    print("✓ test_reranker_initialization passed")
    
    test_rerank_single()
    print("✓ test_rerank_single passed")
    
    test_rerank_batch_parallel()
    print("✓ test_rerank_batch_parallel passed")
    
    test_tuner()
    print("✓ test_tuner passed")
    
    test_l2_similarity()
    print("✓ test_l2_similarity passed")
    
    print("\nAll tests passed!")
