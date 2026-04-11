from __future__ import annotations

"""
Milestone 2: Test suite for query processing
Tests covered:

1. test_single_query_returns_results: basic happy-path query
2. test_fanout_reaches_all_nodes: every partition is queried
3. test_aggregation_deduplicates_candidates: merged set has no duplicates
4. test_reranking_sorted_by_similarity: results in descending cosine sim
5. test_known_similar_image_retrieved: inserted twin is found
6. test_top_k_limit_respected: never returns more than k
7. test_concurrent_queries_correct: parallel queries produce consistent results
8. test_empty_index_returns_empty: graceful handling of empty index
9. test_batch_query: batch API returns correct number of result sets
10. test_query_stats_populated: timing stats are filled in
"""

import sys
import time
import threading
from pathlib import Path
from typing import List

import numpy as np

_SRC = Path(__file__).parent.parent / "src"
_ARHAM = Path(__file__).parent.parent.parent / "Distributed_LSH_Index_Construction" / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ARHAM))
_RAVIHA = Path(__file__).parent.parent.parent / "Index_Sharding_and_Load_Balancing"
sys.path.insert(0, str(_RAVIHA))

from query_processor import DistributedQueryProcessor, QueryResult, QueryStats
from lsh_index import DistributedLSHIndex, LSHConfig

def _make_index_and_features(
    num_images: int = 40,
    dim: int = 512,
    num_nodes: int = 4,
    num_tables: int = 4,
    hash_width: int = 8,
    seed: int = 0,
) -> tuple:
    """Return (DistributedLSHIndex, feature_matrix) with normalised vectors."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((num_images, dim)).astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features /= norms

    config = LSHConfig(
        num_tables=num_tables,
        hash_width=hash_width,
        num_nodes=num_nodes,
        num_threads=2,
    )
    index = DistributedLSHIndex(config)
    index.build_from_features(features)
    return index, features


def _make_processor(
    num_images: int = 40,
    num_nodes: int = 4,
    **kwargs,
) -> tuple:
    index, features = _make_index_and_features(num_images=num_images, num_nodes=num_nodes, **kwargs)
    
    sharder = None
    try:
        from sharding import IndexSharding
        sharder = IndexSharding(num_nodes=num_nodes, virtual_nodes=150)
        sharder.distribute(features)
    except (ImportError, ModuleNotFoundError):
        print("  (Raviha's sharder not found, using modulo fallback)")

    processor = DistributedQueryProcessor(
        lsh_index=index,
        feature_matrix=features,
        sharder=sharder,
        num_nodes=num_nodes,
        use_multiprocess=False,
    )
    return processor, features


def test_single_query_returns_results() -> None:
    """A query against a populated index returns at least one result."""
    processor, features = _make_processor()
    results, _ = processor.query(features[0], k=5)
    assert len(results) > 0, "Expected at least one result"
    print("test_single_query_returns_results")


def test_fanout_reaches_all_nodes() -> None:
    """
    Internal fanout should produce a per-node candidate dict
    with an entry for every node, even if some are empty.
    """
    processor, features = _make_processor(num_nodes=4)
    per_node = processor._fanout_inprocess(features[0])
    assert set(per_node.keys()) == {0, 1, 2, 3}, (
        f"Expected keys 0-3, got {set(per_node.keys())}"
    )
    print("test_fanout_reaches_all_nodes")


def test_aggregation_deduplicates_candidates() -> None:
    """
    Manually craft per-node candidate sets with overlapping IDs
    and verify the merged dict has no duplicates.
    """
    processor, _ = _make_processor()
    per_node = {
        0: {1, 2, 3},
        1: {3, 4, 5}, # 3 appears in both node 0 and node 1
        2: {5, 6},
        3: set(),
    }
    merged = processor._aggregate(per_node)

    # Each image_id should appear exactly once
    assert len(merged) == len(set(merged.keys())), "Duplicates found in merged dict"

    # Specifically, ID 3 should appear exactly once
    assert 3 in merged, "Image 3 missing from merged result"

    # All unique IDs across input sets must be present
    all_ids = {1, 2, 3, 4, 5, 6}
    assert all_ids == set(merged.keys()), f"Merged IDs don't match expected: {set(merged.keys())}"
    print("test_aggregation_deduplicates_candidates")


def test_reranking_sorted_by_similarity() -> None:
    """Results must be in strictly descending order of cosine similarity."""
    processor, features = _make_processor(num_images=60)
    results, _ = processor.query(features[5], k=10)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"Results not sorted by descending similarity: {scores}"
    )
    print("test_reranking_sorted_by_similarity")


def test_known_similar_image_retrieved() -> None:
    """
    Insert an image and a near-duplicate (small Gaussian noise).
    The near-duplicate should appear in the top results when the
    original is used as the query.
    """
    rng = np.random.default_rng(7)
    dim = 512
    num_images = 30

    features = rng.standard_normal((num_images, dim)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    # Create a near-duplicate of image 0 and append it as image num_images
    twin = features[0] + rng.standard_normal(dim).astype(np.float32) * 0.05
    twin /= np.linalg.norm(twin)
    features_with_twin = np.vstack([features, twin[np.newaxis, :]])

    config = LSHConfig(num_tables=10, hash_width=8, num_nodes=4, num_threads=2)
    index = DistributedLSHIndex(config)
    index.build_from_features(features_with_twin)

    processor = DistributedQueryProcessor(
        lsh_index=index,
        feature_matrix=features_with_twin,
        num_nodes=4,
        use_multiprocess=False,
    )

    results, _ = processor.query(features_with_twin[0], k=5)
    retrieved_ids = {r.image_id for r in results}

    twin_id = num_images  # index of the twin
    assert twin_id in retrieved_ids, (
        f"Near-duplicate (id={twin_id}) not found in top-5 results: {retrieved_ids}"
    )
    print("test_known_similar_image_retrieved")


def test_top_k_limit_respected() -> None:
    """Query must never return more than k results."""
    processor, features = _make_processor(num_images=80)
    for k in [1, 5, 10, 20]:
        results, _ = processor.query(features[0], k=k)
        assert len(results) <= k, f"Got {len(results)} results for k={k}"
    print(" test_top_k_limit_respected")


def test_concurrent_queries_correct() -> None:
    """
    Run 10 queries concurrently from separate threads.
    Each query result must be non-empty and sorted correctly.
    Thread-safety of the in-process fanout is validated here.
    """
    processor, features = _make_processor(num_images=80, num_tables=6)
    errors: List[str] = []

    def run_query(idx: int) -> None:
        q_vec = features[idx % len(features)]
        results, _ = processor.query(q_vec, k=5)
        if len(results) == 0:
            errors.append(f"Thread {idx}: got 0 results")
            return
        scores = [r.score for r in results]
        if scores != sorted(scores, reverse=True):
            errors.append(f"Thread {idx}: results not sorted")

    threads = [threading.Thread(target=run_query, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent query errors:\n" + "\n".join(errors)
    print("test_concurrent_queries_correct")


def test_empty_index_returns_empty() -> None:
    """Querying an index with no images should return an empty list gracefully."""
    config = LSHConfig(num_tables=4, hash_width=8, num_nodes=2, num_threads=1)
    index = DistributedLSHIndex(config)
    # Build with zero images
    empty_features = np.zeros((0, 512), dtype=np.float32)
    # Manually set up empty partitions (already the default)

    # Use a fresh processor with a tiny non-empty feature matrix, won't overlap with index
    dummy_features = np.random.randn(1, 512).astype(np.float32)
    dummy_features /= np.linalg.norm(dummy_features, axis=1, keepdims=True)

    processor = DistributedQueryProcessor(
        lsh_index=index,
        feature_matrix=dummy_features,
        num_nodes=2,
        use_multiprocess=False,
    )
    query_vec = dummy_features[0]
    results, _ = processor.query(query_vec, k=5)
    assert results == [], f"Expected empty results, got {results}"
    print("✓ test_empty_index_returns_empty")


def test_batch_query() -> None:
    """batch_query should return exactly Q result-sets for Q input vectors."""
    processor, features = _make_processor(num_images=50)
    Q = 8
    query_vecs = features[:Q]
    batch_results = processor.batch_query(query_vecs, k=5, collect_stats=True)
    assert len(batch_results) == Q, f"Expected {Q} result sets, got {len(batch_results)}"
    for i, (results, stats) in enumerate(batch_results):
        assert isinstance(results, list), f"Query {i}: results is not a list"
        assert isinstance(stats, QueryStats), f"Query {i}: stats is not a QueryStats"
    print(" test_batch_query")


def test_query_stats_populated() -> None:
    """When collect_stats=True, all timing fields must be >= 0."""
    processor, features = _make_processor()
    _, stats = processor.query(features[0], k=5, collect_stats=True)
    assert stats is not None, "Stats should not be None when collect_stats=True"
    assert stats.total_ms       >= 0, "total_ms should be non-negative"
    assert stats.fanout_ms      >= 0, "fanout_ms should be non-negative"
    assert stats.aggregation_ms >= 0, "aggregation_ms should be non-negative"
    assert stats.reranking_ms   >= 0, "reranking_ms should be non-negative"
    assert stats.num_candidates >= 0, "num_candidates should be non-negative"
    assert stats.num_nodes_queried > 0, "num_nodes_queried should be positive"
    print("test_query_stats_populated")


if __name__ == "__main__":
    print("Running M2 query processor tests:\n")

    tests = [
        test_single_query_returns_results,
        test_fanout_reaches_all_nodes,
        test_aggregation_deduplicates_candidates,
        test_reranking_sorted_by_similarity,
        test_known_similar_image_retrieved,
        test_top_k_limit_respected,
        test_concurrent_queries_correct,
        test_empty_index_returns_empty,
        test_batch_query,
        test_query_stats_populated,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f" {test_fn.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{'='*45}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)