from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class ReRankConfig:
    """Configuration for re-ranking."""
    num_threads: int = 4
    similarity_metric: str = "cosine"  # "cosine" or "l2"
    candidate_set_sizes: List[int] = None
    
    def __post_init__(self):
        if self.candidate_set_sizes is None:
            self.candidate_set_sizes = [10, 25, 50, 100, 250, 500]


class ParallelReRanker:
    """
    Re-ranks LSH candidate sets using exact similarity computation.
    Supports parallel re-ranking via thread pools.
    """
    
    def __init__(self, config: ReRankConfig, feature_vectors: np.ndarray):
        """
        Initialize re-ranker with feature vectors.
        
        Args:
            config: ReRankConfig object
            feature_vectors: shape (N, D) float32, L2-normalized or unnormalized
        """
        self.config = config
        self.feature_vectors = feature_vectors.astype(np.float32)
        
        # Normalize if using cosine similarity
        if config.similarity_metric == "cosine":
            norms = np.linalg.norm(self.feature_vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            self.feature_vectors = self.feature_vectors / norms
    
    def rerank_batch_parallel(
        self,
        candidates_batch: List[List[int]],
        query_vectors: np.ndarray,
    ) -> List[List[Tuple[int, float]]]:
        """
        Re-rank a batch of candidate sets using thread pool.
        
        Args:
            candidates_batch: list of candidate lists, each containing image IDs
            query_vectors: shape (B, D) float32, query feature vectors
        
        Returns:
            list of re-ranked results: [(image_id, score), ...] sorted by score
        """
        results = [None] * len(candidates_batch)
        
        def rerank_single(idx: int, query_vec: np.ndarray, candidates: List[int]) -> None:
            ranked = self._rerank_single(query_vec, candidates)
            results[idx] = ranked
        
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            futures = [
                executor.submit(rerank_single, i, query_vectors[i], cands)
                for i, cands in enumerate(candidates_batch)
            ]
            for future in as_completed(futures):
                future.result()
        
        return results
    
    def _rerank_single(
        self,
        query_vector: np.ndarray,
        candidate_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """
        Re-rank a single candidate set using exact similarity.
        
        Args:
            query_vector: shape (D,) float32
            candidate_ids: list of image IDs in candidate set
        
        Returns:
            list of (image_id, similarity_score) sorted by score (descending)
        """
        if not candidate_ids:
            return []
        
        candidate_vectors = self.feature_vectors[candidate_ids]
        
        if self.config.similarity_metric == "cosine":
            # Cosine similarity: dot product (vectors are already normalized)
            scores = np.dot(candidate_vectors, query_vector)
        elif self.config.similarity_metric == "l2":
            # L2 distance (negative so higher = closer)
            scores = -np.linalg.norm(candidate_vectors - query_vector, axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.config.similarity_metric}")
        
        # Sort by score (descending)
        sorted_indices = np.argsort(-scores)
        results = [(candidate_ids[idx], scores[idx]) for idx in sorted_indices]
        return results
    
    def rerank_vectorized(
        self,
        query_vector: np.ndarray,
        candidate_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """
        Re-rank using fully vectorized numpy (no parallelization, fastest for small sets).
        
        Args:
            query_vector: shape (D,) float32
            candidate_ids: list of image IDs
        
        Returns:
            list of (image_id, score) sorted by score
        """
        if not candidate_ids:
            return []
        
        candidate_vectors = self.feature_vectors[candidate_ids]
        
        if self.config.similarity_metric == "cosine":
            scores = np.dot(candidate_vectors, query_vector)
        elif self.config.similarity_metric == "l2":
            scores = -np.linalg.norm(candidate_vectors - query_vector, axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.config.similarity_metric}")
        
        sorted_indices = np.argsort(-scores)
        results = [(candidate_ids[idx], scores[idx]) for idx in sorted_indices]
        return results


class CandidateSetTuner:
    """
    Tunes candidate set size to find optimal accuracy-vs-latency tradeoff.
    """
    
    def __init__(self, reranker: ParallelReRanker, config: ReRankConfig):
        self.reranker = reranker
        self.config = config
        self.results = {}
    
    def tune(
        self,
        query_vectors: np.ndarray,
        candidate_sets: List[List[int]],
        ground_truth_ids: List[int] | None = None,
    ) -> dict:
        """
        Evaluate re-ranking latency and accuracy across candidate set sizes.
        
        Args:
            query_vectors: shape (Q, D) float32
            candidate_sets: list of candidate lists from LSH
            ground_truth_ids: optional true relevant IDs for accuracy measurement
        
        Returns:
            dict with metrics per candidate set size
        """
        metrics = {}
        
        for k in self.config.candidate_set_sizes:
            truncated_candidates = [cands[:k] for cands in candidate_sets]
            
            # Measure latency
            start = time.perf_counter()
            results = self.reranker.rerank_batch_parallel(truncated_candidates, query_vectors)
            latency = time.perf_counter() - start
            
            metrics[k] = {
                "latency_total_sec": latency,
                "latency_per_query_ms": (latency / len(query_vectors)) * 1000,
                "num_queries": len(query_vectors),
                "results_sample": results[0][:5] if results else [],
            }
            
            # Optional: accuracy if ground truth provided
            if ground_truth_ids is not None:
                recall = self._compute_recall(results, ground_truth_ids)
                metrics[k]["recall@k"] = recall
        
        self.results = metrics
        return metrics
    
    def _compute_recall(
        self,
        results: List[List[Tuple[int, float]]],
        ground_truth: List[int],
    ) -> float:
        """Compute average recall across queries."""
        recalls = []
        for query_results in results:
            retrieved_ids = [img_id for img_id, _ in query_results]
            num_relevant = sum(1 for img_id in retrieved_ids if img_id in ground_truth)
            recall = num_relevant / max(len(ground_truth), 1)
            recalls.append(recall)
        return np.mean(recalls) if recalls else 0.0
    
    def print_summary(self) -> None:
        """Print tuning results summary."""
        print("\n=== Candidate Set Size Tuning Results ===")
        print(f"{'K':<10} {'Latency (ms)':<15} {'Latency/Query (ms)':<20}")
        print("-" * 45)
        for k in sorted(self.results.keys()):
            metrics = self.results[k]
            total_latency = metrics["latency_total_sec"] * 1000
            per_query = metrics["latency_per_query_ms"]
            print(f"{k:<10} {total_latency:<15.2f} {per_query:<20.4f}")
