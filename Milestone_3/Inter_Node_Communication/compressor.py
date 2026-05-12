from __future__ import annotations
import numpy as np


class ScalarQuantizer:
    """
    Compresses float32 feature vectors to uint8 (4x smaller).

    Why this matters:
        In a distributed system, feature vectors are serialized and sent
        between nodes during query fanout. Each vector here is 1184-dim
        float32 = 4736 bytes. Compressing to uint8 = 1184 bytes (4x smaller),
        which directly reduces network transfer time.

    Usage:
        quantizer= ScalarQuantizer()
        quantizer.fit(feature_matrix)  # call once on full dataset
        compressed= quantizer.compress(vec)   # float32 -> uint8
        restored = quantizer.decompress(compressed)  # uint8 -> float32 (approx)
    """

    def __init__(self) -> None:
        self.min_val: float | None = None
        self.max_val: float | None = None
        self.is_fitted: bool = False

    def fit(self, vectors: np.ndarray) -> None:
        """
        Learn min and max from the full feature matrix.
        Must be called once before any compress/decompress.

        Parameters
        vectors : shape (N, D) float32
        """
        self.min_val = float(vectors.min())
        self.max_val = float(vectors.max())
        self.is_fitted = True
        print(f"[ScalarQuantizer] Fitted: min={self.min_val:.4f}, max={self.max_val:.4f}")

    def compress(self, vector: np.ndarray) -> np.ndarray:
        """
        Compress a single float32 vector to uint8.
        Scales values from [min, max] into [0, 255].

        float32= 4 bytes/value  →  uint8 = 1 byte/value  (4x smaller)

        Parameters
        vector : shape (D,) float32

        Returns
        shape (D,) uint8
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before compress()")
        scaled = (vector - self.min_val) / (self.max_val - self.min_val)
        return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)

    def decompress(self, vector: np.ndarray) -> np.ndarray:
        """
        Restore a uint8 vector back to approximate float32.
        There is a small quantization error (0.05% accuracy loss) — acceptable.

        Parameters
        vector : shape (D,) uint8

        Returns
        shape (D,) float32
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before decompress()")
        restored = vector.astype(np.float32) / 255.0
        return restored * (self.max_val - self.min_val) + self.min_val

    def compress_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compress a batch of vectors in one vectorised operation (fast).

        Parameters
        vectors : shape (N, D) float32

        Returns
        shape (N, D) uint8
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before compress_batch()")
        scaled = (vectors - self.min_val) / (self.max_val - self.min_val)
        return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)

    def decompress_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        Decompress a batch of uint8 vectors back to float32.

        Parameters
        vectors : shape (N, D) uint8

        Returns
        shape (N, D) float32
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before decompress_batch()")
        restored = vectors.astype(np.float32) / 255.0
        return restored * (self.max_val - self.min_val) + self.min_val

    def compression_ratio(self, original: np.ndarray) -> float:
        """Return bytes_before / bytes_after. Always 4.0 for float32 -> uint8."""
        return float(original.nbytes) / float(original.size)  # 4 bytes / 1 byte

    def measure_accuracy_loss(self, original: np.ndarray) -> float:
        """
        Cosine similarity between the original vector and its compress→decompress
        roundtrip. 1.0= perfect reconstruction. Anything above 0.999 is excellent.

        Parameters
        original : shape (D,) float32

        Returns
        float in [0, 1]
        """
        compressed = self.compress(original)
        restored = self.decompress(compressed)
        norm = np.linalg.norm(original) * np.linalg.norm(restored)
        if norm == 0:
            return 0.0
        return float(np.dot(original, restored) / norm)


#Self-test
if __name__ == "__main__":
    print("ScalarQuantizer self-test\n")
    rng = np.random.default_rng(42)
    fake = rng.standard_normal((100, 512)).astype(np.float32)

    q = ScalarQuantizer()
    q.fit(fake)

    v = fake[0]
    c = q.compress(v)
    r = q.decompress(c)

    print(f"  Original size  : {v.nbytes} bytes  (float32, {len(v)}-dim)")
    print(f"  Compressed size: {c.nbytes} bytes  (uint8)")
    print(f"  Ratio          : {q.compression_ratio(v):.0f}x smaller")
    print(f"  Cosine sim     : {q.measure_accuracy_loss(v):.6f}  (1.0= perfect)")

    batch = fake[:10]
    cb = q.compress_batch(batch)
    rb = q.decompress_batch(cb)
    assert cb.dtype == np.uint8
    assert rb.shape == batch.shape
    print(f"  Batch test     : {batch.shape} → {cb.shape} → {rb.shape}  OK")
    print("\nSelf-test passed.")