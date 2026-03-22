from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deep_features import l2_normalize_rows


def test_l2_normalize_rows_has_unit_norms():
    matrix = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32)
    normalized = l2_normalize_rows(matrix)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, np.ones_like(norms), atol=1e-6)


def test_l2_normalize_rows_handles_zeros():
    matrix = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    normalized = l2_normalize_rows(matrix)
    assert np.isfinite(normalized).all()
    assert np.allclose(normalized[1], np.array([1.0, 0.0], dtype=np.float32))
