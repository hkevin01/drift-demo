"""
# ============================================================
# FILE: src/data/generate_stream.py
# ID: DATA-001
# Purpose: Generate a synthetic binary-classification dataset
#          and simulate a time-ordered stream of batches.
# Requirement: Produce reproducible initial training data and
#              a sequence of stream batches that can be mutated
#              by drift_injector.py before consumption.
# ============================================================
"""

from __future__ import annotations

import numpy as np
from typing import Generator, Tuple, List

from utils.config import (
    RANDOM_SEED, N_FEATURES, N_INITIAL_SAMPLES,
    N_STREAM_BATCHES, BATCH_SIZE,
)


def _make_blobs(
    n_samples: int,
    n_features: int,
    class_sep: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ID: DATA-002
    Purpose: Generate two-class Gaussian blobs centred at +/-class_sep/2.
    Inputs:
        n_samples  - total samples (split evenly between classes).
        n_features - number of continuous input features.
        class_sep  - distance between class centroids on feature-0.
        rng        - numpy random Generator for reproducibility.
    Outputs:
        X - float64 array (n_samples, n_features).
        y - int32 array  (n_samples,) with values in {0, 1}.
    Preconditions: n_samples >= 2, n_features >= 1.
    """
    half = n_samples // 2
    X0 = rng.standard_normal((half, n_features))
    X0[:, 0] -= class_sep / 2.0

    X1 = rng.standard_normal((n_samples - half, n_features))
    X1[:, 0] += class_sep / 2.0

    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(half, dtype=np.int32),
                        np.ones(n_samples - half, dtype=np.int32)])

    shuffle = rng.permutation(n_samples)
    return X[shuffle], y[shuffle]


def get_initial_data(
    class_sep: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ID: DATA-003
    Purpose: Return the fixed initial training dataset.
    Inputs:  class_sep - separation between class centroids (default 2.0).
    Outputs: (X_train, y_train) numpy arrays.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    return _make_blobs(N_INITIAL_SAMPLES, N_FEATURES, class_sep, rng)


def stream_batches(
    class_sep: float = 2.0,
) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
    """
    ID: DATA-004
    Purpose: Yield (batch_index, X_batch, y_batch) tuples in time order.
             Each batch represents one monitoring window in production.
    Inputs:  class_sep - baseline class separation for un-drifted batches.
    Outputs: Generator yielding (int, ndarray, ndarray).
    Notes:   Drift injection is handled externally by drift_injector.py;
             this generator produces clean reference-distribution data.
    """
    rng = np.random.default_rng(RANDOM_SEED + 1)
    for batch_idx in range(N_STREAM_BATCHES):
        X, y = _make_blobs(BATCH_SIZE, N_FEATURES, class_sep, rng)
        yield batch_idx, X, y


def collect_stream_as_list(
    class_sep: float = 2.0,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """
    ID: DATA-005
    Purpose: Materialise the full stream into a list (for inspection/testing).
    Outputs: list of (batch_index, X, y) tuples.
    """
    return list(stream_batches(class_sep))
