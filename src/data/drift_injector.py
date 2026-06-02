"""
# ============================================================
# FILE: src/data/drift_injector.py
# ID: INJ-001
# Purpose: Apply controlled data drift, concept drift, and
#          combined drift to a stream batch.
# Requirement: Each injector function must be idempotent on
#              un-affected batches (i.e. return data unchanged
#              when the batch index is before the drift start).
# ============================================================
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from utils.config import (
    DATA_DRIFT_START_BATCH, CONCEPT_DRIFT_START_BATCH,
    DRIFT_RAMP_BATCHES, N_FEATURES,
)


def _ramp(batch_idx: int, start: int, ramp: int) -> float:
    """
    ID: INJ-002
    Purpose: Compute a [0, 1] ramp factor for gradual drift onset.
    Inputs:
        batch_idx - current batch index.
        start     - batch index where drift begins.
        ramp      - number of batches over which drift ramps to full strength.
    Outputs: float in [0.0, 1.0].
    """
    if batch_idx < start:
        return 0.0
    if batch_idx >= start + ramp:
        return 1.0
    return (batch_idx - start) / float(ramp)


def inject_data_drift(
    batch_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    shift_magnitude: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ID: INJ-003
    Purpose: Shift the input feature distribution to simulate covariate drift.
             Features 0-2 are shifted proportionally to the ramp factor.
    Inputs:
        batch_idx       - current batch index.
        X               - float64 array (n_samples, n_features).
        y               - int32 label array (n_samples,).
        shift_magnitude - maximum mean shift applied to drifted features.
    Outputs: (X_drifted, y) - labels are unchanged.
    Side Effects: None (returns new array, does not mutate inputs).
    """
    alpha = _ramp(batch_idx, DATA_DRIFT_START_BATCH, DRIFT_RAMP_BATCHES)
    if alpha == 0.0:
        return X, y

    X_out = X.copy()
    n_drift_features = min(3, N_FEATURES)
    X_out[:, :n_drift_features] += alpha * shift_magnitude
    return X_out, y


def inject_concept_drift(
    batch_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    flip_fraction: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ID: INJ-004
    Purpose: Flip a fraction of labels to simulate concept drift - the
             decision boundary meaning changes over time.
    Inputs:
        batch_idx    - current batch index.
        X            - float64 array (unchanged, returned as-is).
        y            - int32 label array to be partially flipped.
        flip_fraction - fraction of labels flipped at full drift strength.
    Outputs: (X, y_drifted).
    Side Effects: None (returns new y array).
    """
    alpha = _ramp(batch_idx, CONCEPT_DRIFT_START_BATCH, DRIFT_RAMP_BATCHES)
    if alpha == 0.0:
        return X, y

    rng = np.random.default_rng(batch_idx + 9999)
    n_flip = int(alpha * flip_fraction * len(y))
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    y_out = y.copy()
    y_out[flip_idx] = 1 - y_out[flip_idx]
    return X, y_out


def inject_all_drift(
    batch_idx: int,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    ID: INJ-005
    Purpose: Apply both data drift and concept drift sequentially and
             return a metadata dict describing what was injected.
    Inputs:  batch_idx, X, y - as above.
    Outputs:
        X_out  - feature array after data drift.
        y_out  - label array after concept drift.
        meta   - dict with keys 'data_drift_alpha', 'concept_drift_alpha'.
    """
    data_alpha = _ramp(batch_idx, DATA_DRIFT_START_BATCH, DRIFT_RAMP_BATCHES)
    concept_alpha = _ramp(batch_idx, CONCEPT_DRIFT_START_BATCH, DRIFT_RAMP_BATCHES)

    X_out, y_tmp = inject_data_drift(batch_idx, X, y)
    X_out, y_out = inject_concept_drift(batch_idx, X_out, y_tmp)

    meta = {
        "data_drift_alpha": data_alpha,
        "concept_drift_alpha": concept_alpha,
    }
    return X_out, y_out, meta
