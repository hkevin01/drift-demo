"""
# ============================================================
# FILE: src/model/retrain.py
# ID: RTN-001
# Purpose: Retraining utilities for drift response.
# ============================================================
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from model.classifier import DriftAwareClassifier


def build_retrain_set(
    history: List[Tuple[np.ndarray, np.ndarray]],
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ID: RTN-002
    Purpose: Stack the most recent `window` batches into one retraining set.
    Inputs:
        history - list[(X_batch, y_batch)] in chronological order.
        window  - number of latest batches to include.
    Outputs:
        X_retrain - concatenated feature matrix.
        y_retrain - concatenated label vector.
    Failure modes: raises ValueError when history empty.
    """
    if not history:
        raise ValueError("Cannot build retrain set from empty history.")

    selected = history[-window:] if len(history) >= window else history
    X = np.vstack([x for x, _ in selected])
    y = np.concatenate([yy for _, yy in selected])
    return X, y


def retrain_model(
    clf: DriftAwareClassifier,
    history: List[Tuple[np.ndarray, np.ndarray]],
    window: int,
) -> DriftAwareClassifier:
    """
    ID: RTN-003
    Purpose: Retrain classifier in-place using the recent data window.
    Inputs:
        clf     - existing model wrapper to re-fit.
        history - recent batch history.
        window  - number of batches to use.
    Outputs: retrained classifier (same instance).
    """
    X_new, y_new = build_retrain_set(history, window)
    clf.train(X_new, y_new)
    return clf
