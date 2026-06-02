"""
# ============================================================
# FILE: src/model/classifier.py
# ID: MDL-001
# Purpose: Thin wrapper around a scikit-learn RandomForest that
#          exposes train / predict / evaluate / persist methods.
# ============================================================
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Optional, Tuple

from utils.config import MODEL_PATH, RANDOM_SEED


class DriftAwareClassifier:
    """
    ID: MDL-002
    Purpose: Encapsulate a RandomForest with persistence and scoring helpers
             needed by the drift monitoring pipeline.
    Preconditions: scikit-learn >= 1.3 installed.
    """

    def __init__(self, n_estimators: int = 100) -> None:
        """
        ID: MDL-003
        Inputs: n_estimators - number of trees (default 100).
        """
        self.n_estimators = n_estimators
        self._model: Optional[RandomForestClassifier] = None
        self._baseline_accuracy: float = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray) -> "DriftAwareClassifier":
        """
        ID: MDL-004
        Purpose: Fit a new RandomForest on the supplied data.
        Inputs:
            X - float64 (n_samples, n_features).
            y - int32   (n_samples,).
        Outputs: self (for chaining).
        Side Effects: replaces self._model.
        """
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        self._baseline_accuracy = accuracy_score(y, self._model.predict(X))
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        ID: MDL-005
        Preconditions: train() must have been called.
        Outputs: int32 array (n_samples,).
        Failure modes: raises RuntimeError if model not fitted.
        """
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        ID: MDL-006
        Outputs: float64 array (n_samples, n_classes).
        """
        self._check_fitted()
        return self._model.predict_proba(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        ID: MDL-007
        Purpose: Compute accuracy and accuracy drop vs baseline.
        Outputs: (accuracy, accuracy_drop) both floats in [0, 1].
        """
        self._check_fitted()
        acc = accuracy_score(y, self.predict(X))
        drop = max(0.0, self._baseline_accuracy - acc)
        return acc, drop

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str = MODEL_PATH) -> None:
        """
        ID: MDL-008
        Purpose: Serialise the fitted model to disk with joblib.
        Failure modes: raises ValueError if model not fitted.
        """
        self._check_fitted()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "DriftAwareClassifier":
        """
        ID: MDL-009
        Purpose: Deserialise a previously saved DriftAwareClassifier.
        Outputs: DriftAwareClassifier instance.
        Failure modes: raises FileNotFoundError if path missing.
        """
        return joblib.load(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

    @property
    def baseline_accuracy(self) -> float:
        return self._baseline_accuracy
