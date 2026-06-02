"""
# ============================================================
# FILE: src/drift/detectors.py
# ID: DET-001
# Purpose: Drift detector suite combining KS, PSI, KL, and ADWIN.
# ============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import ks_2samp
from river.drift import ADWIN

from drift.metrics import population_stability_index, kl_divergence
from utils.config import (
    KS_P_VALUE_THRESHOLD,
    PSI_WARNING_THRESHOLD,
    PSI_ALERT_THRESHOLD,
    KL_ALERT_THRESHOLD,
    ADWIN_DELTA,
)


@dataclass
class DriftSignals:
    """
    ID: DET-002
    Purpose: Typed container for detector outputs per batch.
    """
    ks_stat: float
    ks_pvalue: float
    psi: float
    kl_div: float
    adwin_drift: bool
    data_drift_detected: bool
    concept_drift_detected: bool
    model_drift_detected: bool


class DriftDetectorSuite:
    """
    ID: DET-003
    Purpose: Maintain detector state and produce drift signals.
    Inputs:
        reference_X - baseline training feature matrix.
    Outputs:
        detect(...) -> DriftSignals per batch.
    """

    def __init__(self, reference_X: np.ndarray) -> None:
        self.reference_feature = reference_X[:, 0].astype(float)
        self._adwin = ADWIN(delta=ADWIN_DELTA)

    def detect(
        self,
        X_batch: np.ndarray,
        error_rate: float,
        accuracy_drop: float,
    ) -> DriftSignals:
        """
        ID: DET-004
        Purpose: Evaluate all detectors on the incoming batch.
        Inputs:
            X_batch       - current batch features.
            error_rate    - 1 - model accuracy for this batch.
            accuracy_drop - baseline_accuracy - current_accuracy.
        Outputs: DriftSignals dataclass.
        Logic ordering:
            1) Input selection
            2) Statistical tests
            3) ADWIN update
            4) Signal composition
        """
        x_curr = X_batch[:, 0].astype(float)

        ks_stat, ks_pvalue = ks_2samp(self.reference_feature, x_curr)
        psi_value = population_stability_index(self.reference_feature, x_curr)
        kl_value = kl_divergence(self.reference_feature, x_curr)

        self._adwin.update(error_rate)
        adwin_drift = bool(self._adwin.drift_detected)

        data_drift_detected = (
            ks_pvalue < KS_P_VALUE_THRESHOLD
            or psi_value >= PSI_ALERT_THRESHOLD
            or kl_value >= KL_ALERT_THRESHOLD
        )

        concept_drift_detected = adwin_drift

        model_drift_detected = accuracy_drop > 0.0

        return DriftSignals(
            ks_stat=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
            psi=float(psi_value),
            kl_div=float(kl_value),
            adwin_drift=adwin_drift,
            data_drift_detected=bool(data_drift_detected),
            concept_drift_detected=bool(concept_drift_detected),
            model_drift_detected=bool(model_drift_detected),
        )

    @staticmethod
    def severity_from_psi(psi_value: float) -> str:
        """
        ID: DET-005
        Purpose: Map PSI value to qualitative severity band.
        """
        if psi_value < PSI_WARNING_THRESHOLD:
            return "stable"
        if psi_value < PSI_ALERT_THRESHOLD:
            return "warning"
        return "alert"


def signals_to_dict(signals: DriftSignals) -> Dict[str, float | bool]:
    """
    ID: DET-006
    Purpose: Convert DriftSignals dataclass to serialisable dictionary.
    """
    return {
        "ks_stat": signals.ks_stat,
        "ks_pvalue": signals.ks_pvalue,
        "psi": signals.psi,
        "kl_div": signals.kl_div,
        "adwin_drift": signals.adwin_drift,
        "data_drift_detected": signals.data_drift_detected,
        "concept_drift_detected": signals.concept_drift_detected,
        "model_drift_detected": signals.model_drift_detected,
    }
