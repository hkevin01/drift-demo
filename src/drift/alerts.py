"""
# ============================================================
# FILE: src/drift/alerts.py
# ID: ALT-001
# Purpose: Alert formatting and trigger policy for drift events.
# ============================================================
"""

from __future__ import annotations

from typing import Dict

from drift.detectors import DriftSignals
from utils.config import MIN_RETRAIN_ACCURACY_DROP


def build_alerts(signals: DriftSignals, accuracy_drop: float) -> Dict[str, bool | str]:
    """
    ID: ALT-002
    Purpose: Produce action flags from detector outputs.
    Inputs:
        signals       - DriftSignals for current batch.
        accuracy_drop - model accuracy drop vs baseline.
    Outputs:
        dict with boolean flags and summary text.
    """
    data_alert = signals.data_drift_detected
    concept_alert = signals.concept_drift_detected
    model_alert = accuracy_drop >= MIN_RETRAIN_ACCURACY_DROP

    retrain = model_alert or (data_alert and concept_alert)

    if retrain:
        reason = "retrain_triggered"
    elif data_alert or concept_alert:
        reason = "monitor_closely"
    else:
        reason = "stable"

    return {
        "data_alert": bool(data_alert),
        "concept_alert": bool(concept_alert),
        "model_alert": bool(model_alert),
        "retrain": bool(retrain),
        "reason": reason,
    }
