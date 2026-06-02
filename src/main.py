"""
# ============================================================
# FILE: src/main.py
# ID: MAIN-001
# Purpose: End-to-end drift demo pipeline.
#          1) Train baseline model
#          2) Stream batches with injected drift
#          3) Detect data / concept / model drift
#          4) Trigger retraining response
#          5) Save logs, model, and plots
# ============================================================
"""

from __future__ import annotations

import os
from collections import deque
from typing import Deque, Tuple

import numpy as np
import pandas as pd

from data.generate_stream import get_initial_data, stream_batches
from data.drift_injector import inject_all_drift
from model.classifier import DriftAwareClassifier
from model.retrain import retrain_model
from drift.detectors import DriftDetectorSuite, signals_to_dict
from drift.alerts import build_alerts
from visualize.drift_plots import (
    plot_distribution_shift,
    plot_drift_signals,
    plot_accuracy_curve,
    plot_before_after_retraining,
)
from utils.config import RETRAIN_WINDOW, LOG_PATH
from utils.logger import log_batch


def run_pipeline() -> pd.DataFrame:
    """
    ID: MAIN-002
    Purpose: Execute complete demonstration pipeline and return result DataFrame.
    Outputs: DataFrame with one row per stream batch.
    """
    # ------------------------------------------------------------------
    # 1) Initial training
    # ------------------------------------------------------------------
    X_train, y_train = get_initial_data()
    clf = DriftAwareClassifier(n_estimators=200).train(X_train, y_train)

    detector = DriftDetectorSuite(reference_X=X_train)

    # Keep recent batches for retraining
    history: Deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=RETRAIN_WINDOW)

    # Reset previous log file for a clean run
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    # ------------------------------------------------------------------
    # 2) Stream monitoring loop
    # ------------------------------------------------------------------
    rows = []
    for batch_idx, X_batch, y_batch in stream_batches():
        X_drifted, y_drifted, drift_meta = inject_all_drift(batch_idx, X_batch, y_batch)

        acc, acc_drop = clf.evaluate(X_drifted, y_drifted)
        error_rate = 1.0 - acc

        signals = detector.detect(X_drifted, error_rate=error_rate, accuracy_drop=acc_drop)
        signal_dict = signals_to_dict(signals)

        alerts = build_alerts(signals, accuracy_drop=acc_drop)
        retrain_triggered = bool(alerts["retrain"])

        # Add current drifted data to retraining history
        history.append((X_drifted, y_drifted))

        if retrain_triggered and len(history) > 0:
            clf = retrain_model(clf, list(history), window=RETRAIN_WINDOW)

        metrics = {
            "accuracy": float(acc),
            "ks_stat": signal_dict["ks_stat"],
            "ks_pvalue": signal_dict["ks_pvalue"],
            "psi": signal_dict["psi"],
            "kl_div": signal_dict["kl_div"],
            "adwin_drift": signal_dict["adwin_drift"],
            "data_drift_detected": signal_dict["data_drift_detected"],
            "concept_drift_detected": signal_dict["concept_drift_detected"],
            "model_drift_detected": signal_dict["model_drift_detected"],
            "retrain_triggered": retrain_triggered,
        }

        note = (
            f"data_alpha={drift_meta['data_drift_alpha']:.2f};"
            f"concept_alpha={drift_meta['concept_drift_alpha']:.2f};"
            f"alert={alerts['reason']}"
        )
        log_batch(batch_idx, metrics, notes=note)

        rows.append({"batch": batch_idx, **metrics, **drift_meta, "accuracy_drop": acc_drop})

        # Save occasional distribution snapshots
        if batch_idx in (0, 15, 30, 40, 49):
            plot_distribution_shift(X_train[:, 0], X_drifted[:, 0], batch_idx)

    # ------------------------------------------------------------------
    # 3) Persist model + visualisations
    # ------------------------------------------------------------------
    clf.save()

    df = pd.DataFrame(rows)
    plot_drift_signals(df)
    plot_accuracy_curve(df)
    plot_before_after_retraining(df)

    return df


if __name__ == "__main__":
    result_df = run_pipeline()
    print(result_df.head())
    print("\\nPipeline complete. Outputs written to outputs/.")
