"""
# ============================================================
# FILE: src/visualize/drift_plots.py
# ID: VIZ-001
# Purpose: Save static visualisations for drift monitoring.
# ============================================================
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.config import PLOT_DIR

sns.set_theme(style="whitegrid")


def _save(fig, filename: str) -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_shift(ref_feature: Sequence[float], curr_feature: Sequence[float], batch_idx: int) -> None:
    """
    ID: VIZ-002
    Purpose: Compare baseline vs current feature distributions.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.kdeplot(ref_feature, fill=True, alpha=0.4, label="Reference", ax=ax)
    sns.kdeplot(curr_feature, fill=True, alpha=0.4, label="Current", ax=ax)
    ax.set_title(f"Feature Distribution Shift - Batch {batch_idx}")
    ax.legend()
    _save(fig, f"distribution_shift_batch_{batch_idx:03d}.png")


def plot_drift_signals(df: pd.DataFrame) -> None:
    """
    ID: VIZ-003
    Purpose: Plot KS p-value, PSI, KL and ADWIN drift markers over time.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["batch"], df["ks_pvalue"], label="KS p-value")
    ax.plot(df["batch"], df["psi"], label="PSI")
    ax.plot(df["batch"], df["kl_div"], label="KL divergence")

    adwin_batches = df.loc[df["adwin_drift"] == True, "batch"]
    adwin_y = df.loc[df["adwin_drift"] == True, "psi"]
    ax.scatter(adwin_batches, adwin_y, marker="x", s=70, label="ADWIN trigger")

    ax.set_title("Drift Detection Signals Over Time")
    ax.set_xlabel("Batch")
    ax.legend()
    _save(fig, "drift_signals_over_time.png")


def plot_accuracy_curve(df: pd.DataFrame) -> None:
    """
    ID: VIZ-004
    Purpose: Plot model accuracy trajectory and retraining points.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["batch"], df["accuracy"], label="Accuracy", color="tab:blue")

    retrain_df = df[df["retrain_triggered"] == True]
    ax.scatter(retrain_df["batch"], retrain_df["accuracy"], color="red", s=80, label="Retrain")

    ax.set_ylim(0.0, 1.05)
    ax.set_title("Accuracy Degradation and Retraining Events")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    _save(fig, "accuracy_degradation_curve.png")


def plot_before_after_retraining(df: pd.DataFrame) -> None:
    """
    ID: VIZ-005
    Purpose: Plot rolling mean accuracy before/after retraining events.
    """
    tmp = df.copy()
    tmp["rolling_acc"] = tmp["accuracy"].rolling(5, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(tmp["batch"], tmp["rolling_acc"], label="5-batch rolling accuracy")

    for b in tmp.loc[tmp["retrain_triggered"] == True, "batch"]:
        ax.axvline(b, color="red", linestyle="--", alpha=0.4)

    ax.set_ylim(0.0, 1.05)
    ax.set_title("Before/After Retraining Comparison")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Rolling Accuracy")
    ax.legend()
    _save(fig, "before_after_retraining.png")
