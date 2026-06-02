"""
# ============================================================
# FILE: src/drift/metrics.py
# ID: DRM-001
# Purpose: Statistical metrics used for drift measurement.
# ============================================================
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    ID: DRM-002
    Purpose: Compute PSI between expected and actual feature samples.
    Inputs:
        expected - baseline 1D sample array.
        actual   - current  1D sample array.
        bins     - number of histogram bins.
    Outputs: PSI scalar (>=0).
    Notes: Uses common bin edges derived from expected distribution quantiles.
    """
    expected = np.asarray(expected).ravel()
    actual = np.asarray(actual).ravel()

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(expected, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = exp_counts / max(1, exp_counts.sum())
    act_pct = act_counts / max(1, act_counts.sum())

    eps = 1e-8
    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def kl_divergence(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 20,
) -> float:
    """
    ID: DRM-003
    Purpose: Estimate KL(expected || actual) by histogramming both samples.
    Inputs: expected, actual - 1D arrays.
    Outputs: KL divergence scalar (>=0).
    """
    expected = np.asarray(expected).ravel()
    actual = np.asarray(actual).ravel()

    low = min(expected.min(), actual.min())
    high = max(expected.max(), actual.max())
    if low == high:
        return 0.0

    edges = np.linspace(low, high, bins + 1)
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    eps = 1e-8
    exp_prob = np.clip(exp_counts / max(1, exp_counts.sum()), eps, None)
    act_prob = np.clip(act_counts / max(1, act_counts.sum()), eps, None)

    return float(entropy(exp_prob, act_prob))
