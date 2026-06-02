"""
# ============================================================
# FILE: src/utils/logger.py
# ID: LOG-001
# Purpose: Structured CSV logger for per-batch drift metrics
#          and pipeline events.
# ============================================================
"""

import csv
import os
from datetime import datetime
from typing import Dict, Any

from utils.config import LOG_PATH


_FIELDNAMES = [
    "timestamp", "batch", "accuracy", "ks_stat", "ks_pvalue",
    "psi", "kl_div", "adwin_drift", "data_drift_detected",
    "concept_drift_detected", "model_drift_detected",
    "retrain_triggered", "notes",
]


def _ensure_header(path: str) -> None:
    """
    ID: LOG-002
    Purpose: Write CSV header if the file does not yet exist.
    Inputs:  path - absolute path to the log file.
    Outputs: None (side effect: file creation).
    """
    if not os.path.exists(path):
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
            writer.writeheader()


def log_batch(batch: int, metrics: Dict[str, Any], notes: str = "") -> None:
    """
    ID: LOG-003
    Purpose: Append one row of metrics for the given batch to the CSV log.
    Inputs:
        batch   - integer batch index (0-based).
        metrics - dict with keys matching _FIELDNAMES (missing keys default 0/False).
        notes   - optional free-text annotation.
    Outputs: None (side effect: append to LOG_PATH).
    Failure modes: IOError propagated to caller.
    """
    _ensure_header(LOG_PATH)
    row: Dict[str, Any] = {f: metrics.get(f, "") for f in _FIELDNAMES}
    row["timestamp"] = datetime.utcnow().isoformat()
    row["batch"] = batch
    row["notes"] = notes
    with open(LOG_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writerow(row)


def read_log():
    """
    ID: LOG-004
    Purpose: Return all logged rows as a list of dicts.
    Outputs: list[dict] - each dict corresponds to one logged batch.
    """
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", newline="") as fh:
        return list(csv.DictReader(fh))
