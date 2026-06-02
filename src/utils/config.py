"""
# ============================================================
# FILE: src/utils/config.py
# ID: CFG-001
# Purpose: Central configuration for the drift-demo pipeline.
# ============================================================
"""

import os

# ---------------------------------------------------------------------------
# Dataset / stream settings
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
N_FEATURES: int = 10
N_INITIAL_SAMPLES: int = 2_000   # samples used to train the initial model
N_STREAM_BATCHES: int = 50       # number of time-step batches in the stream
BATCH_SIZE: int = 200            # samples per batch

# ---------------------------------------------------------------------------
# Drift injection schedule  (batch index at which each drift begins)
# ---------------------------------------------------------------------------
DATA_DRIFT_START_BATCH: int = 15   # Covariate / data drift
CONCEPT_DRIFT_START_BATCH: int = 30  # Label-mapping concept drift
DRIFT_RAMP_BATCHES: int = 5        # gradual ramp-up length

# ---------------------------------------------------------------------------
# Drift detection thresholds
# ---------------------------------------------------------------------------
KS_P_VALUE_THRESHOLD: float = 0.05
PSI_WARNING_THRESHOLD: float = 0.10
PSI_ALERT_THRESHOLD: float = 0.25
KL_ALERT_THRESHOLD: float = 0.30
ADWIN_DELTA: float = 0.002         # ADWIN sensitivity (smaller => more sensitive)

# ---------------------------------------------------------------------------
# Model / retraining settings
# ---------------------------------------------------------------------------
RETRAIN_WINDOW: int = 5            # batches of recent data to include in retrain
MIN_RETRAIN_ACCURACY_DROP: float = 0.05  # trigger retraining if accuracy drops this much

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")
MODEL_PATH: str = os.path.join(OUTPUT_DIR, "model.joblib")
LOG_PATH: str = os.path.join(OUTPUT_DIR, "drift_log.csv")
PLOT_DIR: str = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
