"""
ml/config.py
Configuration and prototype scoring parameters for Chiller Forensics.

NOTE ON PARAMETERS:
These scoring parameters are heuristic prototype configuration values designed for 
contextual anomaly detection, persistence tracking, and operational severity grading.
They are NOT hardcoded physical equipment failure limits.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "development_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# Dataset Columns
TIMESTAMP_COL = "timestamp"
EQUIPMENT_COL = "equipment_id"
TARGET_COL = "Chiller Energy Consumption (kWh)"

OPERATING_FEATURES = [
    "Chilled Water Rate (L/sec)",
    "Cooling Water Temperature (C)",
    "Building Load (RT)",
]

ENVIRONMENTAL_FEATURES = [
    "Outside Temperature (F)",
    "Dew Point (F)",
    "Humidity (%)",
    "Wind Speed (mph)",
    "Pressure (in)",
]

TIME_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "sin_hour",
    "cos_hour",
]

CONTEXTUAL_VARIABLES = OPERATING_FEATURES + ENVIRONMENTAL_FEATURES
MODEL_FEATURES = CONTEXTUAL_VARIABLES + TIME_FEATURES

# Train / Test Splitting
TRAIN_SPLIT_RATIO = 0.80  # Chronological 80% train, 20% test per chiller

# ==============================================================================
# PROTOTYPE SCORING PARAMETERS (Configurable in one place)
# ==============================================================================
# Primary Anomaly Signal: Statistical residual z-score threshold
# A reading is flagged abnormal if its ML residual is > ANOMALY_Z_THRESHOLD
# standard deviations above the learned normal training residual baseline.
ANOMALY_Z_THRESHOLD = 2.5

# Persistence Count: Number of consecutive abnormal readings to trigger persistent state
PERSISTENCE_COUNT = 3

# Secondary Severity Thresholds (Percentage deviation from expected energy)
WATCH_DEVIATION_PCT = 10.0
INVESTIGATION_DEVIATION = 20.0
PRIORITY_DEVIATION = 35.0

# Reference Window for Context Evidence Comparison (48 readings = 24 hours at 30-min sampling)
REFERENCE_WINDOW_SIZE = 48
