"""
ml/model.py
Equipment-specific expected energy regression models for Chiller Forensics.

CORE PRINCIPLES:
1. Contextual Prediction: Model learns mapping from operating and environmental
   conditions to normal expected energy consumption:
   context (weather + load + water rates + time) -> expected energy (kWh).
2. Equipment Isolation: Separate models per chiller to capture individual physical baselines.
3. Strict Chronological Evaluation: Trained on chronological training partition;
   evaluated on held-out future chronological test partition.
4. Baseline Residual Calibration: Computes residual mean and standard deviation
   on the training partition to serve as the statistical baseline for anomaly scoring.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.config import (
    EQUIPMENT_COL,
    MODEL_DIR,
    MODEL_FEATURES,
    TARGET_COL,
)


@dataclass
class ChillerModelArtifact:
    equipment_id: str
    model: RandomForestRegressor
    features: List[str]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    # Training reference residual distribution parameters
    residual_mean: float
    residual_std: float
    residual_median: float
    residual_iqr: float


def train_chiller_model(
    equipment_id: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str] = MODEL_FEATURES,
    target: str = TARGET_COL,
    n_estimators: int = 100,
    random_state: int = 42,
) -> ChillerModelArtifact:
    """
    Train a dedicated RandomForestRegressor for a specific chiller.
    Calibrates the normal training residual baseline for statistical anomaly detection.
    """
    eq_train = train_df[train_df[EQUIPMENT_COL] == equipment_id].copy()
    eq_test = test_df[test_df[EQUIPMENT_COL] == equipment_id].copy()

    X_train = eq_train[features]
    y_train = eq_train[target]
    X_test = eq_test[features]
    y_test = eq_test[target]

    # Fast, robust random forest regressor (n_jobs=1 avoids Windows multiprocessing overhead)
    # oob_score=True enables out-of-bag prediction evaluation, ensuring residual z-scores
    # are calibrated on realistic out-of-sample predictions rather than overfit in-sample data.
    model = RandomForestRegressor(
        n_estimators=60,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
        oob_score=True,
    )
    model.fit(X_train, y_train)

    # In-sample predictions
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    train_metrics = {
        "mae": float(train_mae),
        "rmse": float(train_rmse),
        "r2": float(train_r2),
    }

    # Out-of-sample test evaluation
    test_metrics = {}
    if len(X_test) > 0:
        y_test_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_metrics = {
            "mae": float(test_mae),
            "rmse": float(test_rmse),
            "r2": float(test_r2),
        }

    # CALIBRATION FIX:
    # Use Out-Of-Bag (OOB) residuals as the reference distribution for anomaly scoring.
    # This prevents artificially compressed in-sample residuals from inflating z-scores,
    # establishing a realistic out-of-sample operational baseline.
    if hasattr(model, "oob_prediction_"):
        ref_residuals = y_train.values - model.oob_prediction_
    else:
        ref_residuals = y_train.values - y_train_pred

    res_mean = float(np.mean(ref_residuals))
    res_std = float(np.std(ref_residuals))
    if res_std <= 1e-6:
        res_std = 1.0  # Guard against division by zero

    q75, q25 = np.percentile(ref_residuals, [75, 25])
    iqr = float(q75 - q25)
    res_median = float(np.median(ref_residuals))

    artifact = ChillerModelArtifact(
        equipment_id=equipment_id,
        model=model,
        features=features,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        residual_mean=res_mean,
        residual_std=res_std,
        residual_median=res_median,
        residual_iqr=iqr if iqr > 1e-6 else 1.0,
    )

    return artifact


def train_all_chillers(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str] = MODEL_FEATURES,
    save_models: bool = True,
) -> Dict[str, ChillerModelArtifact]:
    """Train models for all equipment in the training dataset and save artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, ChillerModelArtifact] = {}

    equipment_list = train_df[EQUIPMENT_COL].unique()
    for eq in sorted(equipment_list):
        artifact = train_chiller_model(
            equipment_id=eq,
            train_df=train_df,
            test_df=test_df,
            features=features,
        )
        artifacts[eq] = artifact

        if save_models:
            save_path = MODEL_DIR / f"{eq}_model.joblib"
            joblib.dump(artifact, save_path)

    return artifacts


def load_chiller_models() -> Dict[str, ChillerModelArtifact]:
    """Load all pre-trained chiller model artifacts from MODEL_DIR."""
    artifacts = {}
    for model_file in MODEL_DIR.glob("*_model.joblib"):
        artifact: ChillerModelArtifact = joblib.load(model_file)
        artifacts[artifact.equipment_id] = artifact
    return artifacts


def predict_expected_energy(
    df: pd.DataFrame,
    models: Dict[str, ChillerModelArtifact],
) -> pd.DataFrame:
    """
    Generate expected energy predictions for a dataframe across all chillers.
    Appends 'expected_energy' column.
    """
    df_out = df.copy()
    df_out["expected_energy"] = np.nan

    for eq, artifact in models.items():
        mask = df_out[EQUIPMENT_COL] == eq
        if mask.sum() > 0:
            X = df_out.loc[mask, artifact.features]
            preds = artifact.model.predict(X)
            # Ensure non-negative expected energy
            preds = np.maximum(preds, 0.0)
            df_out.loc[mask, "expected_energy"] = preds

    return df_out
