import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "development_dataset.csv")

# Global DataFrame Cache
_cached_df: Optional[pd.DataFrame] = None


def load_raw_dataset() -> pd.DataFrame:
    """
    Load and cache development_dataset.csv as the primary Source of Truth.
    """
    global _cached_df
    if _cached_df is not None:
        return _cached_df

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Source of Truth CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    
    # Standardize timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values(["equipment_id", "timestamp"]).reset_index(drop=True)
    
    _cached_df = df
    return _cached_df


def get_available_chiller_ids() -> List[str]:
    """
    Return unique real chiller IDs from the CSV dataset.
    """
    df = load_raw_dataset()
    return sorted(list(df["equipment_id"].dropna().unique()))


def get_fleet_metrics() -> Dict[str, Any]:
    """
    Compute real statistics directly from development_dataset.csv.
    """
    df = load_raw_dataset()
    chiller_ids = get_available_chiller_ids()
    
    date_min = df["timestamp"].min()
    date_max = df["timestamp"].max()
    total_readings = len(df)
    
    chiller_stats = []
    for cid in chiller_ids:
        c_df = df[df["equipment_id"] == cid]
        energy_col = "Chiller Energy Consumption (kWh)"
        
        stats = {
            "equipment": cid,
            "readings": len(c_df),
            "avg_energy": float(c_df[energy_col].mean()) if energy_col in c_df.columns else 0.0,
            "max_energy": float(c_df[energy_col].max()) if energy_col in c_df.columns else 0.0,
            "min_energy": float(c_df[energy_col].min()) if energy_col in c_df.columns else 0.0,
            "total_energy": float(c_df[energy_col].sum()) if energy_col in c_df.columns else 0.0,
        }
        
        # Check if ML output exists for this chiller
        ml_anomaly = get_anomaly_data(cid)
        if ml_anomaly:
            stats.update({
                "status": ml_anomaly.get("severity", "NORMAL"),
                "current_deviation_pct": ml_anomaly.get("deviation_pct", 0.0),
                "persistent": ml_anomaly.get("persistent", False),
                "consecutive_abnormal_readings": ml_anomaly.get("consecutive_abnormal_readings", 0),
                "trend": ml_anomaly.get("trend", "STABLE"),
                "has_ml": True
            })
        else:
            stats.update({
                "status": "UNASSESSED",
                "current_deviation_pct": None,
                "persistent": False,
                "consecutive_abnormal_readings": 0,
                "trend": "PENDING ML",
                "has_ml": False
            })
            
        chiller_stats.append(stats)

    return {
        "total_chillers": len(chiller_ids),
        "total_readings": total_readings,
        "date_min": date_min,
        "date_max": date_max,
        "chiller_stats": chiller_stats
    }


def get_chiller_readings(chiller_id: str) -> pd.DataFrame:
    """
    Adapter: Get raw readings for a specific chiller from the CSV.
    """
    df = load_raw_dataset()
    chiller_df = df[df["equipment_id"] == chiller_id].copy()
    if chiller_df.empty and len(df["equipment_id"].unique()) > 0:
        # Fall back to first available chiller if ID not found
        fallback_id = sorted(df["equipment_id"].unique())[0]
        chiller_df = df[df["equipment_id"] == fallback_id].copy()
    
    return chiller_df.reset_index(drop=True)


def get_context_variables(df: pd.DataFrame) -> List[str]:
    """
    Return available numerical context variable column names from dataset.
    Excludes timestamp, equipment_id, and primary energy consumption.
    """
    exclude = ["timestamp", "equipment_id", "Chiller Energy Consumption (kWh)"]
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def get_expected_energy(chiller_id: str) -> Optional[pd.Series]:
    """
    Adapter: Retrieve Person 1's expected energy model predictions.
    Returns Series matching chiller timestamps if ML output file exists, else None.
    """
    # Check for real Person 1 ML output files
    candidates = [
        os.path.join(BASE_DIR, "expected_energy.csv"),
        os.path.join(BASE_DIR, "output", "expected_energy.csv"),
        os.path.join(BASE_DIR, "anomaly_results.json"),
        os.path.join(BASE_DIR, "output", "anomaly_results.json"),
    ]
    
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                if path.endswith(".csv"):
                    ml_df = pd.read_csv(path)
                    if "equipment_id" in ml_df.columns and "expected_energy" in ml_df.columns:
                        sub = ml_df[ml_df["equipment_id"] == chiller_id]
                        if not sub.empty:
                            return sub.set_index("timestamp")["expected_energy"]
                elif path.endswith(".json"):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict) and data.get("equipment") == chiller_id and "expected_energy" in data:
                            return pd.Series([data["expected_energy"]], index=[data.get("timestamp")])
            except Exception:
                pass
                
    # Check mock_data directory only if explicitly present for testing adapter contract
    mock_path = os.path.join(BASE_DIR, "mock_data", "chiller_timeseries.csv")
    if os.path.exists(mock_path):
        try:
            m_df = pd.read_csv(mock_path)
            if "expected_energy" in m_df.columns:
                m_df["timestamp"] = pd.to_datetime(m_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                return m_df.set_index("timestamp")["expected_energy"]
        except Exception:
            pass

    return None


def get_anomaly_data(chiller_id: str) -> Optional[Dict[str, Any]]:
    """
    Adapter: Retrieve Person 1 / Person 2 anomaly output structure.
    Returns dict if available, else None.
    """
    candidates = [
        os.path.join(BASE_DIR, "anomaly_results.json"),
        os.path.join(BASE_DIR, "output", "anomaly_results.json"),
        os.path.join(BASE_DIR, "mock_data", "anomaly_results.json")
    ]
    
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        req_equip = data.get("equipment", "")
                        if req_equip.upper().replace("_", "-") == chiller_id.upper().replace("_", "-") or not req_equip:
                            return data
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and item.get("equipment", "").upper().replace("_", "-") == chiller_id.upper().replace("_", "-"):
                                return item
            except Exception:
                pass

    return None
