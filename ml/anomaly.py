"""
ml/anomaly.py
Contextual anomaly detection, persistence tracking, trend analysis,
non-causal evidence extraction, and investigation case generation.

CORE DESIGN:
- Primary anomaly signal is the statistical residual z-score:
    residual = actual_energy - expected_energy
    z_score = (residual - train_residual_mean) / train_residual_std
- Events group contiguous sequences of abnormal readings.
- Context evidence computes shifts in operating & environmental conditions
  between the anomaly event and a preceding normal baseline window.
  Statements strictly avoid causal assertions ("coincided with", "notable contextual change").
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm

from ml.config import (
    ANOMALY_Z_THRESHOLD,
    CONTEXTUAL_VARIABLES,
    EQUIPMENT_COL,
    INVESTIGATION_DEVIATION,
    PERSISTENCE_COUNT,
    PRIORITY_DEVIATION,
    REFERENCE_WINDOW_SIZE,
    TARGET_COL,
    TIMESTAMP_COL,
    WATCH_DEVIATION_PCT,
)
from ml.model import ChillerModelArtifact


def compute_residuals_and_scores(
    df: pd.DataFrame,
    artifacts: Dict[str, ChillerModelArtifact],
    z_threshold: float = ANOMALY_Z_THRESHOLD,
) -> pd.DataFrame:
    """
    Computes residuals, deviation percentages, statistical z-scores, and continuous anomaly scores.
    Primary anomaly flag is strictly driven by the residual z-score.
    """
    df_out = df.copy()
    df_out["actual_energy"] = df_out[TARGET_COL]
    df_out["residual"] = df_out["actual_energy"] - df_out["expected_energy"]

    # Calculate percentage deviation from expected model prediction
    # Guard against division by near-zero expected energy
    safe_expected = np.maximum(df_out["expected_energy"], 1.0)
    df_out["deviation_pct"] = (df_out["residual"] / safe_expected) * 100.0

    df_out["residual_zscore"] = np.nan
    df_out["anomaly_score"] = np.nan
    df_out["is_abnormal"] = False

    for eq, artifact in artifacts.items():
        mask = df_out[EQUIPMENT_COL] == eq
        if mask.sum() == 0:
            continue

        eq_residuals = df_out.loc[mask, "residual"]
        # Z-score calibrated strictly against training reference distribution
        z_scores = (eq_residuals - artifact.residual_mean) / artifact.residual_std
        df_out.loc[mask, "residual_zscore"] = z_scores

        # Continuous anomaly score in [0, 1] using standard normal cumulative distribution
        # High positive residuals (unexpected excess consumption) yield scores close to 1.0
        scores = norm.cdf(z_scores)
        df_out.loc[mask, "anomaly_score"] = np.round(scores, 4)

        # Primary anomaly detection: statistical deviation exceeding threshold
        # Focus on excess energy consumption (z > z_threshold)
        abnormal_mask = z_scores >= z_threshold
        df_out.loc[mask, "is_abnormal"] = abnormal_mask

    return df_out


def compute_persistence_and_status(
    df: pd.DataFrame,
    persistence_count: int = PERSISTENCE_COUNT,
    watch_dev: float = WATCH_DEVIATION_PCT,
    investigate_dev: float = INVESTIGATION_DEVIATION,
    priority_dev: float = PRIORITY_DEVIATION,
) -> pd.DataFrame:
    """
    Identifies consecutive abnormal readings per chiller and assigns operational severity:
    - NORMAL: within statistical expectations
    - WATCH: abnormal reading but persistence < persistence_count
    - INVESTIGATE: persistent abnormal behaviour (>= persistence_count readings)
    - PRIORITY: persistent abnormal behaviour with severe deviation
    """
    df_out = df.copy()
    df_out["consecutive_abnormal_readings"] = 0
    df_out["persistent"] = False
    df_out["severity"] = "NORMAL"
    df_out["event_id"] = None

    processed_groups = []

    for eq, group in df_out.groupby(EQUIPMENT_COL, sort=False):
        group = group.sort_values("dt").copy()

        consecutive = 0
        current_event_idx = 0
        in_event = False
        consec_list = []
        persistent_list = []
        severity_list = []
        event_id_list = []

        for idx, row in group.iterrows():
            abnormal = bool(row["is_abnormal"])
            dev_pct = float(row["deviation_pct"])
            z_val = float(row["residual_zscore"])

            if abnormal:
                consecutive += 1
                if not in_event:
                    in_event = True
                    current_event_idx += 1

                event_str = f"EVT_{eq}_{current_event_idx:04d}"
                is_persistent = consecutive >= persistence_count

                # Severity logic: combines statistical persistence with magnitude
                if is_persistent and (dev_pct >= priority_dev or z_val >= 3.5):
                    sev = "PRIORITY"
                elif is_persistent or dev_pct >= investigate_dev:
                    sev = "INVESTIGATE"
                elif dev_pct >= watch_dev or z_val >= 2.0:
                    sev = "WATCH"
                else:
                    sev = "WATCH"

                consec_list.append(consecutive)
                persistent_list.append(is_persistent)
                severity_list.append(sev)
                event_id_list.append(event_str)
            else:
                consecutive = 0
                in_event = False
                consec_list.append(0)
                persistent_list.append(False)
                severity_list.append("NORMAL")
                event_id_list.append(None)

        group["consecutive_abnormal_readings"] = consec_list
        group["persistent"] = persistent_list
        group["severity"] = severity_list
        group["event_id"] = event_id_list
        processed_groups.append(group)

    return pd.concat(processed_groups, ignore_index=True)


def calculate_trend_direction(deviations: np.ndarray) -> str:
    """
    Calculates whether the progressive deviation percentage is INCREASING, DECREASING, or STABLE.
    Uses linear regression slope over the sequence.
    """
    if len(deviations) < 2:
        return "STABLE"

    x = np.arange(len(deviations))
    # Simple linear regression slope
    x_mean = np.mean(x)
    y_mean = np.mean(deviations)
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return "STABLE"

    slope = np.sum((x - x_mean) * (deviations - y_mean)) / denom

    # Slope threshold: +/- 0.5% deviation change per 30-min time step
    if slope > 0.5:
        return "INCREASING"
    elif slope < -0.5:
        return "DECREASING"
    else:
        return "STABLE"


def extract_context_evidence(
    df_equipment: pd.DataFrame,
    event_df: pd.DataFrame,
    reference_window: int = REFERENCE_WINDOW_SIZE,
) -> List[Dict[str, Any]]:
    """
    Compares contextual operating & environmental variables during the anomaly event
    against a preceding NORMAL baseline reference window.
    
    STRICT COMPLIANCE:
    - Never asserts causality or physical root failure.
    - Uses neutral terminology: "notable contextual change", "observed during the anomaly".
    """
    start_dt = event_df["dt"].min()

    # Preceding normal operating window: rows before the event that had severity NORMAL
    prior_df = df_equipment[(df_equipment["dt"] < start_dt) & (df_equipment["severity"] == "NORMAL")]
    if len(prior_df) == 0:
        # Fallback to any prior rows if normal not available
        prior_df = df_equipment[df_equipment["dt"] < start_dt]

    # Use up to reference_window readings preceding the event
    baseline_window = prior_df.tail(reference_window)

    evidence_list = []
    for var in CONTEXTUAL_VARIABLES:
        if var not in event_df.columns:
            continue

        anomaly_mean = float(event_df[var].mean())
        normal_mean = float(baseline_window[var].mean()) if len(baseline_window) > 0 else anomaly_mean

        if abs(normal_mean) > 1e-4:
            change_pct = float(((anomaly_mean - normal_mean) / abs(normal_mean)) * 100.0)
        else:
            change_pct = 0.0

        if change_pct > 3.0:
            direction = "INCREASED"
        elif change_pct < -3.0:
            direction = "DECREASED"
        else:
            direction = "STABLE"

        evidence_list.append({
            "variable": var,
            "normal_value": round(normal_mean, 2),
            "current_value": round(anomaly_mean, 2),
            "normal_mean": round(normal_mean, 2),
            "anomaly_mean": round(anomaly_mean, 2),
            "change_pct": round(change_pct, 1),
            "direction": direction,
            "description": (
                f"Notable contextual change: {var} shifted by {change_pct:+.1f}% "
                f"from reference baseline during the anomalous period."
                if direction != "STABLE"
                else f"{var} remained consistent with reference conditions."
            ),
        })

    # Sort evidence by absolute percentage shift descending
    evidence_list.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    return evidence_list


def generate_investigation_cases(
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Constructs structured Investigation Cases for all significant anomaly events.
    Matches both the backend investigation contract and Person 2 dashboard consumption.
    """
    cases = []
    event_rows = df[df["event_id"].notnull()].copy()

    for (eq, evt_id), event_group in event_rows.groupby([EQUIPMENT_COL, "event_id"], sort=False):
        max_severity = "WATCH"
        if (event_group["severity"] == "PRIORITY").any():
            max_severity = "PRIORITY"
        elif (event_group["severity"] == "INVESTIGATE").any():
            max_severity = "INVESTIGATE"

        persistence_cnt = int(event_group["consecutive_abnormal_readings"].max())

        # Actionable events: persistence >= 2 or severity in INVESTIGATE/PRIORITY
        if persistence_cnt < 2 and max_severity == "WATCH":
            continue

        start_time = str(event_group[TIMESTAMP_COL].iloc[0])
        end_time = str(event_group[TIMESTAMP_COL].iloc[-1])
        num_readings = len(event_group)
        duration_minutes = num_readings * 30
        duration_str = f"{duration_minutes // 60}h {duration_minutes % 60}m"

        actual_energy_mean = float(event_group["actual_energy"].mean())
        expected_energy_mean = float(event_group["expected_energy"].mean())
        peak_dev = float(event_group["deviation_pct"].max())
        avg_dev = float(event_group["deviation_pct"].mean())
        peak_score = float(event_group["anomaly_score"].max())
        avg_residual = float(event_group["residual"].mean())

        trend_dir = calculate_trend_direction(event_group["deviation_pct"].values)

        # Context evidence
        eq_full_df = df[df[EQUIPMENT_COL] == eq]
        context_changes = extract_context_evidence(eq_full_df, event_group)

        # Operational investigation recommendations (Guidance only, non-causal)
        recommended_actions = [
            "Review heat-transfer performance and check for condenser/evaporator tube fouling.",
            "Verify chilled-water and cooling-water sensor calibration and flow rate balancing.",
            "Inspect compressor lift and operating head pressure relative to cooling-water temperature.",
            "Compare with recent normal operating baseline.",
        ]

        summary_text = (
            f"Chiller energy consumption averaged {actual_energy_mean:.1f} kWh vs learned normal "
            f"baseline of {expected_energy_mean:.1f} kWh ({avg_dev:+.1f}% deviation, peaking at {peak_dev:+.1f}%) "
            f"under prevailing operating conditions across {duration_str}."
        )

        case = {
            "case_id": evt_id,
            "equipment": eq,
            "timestamp": start_time,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration_str,
            "actual_energy": round(actual_energy_mean, 2),
            "actual_energy_mean": round(actual_energy_mean, 2),
            "expected_energy": round(expected_energy_mean, 2),
            "expected_energy_mean": round(expected_energy_mean, 2),
            "residual": round(avg_residual, 2),
            "deviation_pct": round(avg_dev, 1),
            "peak_deviation_pct": round(peak_dev, 1),
            "average_deviation_pct": round(avg_dev, 1),
            "anomaly_score": round(peak_score, 3),
            "severity": max_severity,
            "persistent": persistence_cnt >= PERSISTENCE_COUNT,
            "consecutive_abnormal_readings": persistence_cnt,
            "persistence_count": persistence_cnt,
            "trend": trend_dir,
            "context_changes": context_changes[:4],
            "investigation_summary": summary_text,
            "recommended_investigation": recommended_actions,
        }
        cases.append(case)

    # Rank cases by severity (PRIORITY first), peak deviation, and persistence
    severity_order = {"PRIORITY": 3, "INVESTIGATE": 2, "WATCH": 1, "NORMAL": 0}
    cases.sort(
        key=lambda c: (
            severity_order.get(c["severity"], 0),
            c["peak_deviation_pct"],
            c["persistence_count"]
        ),
        reverse=True
    )
    return cases


def calculate_operational_health_index(
    recent_records: pd.DataFrame,
    total_anomalies: int,
    total_records: int,
) -> int:
    """
    DOCUMENTATION & GOVERNANCE:
    The 'Operational Health Index' is a synthetic prototype operational rating (0 - 100)
    derived from anomaly magnitude, statistical persistence, and recent deviation trend.
    It indicates whether recent operational energy behavior conforms to learned baselines.
    It is NOT a direct measurement of physical mechanical health or component wear.

    Scoring formula:
    - Starts at 100
    - Penalized by recent average deviation percentage
    - Penalized by recent persistence flags
    - Penalized by recent trend degradation
    """
    score = 100.0

    if len(recent_records) == 0:
        return 100

    latest_severity = recent_records["severity"].iloc[-1]
    latest_persistent = bool(recent_records["persistent"].iloc[-1])
    recent_avg_dev = float(recent_records["deviation_pct"].tail(10).mean())

    if latest_severity == "PRIORITY":
        score -= 40.0
    elif latest_severity == "INVESTIGATE":
        score -= 25.0
    elif latest_severity == "WATCH":
        score -= 10.0

    if latest_persistent:
        score -= 15.0

    if recent_avg_dev > 20.0:
        score -= 15.0
    elif recent_avg_dev > 10.0:
        score -= 8.0

    # Bounded between 0 and 100
    return int(np.clip(round(score), 0, 100))


def generate_chiller_summaries(
    df: pd.DataFrame,
    investigation_cases: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Produce structured summary per chiller matching Task 11 schema.
    """
    summaries = []

    for eq, group in df.groupby(EQUIPMENT_COL, sort=False):
        group = group.sort_values("dt").reset_index(drop=True)
        total_records = len(group)
        total_anomalies = int(group["is_abnormal"].sum())
        anomaly_rate = round((total_anomalies / max(total_records, 1)) * 100.0, 2)

        latest_row = group.iloc[-1]
        active_investigations = len([
            c for c in investigation_cases
            if c["equipment"] == eq and c["severity"] in ("INVESTIGATE", "PRIORITY")
        ])

        health_idx = calculate_operational_health_index(
            group.tail(48), total_anomalies, total_records
        )

        recent_deviations = group["deviation_pct"].tail(10).values
        trend_val = calculate_trend_direction(recent_deviations)

        summary = {
            "equipment": eq,
            "status": str(latest_row["severity"]),
            "health_index": health_idx,
            "current_deviation_pct": round(float(latest_row["deviation_pct"]), 1),
            "persistent": bool(latest_row["persistent"]),
            "consecutive_abnormal_readings": int(latest_row.get("consecutive_abnormal_readings", 0)),
            "trend": trend_val,
            "active_investigations": active_investigations,
            "total_anomalies_detected": total_anomalies,
            "anomaly_rate_pct": anomaly_rate,
            "health_index_disclaimer": (
                "Operational Health Index is an algorithmic score (0-100) reflecting recent "
                "conformance to learned energy baselines under operating context. "
                "It is not a direct physical equipment health measurement."
            ),
        }
        summaries.append(summary)

    return summaries


def generate_timeline_replay(
    df: pd.DataFrame,
    sample_rate: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Produces chronological anomaly timeline per chiller matching Task 8:
    NORMAL -> WATCH -> INVESTIGATE -> PRIORITY
    """
    timeline_by_chiller: Dict[str, List[Dict[str, Any]]] = {}

    for eq, group in df.groupby(EQUIPMENT_COL, sort=False):
        group = group.sort_values("dt").iloc[::sample_rate]
        chiller_timeline = []

        for _, row in group.iterrows():
            chiller_timeline.append({
                "timestamp": str(row[TIMESTAMP_COL]),
                "status": str(row["severity"]),
                "deviation_pct": round(float(row["deviation_pct"]), 2),
                "actual_energy": round(float(row["actual_energy"]), 2),
                "expected_energy": round(float(row["expected_energy"]), 2),
                "residual": round(float(row["residual"]), 2),
                "residual_zscore": round(float(row["residual_zscore"]), 2),
                "anomaly_score": round(float(row["anomaly_score"]), 3),
                "persistent": bool(row["persistent"]),
                "consecutive_abnormal_readings": int(row["consecutive_abnormal_readings"]),
            })
        timeline_by_chiller[eq] = chiller_timeline

    return timeline_by_chiller
