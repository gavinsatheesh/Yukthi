"""
ml/pipeline.py
Master End-to-End Orchestrator for Chiller Forensics.

RUN COMMAND:
    python ml/pipeline.py

PIPELINE STAGES:
1. Leakage-safe Preprocessing & Chronological Splitting
2. Equipment-Specific Regression Modeling (Context -> Expected Energy)
3. Model Evaluation (MAE, RMSE, R2, Residual Baseline)
4. Statistical Anomaly Detection (Residual Z-Scores & Scores)
5. Anomaly Event Grouping & Persistence Tracking
6. Non-Causal Context Evidence Extraction
7. Structured Investigation Cases Generation
8. Operational Health Index & Summary Calculation
9. Export to CSV & JSON Outputs
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.anomaly import (
    compute_persistence_and_status,
    compute_residuals_and_scores,
    generate_chiller_summaries,
    generate_investigation_cases,
    generate_timeline_replay,
)
from ml.config import (
    ANOMALY_Z_THRESHOLD,
    DATA_PATH,
    EQUIPMENT_COL,
    INVESTIGATION_DEVIATION,
    MODEL_DIR,
    OUTPUT_DIR,
    PERSISTENCE_COUNT,
    PRIORITY_DEVIATION,
    TARGET_COL,
    TIMESTAMP_COL,
    WATCH_DEVIATION_PCT,
)
from ml.model import (
    predict_expected_energy,
    train_all_chillers,
)
from ml.preprocess import prepare_dataset


def run_pipeline() -> dict:
    """Execute the complete Chiller Forensics ML and analysis pipeline."""
    print("=" * 70)
    print("CHILLER FORENSICS — INTELLIGENT ENERGY & EQUIPMENT MONITORING")
    print("=" * 70)

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STAGE 1: LEAKAGE-SAFE PREPROCESSING
    # -------------------------------------------------------------------------
    print(f"\n[Stage 1/6] Preprocessing dataset: {DATA_PATH.name}...")
    full_df, train_df, test_df, impute_stats = prepare_dataset(str(DATA_PATH))
    print(f"  -> Total records: {len(full_df):,} rows")
    print(f"  -> Train partition: {len(train_df):,} rows (80% chronological split)")
    print(f"  -> Test partition:  {len(test_df):,} rows (20% chronological split)")
    print(f"  -> Chillers identified: {sorted(full_df[EQUIPMENT_COL].unique().tolist())}")

    # -------------------------------------------------------------------------
    # STAGE 2: EXPECTED ENERGY MODELING
    # -------------------------------------------------------------------------
    print("\n[Stage 2/6] Training chiller-specific expected energy models...")
    artifacts = train_all_chillers(train_df, test_df, save_models=True)

    print("\n--- MODEL PERFORMANCE & RESIDUAL CALIBRATION ---")
    for eq, art in artifacts.items():
        tr = art.train_metrics
        te = art.test_metrics
        print(f"[{eq}]")
        print(f"  Train: MAE={tr['mae']:.2f} kWh | RMSE={tr['rmse']:.2f} kWh | R²={tr['r2']:.4f}")
        print(f"  Test:  MAE={te.get('mae', 0):.2f} kWh | RMSE={te.get('rmse', 0):.2f} kWh | R²={te.get('r2', 0):.4f}")
        print(f"  Train Residual Reference: Mean={art.residual_mean:.3f} kWh | Std={art.residual_std:.3f} kWh")

    # -------------------------------------------------------------------------
    # STAGE 3: INFERENCE & EXPECTED ENERGY PREDICTION
    # -------------------------------------------------------------------------
    print("\n[Stage 3/6] Generating contextual expected energy predictions...")
    scored_df = predict_expected_energy(full_df, artifacts)

    # -------------------------------------------------------------------------
    # STAGE 4: STATISTICAL ANOMALY DETECTION & PERSISTENCE
    # -------------------------------------------------------------------------
    print(f"\n[Stage 4/6] Computing residual z-scores (threshold z >= {ANOMALY_Z_THRESHOLD})...")
    scored_df = compute_residuals_and_scores(scored_df, artifacts, z_threshold=ANOMALY_Z_THRESHOLD)

    print(f"  -> Applying persistence logic (consecutive >= {PERSISTENCE_COUNT})...")
    analyzed_df = compute_persistence_and_status(
        scored_df,
        persistence_count=PERSISTENCE_COUNT,
        watch_dev=WATCH_DEVIATION_PCT,
        investigate_dev=INVESTIGATION_DEVIATION,
        priority_dev=PRIORITY_DEVIATION,
    )

    # Anomaly statistics
    total_anomalies = analyzed_df["is_abnormal"].sum()
    print(f"  -> Total abnormal readings across all chillers: {total_anomalies:,} ({total_anomalies/len(analyzed_df)*100:.2f}%)")

    for eq, grp in analyzed_df.groupby(EQUIPMENT_COL):
        anom_cnt = grp["is_abnormal"].sum()
        pct = (anom_cnt / len(grp)) * 100.0
        investigate_cnt = (grp["severity"] == "INVESTIGATE").sum()
        priority_cnt = (grp["severity"] == "PRIORITY").sum()
        print(f"  -> {eq}: {anom_cnt} abnormal readings ({pct:.2f}%) | "
              f"INVESTIGATE: {investigate_cnt} | PRIORITY: {priority_cnt}")

    # -------------------------------------------------------------------------
    # STAGE 5: INVESTIGATION CASES & NON-CAUSAL CONTEXT EVIDENCE
    # -------------------------------------------------------------------------
    print("\n[Stage 5/6] Generating structured investigation cases & context evidence...")
    investigation_cases = generate_investigation_cases(analyzed_df)
    print(f"  -> Total actionable investigation events generated: {len(investigation_cases)}")

    # Operational Health Summaries & Timeline Replay
    chiller_summaries = generate_chiller_summaries(analyzed_df, investigation_cases)
    timeline_replay = generate_timeline_replay(analyzed_df, sample_rate=1)

    # -------------------------------------------------------------------------
    # STAGE 6: EXPORTS
    # -------------------------------------------------------------------------
    print("\n[Stage 6/6] Exporting results for frontend integration...")

    # 1. Export Full CSV
    export_cols = [
        TIMESTAMP_COL,
        EQUIPMENT_COL,
        "actual_energy",
        "expected_energy",
        "residual",
        "deviation_pct",
        "residual_zscore",
        "anomaly_score",
        "is_abnormal",
        "consecutive_abnormal_readings",
        "persistent",
        "severity",
        "event_id",
    ]
    csv_path = OUTPUT_DIR / "anomaly_results.csv"
    analyzed_df[export_cols].to_csv(csv_path, index=False)
    print(f"  -> Exported CSV: {csv_path.name} ({len(analyzed_df):,} rows)")

    # 2. Export Chiller Time-series CSV (Matching Person 2's dashboard schema)
    ts_df = analyzed_df.copy()
    ts_df["cooling_water_temp"] = ts_df["Cooling Water Temperature (C)"]
    ts_df["chilled_water_flow"] = ts_df["Chilled Water Rate (L/sec)"]
    ts_df["building_load"] = ts_df["Building Load (RT)"]
    ts_df["outside_temp"] = ts_df["Outside Temperature (F)"]
    ts_df["trend"] = "STABLE"
    # Assign rolling trend where persistent
    if "deviation_pct" in ts_df.columns:
        rolling_diff = ts_df["deviation_pct"].diff()
        ts_df.loc[rolling_diff > 1.0, "trend"] = "INCREASING"
        ts_df.loc[rolling_diff < -1.0, "trend"] = "DECREASING"

    ts_cols = [
        "timestamp",
        "equipment_id",
        "actual_energy",
        "expected_energy",
        "deviation_pct",
        "residual",
        "anomaly_score",
        "severity",
        "persistent",
        "consecutive_abnormal_readings",
        "trend",
        "cooling_water_temp",
        "chilled_water_flow",
        "building_load",
        "outside_temp",
    ]
    timeseries_path = OUTPUT_DIR / "chiller_timeseries.csv"
    ts_df[ts_cols].to_csv(timeseries_path, index=False)
    print(f"  -> Exported CSV: {timeseries_path.name} ({len(ts_df):,} rows)")

    # Also export expected_energy.csv
    expected_path = OUTPUT_DIR / "expected_energy.csv"
    ts_df[["timestamp", "equipment_id", "expected_energy"]].to_csv(expected_path, index=False)
    print(f"  -> Exported CSV: {expected_path.name}")

    # 3. Export Investigation JSON (List of cases for dashboard, with featured case per chiller)
    # Organize so the strongest case for each chiller is immediately accessible to the UI
    chillers = sorted(analyzed_df[EQUIPMENT_COL].unique())
    featured_per_chiller = []
    for c in chillers:
        c_cases = [case for case in investigation_cases if case["equipment"] == c]
        if c_cases:
            featured_per_chiller.append(c_cases[0])
    
    # Remaining cases
    other_cases = [case for case in investigation_cases if case not in featured_per_chiller]
    dashboard_cases = featured_per_chiller + other_cases

    json_path = OUTPUT_DIR / "anomaly_results.json"
    with open(json_path, "w") as f:
        json.dump(dashboard_cases, f, indent=2)
    print(f"  -> Exported JSON: {json_path.name} ({len(dashboard_cases)} cases, list format)")

    # Also export wrapped version with metadata
    wrapped_json_path = OUTPUT_DIR / "all_investigation_cases.json"
    investigation_payload = {
        "metadata": {
            "title": "Chiller Forensics Anomaly Investigation Cases",
            "total_cases": len(investigation_cases),
            "scoring_parameters": {
                "anomaly_z_threshold": ANOMALY_Z_THRESHOLD,
                "persistence_count": PERSISTENCE_COUNT,
                "watch_deviation_pct": WATCH_DEVIATION_PCT,
                "investigate_deviation_pct": INVESTIGATION_DEVIATION,
                "priority_deviation_pct": PRIORITY_DEVIATION,
            },
        },
        "investigation_cases": investigation_cases,
    }
    with open(wrapped_json_path, "w") as f:
        json.dump(investigation_payload, f, indent=2)
    print(f"  -> Exported JSON: {wrapped_json_path.name} (with metadata)")

    # 4. Export Chiller Summaries JSON
    summary_path = OUTPUT_DIR / "chiller_summary.json"
    with open(summary_path, "w") as f:
        json.dump(chiller_summaries, f, indent=2)
    print(f"  -> Exported JSON: {summary_path.name} (3 chillers)")

    # 5. Export Timeline Replay JSON
    timeline_path = OUTPUT_DIR / "timeline_replay.json"
    with open(timeline_path, "w") as f:
        json.dump(timeline_replay, f, indent=2)
    print(f"  -> Exported JSON: {timeline_path.name} (full chronological timeline)")

    # -------------------------------------------------------------------------
    # SUMMARY & HIGHLIGHTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE — HIGHLIGHTS")
    print("=" * 70)

    # Show top / most significant investigation case
    if investigation_cases:
        # Sort by peak deviation descending
        top_case = max(investigation_cases, key=lambda c: (c["severity"] == "PRIORITY", c["peak_deviation_pct"]))
        print("\nMOST SIGNIFICANT INVESTIGATION CASE:")
        print(f"  Case ID:       {top_case['case_id']}")
        print(f"  Equipment:     {top_case['equipment']}")
        print(f"  Window:        {top_case['start_time']} -> {top_case['end_time']} ({top_case['duration']})")
        print(f"  Severity:      {top_case['severity']}")
        print(f"  Actual vs Exp: {top_case['actual_energy_mean']:.1f} kWh vs {top_case['expected_energy_mean']:.1f} kWh")
        print(f"  Peak Dev:      {top_case['peak_deviation_pct']:+.1f}% (Avg: {top_case['average_deviation_pct']:+.1f}%)")
        print(f"  Trend:         {top_case['trend']}")
        print(f"  Persistence:   {top_case['persistence_count']} consecutive readings")
        print("  Context Evidence (Coinciding Operational Shifts):")
        for ctx in top_case["context_changes"]:
            print(f"    - {ctx['variable']}: {ctx['normal_mean']} -> {ctx['anomaly_mean']} ({ctx['change_pct']:+.1f}%, {ctx['direction']})")
        print(f"  Summary:       {top_case['investigation_summary']}")

    print("\nCHILLER OPERATIONAL HEALTH SUMMARIES:")
    for sm in chiller_summaries:
        print(f"  {sm['equipment']}: Status={sm['status']} | Operational Health Index={sm['health_index']}/100 | "
              f"Current Dev={sm['current_deviation_pct']:+.1f}% | Trend={sm['trend']} | "
              f"Active Investigations={sm['active_investigations']}")

    return {
        "investigation_cases": investigation_cases,
        "chiller_summaries": chiller_summaries,
    }


if __name__ == "__main__":
    run_pipeline()
