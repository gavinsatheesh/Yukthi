# Chiller Forensics ❄️
> **Intelligent Monitoring & Anomaly Forensics for Industrial Chillers**

Chiller Forensics moves beyond primitive static thresholds (*Sensor → Threshold → Alarm*) to contextual machine learning monitoring (*Historical Data → Learned Normal Behavior → Contextual Deviation → Persistence Verification → Evidence Forensics → Targeted Recommendations*).

---

## Core Concept & Philosophy

Rather than relying on arbitrary static power thresholds, **Chiller Forensics** learns the unique thermodynamic and operational profile of each chiller under real operating conditions:

$$\text{Context (Building Load, Flow Rates, Ambient Weather, Time)} \xrightarrow{\text{ML Regressor}} \text{Expected Energy (kWh)}$$

A reading is identified as anomalous **only when energy consumption significantly deviates from what the model expects for those exact operating conditions**.

1. **Primary Signal — Statistical Residual Z-Score**:
   $$\text{residual} = y_t - \hat{y}_t, \quad z = \frac{\text{residual} - \mu_{train}}{\sigma_{train}}$$
   Calibrated against the chiller's normal training baseline.
2. **Persistence Tracking**:
   Single deviations trigger `WATCH`. Sustained sequences ($\ge 3$ consecutive readings) elevate to `INVESTIGATE` or `PRIORITY`.
3. **Forensic Context Evidence (Non-Causal)**:
   Extracts coinciding shifts in operating temperatures, flow rates, and ambient humidity during anomalous windows compared to prior normal baselines without making ungrounded causal claims.
4. **Operational Health Index**:
   Algorithmic proxy (0–100) reflecting operational baseline conformity.

---

## Project Structure

```
├── ml/
│   ├── config.py         # Centralized prototype scoring parameters & paths
│   ├── preprocess.py     # Leakage-safe chronological preprocessing & imputation
│   ├── model.py          # Equipment-specific Random Forest regressors & calibration
│   ├── anomaly.py        # Residual z-scores, persistence, events & context evidence
│   └── pipeline.py       # Master one-command execution pipeline
├── app/
│   └── api.py            # FastAPI REST backend for frontend integration
├── dashboard/
│   ├── app.py            # Main Streamlit dashboard application
│   ├── components.py     # Plotly visualizations & forensic cards
│   └── data_loader.py    # Seamless real outputs and fallback loader
├── outputs/              # Generated analysis artifacts
│   ├── anomaly_results.csv
│   ├── anomaly_results.json
│   ├── chiller_summary.json
│   ├── timeline_replay.json
│   ├── chiller_timeseries.csv
│   └── expected_energy.csv
├── mock_data/            # Contract verification fallback data
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the ML Pipeline (One Command)
```powershell
& ".\.venv\Scripts\python.exe" ml/pipeline.py
# Or: python ml/pipeline.py
```
This single command:
1. Loads and cleans `development_dataset.csv` (24,994 records across 3 chillers).
2. Performs chronological 80/20 train/test splits.
3. Trains individual `RandomForestRegressor` models for `CHILLER-01`, `CHILLER-02`, and `CHILLER-03`.
4. Calibrates baseline residual distributions ($\mu_{res}, \sigma_{res}$) on training sets.
5. Evaluates statistical residual z-scores and persistence ($\ge 3$ readings).
6. Generates 124 actionable investigation events with non-causal contextual shift evidence.
7. Exports frontend-ready CSV and JSON files to `outputs/`.

### 3. Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
The dashboard will open in your browser at `http://localhost:8501`.

### 4. (Optional) Run the FastAPI REST Backend
```powershell
& ".\.venv\Scripts\uvicorn.exe" app.api:app --reload --port 8000
```
- `GET /chillers`: Operational health summaries for all chillers.
- `GET /chillers/{equipment_id}/timeline`: Chronological replay timeline.
- `GET /investigations`: Filtered structured investigation cases.
- `GET /investigations/{case_id}`: Detailed case investigation with context evidence.

---

## 🏆 60-Second Judge Presentation Demo Sequence

1. **Fleet Overview (0–10s)**: Point to the top Fleet Overview cards. Show how chillers are monitored with status evaluations and deviation percentages.
2. **Investigation Header (10–20s)**: Switch to the Chiller Investigation tab. Highlight Actual Energy vs Learned Baseline Expected Energy under prevailing conditions.
3. **Actual vs Expected Plotly Graph (20–35s)**: Show the Plotly chart comparing Actual Energy against Expected Baseline Energy, displaying anomaly windows.
4. **Anomaly Replay Slider (35–45s)**: Drag the **Anomaly Replay Slider** in the Anomaly Replay tab to scrub through chronological timestamps.
5. **What Changed & Recommendations (45–60s)**: Highlight the **INVESTIGATION REPORT** section showing coinciding contextual parameter shifts alongside actionable engineering checklist recommendations.

---

## Prototype Scoring Parameters

Configurable in `ml/config.py`:
- `ANOMALY_Z_THRESHOLD = 2.5`: Statistical residual z-score threshold for defining an abnormal observation.
- `PERSISTENCE_COUNT = 3`: Minimum consecutive abnormal readings (90 minutes) required for `INVESTIGATE` status.
- `WATCH_DEVIATION_PCT = 10.0%`: Watch severity threshold.
- `INVESTIGATION_DEVIATION = 20.0%`: Investigation severity threshold.
- `PRIORITY_DEVIATION = 35.0%`: High priority severity threshold.
- `REFERENCE_WINDOW_SIZE = 48`: Historical baseline window (24 hours) for non-causal context evidence.
