# Chiller Forensics ❄️
> **Intelligent Monitoring & Anomaly Forensics for Industrial Chillers**

Chiller Forensics moves beyond primitive static thresholds (*Sensor → Threshold → Alarm*) to contextual machine learning monitoring (*Historical Data → Learned Normal Behavior → Contextual Deviation → Persistence Verification → Evidence Forensics → Targeted Recommendations*).

---

## 🚀 Quick Start (Local Run)

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

Launch the Streamlit dashboard:
```bash
python -m streamlit run dashboard/app.py
```
*(Or `streamlit run dashboard/app.py`)*

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📁 Data Integration & Fallback Architecture

The frontend automatically detects output files from Person 1's ML pipeline in the root directory or `output/` directory:
- `chiller_summary.json` or `chiller_summary.csv`
- `anomaly_results.json` or `anomaly_results.csv`
- `chiller_timeseries.csv`

If Person 1's outputs are not present or still generating, the dashboard seamlessly loads mock data from `mock_data/` without crashing or displaying error stack traces.

---

## 🏆 60-Second Judge Presentation Demo Sequence

1. **Enable Demo Mode**: In the left sidebar, check **"⚡ Enable 60-Second Judge Demo Mode"**.
2. **Fleet Overview (0–10s)**: Point to the top Fleet Overview cards. Show how `Chiller_01` is 🟢 NORMAL (+3.2% deviation), while `Chiller_02` is flagged as 🔴 **INVESTIGATE** (+25.8% excess energy consumption, 7 persistent readings).
3. **Investigation Header (10–20s)**: Scroll to the Investigation Case KPI metrics. Highlight that Actual Energy is **103.7 kWh** vs Learned Baseline Expected Energy of **82.4 kWh**. Point to the backend explanation: *"Energy consumption is substantially above learned normal behaviour for the observed operating conditions."*
4. **Actual vs Expected Plotly Graph (20–35s)**: Show the Plotly chart comparing Actual Energy against Expected Energy, highlighting the red marker region where the deviation escalated.
5. **Anomaly Replay Slider (35–45s)**: Drag the **Anomaly Replay Slider** to scrub through time. Show the transition from `NORMAL` → `WATCH` → `PERSISTENT` → `INVESTIGATE`.
6. **What Changed & Recommendations (45–60s)**: Point out the **"WHAT CHANGED?"** section showing contextual parameter shifts (Cooling Water Temp +8.1% ↑, Chilled Water Flow -5.0% ↓) alongside the actionable **RECOMMENDED INVESTIGATION** checklist.

---

## 🛠️ Created / Modified Files

- `dashboard/app.py`: Main Streamlit application entry point & layout controls.
- `dashboard/components.py`: Plotly charts, Anomaly Replay timeline scrubber, contextual tables, and CSS dark theme.
- `dashboard/data_loader.py`: Safe data ingestion module with fallback support for mock data and live ML outputs.
- `mock_data/anomaly_results.json`: JSON output matching Person 1's data contract.
- `mock_data/chiller_summary.json`: Fleet status JSON summary file.
- `mock_data/chiller_timeseries.csv`: Time-series dataset with energy baselines and anomaly tags.
- `requirements.txt`: Python package requirements.
- `README.md`: System documentation & presentation script.
