import streamlit as st
import pandas as pd
import os
import sys

# Add parent dir to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.data_loader import (
    get_fleet_metrics,
    get_available_chiller_ids,
    get_chiller_readings,
    get_expected_energy,
    get_anomaly_data
)
from dashboard.components import (
    inject_custom_css,
    render_fleet_metrics_summary,
    render_chiller_fleet_table,
    render_actual_energy_chart,
    render_contextual_operating_conditions,
    render_contextual_deviation_section,
    render_persistence_analysis,
    render_investigation_report,
    render_anomaly_replay_view
)

# Page Setup
st.set_page_config(
    page_title="Chiller Forensics — Industrial Monitoring",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
inject_custom_css()

# Title Header
st.title("❄️ CHILLER FORENSICS")
st.markdown("##### *Contextual equipment monitoring — from anomaly detection to investigation*")
st.write("")

# Sidebar Navigation
st.sidebar.markdown("### ⚙️ Navigation & Chiller Selector")

chiller_ids = get_available_chiller_ids()
default_chiller = "CHILLER-01" if "CHILLER-01" in chiller_ids else (chiller_ids[0] if chiller_ids else "CHILLER-01")

selected_chiller = st.sidebar.selectbox(
    "Select Real Chiller Asset:",
    options=chiller_ids,
    index=chiller_ids.index(default_chiller) if default_chiller in chiller_ids else 0
)

st.sidebar.markdown("---")
st.sidebar.caption("📊 **Source of Truth:** `development_dataset.csv` (25,003 readings)")

# Load data for selected equipment
df_chiller = get_chiller_readings(selected_chiller)
expected_energy_series = get_expected_energy(selected_chiller)
anomaly_data = get_anomaly_data(selected_chiller)
fleet_metrics = get_fleet_metrics()

# Main Tabs Navigation
tab_fleet, tab_investigation, tab_replay = st.tabs([
    "🏢 Fleet Overview",
    f"🔎 Chiller Investigation ({selected_chiller})",
    f"🎬 Anomaly Replay ({selected_chiller})"
])

with tab_fleet:
    render_fleet_metrics_summary(fleet_metrics)
    st.write("---")
    render_chiller_fleet_table(fleet_metrics["chiller_stats"])

with tab_investigation:
    st.markdown(f"## 🔎 CHILLER INVESTIGATION VIEW — `{selected_chiller}`")
    st.caption(f"Displaying real readings from `development_dataset.csv` for **{selected_chiller}**")
    st.write("")

    # 1. Actual Energy Over Time Line Chart
    render_actual_energy_chart(df_chiller, expected_series=expected_energy_series, chiller_id=selected_chiller)
    st.write("---")

    # 2. Contextual Operating Conditions & Selectable Variable Chart
    render_contextual_operating_conditions(df_chiller)
    st.write("---")

    # 3. Contextual Deviation & Persistence
    col1, col2 = st.columns(2)
    with col1:
        render_contextual_deviation_section(anomaly_data)
    with col2:
        render_persistence_analysis(anomaly_data)

    st.write("---")

    # 4. Investigation Report
    render_investigation_report(anomaly_data)

with tab_replay:
    render_anomaly_replay_view(df_chiller, chiller_id=selected_chiller)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.82rem;'>"
    "Chiller Forensics v1.0 — Powered by real dataset & Person 1/2 ML integration adapters"
    "</div>",
    unsafe_allow_html=True
)
