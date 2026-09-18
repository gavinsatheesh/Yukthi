import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
from dashboard.data_loader import get_context_variables


def get_status_style(status_str: str) -> Dict[str, str]:
    st_upper = str(status_str).upper()
    if st_upper in ["INVESTIGATE", "HIGH"]:
        return {"icon": "🔴", "label": "INVESTIGATE", "color": "#EF4444", "bg": "rgba(239, 68, 68, 0.15)", "border": "#EF4444"}
    elif st_upper in ["PRIORITY", "CRITICAL"]:
        return {"icon": "🚨", "label": "PRIORITY", "color": "#DC2626", "bg": "rgba(220, 38, 38, 0.2)", "border": "#DC2626"}
    elif st_upper in ["WATCH", "WARNING", "MEDIUM"]:
        return {"icon": "🟡", "label": "WATCH", "color": "#F59E0B", "bg": "rgba(245, 158, 11, 0.15)", "border": "#F59E0B"}
    elif st_upper in ["NORMAL"]:
        return {"icon": "🟢", "label": "NORMAL", "color": "#10B981", "bg": "rgba(16, 185, 129, 0.15)", "border": "#10B981"}
    else:
        return {"icon": "⚪", "label": status_str, "color": "#94A3B8", "bg": "rgba(148, 163, 184, 0.15)", "border": "#64748B"}


def inject_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background-color: #0F172A;
            color: #F8FAFC;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .metric-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }
        
        .metric-title {
            color: #94A3B8;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        
        .metric-value {
            color: #F8FAFC;
            font-size: 1.8rem;
            font-weight: 700;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .fleet-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 16px;
        }

        .rec-item {
            background: rgba(56, 189, 248, 0.08);
            border-left: 4px solid #38BDF8;
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 10px;
            color: #E2E8F0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_fleet_metrics_summary(metrics: Dict[str, Any]):
    st.markdown("### 🏢 Fleet Overview")
    st.caption("Real-time equipment monitoring sourced directly from `development_dataset.csv`")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Chillers</div>
                <div class="metric-value" style="color:#38BDF8;">{metrics['total_chillers']}</div>
                <div style="font-size:0.82rem; color:#94A3B8;">Monitored Assets</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Dataset Readings</div>
                <div class="metric-value" style="color:#F8FAFC;">{metrics['total_readings']:,}</div>
                <div style="font-size:0.82rem; color:#94A3B8;">Half-hourly points</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Date Range</div>
                <div style="font-size:1.05rem; font-weight:700; color:#F8FAFC; margin-top:8px;">{metrics['date_min'][:10]}</div>
                <div style="font-size:0.82rem; color:#94A3B8;">to {metrics['date_max'][:10]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        has_ml = any(c.get("has_ml", False) for c in metrics.get("chiller_stats", []))
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">ML Pipeline Status</div>
                <div style="font-size:1.1rem; font-weight:700; color:{'#10B981' if has_ml else '#F59E0B'}; margin-top:8px;">
                    {'🟢 Model Connected' if has_ml else '🟡 Pending Model Output'}
                </div>
                <div style="font-size:0.82rem; color:#94A3B8;">{'Live anomalies loaded' if has_ml else 'Awaiting Person 1 predictions'}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_chiller_fleet_table(fleet_stats: List[Dict[str, Any]]):
    st.markdown("#### 📋 Fleet Chiller Directory")

    search_query = st.text_input("🔍 Search Chiller ID:", "").strip().upper()
    
    # Filter
    filtered = fleet_stats
    if search_query:
        filtered = [c for c in fleet_stats if search_query in c["equipment"].upper()]

    if not filtered:
        st.info("No chillers matched your search query.")
        return

    cols = st.columns(len(filtered) if len(filtered) <= 3 else 3)
    for idx, item in enumerate(filtered):
        cid = item["equipment"]
        readings = item["readings"]
        avg_e = item["avg_energy"]
        max_e = item["max_energy"]
        status = item.get("status", "UNASSESSED")
        style = get_status_style(status)
        dev = item.get("current_deviation_pct")
        trend = item.get("trend", "N/A")

        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="fleet-card" style="border-left: 4px solid {style['color']};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin: 0; color: #F8FAFC;">{cid}</h3>
                        <span class="status-badge" style="color: {style['color']}; background: {style['bg']}; border: 1px solid {style['border']};">
                            {style['icon']} {style['label']}
                        </span>
                    </div>
                    <hr style="border-color: #334155; margin: 12px 0;" />
                    <div style="font-size: 0.88rem; display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                        <div>Readings: <strong style="color:#F8FAFC;">{readings:,}</strong></div>
                        <div>Avg Energy: <strong style="color:#38BDF8;">{avg_e:.1f} kWh</strong></div>
                        <div>Peak Energy: <strong style="color:#F8FAFC;">{max_e:.1f} kWh</strong></div>
                        <div>Deviation: <strong style="color:{style['color']};">{f'+{dev:.1f}%' if dev is not None else 'Pending ML'}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_actual_energy_chart(df_chiller: pd.DataFrame, expected_series: Optional[pd.Series] = None, chiller_id: str = ""):
    st.markdown(f"### 📈 ACTUAL ENERGY OVER TIME — {chiller_id}")
    st.caption("Time-series energy consumption from `development_dataset.csv` (kWh)")

    if df_chiller.empty:
        st.warning("No time-series readings available for this chiller.")
        return

    energy_col = "Chiller Energy Consumption (kWh)"
    if energy_col not in df_chiller.columns:
        st.error(f"Missing column '{energy_col}' in dataset.")
        return

    fig = go.Figure()

    # Solid line for Actual Energy
    fig.add_trace(
        go.Scatter(
            x=df_chiller["timestamp"],
            y=df_chiller[energy_col],
            name="Actual Energy (Observed)",
            line=dict(color="#F8FAFC", width=2),
            hovertemplate="<b>Timestamp:</b> %{x}<br><b>Actual Energy:</b> %{y:.1f} kWh<extra></extra>"
        )
    )

    # Dashed line for Expected Energy if available
    if expected_series is not None and not expected_series.empty:
        fig.add_trace(
            go.Scatter(
                x=df_chiller["timestamp"],
                y=expected_series.reindex(df_chiller["timestamp"]).values,
                name="Expected Energy (ML Baseline)",
                line=dict(color="#38BDF8", width=2, dash="dash"),
                hovertemplate="<b>Timestamp:</b> %{x}<br><b>Expected Energy:</b> %{y:.1f} kWh<extra></extra>"
            )
        )
    else:
        st.info("ℹ️ **Integration State:** Expected-energy model output will appear here when Person 1 ML pipeline runs.")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        margin=dict(l=40, r=40, t=30, b=40),
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Timestamp", gridcolor="#334155", showgrid=True),
        yaxis=dict(title="Energy Consumption (kWh)", gridcolor="#334155", showgrid=True)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_contextual_operating_conditions(df_chiller: pd.DataFrame):
    st.markdown("### 🌡️ OPERATING CONDITIONS")
    st.caption("Actual numerical context variables sourced directly from the dataset")

    context_cols = get_context_variables(df_chiller)
    if not context_cols:
        st.info("No additional contextual variables available in dataset.")
        return

    # Display KPI summary cards for available variables
    cols = st.columns(min(len(context_cols), 4))
    for idx, col_name in enumerate(context_cols[:4]):
        mean_val = df_chiller[col_name].mean()
        min_val = df_chiller[col_name].min()
        max_val = df_chiller[col_name].max()
        
        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{col_name}</div>
                    <div class="metric-value" style="color:#38BDF8; font-size:1.4rem;">{mean_val:.1f}</div>
                    <div style="font-size:0.8rem; color:#94A3B8; margin-top:4px;">Min: {min_val:.1f} | Max: {max_val:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")
    st.markdown("#### Selectable Operating Context Time-Series Chart")
    
    selected_var = st.selectbox(
        "Select context variable to plot against timestamp:",
        options=context_cols,
        index=0
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_chiller["timestamp"],
            y=df_chiller[selected_var],
            name=selected_var,
            line=dict(color="#10B981", width=2),
            hovertemplate=f"<b>Timestamp:</b> %{{x}}<br><b>{selected_var}:</b> %{{y:.2f}}<extra></extra>"
        )
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        margin=dict(l=40, r=40, t=30, b=40),
        height=320,
        xaxis=dict(title="Timestamp", gridcolor="#334155", showgrid=True),
        yaxis=dict(title=selected_var, gridcolor="#334155", showgrid=True)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_contextual_deviation_section(anomaly_data: Optional[Dict[str, Any]]):
    st.markdown("### ⚖️ CONTEXTUAL DEVIATION")
    
    if not anomaly_data or "expected_energy" not in anomaly_data:
        st.info("ℹ️ **Integration State:** Contextual deviation metrics will populate automatically when Person 1's expected energy model generates outputs.")
        return

    actual = anomaly_data.get("actual_energy", 0.0)
    expected = anomaly_data.get("expected_energy", 0.0)
    dev_pct = anomaly_data.get("deviation_pct", 0.0)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Actual Energy", f"{actual:.1f} kWh")
    with c2:
        st.metric("Expected Energy", f"{expected:.1f} kWh")
    with c3:
        st.metric("Contextual Deviation", f"+{dev_pct:.1f}%", delta=f"{actual-expected:.1f} kWh")

    st.markdown(
        f"""
        <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #F59E0B; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-top: 12px;">
            <p style="margin:0; color:#E2E8F0; font-size:1.02rem;">
                "Energy consumption is <strong>{dev_pct:.1f}% higher than expected</strong> for the observed operating conditions."
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_persistence_analysis(anomaly_data: Optional[Dict[str, Any]]):
    st.markdown("### ⏳ PERSISTENCE ANALYSIS")

    if not anomaly_data:
        st.info("ℹ️ **Integration State:** Persistence timeline and severity evaluation pending Person 2 ML output connection.")
        return

    consec = anomaly_data.get("consecutive_abnormal_readings", 0)
    status = anomaly_data.get("severity", "NORMAL")
    trend = anomaly_data.get("trend", "STABLE")
    style = get_status_style(status)

    st.markdown(
        f"""
        <div style="background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #94A3B8; font-size: 0.85rem;">STATUS EVALUATION</span><br/>
                    <span class="status-badge" style="color: {style['color']}; background: {style['bg']}; border: 1px solid {style['border']}; margin-top: 4px;">
                        {style['icon']} {status}
                    </span>
                </div>
                <div>
                    <span style="color: #94A3B8; font-size: 0.85rem;">CONSECUTIVE ABNORMAL READINGS</span><br/>
                    <strong style="color: #F59E0B; font-size: 1.4rem;">{consec} readings</strong>
                </div>
                <div>
                    <span style="color: #94A3B8; font-size: 0.85rem;">DEVIATION TREND</span><br/>
                    <strong style="color: #F8FAFC; font-size: 1.4rem;">{trend}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_investigation_report(anomaly_data: Optional[Dict[str, Any]]):
    st.markdown("### 📝 INVESTIGATION REPORT")

    if not anomaly_data:
        st.info("ℹ️ **Integration State:** Structured investigation report will display once ML model anomaly scoring is generated.")
        return

    summary = anomaly_data.get("investigation_summary", "Review contextual parameter shifts around anomaly window.")
    recommendations = anomaly_data.get("recommended_investigation", [])

    st.markdown(
        f"""
        <div style="background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
            <h4 style="margin-top:0; color:#38BDF8;">INVESTIGATION REQUIRED</h4>
            <p style="color:#E2E8F0; line-height:1.5;">"{summary}"</p>
            <h5 style="color:#94A3B8; margin-top:16px; text-transform:uppercase; font-size:0.85rem;">Suggested Engineering Actions</h5>
        """,
        unsafe_allow_html=True
    )

    for rec in recommendations:
        st.markdown(
            f"""
            <div class="rec-item">
                <span style="color:#38BDF8; font-weight:700;">✓</span> {rec}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_anomaly_replay_view(df_chiller: pd.DataFrame, chiller_id: str):
    st.markdown(f"### 🎬 ANOMALY REPLAY — {chiller_id}")
    st.caption("Chronological timeline replay using real timestamps from `development_dataset.csv`")

    if df_chiller.empty:
        st.warning("No time-series data available for replay.")
        return

    timestamps = list(df_chiller["timestamp"])
    
    selected_idx = st.slider(
        "⏱️ Drag timeline slider to inspect timestamp:",
        min_value=0,
        max_value=len(timestamps) - 1,
        value=min(100, len(timestamps) - 1),
        format="Reading %d"
    )

    row = df_chiller.iloc[selected_idx]
    energy_col = "Chiller Energy Consumption (kWh)"

    st.markdown(
        f"""
        <div style="background: #0F172A; border: 1px solid #334155; border-radius: 12px; padding: 18px; margin-top: 16px;">
            <h4 style="color:#38BDF8; margin:0 0 12px 0;">TIMESTAMP: {row['timestamp']}</h4>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-size: 0.95rem;">
                <div>Actual Energy: <strong style="color:#F8FAFC;">{row[energy_col]:.1f} kWh</strong></div>
                <div>Cooling Water Temp: <strong style="color:#38BDF8;">{row.get('Cooling Water Temperature (C)', 0.0):.1f} °C</strong></div>
                <div>Chilled Water Rate: <strong style="color:#10B981;">{row.get('Chilled Water Rate (L/sec)', 0.0):.1f} L/s</strong></div>
                <div>Building Load: <strong style="color:#F59E0B;">{row.get('Building Load (RT)', 0.0):.1f} RT</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
