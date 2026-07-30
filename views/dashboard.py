import streamlit as st
import pandas as pd
from utils.database import get_history
from utils.helpers import render_kpi_card, render_section_header
from utils.charts import prepare_dataframe, create_pie_chart, create_defect_bar_chart, create_detections_time_chart


def show_dashboard():
    records = get_history()
    df      = prepare_dataframe(records)

    total     = len(df)
    defective = len(df[df['prediction'] == 'Defective']) if not df.empty else 0
    healthy   = len(df[df['prediction'] == 'Healthy'])   if not df.empty else 0
    avg_conf  = (df['confidence'].mean() * 100)          if total > 0 else 0.0

    # ── Hero Card ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-card">
        <div>
            <div class="hero-card-title">Welcome back to PCB Detect AI</div>
            <div class="hero-card-sub">Industrial Vision Inspection System · Real-time PCB defect analysis</div>
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <span class="hero-stat-value">{total}</span>
                <span class="hero-stat-label">Total Inspected</span>
            </div>
            <div class="hero-stat">
                <span class="hero-stat-value" style="color:#6EE7B7;">{healthy}</span>
                <span class="hero-stat-label">Healthy</span>
            </div>
            <div class="hero-stat">
                <span class="hero-stat-value" style="color:#FCA5A5;">{defective}</span>
                <span class="hero-stat-label">Defective</span>
            </div>
            <div class="hero-stat">
                <span class="hero-stat-value">{avg_conf:.1f}%</span>
                <span class="hero-stat-label">Avg Confidence</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        render_kpi_card("Total Inspected",  str(total),     "Boards scanned",    "🔍", "blue")
    with c2:
        render_kpi_card("Healthy Boards",   str(healthy),   "Passed inspection", "✅", "green")
    with c3:
        render_kpi_card("Defective Boards", str(defective), "Anomalies found",   "⚠️", "red")
    with c4:
        render_kpi_card("Avg Confidence",   f"{avg_conf:.1f}%", "Detection accuracy", "📈", "amber")

    st.write("")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_charts, tab_recent = st.tabs(["📊  Performance Charts", "🕒  Recent Activity"])

    with tab_charts:
        col1, col2 = st.columns(2, gap="small")
        with col1:
            st.plotly_chart(create_pie_chart(df),         use_container_width=True, key="db_pie")
        with col2:
            st.plotly_chart(create_defect_bar_chart(df),  use_container_width=True, key="db_bar")

        st.plotly_chart(create_detections_time_chart(df), use_container_width=True, key="db_trend")

    with tab_recent:
        render_section_header("Latest Detections", "Most recent 10 inspection records")
        if df.empty:
            st.info("No detections yet. Run an inspection on the Image Detection or Batch Detection pages.")
        else:
            recent = df.head(10)[['filename', 'prediction', 'confidence', 'inference_time', 'timestamp']].copy()
            recent['confidence']     = recent['confidence'].apply(lambda x: f"{x*100:.1f}%")
            recent['inference_time'] = recent['inference_time'].apply(lambda x: f"{x:.1f} ms")
            recent['timestamp']      = recent['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            recent.columns = ['Filename', 'Result', 'Confidence', 'Processing Time', 'Timestamp']

            def colour_result(val):
                c = '#15803D' if val == 'Healthy' else '#B91C1C'
                return f'color: {c}; font-weight: 700;'

            st.dataframe(
                recent.style.map(colour_result, subset=['Result']),
                use_container_width=True, hide_index=True
            )
