import streamlit as st
import pandas as pd
from utils.database import get_history
from utils.helpers import render_page_header, render_kpi_card
from utils.charts import (
    prepare_dataframe, create_pie_chart, create_defect_bar_chart,
    create_detections_time_chart, create_daily_hourly_heatmap, create_spatial_heatmap
)


def show_analytics():
    render_page_header(
        "Production Analytics",
        "Analytics",
        "Advanced production metrics, yield rates, and spatial defect heatmaps for QC analysis"
    )

    records = get_history()
    df      = prepare_dataframe(records)

    if df.empty:
        st.info("No records available. Run some inspections first to populate analytics.")
        return

    # ── Compute metrics ───────────────────────────────────────────────────────
    total       = len(df)
    defective_n = len(df[df['prediction'] == 'Defective'])
    yield_rate  = ((total - defective_n) / total * 100) if total > 0 else 100.0
    avg_conf    = df['confidence'].mean() * 100
    avg_latency = df['inference_time'].mean()

    all_defects = []
    for d in df['defects_list'].dropna():
        if d:
            all_defects.extend([x.strip() for x in d.split(',')])

    if all_defects:
        top_defect = pd.Series(all_defects).mode()[0]
        defect_str = top_defect
    else:
        defect_str = "None"

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        color = "green" if yield_rate > 90 else "amber"
        render_kpi_card("Production Yield",  f"{yield_rate:.1f}%", "Pass rate",            "⚙️",  color)
    with c2:
        render_kpi_card("Top Defect",        defect_str,           "Most frequent",         "⚠️",  "red")
    with c3:
        render_kpi_card("Avg Confidence",    f"{avg_conf:.1f}%",   "Detection accuracy",    "📊",  "blue")
    with c4:
        render_kpi_card("Avg Latency",       f"{avg_latency:.1f} ms", "Per inference",      "⚡",  "purple")

    st.write("")

    # ── Yield & Category Charts ───────────────────────────────────────────────
    st.markdown('<div class="section-header"><p class="section-title">Yield &amp; Category Analysis</p><p class="section-subtitle">Overall pass / fail distribution and defect frequency</p></div>', unsafe_allow_html=True)

    cy1, cy2 = st.columns(2, gap="small")
    with cy1:
        st.plotly_chart(create_pie_chart(df),          use_container_width=True, key="an_pie")
    with cy2:
        st.plotly_chart(create_defect_bar_chart(df),   use_container_width=True, key="an_bar")

    # ── Timeline Chart ────────────────────────────────────────────────────────
    st.plotly_chart(create_detections_time_chart(df),  use_container_width=True, key="an_trend")

    # ── Temporal & Spatial ────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><p class="section-title">Temporal &amp; Spatial Diagnostics</p></div>', unsafe_allow_html=True)

    cd1, cd2 = st.columns(2, gap="small")
    with cd1:
        st.plotly_chart(create_daily_hourly_heatmap(df), use_container_width=True, key="an_heatmap")
    with cd2:
        st.plotly_chart(create_spatial_heatmap(df),      use_container_width=True, key="an_spatial")
