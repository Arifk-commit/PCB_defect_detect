import streamlit as st
import pandas as pd
from utils.database import get_history
from utils.helpers import render_kpi_card
from utils.charts import (
    prepare_dataframe, 
    create_pie_chart, 
    create_defect_bar_chart, 
    create_detections_time_chart,
    create_daily_hourly_heatmap,
    create_spatial_heatmap
)

def show_analytics():
    st.markdown("""
        <div class="app-header">
            <h1>Industrial Analytics Dashboard</h1>
            <p>Advanced metrics and spatial hot-spots for PCB manufacturing quality control</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Fetch data
    records = get_history()
    df = prepare_dataframe(records)
    
    if df.empty:
        st.info("No records inside the database. Perform detections first to populate the analytics views.")
        return
        
    # Calculate key analytics metrics
    total_runs = len(df)
    defective_df = df[df['prediction'] == 'Defective']
    defective_count = len(defective_df)
    yield_rate = ((total_runs - defective_count) / total_runs) * 100 if total_runs > 0 else 100.0
    
    # Find most common defect
    all_defects = []
    for d_str in df['defects_list'].dropna():
        if d_str:
            all_defects.extend(d_str.split(','))
            
    if all_defects:
        most_common_defect = pd.Series(all_defects).mode()[0]
        defect_mode_count = pd.Series(all_defects).value_counts().iloc[0]
        defect_stat_str = f"{most_common_defect} ({defect_mode_count} occurrences)"
    else:
        defect_stat_str = "None Recorded"
        
    avg_confidence = df['confidence'].mean() * 100
    
    # KPI cards row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Production Yield", f"{yield_rate:.1f}%", "green" if yield_rate > 90 else "orange", "⚙️")
    with col2:
        render_kpi_card("Most Common Defect", defect_stat_str, "red" if all_defects else "blue", "⚠️")
    with col3:
        render_kpi_card("Avg Inspect Confidence", f"{avg_confidence:.1f}%", "blue", "📊")
    with col4:
        render_kpi_card("Average Latency", f"{df['inference_time'].mean():.1f} ms", "orange", "⚡")
        
    st.write("---")
    
    # Layout sections
    st.subheader("Yield and Category Analytics")
    col_y1, col_y2 = st.columns(2)
    with col_y1:
        st.plotly_chart(create_pie_chart(df), use_container_width=True)
    with col_y2:
        st.plotly_chart(create_defect_bar_chart(df), use_container_width=True)
        
    st.write("---")
    st.plotly_chart(create_detections_time_chart(df), use_container_width=True)
    
    st.write("---")
    st.subheader("Temporal and Spatial Diagnostics")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.plotly_chart(create_daily_hourly_heatmap(df), use_container_width=True)
    with col_d2:
        st.plotly_chart(create_spatial_heatmap(df), use_container_width=True)
