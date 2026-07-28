import streamlit as st
import pandas as pd
from utils.database import get_history
from utils.helpers import render_kpi_card
from utils.charts import prepare_dataframe, create_pie_chart, create_defect_bar_chart, create_detections_time_chart

def show_dashboard():
    st.markdown("""
        <div class="app-header">
            <h1>PCB Detect AI</h1>
            <p>AI-Powered Printed Circuit Board Defect Detection System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Retrieve all records from DB
    records = get_history()
    df = prepare_dataframe(records)
    
    # Calculate stats
    total_processed = len(df)
    defective_count = len(df[df['prediction'] == 'Defective'])
    healthy_count = len(df[df['prediction'] == 'Healthy'])
    
    avg_conf = 0.0
    if total_processed > 0:
        avg_conf = df['confidence'].mean() * 100
        
    # Render KPI Cards in a row of 4 columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Total Inspected", f"{total_processed}", "blue", "🔍")
    with col2:
        render_kpi_card("Defective Boards", f"{defective_count}", "red", "❌")
    with col3:
        render_kpi_card("Healthy Boards", f"{healthy_count}", "green", "✅")
    with col4:
        render_kpi_card("Average Confidence", f"{avg_conf:.1f}%", "orange", "📈")
        
    st.write("---")
    
    # Main Layout split: Charts and Recent Activity
    chart_tab, recent_tab = st.tabs(["📊 Performance Charts", "🕒 Recent Activity Logs"])
    
    with chart_tab:
        # Row 1 of charts: Pie & Defect Bar
        c_col1, c_col2 = st.columns([1, 1])
        with c_col1:
            pie_fig = create_pie_chart(df)
            st.plotly_chart(pie_fig, use_container_width=True, key="dashboard_pie_chart")
        with c_col2:
            bar_fig = create_defect_bar_chart(df)
            st.plotly_chart(bar_fig, use_container_width=True, key="dashboard_defect_bar_chart")
            
        # Row 2: Timeline
        time_fig = create_detections_time_chart(df)
        st.plotly_chart(time_fig, use_container_width=True, key="dashboard_timeline_chart")
        
    with recent_tab:
        st.subheader("Latest Detections")
        if df.empty:
            st.info("No detections registered yet. Use the single or batch detection pages to process images.")
        else:
            # Format and show recent runs (up to 10)
            recent_df = df.head(10)[['filename', 'prediction', 'confidence', 'inference_time', 'timestamp']].copy()
            recent_df['confidence'] = recent_df['confidence'].apply(lambda x: f"{x * 100:.1f}%")
            recent_df['inference_time'] = recent_df['inference_time'].apply(lambda x: f"{x:.1f} ms")
            recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            recent_df.columns = ['Filename', 'Result', 'Confidence Score', 'Processing Latency', 'Timestamp']
            
            # Simple custom HTML table or styled pandas dataframe
            def color_result(val):
                color = '#EF4444' if val == 'Defective' else '#10B981'
                return f'color: {color}; font-weight: bold;'
                
            st.dataframe(
                recent_df.style.map(color_result, subset=['Result']),
                use_container_width=True,
                hide_index=True
            )
