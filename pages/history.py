import streamlit as st
import pandas as pd
import os
from utils.database import get_history, delete_record, clear_all_history
from PIL import Image

def show_history():
    st.markdown("""
        <div class="app-header">
            <h1>Detection History Logs</h1>
            <p>Review, filter, and export historical inspection records</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Filtering Section
    with st.expander("🔍 Search & Advanced Filter Options", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            search_query = st.text_input("Search Filename", placeholder="e.g. pcb_inspect")
            prediction_filter = st.selectbox("Result Status", ["All", "Healthy", "Defective"])
        with col2:
            date_filter = st.date_input("Filter Date", value=None)
            defect_types = ["All", "Missing Hole", "Mouse Bite", "Open Circuit", "Short", "Spur", "Spurious Copper"]
            defect_filter = st.selectbox("Defect Category", defect_types)
        with col3:
            conf_min = st.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.0, 0.05)
            
    # Fetch filtered history records
    records = get_history(
        search_query=search_query,
        date_filter=date_filter,
        confidence_min=conf_min,
        defect_type_filter=defect_filter,
        prediction_filter=prediction_filter
    )
    
    if not records:
        st.info("No inspection records match the current filter criteria.")
        return
        
    df_records = pd.DataFrame(records)
    
    # Action buttons
    act_col1, act_col2 = st.columns([1, 4])
    with act_col1:
        # CSV Export
        csv_df = df_records[['id', 'filename', 'prediction', 'defect_count', 'defects_list', 'confidence', 'inference_time', 'timestamp']].copy()
        csv_df['confidence'] = csv_df['confidence'].apply(lambda x: f"{x * 100:.1f}%")
        csv_data = csv_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Filtered Logs to CSV",
            data=csv_data,
            file_name="pcb_detection_logs.csv",
            mime="text/csv",
            use_container_width=True
        )
    with act_col2:
        # Clear database confirmation
        if st.button("⚠️ Wipe Database Logs", type="secondary"):
            st.session_state.confirm_clear = True
            
    if st.session_state.get('confirm_clear', False):
        st.warning("Are you absolutely sure you want to delete all historical logs and image scans? This action is irreversible.")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("Yes, Clear Everything", type="primary", use_container_width=True):
                clear_all_history()
                st.session_state.confirm_clear = False
                st.success("All logs cleared successfully.")
                st.rerun()
        with col_c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()
                
    st.write("---")
    
    # Show entries with details and image previews
    st.subheader(f"Inspection Results ({len(records)} entries)")
    
    for record in records:
        rec_id = record['id']
        filename = record['filename']
        pred = record['prediction']
        d_count = record['defect_count']
        defects = record['defects_list']
        conf = record['confidence']
        latency = record['inference_time']
        ts = record['timestamp']
        orig_img_path = record['original_image_path']
        anno_img_path = record['annotated_image_path']
        
        # Format label title
        badge = "🟢 HEALTHY" if pred == "Healthy" else f"🔴 DEFECTIVE ({d_count} anomalies)"
        expander_title = f"{ts} - {filename} | Result: {badge} | Conf: {conf*100:.1f}% | Latency: {latency:.1f}ms"
        
        with st.expander(expander_title):
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("**Original Inspection Target:**")
                if orig_img_path and os.path.exists(orig_img_path):
                    st.image(Image.open(orig_img_path), use_container_width=True)
                else:
                    st.warning("Original image file not found on disk.")
                    
            with detail_col2:
                st.markdown("**AI Visual Detections:**")
                if anno_img_path and os.path.exists(anno_img_path):
                    st.image(Image.open(anno_img_path), use_container_width=True)
                else:
                    st.warning("Annotated image file not found on disk.")
                    
            # Row for deletion button
            del_col1, del_col2 = st.columns([5, 1])
            with del_col1:
                if defects:
                    st.markdown(f"**Identified anomalies list:** `{defects}`")
            with del_col2:
                if st.button("🗑️ Delete Record", key=f"del_{rec_id}", use_container_width=True):
                    delete_record(rec_id)
                    st.toast(f"Record {filename} deleted.", icon="🗑️")
                    st.rerun()
