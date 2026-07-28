import streamlit as st
import pandas as pd
from PIL import Image
import io
import time
from utils.predict import predict_image
from utils.database import add_detection

def show_batch_detection():
    st.markdown("""
        <div class="app-header">
            <h1>Batch Detection</h1>
            <p>Upload multiple PCB images for high-throughput automated inspect processing</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize settings if not present
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'use_gpu': False,
            'save_images': True,
            'bbox_thickness': 2,
            'font_size': 14
        }
        
    uploaded_files = st.file_uploader(
        "Upload PCB Board Images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        total_files = len(uploaded_files)
        st.info(f"Loaded {total_files} files for batch inspection.")
        
        if st.button("🚀 Begin Batch Inspection"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_data = []
            
            # Start loop
            for idx, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                status_text.markdown(f"**Processing {idx + 1} of {total_files}:** `{filename}`")
                
                try:
                    # Read image
                    image = Image.open(uploaded_file).convert("RGB")
                    
                    # Predict
                    conf = st.session_state.settings['conf_threshold']
                    iou = st.session_state.settings['iou_threshold']
                    thick = st.session_state.settings['bbox_thickness']
                    f_size = st.session_state.settings['font_size']
                    gpu = st.session_state.settings['use_gpu']
                    
                    prediction, avg_confidence, boxes, proc_time, anno_image = predict_image(
                        image, 
                        conf_threshold=conf, 
                        iou_threshold=iou,
                        bbox_thickness=thick,
                        font_size=f_size,
                        use_gpu=gpu
                    )
                    
                    # Store results
                    results_data.append({
                        "Filename": filename,
                        "Prediction": prediction,
                        "Confidence Score": f"{avg_confidence * 100:.1f}%",
                        "Defects Detected": len(boxes),
                        "Defect Names": ", ".join(list(set([box['label'] for box in boxes]))) if boxes else "None",
                        "Processing Time (ms)": proc_time
                    })
                    
                    # Save results if enabled in settings
                    if st.session_state.settings['save_images']:
                        defect_list = [box['label'] for box in boxes]
                        add_detection(
                            filename=filename,
                            prediction=prediction,
                            defect_count=len(boxes),
                            defects_list=defect_list,
                            confidence=avg_confidence,
                            inference_time=proc_time,
                            original_image=image,
                            annotated_image=anno_image
                        )
                        
                except Exception as e:
                    results_data.append({
                        "Filename": filename,
                        "Prediction": "ERROR",
                        "Confidence Score": "0.0%",
                        "Defects Detected": 0,
                        "Defect Names": f"Failed: {str(e)}",
                        "Processing Time (ms)": 0.0
                    })
                    
                # Update progress
                progress_bar.progress((idx + 1) / total_files)
                
            status_text.success(f"✓ Completed batch inspection of {total_files} PCB boards.")
            
            # Display results dataframe
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Convert results table to csv for download
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Batch Results CSV",
                data=csv_data,
                file_name=f"pcb_batch_results_{int(time.time())}.csv",
                mime="text/csv"
            )
