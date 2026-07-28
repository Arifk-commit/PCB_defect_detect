import streamlit as st
from PIL import Image
import io
import os
from utils.predict import predict_image
from utils.database import add_detection

def show_image_detection():
    st.markdown("""
        <div class="app-header">
            <h1>Single Image Detection</h1>
            <p>Upload a PCB image to run instant defect analysis</p>
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
        
    uploaded_file = st.file_uploader("Upload PCB Board Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Layout columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
                
            # Perform prediction
            with st.spinner("Analyzing image for defects..."):
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
                
            with col2:
                st.subheader("Inspection Result")
                st.image(anno_image, use_container_width=True)
                
            # Prediction Summary Card
            st.write("---")
            st.subheader("Analysis Summary")
            
            sum_col1, sum_col2 = st.columns([1, 2])
            
            with sum_col1:
                # Render clean custom status card
                if prediction == "Healthy":
                    st.markdown("""
                        <div class="prediction-badge badge-healthy">🟢 HEALTHY BOARD</div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="prediction-badge badge-defective">🔴 DEFECTS DETECTED</div>
                    """, unsafe_allow_html=True)
                    
                st.write("")
                st.metric("Total Defect Counts", len(boxes))
                st.metric("Avg Detection Confidence", f"{avg_confidence * 100:.1f}%")
                st.metric("Processing Latency", f"{proc_time} ms")
                
            with sum_col2:
                st.markdown("**Defect Class Confidence Distribution:**")
                if len(boxes) == 0:
                    st.success("Perfect scan. No visual defect anomalies found on the PCB traces or pads.")
                else:
                    # List of defects
                    defect_list = [box['label'] for box in boxes]
                    defect_summary = ", ".join(set(defect_list))
                    st.info(f"**Identified Defects:** {defect_summary}")
                    
                    # Confidence progress bars
                    for box in boxes:
                        label = box['label']
                        val = box['confidence']
                        st.write(f"{label} ({val * 100:.1f}%)")
                        st.progress(val)
                        
            # Save results if enabled
            if st.session_state.settings['save_images']:
                # Save execution
                defect_list = [box['label'] for box in boxes]
                add_detection(
                    filename=uploaded_file.name,
                    prediction=prediction,
                    defect_count=len(boxes),
                    defects_list=defect_list,
                    confidence=avg_confidence,
                    inference_time=proc_time,
                    original_image=image,
                    annotated_image=anno_image
                )
                st.toast("Detection saved to local logs database", icon="💾")
                
            # Download file prep
            buffered = io.BytesIO()
            anno_image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            
            st.download_button(
                label="📥 Download Annotated Image Result",
                data=img_bytes,
                file_name=f"inspected_{uploaded_file.name}",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Failed to process image: {e}")
            st.warning("Please make sure the uploaded image is a valid, uncorrupted image file format.")
