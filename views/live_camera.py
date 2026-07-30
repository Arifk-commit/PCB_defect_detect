import streamlit as st
try:
    import cv2
except ImportError:
    cv2 = None
import time
import numpy as np
from PIL import Image, ImageDraw
from utils.predict import predict_image

def show_live_camera():
    st.markdown("""
        <div class="app-header">
            <h1>Live Camera Inspection</h1>
            <p>Real-time visual inspection from connected manufacturing webcams</p>
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
        
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
        
    # Controls layout
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        if st.button("🟢 Start Live Camera", use_container_width=True, disabled=st.session_state.camera_running):
            st.session_state.camera_running = True
            st.rerun()
    with col_ctrl2:
        if st.button("🔴 Stop Live Camera", use_container_width=True, disabled=not st.session_state.camera_running):
            st.session_state.camera_running = False
            st.rerun()
            
    # Metric Placeholders
    m_col1, m_col2, m_col3 = st.columns(3)
    fps_metric = m_col1.empty()
    defect_metric = m_col2.empty()
    conf_metric = m_col3.empty()
    
    # Video Frame Placeholder
    frame_placeholder = st.empty()
    
    if st.session_state.camera_running:
        camera_failed = False
        cap = None
        if cv2 is None:
            camera_failed = True
            st.warning("⚠️ Physical camera hardware unavailable on server. Initiating simulated production line feed...")
        else:
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    camera_failed = True
                    st.warning("⚠️ Physical webcam device not detected. Initiating simulated production line feed...")
                    if cap:
                        cap.release()
            except Exception:
                camera_failed = True
                st.warning("⚠️ Physical webcam device not available. Initiating simulated production line feed...")
            
        prev_time = time.time()
        
        # Simulation parameters for synthetic PCB panning stream
        sim_frame_idx = 0
        
        # Loop runs as long as camera state is True
        while st.session_state.camera_running:
            start_loop_time = time.time()
            
            if not camera_failed:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from webcam device.")
                    break
                # Convert BGR from OpenCV to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_to_predict = Image.fromarray(frame_rgb)
            else:
                # --- GENERATE SYNTHETIC PCB STREAM FRAME ---
                # A shifting green canvas with gold components moving slightly to simulate conveyor belt movement
                canvas_w, canvas_h = 640, 480
                img = Image.new('RGB', (canvas_w, canvas_h), color='#0F5132')
                draw = ImageDraw.Draw(img)
                
                # Pan offset based on loop counter
                pan_offset = (sim_frame_idx * 5) % canvas_w
                sim_frame_idx += 1
                
                # Draw mock tracks
                for trace_y in range(50, canvas_h, 80):
                    draw.line([(0, trace_y), (canvas_w, trace_y)], fill='#D4AF37', width=2)
                    
                # Draw a microchip that enters the frame
                chip_x = (canvas_w - pan_offset) % (canvas_w + 150) - 100
                chip_y = 150
                draw.rectangle([chip_x, chip_y, chip_x + 120, chip_y + 120], fill='#1E293B', outline='#E2E8F0', width=2)
                draw.text((chip_x + 35, chip_y + 50), "PCB-V11", fill='#FFFFFF')
                
                # Pins
                for pin_idx in range(6):
                    px = chip_x + 10 + (pin_idx * 18)
                    draw.rectangle([px, chip_y - 8, px + 8, chip_y], fill='#D4AF37')
                    draw.rectangle([px, chip_y + 120, px + 8, chip_y + 128], fill='#D4AF37')
                    
                image_to_predict = img
                
            # Perform prediction
            conf = st.session_state.settings['conf_threshold']
            iou = st.session_state.settings['iou_threshold']
            thick = st.session_state.settings['bbox_thickness']
            f_size = st.session_state.settings['font_size']
            gpu = st.session_state.settings['use_gpu']
            
            prediction, avg_confidence, boxes, proc_time, anno_image = predict_image(
                image_to_predict, 
                conf_threshold=conf, 
                iou_threshold=iou,
                bbox_thickness=thick,
                font_size=f_size,
                use_gpu=gpu
            )
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Update metrics
            fps_metric.metric("Real-Time FPS", f"{fps:.1f} FPS")
            defect_metric.metric("Anomalies Detected", f"{len(boxes)}")
            conf_metric.metric("Avg Conf Score", f"{avg_confidence * 100:.1f}%" if len(boxes) > 0 else "N/A")
            
            # Display frame
            frame_placeholder.image(anno_image, use_container_width=True)
            
            # Stop if the user turns off camera
            if not st.session_state.camera_running:
                break
                
            # Add short buffer to match realistic sensor framerate
            time.sleep(0.03)
            
        if not camera_failed and cap is not None:
            cap.release()
            
        frame_placeholder.empty()
        fps_metric.empty()
        defect_metric.empty()
        conf_metric.empty()
        st.success("Camera stream closed successfully.")
    else:
        # Camera is off
        st.info("Live visual feed is currently offline. Press the 'Start Live Camera' button to establish a connection.")
        
        # Display a mockup image of an industrial workbench
        st.image("https://images.unsplash.com/photo-1591405351990-4726e331f141?auto=format&fit=crop&w=1200&q=80", 
                 caption="Industrial Vision Inspection Station Setup", use_container_width=True)
