import streamlit as st

def show_settings():
    st.markdown("""
        <div class="app-header">
            <h1>System Settings</h1>
            <p>Configure model inference thresholds and visual detection settings</p>
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
        
    current_settings = st.session_state.settings
    
    # Use standard form to control inputs cleanly
    with st.form("settings_form"):
        st.subheader("Model Inference Settings")
        
        conf_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.05, 
            max_value=1.00, 
            value=float(current_settings.get('conf_threshold', 0.25)),
            step=0.05,
            help="Minimum confidence score required to draw a bounding box around a defect."
        )
        
        iou_threshold = st.slider(
            "Intersection over Union (IoU) Threshold", 
            min_value=0.05, 
            max_value=1.00, 
            value=float(current_settings.get('iou_threshold', 0.45)),
            step=0.05,
            help="Controls duplicate overlapping boxes. Higher values allow more overlapping boxes."
        )
        
        use_gpu = st.toggle(
            "Enable GPU Processing (CUDA)", 
            value=bool(current_settings.get('use_gpu', False)),
            help="Enables CUDA acceleration for model execution if supported hardware is available."
        )
        
        save_images = st.toggle(
            "Save Scans to Historical Database", 
            value=bool(current_settings.get('save_images', True)),
            help="Saves original and annotated images to local disk to compile history dashboards."
        )
        
        st.write("---")
        st.subheader("Visual Overlay Style Settings")
        
        bbox_thickness = st.slider(
            "Bounding Box Outline Thickness (px)", 
            min_value=1, 
            max_value=8, 
            value=int(current_settings.get('bbox_thickness', 2)),
            step=1,
            help="Controls line thickness of defect boxes drawn on prediction frames."
        )
        
        font_size = st.slider(
            "Label Font Text Size", 
            min_value=8, 
            max_value=24, 
            value=int(current_settings.get('font_size', 14)),
            step=1,
            help="Controls text size of defect category tags."
        )
        
        # Form submission button
        submitted = st.form_submit_button("💾 Save Settings Configuration")
        
        if submitted:
            # Update session state
            st.session_state.settings = {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'use_gpu': use_gpu,
                'save_images': save_images,
                'bbox_thickness': bbox_thickness,
                'font_size': font_size
            }
            st.success("✓ Settings updated successfully!")
            
    st.write("---")
    st.subheader("Current Active Configuration Summary")
    st.json(st.session_state.settings)
