import streamlit as st
import os

# Set page configuration first
st.set_page_config(
    page_title="PCB Detect AI - Vision Inspection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils.database import init_db
from utils.helpers import inject_custom_css, load_logo_base64
from pages.dashboard import show_dashboard
from pages.image_detection import show_image_detection
from pages.batch_detection import show_batch_detection
from pages.live_camera import show_live_camera
from pages.history import show_history
from pages.analytics import show_analytics
from pages.model_info import show_model_info
from pages.settings import show_settings

def main():
    # 1. Initialize SQLite Database (and seed mock historical entries if empty)
    init_db()
    
    # 2. Inject custom premium CSS rules
    inject_custom_css()
    
    # 3. Setup Default Session States
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
        
    # 4. Sidebar Branding & Navigation
    logo_base64 = load_logo_base64()
    st.sidebar.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="data:image/svg+xml;base64,{logo_base64}" width="220" />
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.write("---")
    
    # Navigation list
    menu_options = [
        "Dashboard", 
        "Image Detection", 
        "Batch Detection", 
        "Live Camera", 
        "Detection History", 
        "Analytics", 
        "Model Information", 
        "Settings"
    ]
    
    selected_page = st.sidebar.radio(
        "NAVIGATION MENU", 
        menu_options,
        index=0
    )
    
    st.sidebar.write("---")
    
    # 5. Sidebar System Status Indicator
    st.sidebar.subheader("SYSTEM DIAGNOSTICS")
    
    # Check if a custom weights model is present
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')
    model_exists = os.path.exists(model_path)
    
    if model_exists:
        status_label = "🟢 Ready (YOLOv11 Loaded)"
        model_type_text = "Custom YOLOv11 Model active"
    else:
        status_label = "🔵 Ready (Simulator Mode)"
        model_type_text = "Fallback simulator active"
        
    st.sidebar.markdown(f"""
        <div class="status-indicator">
            <div class="status-dot" style="background-color: {'#10B981' if model_exists else '#3B82F6'}; box-shadow: 0 0 8px {'#10B981' if model_exists else '#3B82F6'};"></div>
            <div class="status-text">{status_label}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.caption(f"Engine: {model_type_text}")
    st.sidebar.caption(f"Device: {'GPU (CUDA)' if st.session_state.settings['use_gpu'] else 'CPU'}")
    st.sidebar.caption("Software Version: v1.0.0")
    
    # 6. Page View Routing
    if selected_page == "Dashboard":
        show_dashboard()
    elif selected_page == "Image Detection":
        show_image_detection()
    elif selected_page == "Batch Detection":
        show_batch_detection()
    elif selected_page == "Live Camera":
        show_live_camera()
    elif selected_page == "Detection History":
        show_history()
    elif selected_page == "Analytics":
        show_analytics()
    elif selected_page == "Model Information":
        show_model_info()
    elif selected_page == "Settings":
        show_settings()
        
    # 7. Global Sticky Footer
    st.markdown("""
        <div class="footer-text">
            <strong>PCB Detect AI</strong> - Industrial Vision Quality Inspection Suite | Built with Streamlit and Ultralytics YOLOv11 | Version 1.0.0
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
