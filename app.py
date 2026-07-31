import streamlit as st
import os

# ── Page config MUST be first ─────────────────────────────────────────────────
st.set_page_config(
    page_title="PCB Detect AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils.database import init_db
from utils.helpers import (
    inject_custom_css, load_logo_base64,
    render_navbar, render_footer
)
from utils.model_loader import get_model
from views.dashboard import show_dashboard
from views.image_detection import show_image_detection
from views.batch_detection import show_batch_detection
from views.live_camera import show_live_camera
from views.history import show_history
from views.analytics import show_analytics
from views.model_info import show_model_info
from views.settings import show_settings

# ── Nav config ────────────────────────────────────────────────────────────────
NAV_ITEMS = [
    ("⬛  Dashboard",          "Dashboard"),
    ("🔍  Image Detection",    "Image Detection"),
    ("📦  Batch Detection",    "Batch Detection"),
    ("📷  Live Camera",        "Live Camera"),
    ("🗂  Detection History",  "Detection History"),
    ("📊  Analytics",          "Analytics"),
    ("🤖  Model Information",  "Model Information"),
    ("⚙️  Settings",           "Settings"),
]

PAGE_TITLES = {
    "Dashboard":         "Dashboard",
    "Image Detection":   "Single Image Detection",
    "Batch Detection":   "Batch Detection",
    "Live Camera":       "Live Camera Feed",
    "Detection History": "Detection History",
    "Analytics":         "Analytics",
    "Model Information": "Model Information",
    "Settings":          "Settings",
}


@st.dialog("Model Loaded Successfully")
def show_model_loaded_dialog():
    st.success("✅ **YOLO AI Model Initialized** (`models/best.pt`)")
    st.markdown("""
    The PCB defect inspection model has been loaded into memory and is ready for real-time analysis.
    - **Weights Path**: `models/best.pt`
    - **Runtime Status**: Active & Ready
    """)
    if st.button("Continue to Dashboard", type="primary", use_container_width=True):
        st.rerun()


def main():
    # 1. Init DB
    init_db()

    # 2. Inject CSS + auto-expand JS
    inject_custom_css()

    # 3. Session state defaults
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'conf_threshold': 0.25,
            'iou_threshold':  0.45,
            'use_gpu':        False,
            'save_images':    True,
            'bbox_thickness': 2,
            'font_size':      14,
        }
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False

    # 4. Model status
    model        = get_model()
    model_loaded = model is not None

    if model_loaded and not st.session_state.get('model_dialog_shown', False):
        st.session_state.model_dialog_shown = True
        st.toast("✅ YOLO Model Loaded Successfully (models/best.pt)", icon="🤖")
        show_model_loaded_dialog()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        logo_b64 = load_logo_base64()

        # Logo
        st.markdown(f"""
        <div style="padding:20px 16px 12px 16px;">
            <img src="data:image/svg+xml;base64,{logo_b64}" width="176" style="display:block;" />
        </div>
        """, unsafe_allow_html=True)

        # Nav label
        st.markdown("""
        <hr style="border:none;border-top:1px solid #1E293B;margin:0 12px 10px;">
        <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;color:#475569;
                    text-transform:uppercase;padding:0 18px 6px;">Navigation</div>
        """, unsafe_allow_html=True)

        # Navigation radio
        labels = [lbl for lbl, _ in NAV_ITEMS]
        names  = [nm  for _, nm  in NAV_ITEMS]

        selected_label = st.radio(
            "Navigation",
            labels,
            index=0,
            label_visibility="collapsed"
        )
        selected_page = names[labels.index(selected_label)]

        # System Status
        if model_loaded:
            dot_color   = "#22C55E"
            status_text = "YOLOv11 Loaded"
        else:
            dot_color   = "#3B82F6"
            status_text = "Simulator Active"

        conf_val = st.session_state.settings['conf_threshold']
        iou_val  = st.session_state.settings['iou_threshold']
        device   = "GPU (CUDA)" if st.session_state.settings['use_gpu'] else "CPU"

        st.markdown(f"""
        <hr style="border:none;border-top:1px solid #1E293B;margin:12px 12px 10px;">
        <div style="padding:0 16px 20px;">
            <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;color:#475569;
                        text-transform:uppercase;margin-bottom:10px;">System Status</div>
            <div style="display:flex;align-items:center;gap:9px;background:#1E293B;
                        border-radius:9px;padding:10px 13px;margin-bottom:12px;">
                <span style="width:8px;height:8px;border-radius:50%;background:{dot_color};
                             box-shadow:0 0 8px {dot_color};display:inline-block;flex-shrink:0;"></span>
                <span style="font-size:13px;font-weight:600;color:#E2E8F0;">{status_text}</span>
            </div>
            <div style="font-size:11.5px;color:#475569;line-height:2.1;padding:0 2px;">
                Device: <span style="color:#94A3B8;">{device}</span><br>
                Confidence: <span style="color:#94A3B8;">{conf_val}</span><br>
                IoU: <span style="color:#94A3B8;">{iou_val}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Top Navbar ────────────────────────────────────────────────────────────
    render_navbar(PAGE_TITLES[selected_page], model_loaded)

    # ── Page Routing ──────────────────────────────────────────────────────────
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

    # ── Footer ────────────────────────────────────────────────────────────────
    render_footer()


if __name__ == "__main__":
    main()
