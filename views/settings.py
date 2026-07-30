import streamlit as st
from utils.helpers import render_page_header


def show_settings():
    render_page_header(
        "Configuration",
        "Settings",
        "Configure inference thresholds, visual overlays, and data storage preferences"
    )

    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'conf_threshold': 0.25, 'iou_threshold': 0.45,
            'use_gpu': False, 'save_images': True,
            'bbox_thickness': 2, 'font_size': 14
        }

    s = st.session_state.settings

    with st.form("settings_form"):
        # ── Inference Card ────────────────────────────────────────────────────
        st.markdown("""
        <div class="settings-card">
            <div class="settings-card-title">🎯 Inference Parameters</div>
            <div class="settings-card-sub">Control how the AI model detects and filters defect candidates</div>
        </div>
        """, unsafe_allow_html=True)

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05, max_value=1.00,
            value=float(s.get('conf_threshold', 0.25)),
            step=0.05,
            help="Minimum confidence score to draw a bounding box. Lower = more detections (may include false positives)."
        )
        iou_threshold = st.slider(
            "IoU Threshold (Non-Maximum Suppression)",
            min_value=0.05, max_value=1.00,
            value=float(s.get('iou_threshold', 0.45)),
            step=0.05,
            help="Controls duplicate box suppression. Lower = fewer overlapping boxes."
        )

        st.write("")

        # ── Hardware Card ─────────────────────────────────────────────────────
        st.markdown("""
        <div class="settings-card">
            <div class="settings-card-title">⚡ Hardware &amp; Compute</div>
            <div class="settings-card-sub">Select compute device and runtime options</div>
        </div>
        """, unsafe_allow_html=True)

        use_gpu = st.toggle(
            "Enable GPU Processing (CUDA)",
            value=bool(s.get('use_gpu', False)),
            help="Uses CUDA-capable GPU for inference. Falls back to CPU if unavailable."
        )

        st.write("")

        # ── Annotation Card ───────────────────────────────────────────────────
        st.markdown("""
        <div class="settings-card">
            <div class="settings-card-title">🎨 Visual Overlay Style</div>
            <div class="settings-card-sub">Customise how bounding boxes and labels appear on detection results</div>
        </div>
        """, unsafe_allow_html=True)

        bbox_thickness = st.slider(
            "Bounding Box Thickness (px)",
            min_value=1, max_value=8,
            value=int(s.get('bbox_thickness', 2)),
            step=1,
            help="Line thickness of defect bounding boxes drawn on output images."
        )
        font_size = st.slider(
            "Label Font Size",
            min_value=8, max_value=24,
            value=int(s.get('font_size', 14)),
            step=1,
            help="Size of the defect class text labels in annotation overlays."
        )

        st.write("")

        # ── Data Card ─────────────────────────────────────────────────────────
        st.markdown("""
        <div class="settings-card">
            <div class="settings-card-title">💾 Data Storage</div>
            <div class="settings-card-sub">Control whether inspection results are saved to the local database</div>
        </div>
        """, unsafe_allow_html=True)

        save_images = st.toggle(
            "Save Inspections to Database",
            value=bool(s.get('save_images', True)),
            help="Saves original and annotated images along with detection metadata to local SQLite."
        )

        st.write("")

        submitted = st.form_submit_button("💾  Save Settings", use_container_width=False)
        if submitted:
            st.session_state.settings = {
                'conf_threshold': conf_threshold,
                'iou_threshold':  iou_threshold,
                'use_gpu':        use_gpu,
                'save_images':    save_images,
                'bbox_thickness': bbox_thickness,
                'font_size':      font_size,
            }
            st.success("✓ Settings saved successfully.")

    # ── Active Config Summary ──────────────────────────────────────────────────
    st.write("")
    st.markdown("""
    <div class="section-header">
        <p class="section-title">Active Configuration</p>
        <p class="section-subtitle">Current runtime values</p>
    </div>
    """, unsafe_allow_html=True)

    sc = st.session_state.settings
    rows_html = "".join([
        f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val">{v}</span></div>'
        for k, v in [
            ("Confidence Threshold",  f"{sc['conf_threshold']:.2f}"),
            ("IoU Threshold",         f"{sc['iou_threshold']:.2f}"),
            ("GPU Processing",        "Enabled" if sc['use_gpu'] else "Disabled"),
            ("Save to Database",      "Yes" if sc['save_images'] else "No"),
            ("BBox Thickness",        f"{sc['bbox_thickness']} px"),
            ("Font Size",             f"{sc['font_size']} pt"),
        ]
    ])
    st.markdown(f'<div class="info-card">{rows_html}</div>', unsafe_allow_html=True)
