import streamlit as st
from utils.helpers import render_page_header, render_info_card, render_metric_progress


def show_model_info():
    render_page_header(
        "AI Model",
        "Model Information",
        "Architecture details, training metrics, and defect class specifications"
    )

    # ── Top row: Config + Metrics ─────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        render_info_card("Model Configuration", [
            ("Model Name",        "YOLOv11 Nano (PCB Custom)"),
            ("Framework",         "PyTorch 2.3.0 + Ultralytics"),
            ("Input Resolution",  "640 × 640 px"),
            ("Parameters",        "3.1 M (Lightweight)"),
            ("Model Size",        "6.2 MB"),
            ("Export Formats",    "ONNX · TensorRT · CoreML"),
            ("Dataset",           "PCB Defects (6 classes)"),
            ("Training Epochs",   "300 epochs"),
        ])

        st.markdown("""
        <div class="info-card" style="border-left:4px solid #2563EB;">
            <div class="info-card-title">💡 Edge Deployment</div>
            <p style="font-size:13px;color:#475569;margin:0;line-height:1.6;">
            YOLOv11 nano is optimised for deployment on low-power edge hardware
            such as <strong>Raspberry Pi 5</strong> and <strong>NVIDIA Jetson Nano</strong>,
            making it ideal for conveyor-belt inspection lines.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="info-card-title">Training Accuracy Metrics</div>', unsafe_allow_html=True)

        render_metric_progress("Precision (P)",      93.4, 100.0, "#2563EB")
        render_metric_progress("Recall (R)",          91.2, 100.0, "#8B5CF6")
        render_metric_progress("mAP@50",              94.8, 100.0, "#22C55E")
        render_metric_progress("mAP@50-95",           68.3, 100.0, "#F59E0B")

        st.markdown('<div class="info-card-title" style="margin-top:18px;">Inference Speed</div>', unsafe_allow_html=True)

        render_metric_progress("GPU (avg)",    4.2,  30.0, "#22C55E")
        render_metric_progress("CPU (avg)",   24.5, 100.0, "#F59E0B")

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # ── Defect Classes ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <p class="section-title">Detectable Defect Classes</p>
        <p class="section-subtitle">The model is trained to identify 6 PCB manufacturing defects</p>
    </div>
    """, unsafe_allow_html=True)

    classes = [
        ("🕳️", "Missing Hole",    "A through-hole or via is missing from the copper pad, preventing component insertion."),
        ("🐭", "Mouse Bite",      "Small circular bites out of copper tracks, reducing current capacity."),
        ("⚡", "Open Circuit",    "Discontinuity in trace prevents signal propagation between components."),
        ("🔗", "Short Circuit",   "Accidental connection between adjacent electrical tracks or vias."),
        ("📌", "Spur",            "Copper projections sticking out from trace borders due to etching defects."),
        ("🟡", "Spurious Copper", "Isolated excess copper blobs remaining after the etching process."),
    ]

    col_a, col_b = st.columns(2, gap="medium")
    for i, (icon, name, desc) in enumerate(classes):
        col = col_a if i % 2 == 0 else col_b
        with col:
            st.markdown(f"""
            <div class="class-card">
                <div class="class-card-icon">{icon}</div>
                <div>
                    <div class="class-card-name">{name}</div>
                    <div class="class-card-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # ── Integration Guide ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="info-card" style="border-left:4px solid #22C55E;">
        <div class="info-card-title">🔧 Custom Model Integration</div>
        <p style="font-size:13px;color:#475569;margin:0;line-height:1.8;">
            To replace the simulator with your own trained model:
        </p>
        <ol style="font-size:13px;color:#475569;margin:8px 0 0;padding-left:18px;line-height:2.0;">
            <li>Export your YOLOv11 weights as <code>best.pt</code>.</li>
            <li>Place the file at <code>models/best.pt</code> in the project root.</li>
            <li>The app will auto-detect the file on next startup and switch from simulator to live inference.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
