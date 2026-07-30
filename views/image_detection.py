import streamlit as st
from PIL import Image
import io
from utils.predict import predict_image
from utils.database import add_detection
from utils.helpers import render_page_header, render_defect_pills, render_metric_progress


def show_image_detection():
    render_page_header(
        "AI Inspection",
        "Single Image Detection",
        "Upload a PCB board image to run instant defect analysis with YOLOv11"
    )

    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'conf_threshold': 0.25, 'iou_threshold': 0.45,
            'use_gpu': False, 'save_images': True,
            'bbox_thickness': 2, 'font_size': 14
        }

    # ── Upload Area ───────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload PCB Board Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG · Max size: 200 MB"
    )

    if uploaded_file is None:
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;color:#94A3B8;">
            <div style="font-size:48px;margin-bottom:12px;">🔍</div>
            <div style="font-size:15px;font-weight:600;color:#475569;margin-bottom:6px;">
                Upload a PCB image to begin analysis
            </div>
            <div style="font-size:13px;">
                The AI model will detect defects such as missing holes, mouse bites, open circuits and more.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Run prediction
        with st.spinner("🔬 Analyzing PCB board for defects…"):
            s = st.session_state.settings
            prediction, avg_confidence, boxes, proc_time, anno_image = predict_image(
                image,
                conf_threshold=s['conf_threshold'],
                iou_threshold=s['iou_threshold'],
                bbox_thickness=s['bbox_thickness'],
                font_size=s['font_size'],
                use_gpu=s['use_gpu']
            )

        # ── Two-column image layout ───────────────────────────────────────────
        col_orig, col_anno = st.columns(2, gap="medium")
        with col_orig:
            st.markdown("**📷 Original Image**")
            st.image(image, use_container_width=True)
        with col_anno:
            st.markdown("**🎯 Detection Result**")
            st.image(anno_image, use_container_width=True)

        st.write("")

        # ── Results Panel ─────────────────────────────────────────────────────
        res_col, detail_col = st.columns([1, 2], gap="medium")

        with res_col:
            is_healthy  = prediction == "Healthy"
            card_class  = "healthy" if is_healthy else "defective"
            status_icon = "✅" if is_healthy else "🔴"
            status_text = "HEALTHY BOARD" if is_healthy else "DEFECTS DETECTED"
            status_cls  = "healthy" if is_healthy else "defective"

            st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="result-status {status_cls}">
                    {status_icon} {status_text}
                </div>
                <div class="result-metric">
                    <div class="result-metric-label">Defects Found</div>
                    <div class="result-metric-value">{len(boxes)}</div>
                </div>
                <div class="result-metric">
                    <div class="result-metric-label">Avg Confidence</div>
                    <div class="result-metric-value">{avg_confidence*100:.1f}%</div>
                </div>
                <div class="result-metric">
                    <div class="result-metric-label">Processing Time</div>
                    <div class="result-metric-value">{proc_time} ms</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with detail_col:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<div class="info-card-title">Defect Analysis</div>', unsafe_allow_html=True)

            if len(boxes) == 0:
                st.success("✓ Perfect scan — no visual defect anomalies detected on traces or pads.")
            else:
                defect_names = [box['label'] for box in boxes]
                st.markdown("**Identified defect classes:**")
                render_defect_pills(list(set(defect_names)))
                st.write("")
                st.markdown("**Confidence per detection:**")
                for box in boxes:
                    render_metric_progress(box['label'], box['confidence'] * 100, 100.0)

            st.markdown("</div>", unsafe_allow_html=True)

        # Save if enabled
        if st.session_state.settings['save_images']:
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
            st.toast("Detection saved to database", icon="💾")

        # Download button
        buf = io.BytesIO()
        anno_image.save(buf, format="PNG")
        st.download_button(
            label="📥 Download Annotated Result",
            data=buf.getvalue(),
            file_name=f"inspected_{uploaded_file.name}",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Failed to process image: {e}")
        st.warning("Please make sure the uploaded file is a valid, uncorrupted image.")
