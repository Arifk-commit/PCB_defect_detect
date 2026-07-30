import streamlit as st
import pandas as pd
from PIL import Image
import io
import time
from utils.predict import predict_image
from utils.database import add_detection
from utils.helpers import render_page_header


def show_batch_detection():
    render_page_header(
        "Batch Processing",
        "Batch Detection",
        "Upload multiple PCB images for high-throughput automated inspection"
    )

    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'conf_threshold': 0.25, 'iou_threshold': 0.45,
            'use_gpu': False, 'save_images': True,
            'bbox_thickness': 2, 'font_size': 14
        }

    uploaded_files = st.file_uploader(
        "Upload PCB Board Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Select multiple images for batch inspection"
    )

    if not uploaded_files:
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;color:#94A3B8;">
            <div style="font-size:48px;margin-bottom:12px;">📦</div>
            <div style="font-size:15px;font-weight:600;color:#475569;margin-bottom:6px;">
                Upload multiple PCB images to begin batch analysis
            </div>
            <div style="font-size:13px;">Select any number of JPG or PNG images for simultaneous inspection.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    n = len(uploaded_files)

    # Summary row
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown(f"""
        <div class="kpi-card blue" style="padding:16px 20px;">
            <div class="kpi-label">Files Queued</div>
            <div class="kpi-value">{n}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card amber" style="padding:16px 20px;">
            <div class="kpi-label">Confidence Threshold</div>
            <div class="kpi-value">{st.session_state.settings['conf_threshold']:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        device = "GPU" if st.session_state.settings['use_gpu'] else "CPU"
        st.markdown(f"""
        <div class="kpi-card purple" style="padding:16px 20px;">
            <div class="kpi-label">Compute Device</div>
            <div class="kpi-value">{device}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    if st.button("🚀  Start Batch Inspection", use_container_width=False):
        progress_bar = st.progress(0)
        status_box   = st.empty()
        results_data = []

        for idx, uf in enumerate(uploaded_files):
            status_box.markdown(
                f'<div class="filter-panel" style="padding:12px 18px;">'
                f'<span style="font-size:13px;font-weight:600;color:#374151;">Processing {idx+1} / {n}: '
                f'<code>{uf.name}</code></span></div>',
                unsafe_allow_html=True
            )

            try:
                img = Image.open(uf).convert("RGB")
                s   = st.session_state.settings
                pred, conf, boxes, proc_t, anno = predict_image(
                    img,
                    conf_threshold=s['conf_threshold'],
                    iou_threshold=s['iou_threshold'],
                    bbox_thickness=s['bbox_thickness'],
                    font_size=s['font_size'],
                    use_gpu=s['use_gpu']
                )
                results_data.append({
                    "Filename":          uf.name,
                    "Prediction":        pred,
                    "Confidence":        f"{conf*100:.1f}%",
                    "Defects":           len(boxes),
                    "Defect Names":      ", ".join(set(b['label'] for b in boxes)) or "None",
                    "Processing (ms)":   proc_t,
                })
                if s['save_images']:
                    add_detection(
                        filename=uf.name, prediction=pred,
                        defect_count=len(boxes),
                        defects_list=[b['label'] for b in boxes],
                        confidence=conf, inference_time=proc_t,
                        original_image=img, annotated_image=anno
                    )
            except Exception as e:
                results_data.append({
                    "Filename": uf.name, "Prediction": "ERROR",
                    "Confidence": "—", "Defects": 0,
                    "Defect Names": str(e), "Processing (ms)": 0
                })

            progress_bar.progress((idx + 1) / n)

        status_box.success(f"✓ Batch inspection complete — {n} images processed.")

        # Results table
        results_df = pd.DataFrame(results_data)
        st.write("")

        # Colour Prediction column
        def colour_pred(val):
            if val == "Healthy":   return "color:#15803D;font-weight:700;"
            if val == "Defective": return "color:#B91C1C;font-weight:700;"
            return "color:#92400E;font-weight:700;"

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<p class="chart-card-title">Batch Inspection Results</p>', unsafe_allow_html=True)
        st.dataframe(
            results_df.style.map(colour_pred, subset=["Prediction"]),
            use_container_width=True, hide_index=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Summary stats
        healthy_n   = sum(1 for r in results_data if r["Prediction"] == "Healthy")
        defective_n = sum(1 for r in results_data if r["Prediction"] == "Defective")
        s1, s2, s3 = st.columns(3, gap="small")
        with s1:
            st.markdown(f'<div class="kpi-card green" style="padding:14px 18px;"><div class="kpi-label">Healthy</div><div class="kpi-value">{healthy_n}</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="kpi-card red" style="padding:14px 18px;"><div class="kpi-label">Defective</div><div class="kpi-value">{defective_n}</div></div>', unsafe_allow_html=True)
        with s3:
            yld = healthy_n / n * 100 if n else 0
            st.markdown(f'<div class="kpi-card blue" style="padding:14px 18px;"><div class="kpi-label">Yield Rate</div><div class="kpi-value">{yld:.1f}%</div></div>', unsafe_allow_html=True)

        st.write("")

        # CSV download
        buf = io.StringIO()
        results_df.to_csv(buf, index=False)
        st.download_button(
            "📥 Download Results CSV",
            data=buf.getvalue(),
            file_name=f"batch_results_{int(time.time())}.csv",
            mime="text/csv"
        )
