import os
import streamlit as st
from utils.helpers import render_page_header, render_info_card, render_metric_progress
from utils.model_loader import get_model, MODEL_PATH
from utils.model_info_helper import (
    get_model_specs,
    get_validation_metrics,
    measure_inference_performance
)


def show_model_info():
    render_page_header(
        "AI Model",
        "Model Information",
        "100% dynamic architecture details, live validation metrics, and model checkpoint metadata"
    )

    # 1. Fetch real dynamic specs & metrics
    model = get_model()
    specs = get_model_specs(model, MODEL_PATH)
    val_metrics = get_validation_metrics(model, MODEL_PATH)
    perf = measure_inference_performance(model)

    # ── Top Row: Model Configuration & Validation Metrics ──────────────────────
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        render_info_card("Model Configuration", [
            ("Runtime Status",    specs['status']),
            ("Checkpoint Path",   specs['path']),
            ("Model Name",        specs['name']),
            ("Framework",         specs['framework']),
            ("Task Type",         specs['task']),
            ("Input Resolution",  specs['resolution']),
            ("Total Parameters",  specs['total_params']),
            ("Trainable Params",  specs['trainable_params']),
            ("Weight Size",      specs['weight_size']),
            ("Classes Count",     specs['class_count']),
        ])

        st.markdown(f"""
        <div class="info-card" style="border-left:4px solid #2563EB;">
            <div class="info-card-title">💡 Edge Deployment &amp; Environment</div>
            <p style="font-size:13px;color:#475569;margin:0 0 8px 0;line-height:1.6;">
            Active Device: <strong>{specs['device']}</strong> ({specs['gpu_name']})<br>
            CUDA Available: <strong>{'Yes' if specs['cuda_available'] else 'No'}</strong><br>
            PyTorch Version: <strong>{specs['pytorch_ver']}</strong> | Ultralytics: <strong>{specs['ultralytics_ver']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        
        if val_metrics['available']:
            st.markdown(f'<div class="info-card-title">Validation Accuracy Metrics ({val_metrics["source"]})</div>', unsafe_allow_html=True)

            render_metric_progress("Precision (P)",      val_metrics['precision'], 100.0, "#2563EB")
            render_metric_progress("Recall (R)",          val_metrics['recall'],    100.0, "#8B5CF6")
            render_metric_progress("mAP@50",              val_metrics['map50'],     100.0, "#22C55E")
            render_metric_progress("mAP@50-95",           val_metrics['map50_95'],  100.0, "#F59E0B")
        else:
            st.markdown('<div class="info-card-title">Validation Accuracy Metrics</div>', unsafe_allow_html=True)
            st.info("Validation metrics unavailable. Run model validation to generate metrics.")
            
            if st.button("🧪 Run Model Validation", use_container_width=True):
                if model is not None:
                    try:
                        with st.spinner("Running model.val() on dataset..."):
                            val_results = model.val()
                            st.success("Validation completed!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Validation execution failed: {e}")
                else:
                    st.warning("Model weights best.pt not loaded. Cannot run validation.")

        st.markdown('<div class="info-card-title" style="margin-top:18px;">Real Measured Inference Performance</div>', unsafe_allow_html=True)

        if perf['gpu_latency'] > 0:
            render_metric_progress("GPU Inference Latency (avg)", perf['gpu_latency'], 50.0, "#22C55E")
        render_metric_progress("CPU Inference Latency (avg)", perf['cpu_latency'], 200.0, "#F59E0B")
        render_metric_progress("Processing Speed (FPS)", perf['fps'], 60.0, "#2563EB")

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # ── Defect Classes Section ────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <p class="section-title">Detectable Defect Categories</p>
        <p class="section-subtitle">Classes loaded dynamically from model.names</p>
    </div>
    """, unsafe_allow_html=True)

    # Dictionary of defect class descriptions & icons
    CLASS_LOOKUP = {
        'missing_hole':    ("🕳️", "Missing Hole",    "A through-hole or via is missing from the copper pad, preventing component pin insertion."),
        'mouse_bite':      ("🐭", "Mouse Bite",      "Small circular bites taken out of copper tracks, reducing electrical current capacity."),
        'open_circuit':    ("⚡", "Open Circuit",    "Discontinuity in a copper trace preventing electrical signal propagation."),
        'short':           ("🔗", "Short Circuit",   "Accidental copper bridge connecting adjacent electrical tracks or vias."),
        'spur':            ("📌", "Spur",            "Jagged copper projection sticking out from trace borders due to etching defects."),
        'spurious_copper': ("🟡", "Spurious Copper", "Isolated excess copper blob remaining on the PCB substrate after etching."),
    }

    col_a, col_b = st.columns(2, gap="medium")
    raw_classes = specs['classes']

    for i, (cls_id, raw_name) in enumerate(raw_classes.items()):
        norm_key = str(raw_name).lower().strip()
        icon, title, desc = CLASS_LOOKUP.get(
            norm_key, 
            ("🔍", str(raw_name).replace('_', ' ').title(), f"Detected class: {raw_name}")
        )
        col = col_a if i % 2 == 0 else col_b
        with col:
            st.markdown(f"""
            <div class="class-card">
                <div class="class-card-icon">{icon}</div>
                <div>
                    <div class="class-card-name">Class {cls_id}: {title}</div>
                    <div class="class-card-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # ── Model Metadata Footer Card ─────────────────────────────────────────────
    st.markdown(f"""
    <div class="info-card" style="border-left:4px solid #22C55E;">
        <div class="info-card-title">⚙️ Checkpoint Metadata</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:13px;color:#475569;">
            <div>Checkpoint Path: <strong>{specs['path']}</strong></div>
            <div>Training/Modification Date: <strong>{specs['training_date']}</strong></div>
            <div>PyTorch Version: <strong>{specs['pytorch_ver']}</strong></div>
            <div>Ultralytics Version: <strong>{specs['ultralytics_ver']}</strong></div>
            <div>CUDA Status: <strong>{'Available' if specs['cuda_available'] else 'Not Available'}</strong></div>
            <div>Hardware Device: <strong>{specs['device']}</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
