import streamlit as st
import pandas as pd
import os
from utils.database import get_history, delete_record, clear_all_history
from utils.helpers import render_page_header
from PIL import Image


def show_history():
    render_page_header(
        "Audit Trail",
        "Detection History",
        "Review, filter and export all historical PCB inspection records"
    )

    # ── Filter Panel ──────────────────────────────────────────────────────────
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.markdown('<div class="filter-title">🔍 Search &amp; Filters</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3, gap="medium")
    with fc1:
        search_query     = st.text_input("Search filename", placeholder="e.g. pcb_board_01")
        prediction_filter = st.selectbox("Result Status", ["All", "Healthy", "Defective"])
    with fc2:
        date_filter  = st.date_input("Filter by Date", value=None)
        defect_types = ["All", "Missing Hole", "Mouse Bite", "Open Circuit", "Short", "Spur", "Spurious Copper"]
        defect_filter = st.selectbox("Defect Category", defect_types)
    with fc3:
        conf_min = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05, format="%.2f")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Fetch records ─────────────────────────────────────────────────────────
    records = get_history(
        search_query=search_query,
        date_filter=date_filter,
        confidence_min=conf_min,
        defect_type_filter=defect_filter,
        prediction_filter=prediction_filter
    )

    if not records:
        st.info("No records match the current filter criteria.")
        return

    # ── Action Row ────────────────────────────────────────────────────────────
    act1, act2, act3 = st.columns([2, 2, 6], gap="small")
    with act1:
        csv_df = pd.DataFrame(records)[
            ['id','filename','prediction','defect_count','defects_list','confidence','inference_time','timestamp']
        ].copy()
        csv_df['confidence'] = csv_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
        st.download_button(
            "📥 Export CSV",
            data=csv_df.to_csv(index=False),
            file_name="pcb_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    with act2:
        if st.button("🗑️ Wipe All Logs", use_container_width=True):
            st.session_state.confirm_clear = True

    if st.session_state.get('confirm_clear', False):
        st.warning("⚠️ This will permanently delete all inspection logs and images. Are you sure?")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("Yes, delete everything", type="primary", use_container_width=True):
                clear_all_history()
                st.session_state.confirm_clear = False
                st.success("All logs cleared.")
                st.rerun()
        with cc2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

    st.write("")

    # ── Record Count ─────────────────────────────────────────────────────────
    st.markdown(
        f'<p style="font-size:13px;font-weight:600;color:#374151;margin-bottom:12px;">'
        f'Showing {len(records)} inspection records</p>',
        unsafe_allow_html=True
    )

    # ── Record List ───────────────────────────────────────────────────────────
    for record in records:
        rec_id  = record['id']
        fname   = record['filename']
        pred    = record['prediction']
        d_count = record['defect_count']
        defects = record['defects_list'] or ""
        conf    = record['confidence']
        latency = record['inference_time']
        ts      = record['timestamp']

        badge_cls  = "badge-healthy" if pred == "Healthy" else "badge-defective"
        badge_icon = "✓" if pred == "Healthy" else "✕"

        title = (
            f'<div style="display:flex;align-items:center;justify-content:space-between;width:100%;">'
            f'<div>'
            f'<span style="font-size:13px;font-weight:700;color:#0F172A;">{fname}</span>'
            f'<span style="font-size:11px;color:#94A3B8;margin-left:10px;">{ts}</span>'
            f'</div>'
            f'<div style="display:flex;gap:8px;align-items:center;">'
            f'<span class="badge {badge_cls}">{badge_icon} {pred}</span>'
            f'<span class="badge badge-neutral">Conf: {conf*100:.1f}%</span>'
            f'<span class="badge badge-neutral">{latency:.0f} ms</span>'
            f'</div>'
            f'</div>'
        )

        with st.expander(f"{fname}  ·  {pred}  ·  {conf*100:.1f}%", expanded=False):
            st.markdown(title, unsafe_allow_html=True)
            st.write("")

            dc1, dc2 = st.columns(2, gap="medium")
            with dc1:
                st.markdown("**Original Image**")
                orig = record.get('original_image_path')
                if orig and os.path.exists(orig):
                    st.image(Image.open(orig), use_container_width=True)
                else:
                    st.caption("Image file not found on disk.")
            with dc2:
                st.markdown("**Annotated Detection**")
                anno = record.get('annotated_image_path')
                if anno and os.path.exists(anno):
                    st.image(Image.open(anno), use_container_width=True)
                else:
                    st.caption("Image file not found on disk.")

            if defects:
                pills = "".join([
                    f'<span class="defect-pill"><span class="defect-pill-dot"></span>{d.strip()}</span>'
                    for d in defects.split(',')
                ])
                st.markdown(f'<div style="margin:10px 0;">{pills}</div>', unsafe_allow_html=True)

            _, del_col = st.columns([6, 1])
            with del_col:
                if st.button("Delete", key=f"del_{rec_id}", use_container_width=True):
                    delete_record(rec_id)
                    st.toast(f"Record deleted.", icon="🗑️")
                    st.rerun()
