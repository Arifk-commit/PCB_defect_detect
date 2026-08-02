import streamlit as st
import os
import base64


# ── CSS & JS Injection ─────────────────────────────────────────────────────────

def inject_custom_css():
    """Reads style.css and injects it. Also injects sidebar auto-expand JS."""
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    # Auto-expand sidebar when collapsed from browser cache
    st.markdown("""
    <script>
    (function() {
        function expandSidebar() {
            var btn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
            if (btn) btn.click();
        }
        var sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
            setTimeout(expandSidebar, 300);
        }
    })();
    </script>
    """, unsafe_allow_html=True)


# ── Logo ───────────────────────────────────────────────────────────────────────

def load_logo_base64():
    """Returns a base64-encoded SVG logo for PCB Detect AI."""
    svg = """
    <svg width="200" height="44" viewBox="0 0 200 44" xmlns="http://www.w3.org/2000/svg">
        <rect width="200" height="44" rx="8" fill="none"/>
        <!-- Icon box -->
        <rect x="2" y="6" width="32" height="32" rx="7" fill="#2563EB"/>
        <!-- PCB trace art -->
        <circle cx="18" cy="22" r="4.5" fill="none" stroke="#10B981" stroke-width="2"/>
        <line x1="22.5" y1="22" x2="30" y2="22" stroke="#10B981" stroke-width="1.8"/>
        <line x1="18" y1="17.5" x2="18" y2="12" stroke="#10B981" stroke-width="1.8"/>
        <circle cx="30" cy="22" r="2" fill="#10B981"/>
        <circle cx="18" cy="12" r="2" fill="#3B82F6"/>
        <!-- Text -->
        <text x="42" y="24" font-family="Inter, sans-serif" font-size="16" font-weight="800" fill="#FFFFFF" letter-spacing="-0.4">PCB <tspan fill="#2563EB">Detect</tspan></text>
        <text x="42" y="37" font-family="Inter, sans-serif" font-size="9" font-weight="500" fill="#475569" letter-spacing="0.08em">VISION INSPECTION</text>
    </svg>
    """
    return base64.b64encode(svg.encode('utf-8')).decode('utf-8')


# ── Top Navbar ─────────────────────────────────────────────────────────────────

def render_navbar(page_title: str, model_loaded: bool = False):
    """Renders the custom sticky top navigation bar."""
    badge_cls  = "active" if model_loaded else "sim"
    badge_text = "YOLOv11m Active" if model_loaded else "Simulator Mode"
    st.markdown(f"""
    <div class="pcb-navbar">
        <div class="pcb-navbar-left">
            <span class="pcb-navbar-brand">PCB <span class="accent">Detect</span> AI</span>
            <div class="pcb-navbar-sep"></div>
            <span class="pcb-navbar-page">{page_title}</span>
        </div>
        <div class="pcb-navbar-right">
            <div class="pcb-status-badge {badge_cls}">
                <span class="pcb-status-dot"></span>
                {badge_text}
            </div>
            <span class="pcb-version">v1.0.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Page Header ────────────────────────────────────────────────────────────────

def render_page_header(eyebrow: str, title: str, subtitle: str = ""):
    """Renders a consistent three-line page heading (eyebrow / title / subtitle)."""
    sub_html = f'<p class="page-header-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
        <span class="page-header-eyebrow">{eyebrow}</span>
        <h1 class="page-header-title">{title}</h1>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


# ── KPI Card ───────────────────────────────────────────────────────────────────

def render_kpi_card(title: str, value: str, subtitle: str = "",
                    icon: str = "📊", color: str = "blue",
                    trend: str = "", trend_dir: str = "flat"):
    """
    Renders a premium KPI metric card.

    Args:
        title     : Label shown above value
        value     : Large primary number / string
        subtitle  : Small description below value
        icon      : Emoji icon
        color     : 'blue' | 'green' | 'red' | 'amber' | 'purple'
        trend     : Optional trend string (e.g. "+3 this week")
        trend_dir : 'up' | 'down' | 'flat'
    """
    trend_html = (
        f'<div class="kpi-trend {trend_dir}">'
        f'{"↑" if trend_dir == "up" else "↓" if trend_dir == "down" else "—"} {trend}'
        f'</div>'
    ) if trend else ""

    st.markdown(f"""
    <div class="kpi-card {color}">
        <div class="kpi-card-header">
            <span class="kpi-label">{title}</span>
            <div class="kpi-icon-box {color}">{icon}</div>
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
        {trend_html}
    </div>
    """, unsafe_allow_html=True)


# ── Section Header ─────────────────────────────────────────────────────────────

def render_section_header(title: str, subtitle: str = ""):
    """Renders a card-section-style heading with optional subtitle."""
    sub_html = f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
        <p class="section-title">{title}</p>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


# ── Chart Card Wrapper ─────────────────────────────────────────────────────────

def chart_card_start(title: str, subtitle: str = ""):
    """Renders the opening of a white chart card with title."""
    sub_html = f'<p class="chart-card-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="chart-card">
        <p class="chart-card-title">{title}</p>
        {sub_html}
    """, unsafe_allow_html=True)


def chart_card_end():
    """Closes the chart card div."""
    st.markdown("</div>", unsafe_allow_html=True)


# ── Badge ──────────────────────────────────────────────────────────────────────

def render_badge(label: str, badge_type: str = "info"):
    """
    Renders an inline status badge pill.
    badge_type: 'healthy' | 'defective' | 'warning' | 'info' | 'neutral'
    """
    st.markdown(
        f'<span class="badge badge-{badge_type}">{label}</span>',
        unsafe_allow_html=True
    )


# ── Defect Pill ────────────────────────────────────────────────────────────────

def render_defect_pills(defect_list: list):
    """Renders a row of defect badge pills."""
    if not defect_list:
        st.markdown('<span class="badge badge-healthy">✓ No Defects Found</span>', unsafe_allow_html=True)
        return
    pills = "".join([
        f'<span class="defect-pill"><span class="defect-pill-dot"></span>{d.strip()}</span>'
        for d in defect_list
    ])
    st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:8px;">{pills}</div>',
                unsafe_allow_html=True)


# ── Info Card (Model Info rows) ────────────────────────────────────────────────

def render_info_card(title: str, rows: list):
    """
    Renders a labelled card with key/value rows.
    rows: list of (key, value) tuples
    """
    rows_html = "".join([
        f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val">{v}</span></div>'
        for k, v in rows
    ])
    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-title">{title}</div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)


# ── Progress Metric ────────────────────────────────────────────────────────────

def render_metric_progress(label: str, value: float, max_val: float = 100.0, color: str = "#2563EB"):
    """Renders a labelled metric with progress bar (value as percentage)."""
    pct = min(value / max_val, 1.0)
    display = f"{value:.1f}%" if max_val == 100.0 else f"{value:.1f}"
    st.markdown(f"""
    <div style="margin-bottom:14px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
            <span style="font-size:13px;font-weight:600;color:#374151;">{label}</span>
            <span style="font-size:13px;font-weight:700;color:#0F172A;">{display}</span>
        </div>
        <div style="background:#E2E8F0;border-radius:4px;height:7px;overflow:hidden;">
            <div style="background:{color};width:{pct*100:.1f}%;height:100%;border-radius:4px;
                        transition:width 0.6s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────

def render_footer():
    """Renders the sticky page footer."""
    st.markdown("""
    <div class="pcb-footer">
        <strong>PCB Detect AI</strong> · Industrial Vision Inspection Suite ·
        Powered by Streamlit &amp; Ultralytics YOLOv11m · v1.0.0
    </div>
    """, unsafe_allow_html=True)


# ── Utilities ──────────────────────────────────────────────────────────────────

def get_image_base64(image_path: str) -> str:
    """Encodes an image file to base64."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ""
