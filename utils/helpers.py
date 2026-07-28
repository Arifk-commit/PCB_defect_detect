import streamlit as st
import os
import base64

def inject_custom_css():
    """Reads style.css and injects it into Streamlit's head via markdown."""
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline basic styles if style.css is not found
        st.markdown("""
            <style>
                .metric-card {
                    background-color: #ffffff;
                    border: 1px solid #E2E8F0;
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 10px;
                }
            </style>
        """, unsafe_allow_html=True)

def render_kpi_card(title, value, border_color="blue", icon="📊"):
    """
    Renders a custom HTML/CSS KPI card matching the design requirements.
    
    Args:
        title (str): Title of the metric
        value (str/int): Numerical value to display
        border_color (str): One of "blue", "green", "red", "orange"
        icon (str): Icon/emoji to show in top right
    """
    # Map input border color to the corresponding CSS class
    border_class = f"metric-card-border-{border_color}"
    
    card_html = f"""
    <div class="metric-card {border_class}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-icon">{icon}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def get_image_base64(image_path):
    """Encodes an image to a base64 string for embedding in HTML."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    return ""

def load_logo_base64():
    """Generates a default logo image or falls back to a clean text layout."""
    # We can embed a simple SVG or return basic base64
    # For PCB Detect, we'll draw a beautiful text-based SVG logo
    svg_logo = """
    <svg width="240" height="50" viewBox="0 0 240 50" xmlns="http://www.w3.org/2000/svg">
        <rect width="240" height="50" rx="8" fill="#1E293B"/>
        <rect x="15" y="10" width="30" height="30" rx="6" fill="#2563EB"/>
        <!-- PCB trace lines -->
        <circle cx="30" cy="25" r="4" fill="#10B981"/>
        <line x1="30" y1="25" x2="45" y2="25" stroke="#10B981" stroke-width="2"/>
        <!-- Logo Text -->
        <text x="55" y="27" font-family="'Outfit', sans-serif" font-size="18" font-weight="800" fill="#FFFFFF">PCB DETECT</text>
        <text x="160" y="27" font-family="'Outfit', sans-serif" font-size="18" font-weight="800" fill="#10B981">AI</text>
        <text x="55" y="40" font-family="'Outfit', sans-serif" font-size="9" font-weight="500" fill="#94A3B8">INDUSTRIAL VISION SYSTEM</text>
    </svg>
    """
    return base64.b64encode(svg_logo.encode('utf-8')).decode('utf-8')
