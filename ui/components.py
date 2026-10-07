"""
UI Components and Styling Module for Streamlit Dashboard.
Provides Glassmorphic CSS styling, custom metric cards, status badges, and alerts.
"""
import streamlit as st


def inject_custom_css():
    """Inject custom dark theme glassmorphic CSS styling with high-visibility contrast."""
    custom_css = """
    <style>
    /* Streamlit Top Header Bar & Decoration Overrides */
    header[data-testid="stHeader"],
    div[data-testid="stHeader"],
    header {
        background: rgba(9, 13, 22, 0.95) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
    }

    [data-testid="stDecoration"] {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc) !important;
        height: 3px !important;
    }

    [data-testid="stToolbar"] *,
    [data-testid="stStatusWidget"] *,
    header * {
        color: #94a3b8 !important;
    }

    /* Dark Theme Core */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111827 50%, #0f172a 100%) !important;
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Force all standard text & headings to be high-contrast white */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp span, .stApp label {
        color: #f8fafc;
    }

    /* Sidebar Background & Visibility */
    section[data-testid="stSidebar"] {
        background: #0b1329 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.5) !important;
    }

    /* Streamlit Native Metric Card Overrides */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #38bdf8 !important;
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    }

    /* Status Badges */
    .status-badge-alert {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff !important;
        padding: 0.6rem 2rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.35rem;
        letter-spacing: 0.06em;
        text-align: center;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.5);
        display: inline-block;
    }

    .status-badge-distracted {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #ffffff !important;
        padding: 0.6rem 2rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.35rem;
        letter-spacing: 0.06em;
        text-align: center;
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.5);
        display: inline-block;
    }

    .status-badge-drowsy {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff !important;
        padding: 0.6rem 2rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.35rem;
        letter-spacing: 0.06em;
        text-align: center;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.7);
        animation: pulse 1.2s infinite;
        display: inline-block;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.04); }
        100% { transform: scale(1); }
    }

    /* Custom Buttons Universal High-Contrast Styling */
    div.stButton > button,
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-primary"],
    button[kind="secondary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
    }

    div.stButton > button:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    }

    /* Force all text inside buttons to be crisp bold white */
    div.stButton > button *,
    button[data-testid="stBaseButton-secondary"] *,
    button[data-testid="stBaseButton-primary"] * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Dataframe Table High Contrast Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        background: rgba(15, 23, 42, 0.9) !important;
    }

    /* Info / Warning Box Styling */
    div.stAlert {
        border-radius: 12px !important;
        font-weight: 600 !important;
    }

    /* Slider / Widget Label Styling */
    label[data-testid="stWidgetLabel"] p {
        color: #f1f5f9 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header():
    """Render top application header banner with dark glassmorphic styling."""
    st.markdown(
        """
        <div style="background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; text-align: center; box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.5);">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 9999px; padding: 4px 14px; margin-bottom: 0.75rem;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #38bdf8; display: inline-block; box-shadow: 0 0 8px #38bdf8;"></span>
                <span style="color: #38bdf8; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;">Real-Time AI Vision System</span>
            </div>
            <h1 style="background: linear-gradient(135deg, #ffffff 0%, #38bdf8 50%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.3rem; font-weight: 900; margin: 0 0 0.4rem 0; letter-spacing: -0.02em;">
                🎓 Student Drowsiness & Attention Analytics Engine
            </h1>
            <p style="color: #94a3b8; font-size: 1.05rem; max-width: 750px; margin: 0 auto; line-height: 1.5; font-weight: 500;">
                Real-Time Computer Vision & Temporal Attention Analytics powered by MediaPipe FaceMesh & 3D Pose Estimation
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(state: str):
    """Render animated status badge indicator (ALERT, DISTRACTED, DROWSY)."""
    if state == "DROWSY":
        badge_html = '<div class="status-badge-drowsy">🚨 DROWSY DETECTED</div>'
    elif state == "DISTRACTED":
        badge_html = '<div class="status-badge-distracted">⚠️ DISTRACTED</div>'
    else:
        badge_html = '<div class="status-badge-alert">✅ ALERT & FOCUSED</div>'

    st.markdown(f'<div style="text-align: center; margin-bottom: 1.5rem;">{badge_html}</div>', unsafe_allow_html=True)
