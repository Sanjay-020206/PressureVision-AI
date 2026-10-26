"""
PressureVision AI - Clinical Pressure Ulcer Staging Intelligence
================================================================
Advanced Deep Learning Decision-Support System:
- Dual-Backbone Feature Extraction (ConvNeXt-Tiny + Swin-Tiny)
- Adaptive Gated Feature Fusion (Learnable Channel-Wise Gate)
- 7-Class NPUAP/EPUAP Staging Classification
- Top Navigation Bar (No Sidebar) & Dark Glassmorphic Theme

Model Checkpoint: pressurevision_exp3_best.pth (56,657,121 parameters | 93.57% Benchmark Test Accuracy | Macro F1: 93.94%)
"""

import os

# Stabilize OpenMP runtime and prevent memory allocator crashes on cloud containers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import re
import streamlit as st
from PIL import Image, ImageOps
from typing import Dict, Any, List, Optional

# Import inference engine and model components
from inference import (
    load_model,
    predict_image,
    generate_gradcam,
    get_device,
    DISPLAY_CLASS_NAMES,
    RAW_CLASS_NAMES,
    CLASS_DESCRIPTIONS,
    IMAGE_SIZE
)
from model import PressureVisionAI, get_model_summary


# =============================================================================
# 1. Streamlit Configuration & Dark Glassmorphic Theme
# =============================================================================
st.set_page_config(
    page_title="PressureVision AI | Clinical Ulcer Staging",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Robust HTML helper that eliminates all markdown indentation / code block issues
def render_html(html_str: str):
    clean_lines = [line.strip() for line in html_str.strip().split("\n") if line.strip()]
    clean_html = " ".join(clean_lines)
    st.markdown(clean_html, unsafe_allow_html=True)


PREMIUM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Typography & Reset */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #E2E8F0;
    }
    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Completely Remove Sidebar */
    [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* App Canvas & Atmospheric Background */
    .stApp {
        background-color: #080C14;
        background-image: 
            radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(20, 184, 166, 0.08) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(99, 102, 241, 0.05) 0px, transparent 50%);
    }

    /* Main Container Constraints */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 3.5rem;
        max-width: 1240px;
    }

    /* Top Global Navbar */
    .top-navbar {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(17, 24, 39, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.1rem 1.8rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(20px);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .nav-brand-group {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .nav-logo-icon {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #0284C7, #06B6D4);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.35);
    }
    .nav-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF, #E2E8F0, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .nav-subtitle {
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 500;
        margin: 0;
        letter-spacing: 0.02em;
    }
    .nav-telemetry-group {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        flex-wrap: wrap;
    }
    .nav-pill {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        padding: 0.35rem 0.85rem;
        font-size: 0.78rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        color: #CBD5E1;
        font-weight: 600;
    }
    .pulse-indicator {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
    }

    /* Custom Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 0.35rem 0.4rem;
        gap: 0.4rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-size: 0.88rem;
        font-weight: 600;
        color: #94A3B8;
        border: 1px solid transparent;
        transition: all 0.2s ease;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.25), rgba(6, 182, 212, 0.25)) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.2);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Glass Panels & Containers */
    .glass-panel {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4);
    }

    /* Result HUD Card */
    .hud-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(17, 24, 39, 0.95) 100%);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(56, 189, 248, 0.25);
        box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.5), 0 0 20px -5px rgba(56, 189, 248, 0.12);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hud-glow-bar {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #38BDF8, #2DD4BF, #818CF8);
    }
    .hud-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .hud-title-badge {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .hud-latency {
        font-size: 0.78rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
    }
    .hud-stage-name {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0.2rem 0 0.8rem 0;
        line-height: 1.1;
    }
    .hud-metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1.2rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hud-metric-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    .hud-metric-label {
        font-size: 0.72rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    .hud-metric-val {
        font-size: 1.35rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Custom Probability Meters */
    .prob-container {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
        padding: 0.5rem 0.85rem;
        border-radius: 10px;
    }
    .prob-row.active {
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
    }
    .prob-row:not(.active) {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid transparent;
    }
    .prob-label-col {
        width: 140px;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .prob-bar-col {
        flex: 1;
        margin: 0 1rem;
    }
    .prob-bar-bg {
        width: 100%;
        height: 10px;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 9999px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 9999px;
    }
    .prob-val-col {
        width: 75px;
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        font-weight: 700;
    }

    /* Gated Fusion Meter */
    .fusion-meter-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-top: 1rem;
    }
    .fusion-bar {
        display: flex;
        height: 12px;
        border-radius: 9999px;
        overflow: hidden;
        margin: 0.6rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .fusion-cnext-part {
        background: linear-gradient(90deg, #38BDF8, #0284C7);
    }
    .fusion-swin-part {
        background: linear-gradient(90deg, #2DD4BF, #0D9488);
    }

    /* Image Preview Container */
    .img-preview-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }

    /* Explainability Card */
    .explain-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px dashed rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
    }

    /* Medical Disclaimer */
    .medical-alert {
        background: rgba(239, 68, 68, 0.06);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        margin-top: 2rem;
        display: flex;
        gap: 1rem;
        align-items: flex-start;
    }
    .medical-alert-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #F87171;
        margin-bottom: 0.2rem;
    }
    .medical-alert-text {
        font-size: 0.82rem;
        color: #CBD5E1;
        line-height: 1.5;
        margin: 0;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2.5rem 0 1rem 0;
        font-size: 0.8rem;
        color: #64748B;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 3rem;
    }
</style>
"""
render_html(PREMIUM_CSS)


# =============================================================================
# 2. Cached Resource Loader (Loaded Once)
# =============================================================================
@st.cache_resource(show_spinner="Initializing PressureVision AI neural engines...")
def get_cached_model():
    """
    Loads model checkpoint once and caches across user interactions.
    """
    model, device, device_info, error_msg = load_model()
    return model, device, device_info, error_msg


model, device, device_info, error_msg = get_cached_model()


# =============================================================================
# 3. Top Global Navigation Bar (Header)
# =============================================================================
dev_str = device_info.get("device", "CPU")
gpu_name = device_info.get("gpu_name", "N/A")
dev_badge_text = f"CUDA ({gpu_name})" if dev_str == "CUDA" else "CPU Device"

nav_html = f"""
<div class="top-navbar">
    <div class="nav-brand-group">
        <div class="nav-logo-icon">🔬</div>
        <div>
            <div class="nav-title">PressureVision™ AI</div>
            <div class="nav-subtitle">Clinical Pressure Ulcer Staging Intelligence &bull; Hybrid ConvNeXt + Swin-Tiny</div>
        </div>
    </div>
    <div class="nav-telemetry-group">
        <div class="nav-pill">
            <div class="pulse-indicator"></div>
            <span style="color:#10B981;">Engine Online</span>
        </div>
        <div class="nav-pill">
            <span style="color:#38BDF8;">⚡ {dev_badge_text}</span>
        </div>
        <div class="nav-pill">
            <span style="color:#CBD5E1;">⚙️ 56.6M Params</span>
        </div>
    </div>
</div>
"""
render_html(nav_html)

# Halt if model loading failed
if error_msg:
    st.error(f"⚠️ **Model Initialization Failure:** {error_msg}")
    st.info("Please verify that `pressurevision_exp3_best.pth` exists in the `models/` directory or is accessible from the Hugging Face repository `Sanjay020206/PressureVision-AI`.")
    st.stop()


# =============================================================================
# 4. Session State Management
# =============================================================================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "gradcam_result" not in st.session_state:
    st.session_state.gradcam_result = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "image_filename" not in st.session_state:
    st.session_state.image_filename = None


# =============================================================================
# 5. Top Tabbed Navigation Structure
# =============================================================================
tab_inspect, tab_arch, tab_explain, tab_reference, tab_specs = st.tabs([
    "🔬 Wound Analysis & Staging",
    "🧠 Neural Architecture & Gating",
    "🔍 Explainability (Grad-CAM)",
    "📚 Staging Guidelines & Criteria",
    "ℹ️ Model Specs & Telemetry"
])


# =============================================================================
# TAB 1: Clinical Wound Analysis & Staging
# =============================================================================
with tab_inspect:
    # Upload & Sample Picker Grid
    col_upload_box, col_sample_box = st.columns([2.2, 1.4], gap="medium")

    with col_upload_box:
        st.markdown("<div style='font-size:0.92rem;font-weight:700;color:#F8FAFC;margin-bottom:0.5rem;'>📤 Patient Wound Photograph</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload clinical image (.jpg, .jpeg, .png, .webp)",
            type=["jpg", "jpeg", "png", "webp"],
            key="clinical_uploader",
            help="High-quality RGB photograph of the affected anatomical area."
        )
        if uploaded_file is not None:
            try:
                raw_img = Image.open(uploaded_file)
                raw_img = ImageOps.exif_transpose(raw_img)
                st.session_state.uploaded_image = raw_img
                st.session_state.image_filename = uploaded_file.name
            except Exception as e:
                st.error(f"Image decode error: {str(e)}")

    with col_sample_box:
        st.markdown("<div style='font-size:0.92rem;font-weight:700;color:#F8FAFC;margin-bottom:0.5rem;'>⚡ Quick Reference Test Cases</div>", unsafe_allow_html=True)
        
        test_samples = {
            "Select a reference case...": None,
            "Case 1: Stage III Pressure Ulcer": r"Dataset\archive\pressure_ulcers_expanded\Stage_III\healing-pressure-ulcer-0006.jpg",
            "Case 2: Stage II Pressure Ulcer": r"Dataset\archive\pressure_ulcers_expanded\Stage_II\32.jpg",
            "Case 3: Stage IV Pressure Ulcer": r"Dataset\archive\pressure_ulcers_expanded\Stage_IV\pressure-ulcer-0062.jpg",
            "Case 4: Suspected Deep Tissue Injury (SDTI)": r"Dataset\archive\pressure_ulcers_expanded\SDTI\12.jpg",
            "Case 5: Unstageable Necrotic Ulcer": r"Dataset\archive\pressure_ulcers_expanded\Unstageable\1.jpg",
        }
        
        valid_samples = {k: v for k, v in test_samples.items() if v is None or os.path.exists(v)}
        
        selected_sample_key = st.selectbox(
            "Quick Reference Test Cases",
            options=list(valid_samples.keys()),
            label_visibility="collapsed"
        )
        
        if selected_sample_key != "Select a reference case..." and valid_samples[selected_sample_key] is not None:
            sample_file_path = valid_samples[selected_sample_key]
            if st.button("Load Reference Case", width="stretch"):
                try:
                    s_img = Image.open(sample_file_path)
                    st.session_state.uploaded_image = s_img
                    st.session_state.image_filename = os.path.basename(sample_file_path)
                    st.session_state.prediction_result = None
                    st.session_state.gradcam_result = None
                    st.rerun()
                except Exception as e:
                    st.warning(f"Failed to load sample: {str(e)}")

    # Main Action Controls
    if st.session_state.uploaded_image is not None:
        curr_image = st.session_state.uploaded_image
        
        st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1.2rem 0;'>", unsafe_allow_html=True)
        
        # Action Toolbar
        c_btn1, c_btn2, c_space = st.columns([1.6, 1.2, 3])
        with c_btn1:
            run_pred = st.button("🔬 Analyze & Stage Wound", type="primary", width="stretch")
        with c_btn2:
            run_reset = st.button("🔄 Clear Analysis", type="secondary", width="stretch")
            
        if run_reset:
            st.session_state.uploaded_image = None
            st.session_state.prediction_result = None
            st.session_state.gradcam_result = None
            st.session_state.image_filename = None
            st.rerun()

        if run_pred:
            with st.spinner("Executing dual-backbone feature extraction and adaptive gated inference..."):
                t_start = time.time()
                try:
                    pred_data = predict_image(model, curr_image, device)
                    pred_data["latency_ms"] = (time.time() - t_start) * 1000.0
                    st.session_state.prediction_result = pred_data
                except Exception as ex:
                    st.error(f"Inference error: {str(ex)}")
                    st.session_state.prediction_result = None

        # Two-Column Stage & Inspection Layout
        col_img_view, col_hud_view = st.columns([1, 1.25], gap="large")

        with col_img_view:
            render_html('<div class="img-preview-card"><div style="font-size:0.88rem;font-weight:700;color:#38BDF8;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.8rem;">📷 Wound Image Inspection</div>')
            st.image(curr_image, width="stretch")
            
            w_px, h_px = curr_image.size
            fn = st.session_state.image_filename or "clinical_upload.jpg"
            
            render_html(f"""
                <div style="display:flex;justify-content:space-between;margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid rgba(255,255,255,0.06);font-size:0.8rem;color:#94A3B8;">
                    <span>📁 <code>{fn}</code></span>
                    <span>📐 {w_px} × {h_px} px</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:0.3rem;font-size:0.8rem;color:#64748B;">
                    <span>Format: {curr_image.format or 'RGB'}</span>
                    <span>Tensor: 224 × 224 RGB</span>
                </div>
            </div>
            """)

        with col_hud_view:
            if st.session_state.prediction_result is not None:
                p_res = st.session_state.prediction_result
                p_stage = p_res["predicted_class"]
                p_conf = p_res["confidence_percentage"]
                p_lat = p_res.get("latency_ms", 0.0)
                
                # Dynamic Stage Colors
                stage_color_palette = {
                    "Invalid": "#94A3B8",
                    "SDTI": "#F43F5E",
                    "Stage I": "#38BDF8",
                    "Stage II": "#FBBF24",
                    "Stage III": "#F97316",
                    "Stage IV": "#EF4444",
                    "Unstageable": "#A855F7"
                }
                theme_accent = stage_color_palette.get(p_stage, "#38BDF8")
                
                # Reliability Badge
                if p_conf >= 75.0:
                    rel_badge = '<span style="background:rgba(16,185,129,0.15);color:#34D399;border:1px solid rgba(16,185,129,0.3);padding:0.2rem 0.6rem;border-radius:6px;font-size:0.75rem;font-weight:700;">HIGH CERTAINTY</span>'
                elif p_conf >= 50.0:
                    rel_badge = '<span style="background:rgba(245,158,11,0.15);color:#FBBF24;border:1px solid rgba(245,158,11,0.3);padding:0.2rem 0.6rem;border-radius:6px;font-size:0.75rem;font-weight:700;">MODERATE CERTAINTY</span>'
                else:
                    rel_badge = '<span style="background:rgba(239,68,68,0.15);color:#F87171;border:1px solid rgba(239,68,68,0.3);padding:0.2rem 0.6rem;border-radius:6px;font-size:0.75rem;font-weight:700;">LOW CERTAINTY</span>'

                # HUD Card
                render_html(f"""
                <div class="hud-card" style="border-color: {theme_accent}40;">
                    <div class="hud-glow-bar" style="background: linear-gradient(90deg, {theme_accent}, #38BDF8);"></div>
                    <div class="hud-header">
                        <span class="hud-title-badge">PREDICTED CLINICAL STAGE</span>
                        <span class="hud-latency">⚡ {p_lat:.1f} ms</span>
                    </div>
                    <div class="hud-stage-name" style="color: {theme_accent};">
                        {p_stage}
                    </div>
                    <div style="margin-bottom:0.5rem;">
                        {rel_badge}
                    </div>
                    <div class="hud-metrics-grid">
                        <div class="hud-metric-box">
                            <div class="hud-metric-label">Softmax Confidence</div>
                            <div class="hud-metric-val" style="color:{theme_accent};">{p_conf:.2f}%</div>
                        </div>
                        <div class="hud-metric-box">
                            <div class="hud-metric-label">Decision Target</div>
                            <div class="hud-metric-val" style="color:#F1F5F9;font-size:1.05rem;">{p_res['predicted_class_raw']}</div>
                        </div>
                    </div>
                </div>
                """)

                # Context Info
                stage_desc_text = CLASS_DESCRIPTIONS.get(p_res["predicted_class_raw"], "")
                if stage_desc_text:
                    render_html(f"""
                    <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:0.85rem 1rem;font-size:0.84rem;color:#CBD5E1;margin-bottom:1rem;">
                        <strong style="color:#38BDF8;">📋 Stage Anatomy:</strong> {stage_desc_text}
                    </div>
                    """)

                # Adaptive Gate Telemetry
                cnext_w = p_res["gate_weights"]["convnext_weight_percent"]
                swin_w = p_res["gate_weights"]["swin_weight_percent"]

                render_html(f"""
                <div class="fusion-meter-box">
                    <div style="display:flex;justify-content:space-between;font-size:0.8rem;font-weight:700;color:#94A3B8;text-transform:uppercase;">
                        <span>Adaptive Gated Balance (α)</span>
                        <span style="color:#38BDF8;">{cnext_w:.1f}% CNN &bull; {swin_w:.1f}% ViT</span>
                    </div>
                    <div class="fusion-bar">
                        <div class="fusion-cnext-part" style="width:{cnext_w}%;"></div>
                        <div class="fusion-swin-part" style="width:{swin_w}%;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#64748B;">
                        <span>ConvNeXt (Local Texture)</span>
                        <span>Swin (Global Context)</span>
                    </div>
                </div>
                """)
            else:
                render_html("""
                <div style="background:rgba(15,23,42,0.5);border:2px dashed rgba(255,255,255,0.08);border-radius:16px;padding:3.5rem 2rem;text-align:center;">
                    <div style="font-size:2rem;margin-bottom:0.5rem;">🔬</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#F8FAFC;margin-bottom:0.3rem;">Awaiting Analysis</div>
                    <div style="font-size:0.85rem;color:#64748B;max-width:300px;margin:0 auto;">
                        Click <strong>[Analyze & Stage Wound]</strong> to evaluate using the hybrid dual-backbone neural network.
                    </div>
                </div>
                """)

        # Full 7-Class Probability Spectrum
        if st.session_state.prediction_result is not None:
            st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1.8rem 0;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:1.15rem;font-weight:800;color:#F8FAFC;margin-bottom:1rem;'>📈 Complete 7-Class Probability Spectrum</div>", unsafe_allow_html=True)
            
            prob_list = st.session_state.prediction_result["probabilities_list"]
            
            spectrum_rows = ""
            for item in prob_list:
                c_name = item["display_name"]
                c_pct = item["percentage"]
                is_target = item["is_predicted"]
                
                active_class = "active" if is_target else ""
                bar_color = "#38BDF8" if is_target else "rgba(148, 163, 184, 0.4)"
                text_color = "#38BDF8" if is_target else "#E2E8F0"
                marker = "▶ " if is_target else ""
                
                spectrum_rows += f"""
                <div class="prob-row {active_class}">
                    <div class="prob-label-col" style="color:{text_color};">{marker}{c_name}</div>
                    <div class="prob-bar-col">
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{min(max(c_pct, 1.0), 100.0)}%; background: {bar_color};"></div>
                        </div>
                    </div>
                    <div class="prob-val-col" style="color:{text_color};">{c_pct:.2f}%</div>
                </div>
                """
                
            full_spectrum_html = f"""
            <div class="prob-container">
                {spectrum_rows}
                <div style="font-size:0.75rem;color:#64748B;margin-top:0.8rem;text-align:right;">
                    Probabilities generated directly via live PyTorch Softmax layer (&sum; P<sub>i</sub> = 100.00%)
                </div>
            </div>
            """
            render_html(full_spectrum_html)

            # Instant Grad-CAM Explainability Preview in Tab 1
            st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1.6rem 0;'>", unsafe_allow_html=True)
            with st.expander("🔍 View Instant ConvNeXt Grad-CAM Explainability Overlay", expanded=False):
                with st.spinner("Backpropagating gradients to generate visual heatmap..."):
                    try:
                        instant_cam = generate_gradcam(
                            model=model,
                            image=curr_image,
                            device=device,
                            target_class=None,
                            overlay_opacity=0.45,
                            colormap_name="turbo"
                        )
                        c_cam_p1, c_cam_p2, c_cam_p3 = st.columns(3)
                        with c_cam_p1:
                            st.caption(f"1. Original Photograph ({w_px} × {h_px} px)")
                            st.image(curr_image, width="stretch")
                        with c_cam_p2:
                            st.caption("2. ConvNeXt Heatmap (Stage 3)")
                            st.image(instant_cam["colored_heatmap_pil"], width="stretch")
                        with c_cam_p3:
                            st.caption(f"3. Overlay ({instant_cam['predicted_class']})")
                            st.image(instant_cam["overlay_pil"], width="stretch")
                        st.caption("💡 *For custom target class explanations, opacity tuning, and colormaps, open the **🔍 Explainability (Grad-CAM)** tab above.*")
                    except Exception as e_cam:
                        st.warning(f"Grad-CAM preview notice: {str(e_cam)}")

    else:
        # Standby View when No Image is Loaded
        render_html("""
        <div style="background:rgba(15,23,42,0.4);border:2px dashed rgba(255,255,255,0.08);border-radius:18px;padding:4rem 2rem;text-align:center;margin-top:1rem;">
            <div style="font-size:2.8rem;margin-bottom:0.75rem;">🩺</div>
            <h3 style="color:#F8FAFC;font-weight:700;margin-bottom:0.4rem;">No Clinical Photograph Selected</h3>
            <p style="color:#94A3B8;max-width:550px;margin:0 auto 1.5rem auto;font-size:0.92rem;line-height:1.6;">
                Upload an RGB wound image from your device or select one of the pre-loaded clinical reference cases above to initialize the staging engine.
            </p>
        </div>
        """)


# =============================================================================
# TAB 2: Neural Architecture & Gating
# =============================================================================
with tab_arch:
    render_html("""
    <div class="glass-panel">
        <div style="font-size:1.25rem;font-weight:800;color:#F8FAFC;margin-bottom:0.6rem;">
            🧠 Dual-Backbone Hybrid Neural Architecture
        </div>
        <p style="color:#CBD5E1;font-size:0.92rem;line-height:1.65;margin-bottom:1.5rem;">
            PressureVision AI addresses the inherent trade-off between micro-level dermatological texture perception 
            and macro-level anatomical context through an integrated dual-stream deep learning network.
        </p>
        
        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(260px, 1fr));gap:1.2rem;margin-bottom:1.5rem;">
            <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(56,189,248,0.25);border-radius:12px;padding:1.2rem;">
                <div style="font-size:0.95rem;font-weight:700;color:#38BDF8;margin-bottom:0.5rem;">1. ConvNeXt-Tiny (CNN)</div>
                <div style="font-size:0.84rem;color:#94A3B8;line-height:1.55;">
                    Utilizes 7&times;7 depthwise separable convolutions to capture high-frequency visual features: 
                    non-blanchable erythema borders, epidermal slough textures, and necrotic eschar margins.
                    <br><strong style="color:#F1F5F9;">Output Dim: 768</strong>
                </div>
            </div>
            <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(45,212,191,0.25);border-radius:12px;padding:1.2rem;">
                <div style="font-size:0.95rem;font-weight:700;color:#2DD4BF;margin-bottom:0.5rem;">2. Swin-Tiny (Hierarchical ViT)</div>
                <div style="font-size:0.84rem;color:#94A3B8;line-height:1.55;">
                    Applies shifted window self-attention across hierarchical resolution levels to model 
                    broader anatomical context, surrounding healthy tissue, and spatial lesion geometry.
                    <br><strong style="color:#F1F5F9;">Output Dim: 768</strong>
                </div>
            </div>
            <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(129,140,248,0.25);border-radius:12px;padding:1.2rem;">
                <div style="font-size:0.95rem;font-weight:700;color:#818CF8;margin-bottom:0.5rem;">3. Adaptive Gated Fusion</div>
                <div style="font-size:0.84rem;color:#94A3B8;line-height:1.55;">
                    Learns channel-wise gating weights (&alpha;) through an MLP gate:
                    <br><code>F = &alpha; &odot; Proj(CNN) + (1 &minus; &alpha;) &odot; Proj(ViT)</code>
                    <br><strong style="color:#F1F5F9;">Fused Dim: 512</strong>
                </div>
            </div>
            <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(244,63,94,0.25);border-radius:12px;padding:1.2rem;">
                <div style="font-size:0.95rem;font-weight:700;color:#F43F5E;margin-bottom:0.5rem;">4. Regularized Head</div>
                <div style="font-size:0.84rem;color:#94A3B8;line-height:1.55;">
                    LayerNorm &rarr; Dropout (0.4) &rarr; Linear(512, 256) &rarr; GELU &rarr; Dropout (0.4) &rarr; Linear(256, 7).
                    <br><strong style="color:#F1F5F9;">Output: 7 Staging Logits</strong>
                </div>
            </div>
        </div>

        <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1.2rem;">
            <div style="font-size:0.88rem;font-weight:700;color:#F8FAFC;margin-bottom:0.5rem;">📊 Benchmark Test Results (pressurevision_exp3_best.pth)</div>
            <div style="display:flex;gap:2rem;flex-wrap:wrap;font-size:0.85rem;color:#94A3B8;">
                <div>Test Set Size: <strong style="color:#F8FAFC;">342 Samples</strong></div>
                <div>Classification Accuracy: <strong style="color:#10B981;">93.57%</strong></div>
                <div>Macro Precision: <strong style="color:#38BDF8;">94.04%</strong></div>
                <div>Macro Recall: <strong style="color:#38BDF8;">93.87%</strong></div>
                <div>Macro F1-Score: <strong style="color:#38BDF8;">93.94%</strong></div>
                <div>Weighted F1-Score: <strong style="color:#34D399;">93.60%</strong></div>
            </div>
        </div>
    </div>
    """)


# =============================================================================
# TAB 3: Explainability (Grad-CAM)
# =============================================================================
with tab_explain:
    render_html("""
    <div class="glass-panel" style="margin-bottom: 1.5rem;">
        <div style="font-size:1.25rem;font-weight:800;color:#F8FAFC;margin-bottom:0.4rem;">
            🔍 Grad-CAM Visual Explainability Engine
        </div>
        <p style="color:#CBD5E1;font-size:0.92rem;line-height:1.6;margin-bottom:0.2rem;">
            Gradient-weighted Class Activation Mapping (Grad-CAM) visualizes the spatial focus of the 
            <strong>ConvNeXt-Tiny local feature extraction branch</strong> (Stage 3: 768 channels, 7&times;7 feature grid) 
            w.r.t the unnormalized logit score of the complete PressureVision AI hybrid model.
        </p>
    </div>
    """)

    if st.session_state.uploaded_image is None:
        render_html("""
        <div style="background:rgba(15,23,42,0.4);border:2px dashed rgba(255,255,255,0.08);border-radius:18px;padding:3.5rem 2rem;text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;">🩺</div>
            <h4 style="color:#F8FAFC;font-weight:700;margin-bottom:0.4rem;">No Ulcer Photograph Loaded</h4>
            <p style="color:#94A3B8;max-width:520px;margin:0 auto 1.2rem auto;font-size:0.9rem;line-height:1.6;">
                Please upload a wound photograph or pick a clinical reference case in the 
                <strong>🔬 Wound Analysis & Staging</strong> tab to generate Grad-CAM heatmaps.
            </p>
        </div>
        """)
    else:
        cam_img = st.session_state.uploaded_image
        
        # Interactive Controls
        render_html("<div style='font-size:1rem;font-weight:700;color:#38BDF8;margin-bottom:0.8rem;'>🎛️ Grad-CAM Configuration</div>")
        
        c_cam_ctrl1, c_cam_ctrl2, c_cam_ctrl3, c_cam_btn = st.columns([1.5, 1.2, 1.2, 1.4], gap="medium")
        
        # Determine default target class based on current prediction
        pred_label = "Stage III"
        if st.session_state.prediction_result is not None:
            pred_label = st.session_state.prediction_result["predicted_class"]
            
        class_options = [f"Predicted Class ({pred_label})"] + DISPLAY_CLASS_NAMES
        
        with c_cam_ctrl1:
            sel_target = st.selectbox(
                "Target Class to Explain",
                options=class_options,
                index=0,
                help="Class score whose gradient determines the activation map."
            )
            
        with c_cam_ctrl2:
            cam_opacity = st.slider(
                "Overlay Opacity (α)",
                min_value=0.10,
                max_value=0.90,
                value=0.45,
                step=0.05,
                help="Blending opacity between original RGB image and colored activation map."
            )
            
        with c_cam_ctrl3:
            cam_cmap = st.selectbox(
                "Colormap",
                options=["turbo", "jet", "plasma", "inferno", "viridis"],
                index=0,
                help="Color map palette for intensity rendering."
            )
            
        with c_cam_btn:
            st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
            compute_cam_clicked = st.button("⚡ Compute Grad-CAM", type="primary", width="stretch")

        # Parse target class
        if sel_target.startswith("Predicted Class"):
            chosen_target = None
        else:
            chosen_target = sel_target

        # Check if we need to auto-generate or user clicked compute
        should_compute = compute_cam_clicked or (st.session_state.gradcam_result is None)

        if should_compute:
            with st.spinner("Backpropagating gradients through hybrid architecture to ConvNeXt Stage 3..."):
                try:
                    t_cam_start = time.time()
                    cam_res = generate_gradcam(
                        model=model,
                        image=cam_img,
                        device=device,
                        target_class=chosen_target,
                        overlay_opacity=cam_opacity,
                        colormap_name=cam_cmap
                    )
                    cam_res["latency_ms"] = (time.time() - t_cam_start) * 1000.0
                    st.session_state.gradcam_result = cam_res
                except Exception as ex:
                    st.error(f"Grad-CAM error: {str(ex)}")

        # Render 3-Panel Inspection if result exists
        if st.session_state.gradcam_result is not None:
            res = st.session_state.gradcam_result
            
            st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:1.4rem 0;'>", unsafe_allow_html=True)
            
            # Status Banner
            render_html(f"""
            <div style="background:rgba(15,23,42,0.7);border:1px solid rgba(56,189,248,0.25);border-radius:14px;padding:0.9rem 1.4rem;margin-bottom:1.4rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
                <div style="display:flex;align-items:center;gap:1.2rem;">
                    <span style="font-size:1.4rem;">🎯</span>
                    <div>
                        <div style="font-size:0.78rem;font-weight:700;color:#94A3B8;text-transform:uppercase;">Class Explanation Target</div>
                        <div style="font-size:1.15rem;font-weight:800;color:#38BDF8;">{res['target_class']}</div>
                    </div>
                </div>
                <div style="display:flex;gap:1.8rem;font-size:0.82rem;color:#CBD5E1;">
                    <div>Prediction: <strong style="color:#F8FAFC;">{res['predicted_class']} ({res['confidence_percentage']:.1f}%)</strong></div>
                    <div>Target Module: <strong style="color:#38BDF8;">model.convnext.stages[-1]</strong></div>
                    <div>Activation Grid: <strong style="color:#34D399;">7 &times; 7 (768-D)</strong></div>
                    <div>Latency: <strong style="color:#FBBF24;">{res.get('latency_ms', 0.0):.1f} ms</strong></div>
                </div>
            </div>
            """)

            # 3-Panel Display
            p1_col, p2_col, p3_col = st.columns(3, gap="medium")
            
            with p1_col:
                render_html("""
                <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1rem;text-align:center;">
                    <div style="font-size:0.85rem;font-weight:700;color:#94A3B8;text-transform:uppercase;margin-bottom:0.6rem;">1. Original RGB Input</div>
                """)
                st.image(cam_img, width="stretch")
                w_px, h_px = cam_img.size
                render_html(f"""
                    <div style="font-size:0.75rem;color:#64748B;margin-top:0.5rem;">{w_px} &times; {h_px} px &bull; Input Resolution</div>
                </div>
                """)

            with p2_col:
                render_html(f"""
                <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1rem;text-align:center;">
                    <div style="font-size:0.85rem;font-weight:700;color:#38BDF8;text-transform:uppercase;margin-bottom:0.6rem;">2. ConvNeXt Activation Heatmap</div>
                """)
                st.image(res["colored_heatmap_pil"], width="stretch")
                render_html(f"""
                    <div style="font-size:0.75rem;color:#64748B;margin-top:0.5rem;">Normalized [0.0, 1.0] &bull; Colormap: <code>{cam_cmap}</code></div>
                </div>
                """)

            with p3_col:
                render_html(f"""
                <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(56,189,248,0.2);border-radius:14px;padding:1rem;text-align:center;">
                    <div style="font-size:0.85rem;font-weight:700;color:#34D399;text-transform:uppercase;margin-bottom:0.6rem;">3. Blended Overlay</div>
                """)
                st.image(res["overlay_pil"], width="stretch")
                render_html(f"""
                    <div style="font-size:0.75rem;color:#64748B;margin-top:0.5rem;">Alpha blend (α = {cam_opacity:.2f}) &bull; Focused spatial regions</div>
                </div>
                """)

            # Mathematical Formulation & Explainability Limitation Notice
            render_html("""
            <div style="margin-top:1.5rem;background:rgba(15,23,42,0.5);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:1.2rem;">
                <div style="font-size:0.9rem;font-weight:700;color:#F8FAFC;margin-bottom:0.6rem;">
                    📐 Gradient-Weighted Formulation & Clinical Interpretability Notice
                </div>
                <div style="font-size:0.84rem;color:#94A3B8;line-height:1.7;">
                    The activation map is calculated via spatial Global Average Pooling of backpropagated gradients:
                    <br>
                    <code style="color:#38BDF8;">α<sub>k</sub><sup>c</sup> = (1 / (H &times; W)) &sum;<sub>i</sub> &sum;<sub>j</sub> (∂y<sup>c</sup> / ∂A<sub>i,j</sub><sup>k</sup>)</code>
                    &nbsp;&nbsp;&bull;&nbsp;&nbsp;
                    <code style="color:#34D399;">L<sub>Grad-CAM</sub><sup>c</sup> = ReLU(&sum;<sub>k</sub> α<sub>k</sub><sup>c</sup> A<sup>k</sup>)</code>
                    <br><br>
                    <strong style="color:#FBBF24;">⚠️ Clinical Decision-Support Notice:</strong>
                    This visualization illustrates the local textural and morphological cues in the ConvNeXt branch that influenced the full hybrid model's unnormalized logit score for the target class. It does <em>not</em> represent histological depth, clinical wound edge margin delineation, or automated surgical boundaries.
                </div>
            </div>
            """)


# =============================================================================
# TAB 4: Staging Guidelines & Criteria
# =============================================================================
with tab_reference:
    ref_cards_html = ""
    for r_name, d_name in zip(RAW_CLASS_NAMES, DISPLAY_CLASS_NAMES):
        desc = CLASS_DESCRIPTIONS.get(r_name, "")
        ref_cards_html += f"""
        <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.05);border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem;">
            <div style="font-size:0.95rem;font-weight:700;color:#38BDF8;margin-bottom:0.3rem;">{d_name}</div>
            <div style="font-size:0.85rem;color:#CBD5E1;line-height:1.5;">{desc}</div>
        </div>
        """

    full_ref_html = f"""
    <div class="glass-panel">
        <div style="font-size:1.25rem;font-weight:800;color:#F8FAFC;margin-bottom:0.6rem;">
            📚 Clinical Pressure Ulcer Staging Guidelines
        </div>
        <p style="color:#CBD5E1;font-size:0.92rem;line-height:1.6;margin-bottom:1.2rem;">
            Pressure ulcer staging follows the standard clinical definitions established by the National Pressure Injury Advisory Panel (NPIAP) and European Pressure Ulcer Advisory Panel (EPUAP).
        </p>
        {ref_cards_html}
    </div>
    """
    render_html(full_ref_html)


# =============================================================================
# TAB 5: Model Specs & Telemetry
# =============================================================================
with tab_specs:
    specs_html = f"""
    <div class="glass-panel">
        <div style="font-size:1.25rem;font-weight:800;color:#F8FAFC;margin-bottom:0.6rem;">
            ⚙️ Deep Learning Engine Specifications
        </div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:1.2rem;margin-top:1.2rem;">
            <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1.2rem;">
                <div style="font-size:0.75rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.05em;">Model Checkpoint</div>
                <div style="font-size:1rem;font-weight:700;color:#38BDF8;margin-top:0.2rem;">pressurevision_exp3_best.pth</div>
                <div style="font-size:0.82rem;color:#64748B;margin-top:0.4rem;">Pre-trained weights file (680 MB)</div>
            </div>
            <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1.2rem;">
                <div style="font-size:0.75rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.05em;">Total Parameters</div>
                <div style="font-size:1.4rem;font-weight:800;color:#F8FAFC;margin-top:0.2rem;">56,657,121</div>
                <div style="font-size:0.82rem;color:#10B981;margin-top:0.4rem;">100% Strict Weight Match</div>
            </div>
            <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1.2rem;">
                <div style="font-size:0.75rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.05em;">Benchmark Test Accuracy</div>
                <div style="font-size:1.4rem;font-weight:800;color:#10B981;margin-top:0.2rem;">93.57%</div>
                <div style="font-size:0.82rem;color:#64748B;margin-top:0.4rem;">Macro F1: 93.94% | Precision: 94.04% | Recall: 93.87%</div>
            </div>
            <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1.2rem;">
                <div style="font-size:0.75rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.05em;">Inference Resolution</div>
                <div style="font-size:1.4rem;font-weight:800;color:#F8FAFC;margin-top:0.2rem;">224 × 224</div>
                <div style="font-size:0.82rem;color:#64748B;margin-top:0.4rem;">ImageNet Normalized RGB Tensor</div>
            </div>
        </div>
    </div>
    """
    render_html(specs_html)


# =============================================================================
# 6. Regulatory & Medical Disclaimer
# =============================================================================
render_html("""
<div class="medical-alert">
    <div style="font-size:1.4rem;line-height:1;">⚠️</div>
    <div>
        <div class="medical-alert-title">Clinical Decision-Support & Diagnostic Notice</div>
        <p class="medical-alert-text">
            PressureVision AI is an artificial intelligence decision-support tool developed for research, academic evaluation, 
            and clinical workflow demonstration. It is not an FDA/CE-cleared primary diagnostic device and must not replace 
            the clinical judgment, physical examination, or diagnostic protocols of licensed healthcare practitioners.
        </p>
    </div>
</div>
""")


# =============================================================================
# 7. Enterprise Footer
# =============================================================================
render_html("""
<div class="app-footer">
    <strong>PressureVision™ AI</strong> &bull; Clinical Pressure Ulcer Staging Intelligence &bull; Powered by ConvNeXt-Tiny & Swin-Tiny Hybrid Architecture<br>
    Model Checkpoint: <code>pressurevision_exp3_best.pth</code> (56.6M Parameters | 93.57% Benchmark Test Accuracy | Macro F1: 93.94%)
</div>
""")
