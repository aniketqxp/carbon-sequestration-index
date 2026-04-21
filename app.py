"""
app.py — Carbon Sequestration Analysis Platform
==============================================
Streamlit application entry point.

Run with:
    streamlit run app.py

This file contains ONLY UI orchestration (layout, widgets, session state).
All scientific logic lives under src/.
"""

from __future__ import annotations

import io as python_io
import warnings
import sys
import os

# Force UTF-8 output for Windows support
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import threading
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx

from src.carbon.species import create_species_data, get_species_data_table
from src.utils.image_io import save_uploaded_file, process_image

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Carbon Sequestration Analysis Platform",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    /* Global Font & Spacing */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }
    
    .main { 
        padding: 3rem; 
        background-color: #ffffff; 
    }

    /* Professional Typography */
    .title { 
        font-size: 2.25rem; 
        font-weight: 700;
        margin-bottom: 0.25rem; 
        color: #0f172a;
        letter-spacing: -0.02em;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 3rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* Primary Action Button - Sharp & Professional */
    div.stButton > button:first-child {
        width: 100%;
        height: 3.25rem;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        background-color: #0f172a;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    div.stButton > button:first-child:hover {
        background-color: #1e293b;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }
    div.stButton > button:first-child:active {
        transform: translateY(0px);
    }

    /* Refined Containers */
    .report-card {
        border-radius: 12px;
        padding: 2.5rem;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        margin-bottom: 2.5rem;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    .metric-title { 
        font-size: 0.75rem; 
        text-transform: uppercase; 
        letter-spacing: 0.1em; 
        color: #64748b; 
        margin-bottom: 0.75rem; 
        font-weight: 700; 
    }
    .metric-value { 
        font-size: 1.75rem;  
        font-weight: 800; 
        color: #0f172a; 
        letter-spacing: -0.02em;
    }
    .carbon-highlight { 
        font-size: 2.5rem; 
        color: #059669; 
        font-weight: 900; 
        letter-spacing: -0.04em;
        line-height: 1;
    }

    /* CSI Scale */
    .csi-scale {
        height: 10px;
        background: linear-gradient(90deg, 
            #ef4444 0%, 
            #f59e0b 20%, 
            #f1c40f 40%, 
            #2ecc71 60%, 
            #22c55e 80%, 
            #16a085 100%
        );
        border-radius: 5px;
        margin: 1.5rem 0 0.75rem 0;
        position: relative;
        overflow: visible;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);
    }
    .csi-marker {
        position: absolute;
        top: -6px;
        width: 6px;
        height: 22px;
        background: #ffffff;
        border: 2px solid #0f172a;
        border-radius: 3px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        z-index: 2;
        transition: left 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .csi-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Timer display */
    .timer-display {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .timer-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .timer-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        font-variant-numeric: tabular-nums;
        line-height: 1;
    }
    .timer-pulse {
        font-size: 0.8rem;
        color: #10b981;
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Clean up streamlit standard components */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0;
        border: none;
        color: #94a3b8;
        font-weight: 500;
        padding: 0;
    }
    .stTabs [aria-selected="true"] {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "processed" not in st.session_state:
    st.session_state.processed = False
if "results" not in st.session_state:
    st.session_state.results = None


# ---------------------------------------------------------------------------
# Helper: display editable species table
# ---------------------------------------------------------------------------

def render_species_editor() -> dict:
    """
    Render the species parameters table and an expandable coverage editor.
    """
    species_data = create_species_data()
    st.markdown("### Vegetation Species Parameters")
    st.dataframe(get_species_data_table(), use_container_width=True)

    with st.expander("Adjust Species Coverage", expanded=False):
        st.markdown(
            "Adjust the species distribution percentages below "
            "(values must sum to 100%):"
        )
        _, _, content_col, _, _ = st.columns([1, 1, 2, 1, 1])

        with content_col:
            coverage_values: dict[str, float] = {}
            total = 0.0
            for species, data in species_data.items():
                coverage_values[species] = st.slider(
                    f"{species} coverage",
                    min_value=0.0,
                    max_value=100.0,
                    value=data["default_coverage"] * 100,
                    step=0.5,
                    key=f"coverage_{species}",
                )
                total += coverage_values[species]

            if abs(total - 100.0) > 0.1:
                st.warning(f"Current total: {total:.1f}%. Adjust to sum to exactly 100%.")
                st.stop()

            for species, pct in coverage_values.items():
                species_data[species]["default_coverage"] = pct / 100.0

    return species_data


# ---------------------------------------------------------------------------
# Helper: advanced detection settings
# ---------------------------------------------------------------------------

def render_detection_settings() -> tuple[float, float]:
    """
    Render sliders for detectree confidence and patch area filtering.
    """
    with st.expander("Advanced Detection Settings", expanded=False):
        st.info("Adjust these if the vegetation mask appears empty or includes non-vegetation.")
        conf = st.slider(
            "Detection Confidence Threshold",
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.01,
            help="Lower values detect more vegetation but may include noise."
        )
        area = st.slider(
            "Minimum Patch Area (pixels)",
            min_value=10,
            max_value=2000,
            value=700,
            step=10,
            help="Smaller patches will be filtered out as noise."
        )
    return conf, area


# ---------------------------------------------------------------------------
# Helper: render the carbon report card
# ---------------------------------------------------------------------------

def render_report_card(results: dict) -> None:
    """Render the styled carbon sequestration report card."""
    r = results["results"]
    csi_float = float(r["Carbon Sequestration Index (CSI)"].split()[0])
    csi_percent = min(csi_float * 20, 100)  # scale 0-5 to 0-100%

    st.markdown("### Carbon Sequestration Report")
    st.markdown(
        f"""
<div style="position: relative;">
  <div class="report-card">
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
      <div class="metric-card">
        <div class="metric-title">Vegetated Area</div>
        <div class="metric-value">{r["Vegetated Area"]}</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Sequestration Rate (Vegetated)</div>
        <div class="metric-value">{r["Areal Rate (Vegetated)"]}</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Sequestration Rate (Total Region)</div>
        <div class="metric-value">{r["Areal Rate (Total Region)"]}</div>
      </div>
      <div class="metric-card" style="grid-column: span 2;">
        <div class="metric-title">Total Annual Carbon Sequestration</div>
        <div class="carbon-highlight">{r["Total Annual Sequestration"]}</div>
      </div>
    </div>
    <div style="margin-top: 1.5rem;">
      <div style="display: flex; align-items: center; justify-content: space-between;">
        <div style="font-weight: 600; color: #1e293b; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem;">Carbon Sequestration Index (CSI)</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #10b981;">{r["Carbon Sequestration Index (CSI)"]}</div>
      </div>
      <div class="csi-scale">
        <div class="csi-marker" style="left: {csi_percent:.1f}%;"></div>
      </div>
      <div class="csi-labels">
        <div>0.0</div><div>1.0</div><div>2.0</div><div>3.0</div><div>4.0</div><div>5.0+</div>
      </div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("Interpretation Guide", expanded=False):
        legend_colors = ["#ef4444", "#f59e0b", "#f1c40f", "#2ecc71", "#22c55e", "#16a085"]
        bands = list(results["interpretation"].items())
        for (band, label), color in zip(bands, legend_colors):
            st.markdown(
                f'<div style="display:flex;align-items:center;margin-bottom:0.4rem;">'
                f'<div style="width:12px;height:12px;background:{color};border-radius:2px;'
                f'margin-right:0.5rem;"></div>'
                f'<div style="font-size: 0.85rem;"><strong style="color:#0f172a;">{band}</strong>: {label}</div></div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Helper: render output images
# ---------------------------------------------------------------------------

def render_output_images(results: dict) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(results["vegetation_mask"],   caption="Detectree Vegetation Detection",        use_container_width=True)
    with col2:
        st.image(results["segmentation_mask"], caption="Segmentation Mask (from TIF)",          use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(results["ndvi_image"],        caption="NDVI (Pseudo — RGB proxy)",             use_container_width=True)
    with col2:
        st.image(
            results["carbon_map"],
            caption=f"Carbon Sequestration Map  "
                    f"(Total: {results['total_carbon'] / 1000:.1f} tCO2e/yr)",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Helper: download buttons
# ---------------------------------------------------------------------------

def render_downloads(results: dict) -> None:
    st.markdown("### Export Analysis")

    def _to_png_bytes(arr) -> bytes:
        buf = python_io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download NDVI Map",
            data=_to_png_bytes(results["ndvi_image"]),
            file_name="ndvi_map.png",
            mime="image/png",
            use_container_width=True
        )
    with col2:
        st.download_button(
            "Download Carbon Map",
            data=_to_png_bytes(results["carbon_map"]),
            file_name="carbon_map.png",
            mime="image/png",
            use_container_width=True
        )
    with col3:
        csv_bytes = pd.DataFrame([results["results"]]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Report (CSV)",
            data=csv_bytes,
            file_name="carbon_sequestration_report.csv",
            mime="text/csv",
            use_container_width=True
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown("<h1 class='title'>Carbon Sequestration Analysis Platform</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Quantifying biomass and carbon offset capacity from multi-spectral imagery</p>", 
        unsafe_allow_html=True
    )

    tab_sample, tab_upload = st.tabs(["Select GeoNRW Sample", "Upload Custom Image"])

    jp2_path = None
    tif_path = None

    with tab_sample:
        st.markdown("### Browse Pre-loaded Samples")
        sample_dir = os.path.join(os.path.dirname(__file__), "geonrw_samples")
        if os.path.exists(sample_dir):
            all_files = os.listdir(sample_dir)
            base_names = sorted(list(set([f.replace("_rgb.jp2", "").replace("_seg.tif", "") for f in all_files if f.endswith(("_rgb.jp2", "_seg.tif"))])))
            
            selected_base = st.selectbox("Select a region to analyze:", base_names, index=0)
            if selected_base:
                t_jp2 = os.path.join(sample_dir, f"{selected_base}_rgb.jp2")
                t_tif = os.path.join(sample_dir, f"{selected_base}_seg.tif")
                if os.path.exists(t_jp2) and os.path.exists(t_tif):
                    jp2_path = t_jp2
                    tif_path = t_tif
                    st.session_state.is_temp = False
        else:
            st.error("Sample directory not found.")

    with tab_upload:
        st.markdown("### Upload Custom Imagery")
        st.info("Input requirements: JPEG2000 (.jp2) aerial image and matching GeoTIFF (.tif) segmentation mask.")
        col1, col2 = st.columns(2)
        with col1:
            jp2_file = st.file_uploader(
                "Aerial Imagery (JP2)",
                type=["jp2"],
                key="jp2_uploader"
            )
        with col2:
            tif_file = st.file_uploader(
                "Segmentation Mask (TIF)",
                type=["tif", "tiff"],
                key="tif_uploader"
            )
        
        if jp2_file and tif_file:
            jp2_path = save_uploaded_file(jp2_file)
            tif_path = save_uploaded_file(tif_file)
            st.session_state.is_temp = True


    if jp2_path is not None and os.path.exists(jp2_path):
        st.markdown("### Source Imagery Preview")
        try:
            img_bgr = cv2.imread(jp2_path)
            if img_bgr is None:
                import rasterio as rio
                with rio.open(jp2_path) as src:
                    img = src.read()[:3].transpose(1, 2, 0)
            else:
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            p2, p98 = np.percentile(img, [2, 98])
            img_display = np.clip(img, p2, p98)
            img_display = ((img_display - p2) / (p98 - p2) * 255).astype("uint8")
            st.image(img_display, caption=f"Selected: {os.path.basename(jp2_path)}", use_container_width=True)
        except Exception as exc:
            st.error(f"Image preview error: {exc}")

    if jp2_path is not None and tif_path is not None:
        species_data = render_species_editor()
        conf, area   = render_detection_settings()

        st.markdown("<br>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1, 2, 1])
        
        with btn_col:
            if st.button("RUN ANALYSIS", use_container_width=True):
                class ThreadResult:
                    def __init__(self):
                        self.result = None
                        self.error = None
                        self.done = False

                def background_processing(container, jp2, tif, c_thresh, m_area):
                    try:
                        container.result = process_image(
                            jp2, tif, 
                            confidence_thresh=c_thresh, 
                            min_area=m_area
                        )
                    except Exception as e:
                        container.error = e
                    finally:
                        container.done = True

                result_container = ThreadResult()
                t = threading.Thread(
                    target=background_processing, 
                    args=(result_container, jp2_path, tif_path, conf, area)
                )
                add_script_run_ctx(t)
                
                timer_placeholder = st.empty()
                t.start()
                start_time = time.time()
                
                while not result_container.done:
                    elapsed = time.time() - start_time
                    mins, secs = divmod(int(elapsed), 60)
                    timer_placeholder.markdown(
                        f"<div class='timer-display'>"
                        f"<div class='timer-label'>Analysis in Progress</div>"
                        f"<div class='timer-value'>{mins:02d}:{secs:02d}</div>"
                        f"<div class='timer-pulse'>Processing spatial bands</div>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                    time.sleep(0.5)
                
                timer_placeholder.empty()

                if result_container.error:
                    st.error(f"Analysis failed: {result_container.error}")
                else:
                    st.session_state.results = result_container.result
                    st.session_state.processed = True
                    st.info(f"Analysis cycle complete ({time.time() - start_time:.1f}s)")
                    
                    if getattr(st.session_state, 'is_temp', False):
                        if os.path.exists(jp2_path): os.unlink(jp2_path)
                        if os.path.exists(tif_path): os.unlink(tif_path)
                    
                    time.sleep(0.5)
                    st.rerun()

    if st.session_state.processed and st.session_state.results is not None:
        st.markdown("---")
        results = st.session_state.results
        render_report_card(results)
        render_output_images(results)
        render_downloads(results)


if __name__ == "__main__":
    main()
