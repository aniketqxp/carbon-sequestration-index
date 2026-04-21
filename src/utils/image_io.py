"""
utils/image_io.py
-----------------
I/O helpers and the master image processing pipeline.

Responsibilities:
  - Persist Streamlit UploadedFile objects to disk so that rasterio / detectree
    (which require real file paths) can read them.
  - Orchestrate the end-to-end pipeline: load → detect → NDVI → carbon →
    visualise → return result dict consumed by the Streamlit UI.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.vegetation.ndvi import calculate_ndvi_from_rgb
from src.vegetation.detection import get_red_contour_mask
from src.carbon.estimation import estimate_carbon, calculate_csi, format_results


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def save_uploaded_file(uploaded_file) -> str:
    """
    Write a Streamlit UploadedFile to a named temporary file on disk.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Absolute path to the temporary file (caller is responsible for
        deleting it via ``os.unlink`` when done).
    """
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def load_jp2_as_rgb(jp2_path: str) -> np.ndarray:
    """
    Load a JP2 (JPEG2000) file with OpenCV and return an HxWx3 uint8 array.

    Args:
        jp2_path: Absolute path to the JP2 file.

    Returns:
        HxWx3 numpy array (BGR order, converted to RGB).
    """
    image_bgr = cv2.imread(jp2_path)
    if image_bgr is None:
        # Fallback to rasterio only if OpenCV fails
        import rasterio as rio
        with rio.open(jp2_path) as src:
            image = src.read()
            image = np.transpose(image, (1, 2, 0))
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] > 3:
                image = image[:, :, :3]
            return image

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_image(
    jp2_path: str,
    tif_path: str,
    confidence_thresh: float = 0.1,
    min_area: float = 700.0
) -> dict:
    """
    End-to-end pipeline: raw imagery → carbon sequestration metrics + visuals.

    Steps
    -----
    1.  Load RGB image from JP2 with rasterio.
    2.  Write a temporary GeoTIFF so detectree can classify trees.
    3.  Build the refined vegetation mask (red-contour method).
    4.  Compute and normalise pseudo-NDVI over the masked region.
    5.  Load the segmentation TIF and colourise it.
    6.  Derive pixel area from the JP2's geotransform (fallback: 1 m²).
    7.  Estimate per-pixel carbon sequestration and aggregate totals.
    8.  Colourise the NDVI and carbon maps for display.

    Args:
        jp2_path: Path to the aerial JP2 image.
        tif_path: Path to the segmentation mask TIF.
        confidence_thresh: Minimum detectree probability to consider [0-1].
        min_area: Minimum area of a vegetation patch (pixels).

    Returns:
        Dict with keys listed in docstring...
    """
    # ------------------------------------------------------------------
    # 1. Load RGB image
    # ------------------------------------------------------------------
    image = load_jp2_as_rgb(jp2_path)
    height, width = image.shape[:2]

    # ------------------------------------------------------------------
    # 2. Write temp GeoTIFF for detectree (Normalised)
    # ------------------------------------------------------------------
    # Some models expect specific dynamic ranges. Pre-stretching the image
    # from its 1%-99% range to 0-255 uint8 helps ensure the forest
    # classifier sees a consistent signal.
    def _stretch(img_chan):
        p1, p99 = np.percentile(img_chan, [1, 99])
        if p99 == p1: return img_chan
        rescaled = np.clip(img_chan, p1, p99)
        return ((rescaled - p1) / (p99 - p1) * 255).astype(np.uint8)
    
    proc_image = np.zeros_like(image)
    for i in range(3):
        proc_image[..., i] = _stretch(image[..., i])

    tmp_tif = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
    try:
        import rasterio as rio
        with rio.open(
            tmp_tif, "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype=np.uint8,
        ) as dst:
            dst.write(np.transpose(proc_image, (2, 0, 1)))

        # ------------------------------------------------------------------
        # 3. Vegetation detection
        # ------------------------------------------------------------------
        import detectree as dtr
        y_pred = dtr.Classifier().predict_img(tmp_tif)
        
        # Logging stats to diagnose blank masks
        print(f"Detectree Raw Stats: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}")
    finally:
        os.unlink(tmp_tif)

    biased_y_pred = np.where(y_pred > confidence_thresh, 1, 0)
    
    # Adaptive fallback if detection is completely empty but there's "some" confidence
    if np.sum(biased_y_pred) == 0 and y_pred.max() > 0.01:
        print(f"Warning: No vegetation found at confidence > {confidence_thresh}. Falling back to 0.02.")
        biased_y_pred = np.where(y_pred > 0.02, 1, 0)
        
    px_count = np.sum(biased_y_pred)
    print(f"Thresholded ({confidence_thresh}) raw pixels: {px_count} ({px_count/(height*width)*100:.2f}%)")
    
    binary_mask   = (biased_y_pred * 255).astype(np.uint8)

    red_contour_mask = get_red_contour_mask(biased_y_pred, binary_mask, min_area=min_area)

    # ------------------------------------------------------------------
    # 4. NDVI (masked & normalised)
    # ------------------------------------------------------------------
    ndvi        = calculate_ndvi_from_rgb(image)
    masked_ndvi = ndvi * red_contour_mask

    masked_values = masked_ndvi[red_contour_mask > 0]
    if len(masked_values) > 0:
        min_val, max_val = masked_values.min(), masked_values.max()
        if max_val > min_val:
            masked_ndvi = np.where(
                red_contour_mask > 0,
                (masked_ndvi - min_val) / (max_val - min_val),
                0,
            )

    # ------------------------------------------------------------------
    # 5. Segmentation TIF → colourised array
    # ------------------------------------------------------------------
    seg_data = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if seg_data is None:
        import rasterio as rio
        with rio.open(tif_path) as seg_src:
            seg_data = seg_src.read(1)

    norm_seg    = (seg_data - seg_data.min()) / (seg_data.max() - seg_data.min() + 1e-8)
    colored_seg = (plt.cm.viridis(norm_seg)[:, :, :3] * 255).astype(np.uint8)

    # Colourised vegetation output: Apply viridis colormap to the final mask.
    # Cast to float32 first so 0.0 maps to purple and 1.0 maps to yellow.
    veg_colored = (plt.cm.viridis(red_contour_mask.astype(np.float32))[:, :, :3] * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # 6. Pixel area from metadata (with fallback)
    # ------------------------------------------------------------------
    pixel_area = 100.0   # Default: Sentinel-2 (10m x 10m)
    try:
        import rasterio as rio
        with rio.open(jp2_path) as src:
            transform = src.transform
            pixel_area = abs(transform[0]) * abs(transform[4])
    except:
        pass  # keep default if rasterio hangs/fails

    if pixel_area < 0.01:
        pixel_area = 1.0  # fallback for unscaled imagery

    # ------------------------------------------------------------------
    # 7. Carbon estimation
    # ------------------------------------------------------------------
    total_carbon, carbon_map = estimate_carbon(
        red_contour_mask, masked_ndvi, pixel_area
    )

    vegetated_area_m2 = np.sum(red_contour_mask > 0) * pixel_area
    csi               = calculate_csi(total_carbon, vegetated_area_m2)
    results           = format_results(total_carbon, vegetated_area_m2, csi)

    # ------------------------------------------------------------------
    # 8. Visualisation arrays
    # ------------------------------------------------------------------
    # NDVI → JET colourmap (BGR→RGB)
    ndvi_viz    = (masked_ndvi * 255).astype(np.uint8)
    ndvi_bgr    = cv2.applyColorMap(ndvi_viz, cv2.COLORMAP_JET)
    ndvi_bgr[red_contour_mask == 0] = 0
    ndvi_rgb    = cv2.cvtColor(ndvi_bgr, cv2.COLOR_BGR2RGB)

    # Carbon map → VIRIDIS colourmap (BGR→RGB)
    carbon_norm = carbon_map / (np.max(carbon_map) + 1e-10)
    carbon_viz  = (carbon_norm * 255).astype(np.uint8)
    carbon_bgr  = cv2.applyColorMap(carbon_viz, cv2.COLORMAP_VIRIDIS)
    carbon_rgb  = cv2.cvtColor(carbon_bgr, cv2.COLOR_BGR2RGB)

    from src.carbon.estimation import CSI_INTERPRETATION  # avoid circular at module level

    return {
        "original_image":    image,
        "vegetation_mask":   veg_colored,
        "segmentation_mask": colored_seg,
        "ndvi_image":        ndvi_rgb,
        "carbon_map":        carbon_rgb,
        "results":           results,
        "interpretation":    CSI_INTERPRETATION,
        "csi":               csi,
        "total_carbon":      total_carbon,
        "vegetated_area":    vegetated_area_m2,
    }
