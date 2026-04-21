import os
import cv2
import numpy as np
import detectree as dtr
import rasterio as rio
import tempfile
from src.utils.image_io import load_jp2_as_rgb
from src.vegetation.detection import get_red_contour_mask

# Selective sample
sample_dir = "geonrw_samples"
all_files = os.listdir(sample_dir)
base_names = sorted(list(set([f.replace("_rgb.jp2", "").replace("_seg.tif", "") for f in all_files if f.endswith(("_rgb.jp2", "_seg.tif"))])))
if not base_names:
    print("No samples found")
    exit()

selected_base = base_names[0]
jp2_path = os.path.join(sample_dir, f"{selected_base}_rgb.jp2")

print(f"Testing on {jp2_path}")
image = load_jp2_as_rgb(jp2_path)
height, width = image.shape[:2]

tmp_tif = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
try:
    with rio.open(
        tmp_tif, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=image.dtype,
    ) as dst:
        dst.write(np.transpose(image, (2, 0, 1)))

    y_pred = dtr.Classifier().predict_img(tmp_tif)
    print(f"Detectree Raw Stats: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}")
    
    # Use user's problematic values
    conf_thresh = 0.05
    min_area = 70
    
    biased_y_pred = np.where(y_pred > conf_thresh, 1, 0)
    print(f"Biased pixels at {conf_thresh}: {np.sum(biased_y_pred)}")
    
    binary_mask = (biased_y_pred * 255).astype(np.uint8)
    refined_mask = get_red_contour_mask(biased_y_pred, binary_mask, min_area=min_area)
    print(f"Refined mask sum: {np.sum(refined_mask)}")

finally:
    if os.path.exists(tmp_tif):
        os.unlink(tmp_tif)
