import sys
import os

# Force UTF-8 output so that CO₂e and other unicode strings print correctly
# on Windows (default cp1252 codec does not support ₂)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import cv2
import numpy as np
import rasterio as rio
from src.utils.image_io import process_image, load_jp2_as_rgb
from PIL import Image

# Use one of the samples provided by the user
RGB_PATH = r"d:\Desktop\carbon-sequestration\geonrw_samples\aachen_296_5625_rgb.jp2"
SEG_PATH = r"d:\Desktop\carbon-sequestration\geonrw_samples\aachen_296_5625_seg.tif"
OUTPUT_DIR = r"d:\Desktop\carbon-sequestration\small_test_outputs"

def run_test():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Creating small crop of {RGB_PATH}...")
    # Create small crop to speed up detectree
    with rio.open(RGB_PATH) as src:
        img = src.read()[:, :200, :200]
        meta = src.meta.copy()
        meta.update(width=200, height=200, count=src.count)
        
    small_rgb = 'small_rgb.jp2'
    with rio.open(small_rgb, 'w', **meta) as dst:
        dst.write(img)
        
    with rio.open(SEG_PATH) as src:
        img = src.read()[:, :200, :200]
        meta = src.meta.copy()
        meta.update(width=200, height=200)
        
    small_seg = 'small_seg.tif'
    with rio.open(small_seg, 'w', **meta) as dst:
        dst.write(img)
        
    print("Running process_image on crop...")
    results = process_image(small_rgb, small_seg)
    
    print("\n--- Results (200x200 crop) ---")
    for k, v in results['results'].items():
        print(f"{k}: {v}")
        
    # Save visualizations
    Image.fromarray(results['ndvi_image']).save(os.path.join(OUTPUT_DIR, "ndvi_map.png"))
    Image.fromarray(results['carbon_map']).save(os.path.join(OUTPUT_DIR, "carbon_map.png"))
    
    print(f"\nVisualizations saved to {OUTPUT_DIR}")
    
    # Cleanup small files
    os.unlink(small_rgb)
    os.unlink(small_seg)

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        import traceback
        traceback.print_exc()
