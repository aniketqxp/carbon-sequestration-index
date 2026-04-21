import os
import cv2
from src.utils.image_io import process_image
from PIL import Image

# Use one of the samples provided by the user
RGB_PATH = r"d:\Desktop\carbon-sequestration\geonrw_samples\aachen_296_5625_rgb.jp2"
SEG_PATH = r"d:\Desktop\carbon-sequestration\geonrw_samples\aachen_296_5625_seg.tif"
OUTPUT_DIR = r"d:\Desktop\carbon-sequestration\test_outputs"

def run_test():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Opening sample: {RGB_PATH}")
    results = process_image(RGB_PATH, SEG_PATH)
    
    print("\n--- Results ---")
    for k, v in results['results'].items():
        print(f"{k}: {v}")
        
    # Save visualizations
    Image.fromarray(results['ndvi_image']).save(os.path.join(OUTPUT_DIR, "ndvi_map.png"))
    Image.fromarray(results['carbon_map']).save(os.path.join(OUTPUT_DIR, "carbon_map.png"))
    Image.fromarray(results['vegetation_mask']).save(os.path.join(OUTPUT_DIR, "veg_detection.png"))
    
    print(f"\nVisualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        import traceback
        traceback.print_exc()
