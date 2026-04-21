import os
import numpy as np
import cv2
from PIL import Image
from unittest.mock import patch

# Mock detectree so we can verify the rest of the pipeline works
class MockClassifier:
    def predict_img(self, path):
        # Return a dummy mask (say, consistent 80% tree probability)
        img = cv2.imread(path)
        return np.ones((img.shape[0], img.shape[1]), dtype='float32') * 0.8

# Replace detectree.Classifier with our mock
with patch('detectree.Classifier', return_value=MockClassifier()):
    # Import process_image AFTER patching
    from src.utils.image_io import process_image
    
    RGB_PATH = r"d:\Desktop\carbon-sequestration\geonrw_samples\aachen_296_5625_rgb.jp2"
    SEG_PATH = r"d:\Desktop\carbon-sequestration\geonrw_samples\aachen_296_5625_seg.tif"
    OUTPUT_DIR = r"d:\Desktop\carbon-sequestration\mock_test_outputs"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Running carbon pipeline test on: {RGB_PATH} (Mocked classification)")
    results = process_image(RGB_PATH, SEG_PATH)
    
    print("\n--- Pipeline Analytics ---")
    for k, v in results['results'].items():
        print(f"{k}: {v}")
        
    # Save visualizations
    Image.fromarray(results['ndvi_image']).save(os.path.join(OUTPUT_DIR, "ndvi_map.png"))
    Image.fromarray(results['carbon_map']).save(os.path.join(OUTPUT_DIR, "carbon_map.png"))
    
    print(f"\nVerification complete. Visuals saved to {OUTPUT_DIR}")
