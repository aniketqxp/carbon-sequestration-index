import detectree as dtr
import os
import numpy as np
import rasterio as rio

# Simple tiny 100x100 RGB image
img = np.zeros((100, 100, 3), dtype='uint8')
with rio.open('test.tif', 'w', driver='GTiff', height=100, width=100, count=3, dtype='uint8') as dst:
    dst.write(img.transpose(2,0,1))

print("Predicting with detectree...")
try:
    p = dtr.Classifier().predict_img('test.tif')
    print(f"Predict success: {p.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists('test.tif'):
        os.remove('test.tif')
