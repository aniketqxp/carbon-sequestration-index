
import time
import sys

def test_import(name):
    start = time.time()
    print(f"Importing {name}...", end=" ", flush=True)
    try:
        if name == "cv2": import cv2
        elif name == "np": import numpy as np
        elif name == "pd": import pandas as pd
        elif name == "st": import streamlit as st
        elif name == "rio": import rasterio as rio
        elif name == "dtr": import detectree as dtr
        print(f"Done in {time.time() - start:.2f}s")
    except Exception as e:
        print(f"FAILED: {e}")

test_import("cv2")
test_import("np")
test_import("pd")
test_import("st")
test_import("rio")
test_import("dtr")
print("Finished.")
