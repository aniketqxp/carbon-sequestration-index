"""
vegetation/ndvi.py
------------------
Pseudo-NDVI calculation from RGB aerial imagery.

Since true NDVI requires a Near-Infrared (NIR) band, this module computes a
Green-Red-Blue proxy that correlates with vegetation density for standard RGB
or multispectral JP2 inputs.
"""

import numpy as np


def calculate_ndvi_from_rgb(image: np.ndarray) -> np.ndarray:
    """
    Calculate pseudo-NDVI from an RGB image array.

    For standard RGB imagery (no NIR band available) the formula uses the
    green channel as a NIR proxy:

        pseudo-NDVI = (Green - Red) / (Green + Red + Blue + ε)

    Args:
        image: HxWx3 uint8 or float numpy array in RGB channel order.

    Returns:
        2-D float array of the same spatial dimensions with values in [-1, 1].
    """
    blue  = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    red   = image[:, :, 2].astype(float)

    epsilon = 1e-10  # Avoid division by zero
    pseudo_ndvi = (green - red) / (green + red + blue + epsilon)
    return pseudo_ndvi
