"""
vegetation/detection.py
-----------------------
Tree / vegetation detection helpers built on top of the detectree library.

The main outputs are binary masks that identify vegetated pixels in an aerial
image.  These masks are then used by the carbon estimation pipeline.
"""

import cv2
import numpy as np


def find_buffered_black_region_contours(
    cleaned_mask: np.ndarray,
    buffer_distance: float = 10,
    min_area: float = 100,
):
    """
    Expand non-vegetation (black) regions outward by *buffer_distance* pixels
    using a distance transform, then find contours of the resulting buffered
    regions.

    Args:
        cleaned_mask:     Binary mask (0 = non-veg, 255 = vegetation).
        buffer_distance:  Expansion radius in pixels.
        min_area:         Contours smaller than this (px²) are discarded.

    Returns:
        filtered_contours: List of OpenCV contours that meet the area threshold.
        buffered_mask:     The buffered uint8 mask used to derive the contours.
        dist_transform:    The distance-transform array (float32).
    """
    inverted = cv2.bitwise_not(cleaned_mask)
    dist_transform = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)
    buffered_mask = np.where(dist_transform < buffer_distance, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(buffered_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    return filtered_contours, buffered_mask, dist_transform


def calculate_intersection_area(
    cnt_red: np.ndarray,
    green_contours: list,
    binary_mask: np.ndarray,
) -> float:
    """
    Compute the fractional overlap between a single 'red' contour and the
    union of all 'green' (vegetation) contours.

    Args:
        cnt_red:        One OpenCV contour (the candidate region).
        green_contours: List of OpenCV contours representing detected vegetation.
        binary_mask:    Reference array used only for shape / dtype.

    Returns:
        Overlap ratio in [0, 1].  Returns 0 if cnt_red has zero area.
    """
    mask_red   = np.zeros_like(binary_mask, dtype=np.uint8)
    mask_green = np.zeros_like(binary_mask, dtype=np.uint8)

    cv2.drawContours(mask_red,   [cnt_red],       -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask_green, green_contours,  -1, 255, thickness=cv2.FILLED)

    intersection = cv2.bitwise_and(mask_red, mask_green)
    red_area = np.count_nonzero(mask_red)
    return np.count_nonzero(intersection) / red_area if red_area > 0 else 0.0


def get_red_contour_mask(
    biased_y_pred: np.ndarray,
    binary_mask: np.ndarray,
    buffer_distance: float = 10,
    min_area: float = 700,
    overlap_threshold: float = 0.01,
) -> np.ndarray:
    """
    Build a refined vegetation mask from the thresholded detectree prediction.

    Applies a small morphological clean-up pass to remove single-pixel noise,
    then keeps only connected components (contours) larger than `min_area`.

    Args:
        biased_y_pred:    Thresholded detectree prediction (0 or 1 per pixel).
        binary_mask:      uint8 version of biased_y_pred (* 255).
        buffer_distance:  Unused – kept for API compatibility.
        min_area:         Minimum contour area (px²) to retain.
        overlap_threshold: Unused – kept for API compatibility.

    Returns:
        Binary uint8 mask (0 or 1) with accepted vegetation regions filled.
    """
    # --- Small morphological clean-up to remove salt-and-pepper noise ---
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)

    # --- Find vegetation contours and keep those above min_area ---
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_mask = np.zeros_like(binary_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return (final_mask > 0).astype(np.uint8)
