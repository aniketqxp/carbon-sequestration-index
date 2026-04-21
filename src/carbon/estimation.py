"""
carbon/estimation.py
--------------------
Core carbon sequestration estimation algorithms.

Pipeline:
  1. For each vegetated pixel, assign species probabilities from NDVI.
  2. Compute annual carbon sequestration (kgCO₂e) per pixel.
  3. Aggregate to region totals.
  4. Derive the Carbon Sequestration Index (CSI) — a logarithmic scalar
     analogous to the Richter scale.

All carbon values are expressed in kg CO₂ equivalent (kgCO₂e) by default.
"""

from __future__ import annotations

import numpy as np

from src.carbon.species import create_species_data, calculate_species_probabilities


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CO2_CONVERSION_FACTOR = 3.67   # kgCO₂ per kgC (molecular weight ratio)
URBAN_EFFICIENCY_FACTOR = 0.65  # Accounts for stress / maintenance losses in
                                 # suburban environments (conservative estimate)


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------

def estimate_carbon(
    vegetation_mask: np.ndarray,
    ndvi: np.ndarray,
    pixel_area_m2: float = 0.25,
    return_co2: bool = True,
) -> tuple[float, np.ndarray]:
    """
    Estimate annual carbon sequestration for each vegetated pixel.

    For each pixel *p* in the vegetation mask:
      - Species probabilities P(s|NDVI_p) are computed (Gaussian model).
      - Per-species annual sequestration is: biomass(s) × ndvi_factor × seq_rate(s).
      - Weighted average is scaled to pixel area and the urban efficiency factor.
      - Optionally converted from kgC to kgCO₂e.

    Args:
        vegetation_mask: 2-D binary array (0 = non-veg, 1 = vegetation).
        ndvi:            2-D float array of NDVI values (same shape).
        pixel_area_m2:   Physical area of one pixel in m².
        return_co2:      If True, output is in kgCO₂e; otherwise kgC.

    Returns:
        total_carbon: Scalar total sequestration across all vegetated pixels.
        carbon_map:   2-D float32 array of per-pixel sequestration values.
    """
    species_data = create_species_data()
    total_carbon = 0.0
    carbon_map = np.zeros_like(vegetation_mask, dtype=np.float32)

    veg_pixels = np.where(vegetation_mask > 0)

    for y, x in zip(*veg_pixels):
        pixel_ndvi = ndvi[y, x]
        if pixel_ndvi <= 0:
            continue

        probs = calculate_species_probabilities(pixel_ndvi, species_data)
        pixel_carbon = 0.0

        for species, prob in probs.items():
            data = species_data[species]
            max_ndvi = data["ndvi_range"][1]
            ndvi_factor = np.clip(pixel_ndvi / max_ndvi, 0.3, 1.0)
            adj_biomass = data["biomass"] * ndvi_factor
            annual_seq = adj_biomass * data["seq_rate"]
            pixel_carbon += prob * annual_seq

        pixel_carbon *= pixel_area_m2 * URBAN_EFFICIENCY_FACTOR
        if return_co2:
            pixel_carbon *= CO2_CONVERSION_FACTOR

        carbon_map[y, x] = pixel_carbon
        total_carbon += pixel_carbon

    return total_carbon, carbon_map


# ---------------------------------------------------------------------------
# Carbon Sequestration Index
# ---------------------------------------------------------------------------

def calculate_csi(total_carbon_kg: float, vegetated_area_m2: float) -> float:
    """
    Compute the Carbon Sequestration Index (CSI) on a logarithmic scale.

    CSI is inspired by the Richter magnitude scale:

        CSI = log10( carbon_t_ha_yr / 0.25 + 1 )

    where carbon_t_ha_yr is in metric tonnes of CO₂e per hectare per year.
    A reference of 0.25 t/ha/yr anchors the scale so that:

        CSI ≈ 1  →  Grasslands / sparse vegetation
        CSI ≈ 2  →  Shrublands / young plantations
        CSI ≈ 3  →  Established forests
        CSI ≈ 4  →  Mature native forests

    Args:
        total_carbon_kg:   Total annual sequestration in kg (CO₂e or C).
        vegetated_area_m2: Total vegetated area in m².

    Returns:
        CSI value (float ≥ 0, typically 0–4, unbounded above).
    """
    area_ha = vegetated_area_m2 / 10_000
    if area_ha == 0:
        return 0.0

    carbon_t_ha_yr = (total_carbon_kg / 1_000) / area_ha
    return float(np.log10(carbon_t_ha_yr / 0.25 + 1))


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

# CSI interpretation bands
CSI_INTERPRETATION = {
    "0.0–0.5":  "Degraded / Bare land",
    "0.5–1.5":  "Grasslands / Sparse vegetation",
    "1.5–2.5":  "Shrublands / Young plantations",
    "2.5–3.5":  "Established forests",
    "3.5–4.5":  "Mature native forests",
    ">4.5":     "Exceptional carbon sinks",
}


def format_results(
    total_carbon: float,
    vegetated_area_m2: float,
    csi: float,
    reference_area_ha: float = 100.0,
) -> dict[str, str]:
    """
    Format carbon metrics into a human-readable results dictionary.

    Args:
        total_carbon:       Total annual CO₂e sequestration (kg).
        vegetated_area_m2:  Vegetated area (m²).
        csi:                Pre-computed CSI value.
        reference_area_ha:  Total study region area for the regional rate
                            (default 100 ha = 1 km²).

    Returns:
        Ordered dict of {metric_label: formatted_string}.
    """
    area_ha = vegetated_area_m2 / 10_000
    carbon_t_ha_yr = (total_carbon / 1_000) / area_ha if area_ha > 0 else 0.0
    carbon_t_total_region = (total_carbon / 1_000) / reference_area_ha

    return {
        "Vegetated Area":                    f"{area_ha:.2f} hectares",
        "Total Annual Sequestration":        f"{total_carbon / 1_000:.1f} metric tons CO₂e",
        "Areal Rate (Vegetated)":            f"{carbon_t_ha_yr:.2f} tCO₂e/ha/yr",
        "Areal Rate (Total Region)":         f"{carbon_t_total_region:.2f} tCO₂e/ha/yr",
        "Carbon Sequestration Index (CSI)":  f"{csi:.2f}",
    }
