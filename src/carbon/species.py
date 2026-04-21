"""
carbon/species.py
-----------------
Synthetic vegetation species data model and NDVI-based species probability
assignment.

The species parameters (biomass, sequestration rate, NDVI range, etc.) are
literature-informed estimates for a temperate/suburban environment.  They are
intentionally conservative to avoid over-reporting carbon sequestration for
urban canopy analyses.

References:
  - Jenkins et al. (2003). National-scale biomass estimators for United States
    tree species. Forest Science, 49(1), 12-35.
  - Nowak & Crane (2002). Carbon storage and sequestration by urban trees in
    the USA. Environmental Pollution, 116(3), 381-389.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Species data
# ---------------------------------------------------------------------------

def create_species_data() -> dict:
    """
    Return a dictionary of synthetic species parameters.

    Each entry represents a vegetation type commonly found in suburban / peri-
    urban landscapes.  Parameters are used by the carbon estimation pipeline.

    Schema per species:
        type            (str)   : 'Tree', 'Shrub', or 'Grass'
        ndvi_range      (tuple) : (min_ndvi, max_ndvi) typical for the species
        biomass         (float) : Above-ground carbon biomass in kgC/m²
        seq_rate        (float) : Annual sequestration as a fraction of biomass
        height          (tuple) : (min_height_m, max_height_m)
        canopy_factor   (float) : Relative canopy density multiplier
        default_coverage(float) : Prior probability weight for NDVI assignment
    """
    return {
        "Oak": {
            "type": "Tree",
            "ndvi_range": (0.7, 0.9),
            "biomass": 18.0,           # kgC/m²
            "seq_rate": 0.025,         # 2.5 % of biomass / year
            "height": (15, 25),
            "canopy_factor": 1.8,
            "default_coverage": 0.30,
        },
        "Pine": {
            "type": "Tree",
            "ndvi_range": (0.6, 0.8),
            "biomass": 12.5,
            "seq_rate": 0.03,          # Faster growth rate
            "height": (10, 20),
            "canopy_factor": 1.5,
            "default_coverage": 0.25,
        },
        "Maple": {
            "type": "Tree",
            "ndvi_range": (0.5, 0.7),
            "biomass": 10.0,
            "seq_rate": 0.02,
            "height": (8, 15),
            "canopy_factor": 1.2,
            "default_coverage": 0.15,
        },
        "Rhododendron": {
            "type": "Shrub",
            "ndvi_range": (0.4, 0.6),
            "biomass": 3.0,
            "seq_rate": 0.015,
            "height": (2, 4),
            "canopy_factor": 0.7,
            "default_coverage": 0.15,
        },
        "Juniper": {
            "type": "Shrub",
            "ndvi_range": (0.3, 0.5),
            "biomass": 2.0,
            "seq_rate": 0.01,
            "height": (1, 3),
            "canopy_factor": 0.5,
            "default_coverage": 0.10,
        },
        "Fescue": {
            "type": "Grass",
            "ndvi_range": (0.1, 0.3),
            "biomass": 0.5,
            "seq_rate": 0.02,          # Fast turnover compensates low biomass
            "height": (0.1, 0.3),
            "canopy_factor": 0.2,
            "default_coverage": 0.05,
        },
    }


def get_species_data_table() -> pd.DataFrame:
    """
    Return species parameters as a tidy DataFrame suitable for display.

    Returns:
        DataFrame with columns: Species, Type, NDVI Range, Biomass (kg/m²),
        Seq Rate (kgC/m²/yr), Height (m), Coverage (%).
    """
    species_data = create_species_data()
    rows = []
    for species, data in species_data.items():
        rows.append(
            {
                "Species": species,
                "Type": data["type"],
                "NDVI Range": f"{data['ndvi_range'][0]:.1f}–{data['ndvi_range'][1]:.1f}",
                "Biomass (kg/m²)": data["biomass"],
                "Seq Rate (kgC/m²/yr)": data["seq_rate"],
                "Height (m)": f"{data['height'][0]}–{data['height'][1]}",
                "Coverage (%)": f"{data['default_coverage'] * 100:.0f}%",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Species probability assignment
# ---------------------------------------------------------------------------

def calculate_species_probabilities(ndvi: float, species_data: dict) -> dict[str, float]:
    """
    Assign per-species probabilities for a single pixel using a Gaussian
    likelihood weighted by each species' prior coverage.

    The probability of species *s* at NDVI value *v* is:

        P(s | v) ∝ coverage(s) · exp( -(v - μ_s)² / (2 σ²) )

    where μ_s is the midpoint of the species' NDVI range and σ = 0.15.

    Args:
        ndvi:         NDVI value for the pixel (float in [-1, 1]).
        species_data: Dict returned by :func:`create_species_data`.

    Returns:
        Normalised probability dict {species_name: probability}.
    """
    sigma = 0.15
    probs: dict[str, float] = {}
    for name, data in species_data.items():
        mu = np.mean(data["ndvi_range"])
        probs[name] = data["default_coverage"] * np.exp(
            -((ndvi - mu) ** 2) / (2 * sigma ** 2)
        )

    total = sum(probs.values())
    if total == 0:
        # Uniform fallback to avoid division by zero
        n = len(probs)
        return {k: 1.0 / n for k in probs}

    return {k: v / total for k, v in probs.items()}
