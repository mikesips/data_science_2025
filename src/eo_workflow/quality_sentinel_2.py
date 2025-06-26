# ==============================================================================
# File: quality_assessment.py
# Purpose: Sentinel-2 Quality Assessment Using Scene Classification Layer (SCL)
# Author: Mike Sips
# ==============================================================================

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eo_workflow import util
from eo_workflow import visualization_sentinel_2

# ------------------------------------------------------------------------------
# Function: plot_scl_histogram
# Purpose : Plot a histogram of SCL class frequencies for a single scene
# ------------------------------------------------------------------------------
def plot_scl_histogram(
    scene: xr.DataArray,
    valid_classes: list[int] = list(range(12)),
    title: str = "SCL Class Distribution",
    aggregated: bool = True
) -> None:
    """
    Plot the pixel distribution of SCL classes for a Sentinel-2 scene.

    Parameters
    ----------
    scene : xr.DataArray
        A 2D image slice from the SCL band, typically selected via:
        `ds["scl"].isel(time=0)`.
    valid_classes : list of int, optional
        List of class codes to include in the plot. Defaults to [0–11].
    title : str, optional
        Title for the plot. Default: "SCL Class Distribution".
    aggregated : bool, optional
        If True, use only the date in the output filename. Default: True.

    Notes
    -----
    ESA Sentinel-2 SCL class codes:
        0  = No Data
        1  = Saturated/Defective
        2  = Dark Area Pixels
        3  = Cloud Shadow
        4  = Vegetation
        5  = Bare Soils
        6  = Water
        7  = Cloud Low Probability / Unclassified
        8  = Cloud Medium Probability
        9  = Cloud High Probability
        10 = Thin Cirrus
        11 = Snow or Ice

    Returns
    -------
    None
        Displays and saves a histogram plot as a PNG file.
    """
    classes, frequencies = calculate_scl_histogram(scene, valid_classes)

    plt.figure(figsize=(10, 6))
    plt.bar(classes, frequencies, tick_label=classes)
    plt.xlabel("SCL Class")
    plt.ylabel("Number of Pixels")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    timestamp = pd.to_datetime(scene.time.values)
    suffix = timestamp.strftime("%Y-%m-%d" if aggregated else "%Y-%m-%dT%H-%M-%S")
    outputfile = f"{suffix}.png"

    plt.savefig(outputfile)
    plt.close()

# ------------------------------------------------------------------------------
# Function: assess_sentinel2_quality
# Purpose : Evaluate pixel-level data quality using the Scene Classification Layer (SCL)
# ------------------------------------------------------------------------------
def assess_sentinel2_quality(
    data_set: xr.Dataset,
    valid_classes: list[int] = list(range(1, 12)),
    aggregated: bool = True,
    verbose: bool = True
) -> dict:
    """
    Assess Sentinel-2 data quality using the Scene Classification Layer (SCL).

    Parameters
    ----------
    data_set : xr.Dataset
        xarray.Dataset containing the 'scl' band with a 'time' dimension.
    valid_classes : list of int, optional
        SCL class codes considered valid (default: [1–11], excludes 'No Data').
    aggregated : bool, optional
        If True, uses date-only format for output timestamps. Default: True.
    verbose : bool, optional
        If True, print statistics for each scene. Default: True.

    Returns
    -------
    dict
        Dictionary mapping scene index to:
        - 'valid_ratio' (float): Proportion of valid pixels.
        - 'valid_pixels' (int): Count of valid pixels.
        - 'total_pixels' (int): Count of total pixels.
        - 'coverage' (float): Estimated coverage excluding cloud pixels.
    """
    print("[INFO] Generating Quality Report...")

    if not isinstance(data_set, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset.")

    if "scl" not in data_set:
        raise ValueError("Dataset does not contain a 'scl' band.")

    scl = data_set["scl"]
    if "time" not in scl.dims:
        raise ValueError("SCL input must have a 'time' dimension.")

    quality_report = {}

    for scene_id in range(scl.sizes["time"]):
        scl_scene = scl.isel(time=scene_id)

        classes, frequencies = util.calculate_scl_histogram(scl_scene)
        total_pixels = sum(frequencies)
        valid_pixels = sum(frequencies[1:])  # skip class 0 (No Data)
        valid_ratio = valid_pixels / total_pixels if total_pixels else 0

        cloud_pixels = sum(frequencies[7:11])
        coverage = 1 - (cloud_pixels / frequencies[4]) if frequencies[4] else 0
        coverage = coverage if (cloud_pixels < frequencies[4]) else 0

        timestamp = pd.to_datetime(scl_scene.time.values)
        timestamp_str = timestamp.strftime("%Y-%m-%d" if aggregated else "%Y-%m-%dT%H:%M:%S")

        if verbose:
            print(f"[INFO] {timestamp_str}: total_pixels={total_pixels}; valid_pixels={valid_pixels}; valid ratio={valid_ratio:.2%}; cloud_pixels={cloud_pixels}; coverage={coverage:.2%}")

        quality_report[scene_id] = {
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixels,
            "valid_ratio": valid_ratio,
            "coverage": coverage
        }

    print("[INFO] Generating Quality Report Successful.")

    return quality_report
