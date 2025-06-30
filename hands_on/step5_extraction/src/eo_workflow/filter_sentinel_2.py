# ==============================================================================
# File: filter_scenes.py
# Purpose: Filter Sentinel-2 scenes in an xarray.Dataset based on pixel validity
#          and spatial coverage quality metrics derived from the Scene Classification Layer (SCL).
# Author: Mike Sips
# ==============================================================================

# ------------------------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------------------------
import xarray as xr
import pandas as pd
import numpy as np
import os
import yaml

from eo_workflow import util

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
REQUIRED_KEYS = ["validity_threshold", "coverage_threshold"]

# ------------------------------------------------------------------------------
# Function: load_filter_parameters
# Purpose : Load SCL-based filtering thresholds from a YAML configuration file
# ------------------------------------------------------------------------------
def load_filter_parameters(yaml_path: str = "filter_parameters.yml") -> dict:
    """
    Load quality filter thresholds for Sentinel-2 SCL evaluation.

    Parameters
    ----------
    yaml_path : str, optional
        Path to the YAML configuration file (default: 'filter_parameters.yml').

    Returns
    -------
    dict
        Dictionary containing required filtering thresholds:
        - 'validity_threshold' (float): minimum acceptable valid pixel ratio.
        - 'coverage_threshold' (float): minimum required spatial coverage ratio.

    Raises
    ------
    ValueError
        If any required threshold keys are missing from the configuration file.
    """
    yaml_path = os.path.abspath(os.path.expanduser(yaml_path))

    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)

    for key in REQUIRED_KEYS:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in {yaml_path}")

    return config

# ------------------------------------------------------------------------------
# Function: filter_scenes_by_validity_ratio
# Purpose : Filter xarray time steps based on SCL quality metrics
# ------------------------------------------------------------------------------
def filter_scenes_by_validity_ratio(
    data_set: xr.Dataset,
    quality_report: dict,
    validity_threshold: float = 0.6,
    coverage_threshold: float = 0.8,
    aggregation: bool = True,
    verbose: bool = True
) -> xr.Dataset:
    """
    Filter Sentinel-2 scenes based on SCL-derived quality metrics.

    Removes scenes from an xarray.Dataset where the ratio of valid pixels
    or spatial coverage falls below defined thresholds.

    Parameters
    ----------
    data_set : xr.Dataset
        xarray.Dataset containing a 'scl' band and a 'time' dimension.

    quality_report : dict
        Mapping from scene index to a dictionary of quality metrics:
        - 'valid_ratio' (float): Proportion of valid SCL pixels.
        - 'coverage' (float): Proportion of spatial coverage retained.

    validity_threshold : float, optional
        Minimum ratio of valid pixels to retain a scene (default: 0.6).

    coverage_threshold : float, optional
        Minimum ratio of scene coverage to retain a scene (default: 0.8).

    aggregation : bool, optional
        If True, print scene timestamps in daily format (default: True).

    verbose : bool, optional
        If True, log filtering outcome for each scene (default: True).

    Returns
    -------
    xr.Dataset
        Filtered dataset including only time steps that meet both quality thresholds.

    Raises
    ------
    TypeError
        If the input is not an xarray.Dataset.

    ValueError
        If required variables ('scl', 'time') are missing or no valid scenes remain.
    """
    print("[INFO] Filter Scenes...")

    # Validate input
    if not isinstance(data_set, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset.")
    if "scl" not in data_set:
        raise ValueError("Dataset does not contain a 'scl' band.")

    scl = data_set["scl"]
    if "time" not in scl.dims:
        raise ValueError("SCL data must have a 'time' dimension.")

    # Iterate through scenes and evaluate quality metrics
    keep_indices = []

    for scene_id, quality_stats in quality_report.items():
        valid = quality_stats.get("valid_ratio", 0.0) >= validity_threshold
        coverage = quality_stats.get("coverage", 0.0) >= coverage_threshold

        scene = util.get_scene_by_scene_id(data_set, scene_id)
        timestamp = pd.to_datetime(scene.time.values)
        timestamp_str = timestamp.strftime("%Y-%m-%d" if aggregation else "%Y-%m-%dT%H:%M:%S")

        if valid and coverage:
            keep_indices.append(scene_id)
            if verbose:
                print(f"[INFO] {timestamp_str}: valid = {quality_stats['valid_ratio']:.2%} — scene kept")
        elif verbose:
            print(f"[INFO] {timestamp_str}: valid = {quality_stats['valid_ratio']:.2%} — scene removed")

    if not keep_indices:
        raise ValueError("No valid scenes found above the specified thresholds.")

    print("[INFO] Filter Scenes Successful.")
    return data_set.isel(time=keep_indices)
