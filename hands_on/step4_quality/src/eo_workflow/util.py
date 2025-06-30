# ==============================================================================
# File: scl_histogram_utils.py
# Purpose: Utility functions for extracting and analyzing the Scene Classification
#          Layer (SCL) from Sentinel-2 STAC datasets using xarray.
#          Includes pixel-level histogram computation and scene access.
# Autor: Mike Sips
# ==============================================================================

import xarray as xr
import numpy as np
from typing import Tuple

def calculate_nvdi_histogram(
    ndvi: xr.DataArray,
    bins: list[float] = [-1.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a histogram of NDVI values using defined class bins.

    Parameters
    ----------
    ndvi : xr.DataArray
        NDVI values as a 2D or 3D xarray DataArray.

    bins : list of float, optional
        Bin edges to classify NDVI values (default: standard interpretation bins).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - bin_edges: The NDVI bin edges used
        - frequencies: The number of pixels in each NDVI bin
    """
    if not isinstance(ndvi, xr.DataArray):
        raise TypeError("ndvi must be an xarray.DataArray")

    # Flatten the NDVI values
    flat_values = ndvi.values.ravel()

    # Remove NaNs before histogram computation
    valid_values = flat_values[~np.isnan(flat_values)]

    # Compute histogram
    frequencies, bins = np.histogram(valid_values, bins=bins)

    # Convert to lists
    return bins.tolist(), frequencies.tolist()

# ------------------------------------------------------------------------------
# Function: calculate_scl_histogram
# Purpose : Compute pixel counts for each valid class in a 2D SCL scene
# ------------------------------------------------------------------------------
def calculate_scl_histogram(
    scl_scene: xr.DataArray,
    valid_classes: list[int] = list(range(12))
) -> Tuple[list[int], list[int]]:
    """
    Calculate a histogram of SCL class frequencies from a 2D scene.

    Parameters
    ----------
    scl_scene : xr.DataArray
        A 2D array (single time slice) from the Sentinel-2 Scene Classification Layer (SCL).
        Each pixel represents a classification code (e.g., cloud, vegetation, shadow).

    valid_classes : list of int, optional
        List of SCL classification codes to count.
        Default: list(range(12)) includes classes 0–11.

    Returns
    -------
    tuple of (list[int], list[int])
        - classes: Sorted list of SCL codes that were counted.
        - frequencies: Corresponding pixel counts per SCL class.

    Raises
    ------
    TypeError
        If input is not a 2D xarray.DataArray or valid_classes is not a list of ints.
    """
    if not isinstance(scl_scene, xr.DataArray):
        raise TypeError("scl_scene must be an xarray.DataArray.")
    if scl_scene.ndim != 2:
        raise ValueError("scl_scene must be a 2D array (single scene).")
    if not isinstance(valid_classes, list) or not all(isinstance(c, int) for c in valid_classes):
        raise TypeError("valid_classes must be a list of integers.")

    values = scl_scene.values.ravel()
    counts = {cls: np.count_nonzero(values == cls) for cls in valid_classes}
    classes = sorted(counts.keys())
    frequencies = [counts[cls] for cls in classes]
    return classes, frequencies

# ------------------------------------------------------------------------------
# Function: get_scene_by_scene_id
# Purpose : Retrieve a single scene from the SCL band using its time index
# ------------------------------------------------------------------------------
def get_scene_by_scene_id(
    data_set: xr.Dataset,
    scene_id: int,
    scl_band: str = "scl"
) -> xr.DataArray:
    """
    Extract a specific scene from the SCL band in an xarray Dataset.

    Parameters
    ----------
    data_set : xr.Dataset
        The input xarray Dataset containing a time series of Sentinel-2 data.

    scene_id : int
        Index of the time step (scene) to extract.

    scl_band : str, optional
        Name of the SCL band in the dataset. Default is "scl".

    Returns
    -------
    xr.DataArray
        A 2D DataArray representing the SCL values at the specified time step.

    Raises
    ------
    TypeError
        If data_set is not an xarray.Dataset or scene_id is not an integer.

    ValueError
        If scl_band is missing, or scene_id is out of bounds.
    """
    if not isinstance(data_set, xr.Dataset):
        raise TypeError("data_set must be an xarray.Dataset.")
    if not isinstance(scene_id, int):
        raise TypeError("scene_id must be an integer.")
    if scl_band not in data_set:
        raise ValueError(f"Dataset does not contain the SCL band: '{scl_band}'")

    scl = data_set[scl_band]

    if "time" not in scl.dims:
        raise ValueError("SCL band must have a 'time' dimension.")
    if scene_id < 0 or scene_id >= scl.sizes["time"]:
        raise ValueError(f"scene_id {scene_id} is out of bounds (0 to {scl.sizes['time'] - 1})")

    return scl.isel(time=scene_id)
