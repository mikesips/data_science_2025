# ==============================================================================
# File: extract_vegetation_ts_sentinel_2.py
# Purpose: Extract vegetation surface area over time from Sentinel-2 SCL scenes
# Author: Mike Sips
# ==============================================================================

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import xarray as xr
import numpy as np
import pandas as pd

from eo_workflow import util

# ------------------------------------------------------------------------------
# Function: vegetation_time_series
# Purpose : Calculate vegetation surface area from SCL band in Sentinel-2 time series
# ------------------------------------------------------------------------------

def vegetation_time_series(
    data_set: xr.Dataset, 
    pixel_size: float = 10.0, 
    min_coverage: float = 0.6,
    aggregated: bool = True
) -> pd.DataFrame:
    """
    Generate a time series of vegetation surface area using Sentinel-2 SCL band.

    Parameters
    ----------
    data_set : xr.Dataset
        xarray Dataset containing a 'scl' band and a 'time' dimension.
        The 'scl' band must include scene classification codes (e.g., vegetation = 4).

    pixel_size : float, optional
        Spatial resolution of the data in meters. Default is 10.0 (for Sentinel-2 Level-2A SCL).

    min_coverage : float, optional
        Minimum valid pixel ratio to consider a scene for analysis. Currently unused,
        but reserved for future filtering logic.

    aggregated : bool, optional
        If True, timestamps will be aggregated to daily resolution (YYYY-MM-DD).
        If False, full timestamps are retained (YYYY-MM-DDTHH:MM:SS).

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame with columns:
        - 'Date': timestamp of the scene
        - 'Vegetation Surface Area': estimated vegetation area in km²

    Raises
    ------
    ValueError
        If the dataset does not contain the required 'scl' band or time dimension.

    Notes
    -----
    Vegetation is identified using SCL class code 4 (vegetation). Surface area is
    computed by multiplying the number of valid vegetation pixels with pixel area.

    Example
    -------
    >>> df = vegetation_time_series(dataset, pixel_size=10)
    >>> df.head()
           Date  Vegetation Surface Area
    0  2023-05-01                   12.34
    1  2023-05-11                   13.02
    """
    if not isinstance(data_set, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset.")
    if "scl" not in data_set:
        raise ValueError("Dataset does not contain a 'scl' band.")
    
    scl = data_set["scl"]
    if "time" not in scl.dims:
        raise ValueError("SCL band must have a 'time' dimension.")

    # Output containers
    vegetation_areas = []
    date_labels = []

    # Total pixels and total area (in km²) for a full scene
    total_pixels = data_set.sizes["y"] * data_set.sizes["x"]
    total_area_km2 = (total_pixels * pixel_size**2) / 1e6  # m² to km²

    for scene_id in range(scl.sizes["time"]):
        scene = scl.isel(time=scene_id)
        timestamp = pd.to_datetime(scene.time.values)
        timestamp_str = timestamp.strftime("%Y-%m-%d" if aggregated else "%Y-%m-%dT%H:%M:%S")

        # Compute SCL histogram
        classes, frequencies = util.calculate_scl_histogram(scene)

        # Extract vegetation pixel count (class 4)
        try:
            vegetation_pixel_count = frequencies[classes.index(4)]
        except ValueError:
            vegetation_pixel_count = 0

        vegetation_area_km2 = (vegetation_pixel_count * pixel_size**2) / 1e6

        print(f"[INFO] {timestamp_str}: vegetation pixel = {vegetation_pixel_count}; vegetation area = {vegetation_area_km2:.2f} km^2")

        # Store results
        vegetation_areas.append(vegetation_area_km2)
        date_labels.append(timestamp_str)

    # Construct output DataFrame
    time_series = {
        "Date": date_labels,
        "Vegetation Surface Area": vegetation_areas
    }

    return pd.DataFrame(time_series)
