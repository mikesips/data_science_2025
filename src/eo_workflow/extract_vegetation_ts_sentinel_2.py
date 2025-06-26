import numpy as np
import pandas as pd
import xarray as xr

from eo_workflow import util

# ------------------------------------------------------------------------------
# Function: vegetation_time_series
# Purpose: Count vegetation pixels (class 4 in SCL) in Sentinel-2 scene classification
#          data, compute surface area, and filter based on scene coverage quality.
# ------------------------------------------------------------------------------

def vegetation_time_series(
    data_set: xr.Dataset, 
    pixel_size: float = 10.0, 
    min_coverage: float = 0.6,
    aggregated: bool = True
) -> pd.DataFrame:
    """
    Count vegetation-classified pixels from the SCL (Scene Classification Layer)
    of Sentinel-2 data, compute surface area, and return a time series.

    Parameters:
        data (xr.Dataset): xarray dataset loaded from STAC with 'scl' band present.
        pixel_size (float): Spatial resolution in meters (default is 10m for SCL).
        min_coverage (float): Minimum valid coverage ratio to include a date.

    Returns:
        pd.DataFrame: DataFrame with vegetation area per date, coverage ratio, and timestamp.
    """
    # ------------------------------------------------------------------------------
    # Check type of data_set and Extract SCL band from data_set
    # ------------------------------------------------------------------------------
    if isinstance(data_set, xr.Dataset):
        if "scl" not in data_set:
            raise ValueError("Dataset does not contain a 'scl' band.")
        scl = data_set["scl"]
    else:
        raise TypeError("Input must be an xarray.Dataset or xarray.DataArray.")

    if "time" not in scl.dims:
        raise ValueError("SCL input must have a 'time' dimension.")
    
    # Output containers
    vegetation_counts = []
    vegetation_areas = []
    coverage_ratios = []
    date_labels = []

    # Total pixels and total area (in km²) for one time slice
    total_pixels = data_set.sizes["y"] * data_set.sizes["x"]
    total_area_km2 = (total_pixels * pixel_size**2) / 1e6  # convert m² to km²

    for scene_id in range(scl.sizes["time"]):
        scene = scl.isel(time=scene_id)

        classes, frequencies = util.calculate_scl_histogram(scene)

        # Extract timestamp label
        timestamp = pd.to_datetime(scene.time.values)
        timestamp_str = timestamp.strftime("%Y-%m-%d")

        # Count vegetation-classified pixels (code 4 = vegetation)
        vegetation_pixel_count = frequencies[4]
        vegetation_area_km2 = (vegetation_pixel_count * pixel_size**2) / 1e6

        print(f"[INFO] {timestamp_str}: vegetation pixel: {vegetation_pixel_count}; vegetation_area: {vegetation_area_km2}")

        # Append metrics to lists
        vegetation_counts.append(vegetation_pixel_count)
        vegetation_areas.append(vegetation_area_km2)
        date_labels.append(timestamp_str)

    # Construct output DataFrame
    time_series = {
        "Date": date_labels,
        "Vegetation Surface Area": vegetation_areas
    }

    return pd.DataFrame(time_series)
