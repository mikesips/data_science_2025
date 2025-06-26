# ==============================================================================
# File: extract_ndvi_ts_sentinel_2.py
# Purpose: Compute NDVI-based vegetation surface area time series from Sentinel-2 data
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
# Function: calculate_ndvi
# Purpose : Compute the NDVI (Normalized Difference Vegetation Index) from NIR and Red bands
# ------------------------------------------------------------------------------

def calculate_ndvi(
    nir: xr.DataArray,
    red: xr.DataArray 
) -> xr.DataArray:
    """
    Calculate the Normalized Difference Vegetation Index (NDVI) from Sentinel-2 data.

    Parameters
    ----------
    nir : xr.DataArray
        Near-infrared reflectance band as an xarray DataArray.

    red : xr.DataArray
        Red reflectance band as an xarray DataArray.

    Returns
    -------
    xr.DataArray
        NDVI values as an xarray DataArray with the same shape and coordinates
        as the input bands. Values range from -1.0 to 1.0.

    Raises
    ------
    ValueError
        If the input arrays have incompatible dimensions.

    Notes
    -----
    NDVI is computed as (NIR - Red) / (NIR + Red). Where the denominator is zero,
    NDVI is set to NaN to avoid division errors.
    """
    # Ensure matching dimensions
    if nir.shape != red.shape:
        raise ValueError("NIR and Red bands must have the same shape.")

    # Calculate denominator
    denominator = nir + red

    # Mask invalid pixels (e.g., where denominator is 0)
    valid_mask = xr.apply_ufunc(
        lambda x: ~np.isclose(x, 0.0),
        denominator,
        dask="parallelized",
        output_dtypes=[bool]
    )

    # NDVI formula, applied only to valid pixels
    ndvi = xr.where(
        valid_mask,
        (nir - red) / denominator,
        np.nan
    )

    # Clip NDVI values to valid range
    ndvi = ndvi.clip(min=-1.0, max=1.0)

    # Annotate
    ndvi.name = "NDVI"
    ndvi.attrs["long_name"] = "Normalized Difference Vegetation Index"

    return ndvi


# ------------------------------------------------------------------------------
# Function: nvdi_time_series
# Purpose : Compute a vegetation time series using NDVI values from Sentinel-2 bands
# ------------------------------------------------------------------------------

def nvdi_time_series(
    data_set: xr.Dataset, 
    pixel_size: float = 10.0, 
    aggregated: bool = True
) -> pd.DataFrame:
    """
    Compute a vegetation surface area time series from Sentinel-2 NDVI data.

    Parameters
    ----------
    data_set : xr.Dataset
        xarray dataset loaded from STAC with 'nir', 'red', and 'scl' bands.

    pixel_size : float, optional
        Spatial resolution in meters (default is 10.0 for 10m bands).

    min_coverage : float, optional
        Minimum valid pixel coverage required to include the scene (default: 0.6).

    aggregated : bool, optional
        If True, NDVI histograms are aggregated into vegetation bins (default: True).

    Returns
    -------
    pd.DataFrame
        Time series DataFrame containing vegetation area estimates per date.
    """
    if not isinstance(data_set, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset.")

    if not {"nir", "red", "scl"}.issubset(data_set.data_vars):
        raise ValueError("Dataset must include 'nir', 'red', and 'scl' bands.")

    scl = data_set["scl"]
    nir = data_set["nir"]
    red = data_set["red"]

    if "time" not in scl.dims:
        raise ValueError("Dataset must include a 'time' dimension.")

    vegetation_areas = []
    date_labels = []

    total_pixels = data_set.sizes["y"] * data_set.sizes["x"]
    total_area_km2 = (total_pixels * pixel_size**2) / 1e6

    for scene_id in range(nir.sizes["time"]):
        nir_scene = nir.isel(time=scene_id)
        red_scene = red.isel(time=scene_id)
        scl_scene = scl.isel(time=scene_id)

        # Calculate NDVI and remove NaNs
        ndvi_scene = calculate_ndvi(nir_scene, red_scene)
        mask = ~np.isnan(ndvi_scene).compute()
        ndvi_valid = ndvi_scene.where(mask, drop=True)

        # Compute histogram
        bins, frequencies = util.calculate_nvdi_histogram(ndvi_valid)

        # Extract timestamp
        timestamp = pd.to_datetime(nir_scene.time.values)
        timestamp_str = timestamp.strftime("%Y-%m-%d")

        # Count vegetation-classified pixels (e.g., NDVI >= 0.3 assumed vegetation)
        vegetation_pixel_count = sum(frequencies[3:])
        vegetation_area_km2 = (vegetation_pixel_count * pixel_size**2) / 1e6

        print(f"[INFO] {timestamp_str}: vegetation pixel = {vegetation_pixel_count}; vegetation area = {vegetation_area_km2:.2f} km^2")

        vegetation_areas.append(vegetation_area_km2)
        date_labels.append(timestamp_str)

    return pd.DataFrame({
        "Date": date_labels,
        "Vegetation Surface Area": vegetation_areas
    })