# ==============================================================================
# File: clip_dataset.py
# Purpose: Clip Sentinel-2 time-series datasets to a geographic bounding box
#          using rioxarray and GeoPandas geometry operations.
# Author: [Your Name]
# Date: [YYYY-MM-DD]
# ==============================================================================

# ------------------------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------------------------
import xarray as xr
import geopandas as gpd
from shapely import box
import rioxarray

# ------------------------------------------------------------------------------
# Function: clip_dataset_to_bbox
# Purpose : Clip a multi-band time-series xarray.Dataset to a bounding box
# ------------------------------------------------------------------------------

def clip_dataset_to_bbox(
    dataset: xr.Dataset,
    bbox: list[float],
    crs: str = "EPSG:4326"
) -> xr.Dataset:
    """
    Clip all spatial bands in a Sentinel-2 dataset to a bounding box for each time step.

    This function uses rioxarray and geopandas to apply a spatial mask to the dataset,
    returning a clipped dataset that retains all bands and time information.

    Parameters
    ----------
    dataset : xr.Dataset
        A multi-band dataset (e.g., from Sentinel-2) containing 'y' and 'x' spatial dimensions,
        as well as an optional 'time' dimension.

    bbox : list[float]
        Bounding box in the format [minx, miny, maxx, maxy], specified in the CRS defined by `crs`.

    crs : str, optional
        Coordinate reference system (CRS) of the bounding box. Default is "EPSG:4326" (WGS 84).

    Returns
    -------
    xr.Dataset
        A spatially clipped version of the input dataset, retaining the original data variables,
        metadata, and time dimension (if present).

    Raises
    ------
    ValueError
        If the dataset is None or does not contain rioxarray-enabled data variables.
    """
    if dataset is None:
        raise ValueError("Input dataset is None. Please ensure the dataset is loaded correctly.")

    # ------------------------------------------------------------------------------
    # Step 1: Identify a representative rioxarray-enabled variable to obtain CRS
    # ------------------------------------------------------------------------------
    example_var = next((v for v in dataset.data_vars if hasattr(dataset[v], "rio")), None)
    if example_var is None:
        raise ValueError("No rioxarray-enabled variable found in the dataset.")

    # ------------------------------------------------------------------------------
    # Step 2: Construct the clipping geometry from the input bounding box
    # ------------------------------------------------------------------------------
    geom = box(*bbox)
    gdf = gpd.GeoDataFrame(geometry=[geom], crs=crs).to_crs(dataset.rio.crs)

    # ------------------------------------------------------------------------------
    # Step 3: Iterate through time slices (if applicable) and clip each one
    # ------------------------------------------------------------------------------
    has_time = "time" in dataset.dims
    time_values = dataset.time.values if has_time else [None]
    clipped_list = []

    for t in time_values:
        ds_t = dataset.sel(time=t) if t is not None else dataset
        clipped_t = ds_t.rio.clip(gdf.geometry.values, gdf.crs, drop=True)

        # Add back time coordinate if time dimension exists
        if t is not None:
            clipped_t = clipped_t.expand_dims(time=[t])

        clipped_list.append(clipped_t)

    # ------------------------------------------------------------------------------
    # Step 4: Return the clipped dataset (concatenate over time if needed)
    # ------------------------------------------------------------------------------
    clipped_ds = xr.concat(clipped_list, dim="time") if has_time else clipped_list[0]
    return clipped_ds
