# ==============================================================================
# File: clip_dataset.py
# Purpose: Clip Sentinel-2 datasets to a geographic bounding box
#          using rioxarray and GeoPandas geometry operations.
# Author: Mike Sips
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
    print("[INFO] Starting dataset clipping...")

    # ------------------------------------------------------------------------------
    # Step 1: Identify a representative rioxarray-enabled variable to obtain CRS
    # ------------------------------------------------------------------------------
    example_var = next((v for v in dataset.data_vars if hasattr(dataset[v], "rio")), None)
    if example_var is None:
        raise ValueError("No rioxarray-enabled variable found in the dataset.")
    print(f"[INFO] Identified rioxarray-enabled variable: '{example_var}'")

    # ------------------------------------------------------------------------------
    # Step 2: Construct the clipping geometry from the input bounding box
    # ------------------------------------------------------------------------------
    geom = box(*bbox)
    print(f"[INFO] Created clipping geometry from bounding box: {bbox}")

    gdf = gpd.GeoDataFrame(geometry=[geom], crs=crs).to_crs(dataset.rio.crs)
    print(f"[INFO] Reprojected bounding box to match dataset CRS: {dataset.rio.crs}")

    # ------------------------------------------------------------------------------
    # Step 3: Iterate through time slices (if applicable) and clip each one
    # ------------------------------------------------------------------------------
    has_time = "time" in dataset.dims
    time_values = dataset.time.values if has_time else [None]
    clipped_list = []

    for t in time_values:
        if t is not None:
            ds_t = dataset.sel(time=t)
        else:
            ds_t = dataset

        clipped_t = ds_t.rio.clip(gdf.geometry.values, gdf.crs, drop=True)

        if t is not None:
            clipped_t = clipped_t.expand_dims(time=[t])

        clipped_list.append(clipped_t)

    # ------------------------------------------------------------------------------
    # Step 4: Return the clipped dataset (concatenate over time if needed)
    # ------------------------------------------------------------------------------
    clipped_ds = xr.concat(clipped_list, dim="time") if has_time else clipped_list[0]
    print("[INFO] Clipping completed.")
    return clipped_ds

