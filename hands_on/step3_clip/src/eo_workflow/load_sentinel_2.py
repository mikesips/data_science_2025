# ==============================================================================
# File: stac_loader.py
# Purpose: Load Sentinel-2 STAC items into xarray datasets using the odc-stac library.
#          This module provides utilities to configure, filter, and load STAC metadata
#          and associated image bands (e.g., reflectance, SCL) into analysis-ready formats.
# Author: Mike Sips
# ==============================================================================

import os
import yaml
import xarray as xr
import pystac
from odc.stac import load

# Required configuration keys for loading
REQUIRED_KEYS = ["bands", "resolution", "aggregation", "chunks"]

# ------------------------------------------------------------------------------
# Function: load_stac_load_parameters
# Purpose : Load configuration parameters for odc.stac.load from a YAML file
# ------------------------------------------------------------------------------
def load_stac_load_parameters(config_filename: str = "search_parameters.yml") -> dict:
    """
    Load Sentinel-2 loading configuration for `odc.stac.load` from a YAML file.

    Parameters
    ----------
    config_filename : str
        Path to the YAML file.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'bands' (list of str)
        - 'resolution' (int)
        - 'aggregation' (bool)
        - 'chunks' (dict)

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If required keys are missing or have invalid types.
    yaml.YAMLError
        If the YAML syntax is invalid.
    """
    if not isinstance(config_filename, str):
        raise TypeError("config_filename must be a string")

    config_path = os.path.abspath(os.path.expanduser(config_filename))
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("YAML content must be a dictionary")

    for key in REQUIRED_KEYS:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")

    if not isinstance(config["bands"], list) or not all(isinstance(b, str) for b in config["bands"]):
        raise ValueError("bands must be a list of strings")
    if not isinstance(config["resolution"], int):
        raise ValueError("resolution must be an integer")
    if not isinstance(config["aggregation"], bool):
        raise ValueError("aggregation must be a boolean")
    if not isinstance(config["chunks"], dict):
        raise ValueError("chunks must be a dictionary")

    return config

# ------------------------------------------------------------------------------
# Function: print_stac_load_parameters
# Purpose : Print configuration used to load Sentinel-2 STAC items
# ------------------------------------------------------------------------------
def print_stac_load_parameters(loader_config: dict) -> None:
    """
    Print the configuration parameters used for loading Sentinel-2 data.

    Parameters
    ----------
    loader_config : dict
        Dictionary with keys:
        - 'bands': list of str
        - 'resolution': int
        - 'aggregation': bool
        - 'chunks': dict

    Raises
    ------
    TypeError
        If loader_config is not a dictionary.
    """
    if not isinstance(loader_config, dict):
        raise TypeError("loader_config must be a dictionary")

    print("\n[INFO] Load Configuration Parameters")
    print("=======================================")
    print(f"Bands to load:         {loader_config.get('bands', [])}")
    print(f"Spatial resolution:    {loader_config.get('resolution')} m")
    print(f"Aggregation enabled:   {loader_config.get('aggregation')}")

    chunks = loader_config.get("chunks", {})
    print(f"Dask chunks:           {chunks if chunks else '(default / automatic)'}\n")

# ------------------------------------------------------------------------------
# Function: load_sentinel2_xarray
# Purpose : Load Sentinel-2 STAC items into an xarray.Dataset using odc.stac
# ------------------------------------------------------------------------------
def load_sentinel2_xarray(
    items: list,
    band_keys: list[str] = ["red", "nir", "scl"],
    resolution: int = 60,
    aggregation: bool = True
) -> xr.Dataset:
    """
    Load Sentinel-2 STAC items into an xarray.Dataset using `odc.stac.load`.

    Parameters
    ----------
    items : list of pystac.Item
        List of STAC items to load.
    band_keys : list of str
        List of band identifiers (e.g., ["red", "nir", "scl"]).
    resolution : int
        Spatial resolution in meters.
    aggregation : bool
        If True, group by solar day.

    Returns
    -------
    xarray.Dataset
        Loaded dataset.

    Raises
    ------
    ValueError
        If inputs are empty or improperly typed.
    """
    if not isinstance(items, list) or not all(isinstance(i, pystac.Item) for i in items):
        raise TypeError("items must be a list of pystac.Item objects")
    if not items:
        raise ValueError("No STAC items provided for loading")
    if not isinstance(band_keys, list) or not all(isinstance(b, str) for b in band_keys):
        raise TypeError("band_keys must be a list of strings")
    if not isinstance(resolution, int):
        raise TypeError("resolution must be an integer")
    if not isinstance(aggregation, bool):
        raise TypeError("aggregation must be a boolean")

    print("[INFO] Loading data into xarray.Dataset...")
    dataset = load(
        items,
        bands=band_keys,
        resolution=resolution,
        groupby="solar_day" if aggregation else None,
        chunks={}  # Enable Dask lazy loading
    )
    print("[INFO] Loading Successful.")

    return dataset
