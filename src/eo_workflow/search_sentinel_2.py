# ==============================================================================
# File: search_sentinel_2.py
# Purpose: Perform Sentinel-2 STAC search queries, validate configuration files,
#          and display metadata summaries including item properties and asset bands.
# Author: Mike Sips
# ==============================================================================

from pystac_client import Client
import pystac
import os
import yaml

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
REQUIRED_KEYS = ["catalog_url", "bbox", "date_range", "cloud_cover_threshold"]

# ------------------------------------------------------------------------------
# Function: load_stac_search_parameters
# Purpose : Load STAC search parameters from a YAML config file and validate them
# ------------------------------------------------------------------------------
def load_stac_search_parameters(config_filename: str = "search_parameters.yml") -> dict:
    """
    Load Sentinel-2 STAC search parameters from a YAML configuration file.

    Parameters
    ----------
    config_filename : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Dictionary with keys: catalog_url, bbox, date_range, cloud_cover_threshold.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If required keys are missing or misformatted.
    yaml.YAMLError
        If the YAML content is invalid.
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
            raise ValueError(f"Missing required configuration key: '{key}'")

    if not isinstance(config["catalog_url"], str):
        raise ValueError("catalog_url must be a string")
    if not (isinstance(config["bbox"], list) and len(config["bbox"]) == 4 and all(isinstance(x, (int, float)) for x in config["bbox"])):
        raise ValueError("bbox must be a list of four numbers [min_lon, min_lat, max_lon, max_lat]")
    if not isinstance(config["date_range"], str):
        raise ValueError("date_range must be a string in ISO 8601 format")
    if not isinstance(config["cloud_cover_threshold"], (int, float)):
        raise ValueError("cloud_cover_threshold must be a number")

    return config

# ------------------------------------------------------------------------------
# Function: print_stac_search_parameters
# Purpose : Display the loaded STAC search parameters in readable format
# ------------------------------------------------------------------------------
def print_stac_search_parameters(search_config: dict) -> None:
    """
    Print the STAC search parameters in a readable format.

    Parameters
    ----------
    search_config : dict
        Dictionary of loaded STAC search parameters.

    Raises
    ------
    TypeError
        If the input is not a dictionary.
    """
    if not isinstance(search_config, dict):
        raise TypeError("search_config must be a dictionary")

    print("\n[INFO] STAC Search Configuration")
    print("=================================")
    for key in REQUIRED_KEYS:
        print(f"{key.replace('_', ' ').capitalize():<23}: {search_config.get(key, '[Missing]')}")
    print()

# ------------------------------------------------------------------------------
# Function: print_stac_items
# Purpose : Print summary metadata and available bands for a list of STAC items
# ------------------------------------------------------------------------------
def print_stac_items(items: list, cloud_cover_threshold: float = 1.0) -> None:
    """
    Print summary metadata for a list of STAC Items.

    Parameters
    ----------
    items : list of pystac.Item
        List of STAC items to be displayed.
    cloud_cover_threshold : float
        Cloud cover threshold used in the query.

    Raises
    ------
    TypeError
        If the input is not a list of pystac.Item.
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if not all(isinstance(item, pystac.Item) for item in items):
        raise TypeError("items must contain only pystac.Item objects")

    print(f"[INFO] Found {len(items)} items with cloud cover < {cloud_cover_threshold}%\n")
    for item in items:
        print(f"ID:        {item.id}")
        print(f"Datetime:  {item.datetime}")
        print(f"BBox:      {item.bbox}")
        print(f"Assets:    {list(item.assets.keys())}")
        print("-" * 60)

# ------------------------------------------------------------------------------
# Function: search_sentinel2
# Purpose : Perform a STAC query for Sentinel-2 imagery matching specified filters
# ------------------------------------------------------------------------------
def search_sentinel2(
    catalog_url: str,
    bbox: list,
    date_range: str,
    cloud_cover_threshold: float = 1.0
) -> list:
    """
    Query a STAC endpoint for Sentinel-2 Level-2A imagery with low cloud cover.

    Parameters
    ----------
    catalog_url : str
        URL of the STAC API catalog (e.g., EarthSearch v1).
    bbox : list of float
        Bounding box [min_lon, min_lat, max_lon, max_lat].
    date_range : str
        ISO 8601 date interval (e.g., "2020-06-01/2020-12-30").
    cloud_cover_threshold : float
        Maximum cloud cover percentage (default: 1.0).

    Returns
    -------
    list of pystac.Item
        List of matching STAC items.

    Raises
    ------
    TypeError, ValueError
        For invalid input types or values.
    """
    if not isinstance(catalog_url, str):
        raise TypeError("catalog_url must be a string")
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
        raise ValueError("bbox must be a list of four floats or ints")
    if not isinstance(date_range, str):
        raise TypeError("date_range must be a string in ISO 8601 format")
    if not isinstance(cloud_cover_threshold, (int, float)):
        raise TypeError("cloud_cover_threshold must be a number")
    if not (0 <= cloud_cover_threshold <= 100):
        raise ValueError("cloud_cover_threshold must be between 0 and 100")

    print("[INFO] Connecting to STAC catalog...")
    catalog = Client.open(catalog_url)

    print("[INFO] Performing search query...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_threshold}}
    )

    print("[INFO] Search Completed.")
    print(f"[INFO] Number of matched items: {search.matched()}")

    return list(search.items())