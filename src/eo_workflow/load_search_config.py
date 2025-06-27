# ==============================================================================
# File: load_search_config.py
# Purpose: Load Sentinel-2 STAC search configuration from a YAML file.
#          This module provides a utility function to validate and return
#          parameters necessary for querying a STAC API, such as catalog URL,
#          spatial bounding box, date range, and cloud cover threshold.
# Author: Mike Sips
# ==============================================================================

# ------------------------------------------------------------------------------
# Function: load_search_config
# Purpose: Load STAC search parameters from a YAML configuration file.
# ------------------------------------------------------------------------------

import yaml

def load_config(config_path: str = "search_config.yml") -> dict:
    """
    Load Sentinel-2 STAC search configuration from a YAML file.

    Parameters:
        config_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary containing search parameters:
              - catalog_url (str)
              - bbox (list[float])
              - date_range (str)
              - cloud_cover_threshold (float)
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Basic validation
        if not isinstance(config.get("catalog_url"), str):
            raise ValueError("Missing or invalid 'catalog_url'")
        if not (isinstance(config.get("bbox"), list) and len(config["bbox"]) == 4):
            raise ValueError("Missing or invalid 'bbox'")
        if not isinstance(config.get("date_range"), str):
            raise ValueError("Missing or invalid 'date_range'")
        if not isinstance(config.get("cloud_cover_threshold"), (int, float)):
            raise ValueError("Missing or invalid 'cloud_cover_threshold'")

        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"YAML parsing error in {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {e}")
