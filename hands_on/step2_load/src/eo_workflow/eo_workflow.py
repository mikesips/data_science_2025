# ==============================================================================
# File: eo_workflow_pipeline.py
# Purpose: EOWorkflow class for orchestrating Sentinel-2 STAC data processing
#          using STAC search, xarray loading, clipping, quality assessment,
#          filtering, time series extraction, and visualization
# Author: Mike Sips
# ==============================================================================

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from pystac_client import Client
from odc.stac import load
import pystac
import xarray as xr
import pandas as pd
import numpy as np
import sys
import os

from eo_workflow import (
    search_sentinel_2,
    load_sentinel_2
)

# ------------------------------------------------------------------------------
# Class: EOWorkflow
# Purpose: Encapsulate the end-to-end Earth Observation processing pipeline
# ------------------------------------------------------------------------------
class EOWorkflow:
    def __init__(self, config_dir: str):
        """
        Initialize the EO workflow with a directory containing YAML configuration files.

        Parameters
        ----------
        config_dir : str
            Path to the directory containing search, load, and filter parameter files.
        """
        self.config_dir = os.path.abspath(os.path.expanduser(config_dir))
        self.search_config = None
        self.load_config = None
        self.filter_config = None

    def perform(self):
        """
        Execute the complete Sentinel-2 Earth Observation processing workflow:
        1. Load search configuration
        """
        # Step 1: Load search parameters
        self.search_config = search_sentinel_2.load_stac_search_parameters(
            os.path.join(self.config_dir, "search_parameters.yml")
        )
        search_sentinel_2.print_stac_search_parameters(self.search_config)

        # Step 2: Search STAC items
        items = search_sentinel_2.search_sentinel2(
            catalog_url=self.search_config["catalog_url"],
            bbox=self.search_config["bbox"],
            date_range=self.search_config["date_range"],
            cloud_cover_threshold=self.search_config["cloud_cover_threshold"]
        )

        search_sentinel_2.print_stac_items(items)

        # Step 3: Load STAC items
        self.load_config = load_sentinel_2.load_stac_load_parameters(
            os.path.join(self.config_dir, "load_parameters.yml")
        )
        load_sentinel_2.print_stac_load_parameters(self.load_config)

        data_set = load_sentinel_2.load_sentinel2_xarray(
            items=items,
            band_keys=self.load_config["bands"],
            resolution=self.load_config["resolution"],
            aggregation=self.load_config["aggregation"]
        )

