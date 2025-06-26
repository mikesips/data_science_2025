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
    util,
    search_sentinel_2,
    load_sentinel_2,
    clip_sentinel_2,
    quality_sentinel_2,
    filter_sentinel_2,
    extract_vegetation_ts_sentinel_2,
    visualization_sentinel_2
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
        2. Perform STAC search
        3. Load Sentinel-2 bands into xarray.Dataset
        4. Clip data to bounding box
        5. Assess scene quality
        6. Filter scenes by quality metrics
        7. Extract vegetation time series
        8. Visualize results
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

        # Step 4: Clip to bounding box
        clipped_data_set = clip_sentinel_2.clip_dataset_to_bbox(
            dataset=data_set,
            bbox=self.search_config["bbox"]
        )

        # Step 5: Assess quality
        quality_report = quality_sentinel_2.assess_sentinel2_quality(clipped_data_set)

        # Step 6: Filter by quality
        self.filter_config = filter_sentinel_2.load_filter_parameters(
            os.path.join(self.config_dir, "filter_parameters.yml")
        )

        filtered_data_set = filter_sentinel_2.filter_scenes_by_validity_ratio(
            data_set=clipped_data_set,
            quality_report=quality_report,
            validity_threshold=self.filter_config["validity_threshold"],
            coverage_threshold=self.filter_config["coverage_threshold"],
            aggregation=self.load_config["aggregation"]
        )

        # Step 7: Extract vegetation time series
        vegetation_time_series = extract_vegetation_ts_sentinel_2.vegetation_time_series(
            data_set=filtered_data_set,
            pixel_size=self.load_config['resolution']
        )

        # Step 8: Visualize results
        visualization_sentinel_2.plot_vegetation_time_series(
            table=vegetation_time_series
        )
