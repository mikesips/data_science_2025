#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# File: ca_wildfire.py
# Purpose: Entry point for executing the Sentinel-2 EO processing pipeline.
#          Instantiates and runs the EOWorkflow class using configuration files.
# Author: Mike Sips
# ==============================================================================

# ------------------------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------------------------
import sys
import os

# ------------------------------------------------------------------------------
# Add 'src' Directory to sys.path
# Allows importing the 'eo_workflow' package from src/eo_workflow/
# ------------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath("src"))

# ------------------------------------------------------------------------------
# Import Workflow Class
# ------------------------------------------------------------------------------
from eo_workflow import eo_workflow

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Execute the EO processing pipeline by creating and running an EOWorkflow object.
    
    The workflow uses a set of YAML configuration files located in the 
    './config/eo_workflow' directory. These include:
        - search_parameters.yml
        - load_parameters.yml
        - filter_parameters.yml

    The workflow performs:
        1. STAC search for Sentinel-2 scenes
        2. STAC data loading
        3. Spatial clipping
        4. Quality assessment and filtering
        5. Vegetation time series extraction
        6. Result visualization
    """
    workflow = eo_workflow.EOWorkflow("./config/eo_workflow")
    workflow.perform()

