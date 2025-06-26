#!/bin/bash

# SPDX-FileCopyrightText: 2025 Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: EUPL-1.2

# ==============================================================================
# Script: setup_conda_env.sh
# Purpose: Create a conda environment for Sentinel-2 Earth Observation workflow
# Usage: ./setup_conda_env.sh environment.yml sentinel2_env
# Author: Mike Sips
# Date: 2025-06-16
# ==============================================================================

# ------------------------------------------------------------------------------
# Step 0: Parse input arguments
# ------------------------------------------------------------------------------
YML_FILE=$1
ENV_NAME=$2

if [ -z "$YML_FILE" ] || [ -z "$ENV_NAME" ]; then
  echo "[USAGE] $0 <path_to_environment.yml> <environment_name>"
  exit 1
fi

if [ ! -f "$YML_FILE" ]; then
  echo "[FAIL] File '$YML_FILE' not found."
  exit 1
fi

# ------------------------------------------------------------------------------
# Function: print_header
# Utility for formatted step headers
# ------------------------------------------------------------------------------
print_header() {
  echo
  echo "======================================================================"
  echo "$1"
  echo "======================================================================"
}

# ------------------------------------------------------------------------------
# Step 1: Check if conda is available
# ------------------------------------------------------------------------------
print_header "STEP 1: Checking Conda installation"
if ! command -v conda &> /dev/null; then
  echo "[FAIL] Conda is not installed or not in PATH."
  echo "       Please install Miniconda or Anaconda first."
  exit 1
fi
echo "[ OK ] Conda is available."

# ------------------------------------------------------------------------------
# Step 2: Create or update environment
# ------------------------------------------------------------------------------
print_header "STEP 2: Creating or updating conda environment: $ENV_NAME"

conda env create --name "$ENV_NAME" --file "$YML_FILE"
echo "[ OK ] Conda environment '$ENV_NAME' created from $YML_FILE"

# ------------------------------------------------------------------------------
# Step 3: Activation hint and success message
# ------------------------------------------------------------------------------
print_header "STEP 3: Activating environment and confirming success"

echo "[INFO] To activate this environment, run:"
echo "       conda activate $ENV_NAME"

# ------------------------------------------------------------------------------
# Done
# ------------------------------------------------------------------------------
print_header "SETUP COMPLETE"
echo "[SUCCESS] Your Conda environment '$ENV_NAME' is ready."

