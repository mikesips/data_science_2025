#!/bin/bash

# SPDX-FileCopyrightText: 2025 Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: EUPL-1.2

# ==============================================================================
# Script: setup_venv.sh
# Purpose: Set up a Python virtual environment and install dependencies from a
#          requirements.yml file using a helper Python script.
# Author: Mike Sips
# Date: 2025-06-16
# ==============================================================================

set -e  # Exit immediately if a command fails

# ------------------------------------------------------------------------------
# Step 0: Read input arguments (env name and YAML file)
# ------------------------------------------------------------------------------
ENV_NAME=${1:-venv}
YML_FILE=${2:-requirements.yml}

# ------------------------------------------------------------------------------
# Function: print_header
# Utility for printing formatted step headers
# ------------------------------------------------------------------------------
print_header() {
  echo
  echo "======================================================================"
  echo "$1"
  echo "======================================================================"
}

# ------------------------------------------------------------------------------
# Step 1: Check for ensurepip and create virtual environment
# ------------------------------------------------------------------------------
print_header "STEP 1: Creating virtual environment ($ENV_NAME)"

if ! python3 -m ensurepip --version &>/dev/null; then
  echo "[FAIL] Python's ensurepip module is not available."
  echo "       On Debian/Ubuntu, install it with:"
  echo "       sudo apt install python3-venv"
  exit 1
fi

if python3 -m venv "$ENV_NAME"; then
  echo "[ OK ] Virtual environment created at ./$ENV_NAME"
else
  echo "[FAIL] Failed to create virtual environment"
  exit 1
fi

# ------------------------------------------------------------------------------
# Step 2: Activate the virtual environment
# ------------------------------------------------------------------------------
print_header "STEP 2: Activating virtual environment"
source "$ENV_NAME/bin/activate"
echo "[ OK ] Virtual environment '$ENV_NAME' activated"

# ------------------------------------------------------------------------------
# Step 3: Upgrade pip and install PyYAML
# ------------------------------------------------------------------------------
print_header "STEP 3: Installing base tools"
pip install --upgrade pip
pip install pyyaml
echo "[ OK ] pip and pyyaml installed"

# ------------------------------------------------------------------------------
# Step 4: Install packages from YAML using helper script
# ------------------------------------------------------------------------------
print_header "STEP 4: Installing packages from $YML_FILE"

if [ ! -f "setup/python/install_from_yaml.py" ]; then
  echo "[FAIL] install_from_yaml.py not found in current directory."
  echo "       Please provide this script or refer to documentation."
  exit 1
fi

if [ ! -f "$YML_FILE" ]; then
  echo "[FAIL] Requirements file '$YML_FILE' not found."
  exit 1
fi

python setup/python/install_from_yaml.py "$YML_FILE"
echo "[ OK ] Dependencies installed from $YML_FILE"

# ------------------------------------------------------------------------------
# Completion Message
# ------------------------------------------------------------------------------
print_header "SETUP COMPLETE"
echo "[SUCCESS] Your virtual environment '$ENV_NAME' is ready."
echo "To activate it later, run: source $ENV_NAME/bin/activate"