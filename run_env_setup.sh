#!/bin/bash

# SPDX-FileCopyrightText: 2025 Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: EUPL-1.2

# ==============================================================================
# Script: run_venv_setup.sh
# Purpose: Call the Python virtual environment setup script with parameters
# Author: Mike Sips
# Date: 2025-06-16
# ==============================================================================

# ------------------------------------------------------------------------------
# Default parameters (can be overridden via command line)
# ------------------------------------------------------------------------------
ENV_NAME=${1:-venv}
REQUIREMENTS_YML=${2:-requirements.yml}

# ------------------------------------------------------------------------------
# Call the setup script located in setup/python
# ------------------------------------------------------------------------------
bash setup/python/setup_venv.sh "$ENV_NAME" "$REQUIREMENTS_YML"