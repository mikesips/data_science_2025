#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Script: install_from_yaml.py
# Purpose: Load a YAML-formatted requirements file and install each package
#          using pip. Each package is installed with a minimum version constraint.
# Author: Mike Sips
# Date: 2025-06-12
# ==============================================================================

import subprocess
import sys
import yaml

# ------------------------------------------------------------------------------
# Step 1: Load Dependencies from YAML File
# ------------------------------------------------------------------------------

REQUIREMENTS_FILE = sys.argv[1]

try:
    with open(REQUIREMENTS_FILE, "r") as file:
        requirements = yaml.safe_load(file)
        if not isinstance(requirements, dict):
            raise ValueError("YAML file must contain a dictionary of package: version entries.")
except FileNotFoundError:
    print(f"[FAIL] Requirements file '{REQUIREMENTS_FILE}' not found.")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"[FAIL] YAML syntax error in '{REQUIREMENTS_FILE}': {e}")
    sys.exit(1)
except Exception as e:
    print(f"[FAIL] Unexpected error loading requirements: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Step 2: Install Each Package via pip with Version Constraint
# ------------------------------------------------------------------------------

print("\n[INFO] Installing packages listed in requirements.yml\n")

for pkg, version in requirements.items():
    try:
        # Construct version specification (e.g., "numpy>=1.16")
        pkg_spec = f"{pkg}>={version}"
        print(f"[INFO] Installing {pkg_spec}...")

        # Call pip via subprocess, using the same Python interpreter
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_spec])

        print(f"[ OK ] {pkg} installed successfully.\n")

    except subprocess.CalledProcessError:
        print(f"[FAIL] Installation failed for: {pkg_spec}")
    except Exception as e:
        print(f"[FAIL] Unexpected error for {pkg_spec}: {e}")

# ------------------------------------------------------------------------------
# Completion Message
# ------------------------------------------------------------------------------

print("[SUCCESS] All installable packages have been processed.")
