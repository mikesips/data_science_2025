#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: EUPL-1.2

# -*- coding: utf-8 -*-

# ==============================================================================
# Script: check_environment.py
# Purpose: Check Python environment and dependencies listed in requirements.yml.
#          Auto-installs missing core packages ('packaging', 'pyyaml').
# Author: Mike Sips
# ==============================================================================

import sys
import importlib
from typing import Optional

# ------------------------------------------------------------------------------
# Terminal output formatting
# ------------------------------------------------------------------------------
OK = "\x1b[42m[ OK ]\x1b[0m"
FAIL = "\x1b[41m[FAIL]\x1b[0m"
INFO = "\x1b[44m[INFO]\x1b[0m"

# ------------------------------------------------------------------------------
# Auto-install helper function
# ------------------------------------------------------------------------------
def ensure_package_installed(package_name: str, import_name: Optional[str] = None):
    """
    Ensure that a required package is installed. If not, attempt to install it via pip.
    """
    import_name = import_name or package_name
    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(FAIL, f"'{package_name}' not found. Attempting to install via pip...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(OK, f"'{package_name}' installed successfully.")
            return importlib.import_module(import_name)
        except Exception as e:
            print(FAIL, f"Automatic installation of '{package_name}' failed: {e}")
            sys.exit(1)

# ------------------------------------------------------------------------------
# Ensure 'packaging' and 'pyyaml' are available
# ------------------------------------------------------------------------------
packaging = ensure_package_installed("packaging")
from packaging.version import Version

yaml = ensure_package_installed("pyyaml", "yaml")

# ------------------------------------------------------------------------------
# Check Python version compatibility
# ------------------------------------------------------------------------------
print(f"{INFO} Checking Python interpreter")
print("Using Python from:", sys.prefix)
print(sys.version)
current_py_version = Version(f"{sys.version_info.major}.{sys.version_info.minor}")
REQUIRED_PYTHON_VERSION = Version("3.8")

if current_py_version < REQUIRED_PYTHON_VERSION:
    print(FAIL, f"Python {REQUIRED_PYTHON_VERSION} or higher required, but {current_py_version} is installed.")
    sys.exit(1)
print(OK, f"Python {current_py_version} is compatible.\n")

# ------------------------------------------------------------------------------
# Function: import_and_check_version
# ------------------------------------------------------------------------------
def import_and_check_version(pkg: str, min_version: str, fail_hint: str = "") -> Optional[object]:
    """
    Import a module and check that its version meets a required minimum.
    """
    try:
        module = importlib.import_module(pkg)

        # Special handling for PIL (Pillow)
        if pkg == "PIL":
            version = getattr(module, "__version__", None) or \
                      getattr(module, "VERSION", None) or \
                      getattr(module, "PILLOW_VERSION", None)
        else:
            version = getattr(module, "__version__", None)

        if version is None:
            print(FAIL, f"Cannot determine version for {pkg}.")
            return None

        if Version(version) < Version(min_version):
            print(FAIL, f"{pkg} version {min_version} or higher required, but {version} is installed.")
        else:
            print(OK, f"{pkg} version {version}")
        return module

    except ImportError:
        print(FAIL, f"{pkg} is not installed. {fail_hint}")
        return None

# ------------------------------------------------------------------------------
# Load Requirements from YAML File
# ------------------------------------------------------------------------------
def load_requirements_from_yaml(path: str = "requirements.yml") -> dict:
    """
    Load a dictionary of package:version from a YAML file.
    """
    try:
        with open(path, "r") as file:
            requirements = yaml.safe_load(file)
            if not isinstance(requirements, dict):
                raise ValueError("Requirements YAML must contain a dictionary of package names and versions.")
            return requirements
    except FileNotFoundError:
        print(FAIL, f"Requirements file '{path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(FAIL, f"YAML syntax error in '{path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(FAIL, f"Unexpected error reading requirements: {e}")
        sys.exit(1)

# ------------------------------------------------------------------------------
# Main Dependency Check Logic
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    REQUIREMENTS_FILE = sys.argv[1]

    print(f"{INFO} Loading requirements from requirements.yml\n")
    requirements = load_requirements_from_yaml(REQUIREMENTS_FILE)

    print(f"{INFO} Checking installed packages\n")
    for package, required_version in requirements.items():
        import_and_check_version(package, required_version)

    print(f"\n{INFO} Environment check complete. All installable requirements have been processed.")
