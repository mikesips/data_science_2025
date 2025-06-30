# ==============================================================================
# File: visualize_vegetation_ts.py
# Purpose: Load configuration and plot Sentinel-2 vegetation time series
# Author : Mike Sips
# ==============================================================================

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Function: load_visualize_scl_parameters
# Purpose : Load configuration parameters for visualizing SCL and time-series plots
# ------------------------------------------------------------------------------

def load_visualize_vegetation_ts_parameters(config_path: str = "scl_plot_config.yml") -> dict:
    """
    Load plot configuration for saving SCL and vegetation time series plots.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file (default: 'scl_plot_config.yml').

    Returns
    -------
    dict
        Dictionary containing expected plot-related paths:
            - original_scl_save_dir : str
            - clipped_scl_save_dir  : str
            - vegetation_ts_save_path : str

    Raises
    ------
    FileNotFoundError
        If the specified YAML file does not exist.
    ValueError
        If required keys are missing or have incorrect types.
    yaml.YAMLError
        If the YAML content cannot be parsed.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("YAML content must be a dictionary")

    required_keys = ["vegetation_ts_save_dir"]
    for key in required_keys:
        if key not in config or not isinstance(config[key], str):
            raise ValueError(f"Missing or invalid value for key: '{key}'")

    return config

# ------------------------------------------------------------------------------
# Function: plot_vegetation_time_series
# Purpose : Plot surface area of vegetation over time from a DataFrame
# ------------------------------------------------------------------------------

def plot_vegetation_time_series(
    table: pd.DataFrame,
    save_dir: str
) -> None:
    """
    Plot vegetation surface area as a time series.

    Parameters
    ----------
    table : pd.DataFrame
        DataFrame with at least two columns:
            - 'Date' (str or datetime): Observation dates.
            - 'Vegetation Surface Area' (float): Corresponding surface area in km².

    save_path : str, optional
        File path to save the generated plot (default: 'vegetation_time_series.png').

    Returns
    -------
    None
    """
    # Ensure 'Date' is in datetime format
    table['Date'] = pd.to_datetime(table['Date'])
    table.set_index('Date', inplace=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(table.index, table['Vegetation Surface Area'], marker='o', linestyle='-', color='green')

    plt.title("Vegetation Surface Area Over Time", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Vegetation Surface Area (km²)", fontsize=14)
    plt.grid(True)

    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())
    plt.xticks(rotation=45)

    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "vegetation_time_series.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Vegetation time series plot saved to: {save_path}")
