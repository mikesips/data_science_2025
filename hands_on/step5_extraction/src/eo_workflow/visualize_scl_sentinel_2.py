# ==============================================================================
# File    : visualize_scl_sentinel.py
# Purpose : Plot Sentinel-2 Scene Classification (SCL) data.
# Author  : Mike Sips
# ==============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import xarray as xr
import numpy as np
import pandas as pd

# ==============================================================================
# Function: load_visualize_scl_parameters
# Purpose : Load configuration parameters for visualizing SCL layer plots
# Author  : Mike Sips
# ==============================================================================

import os
import yaml

def load_visualize_scl_parameters(config_path: str = "scl_plot_config.yml") -> dict:
    """
    Load configuration for SCL plot visualization from a YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file (default: 'scl_plot_config.yml').

    Returns
    -------
    dict
        Dictionary containing at least:
            - save_path (str): Path to save the generated plot.

    Raises
    ------
    FileNotFoundError
        If the specified config file does not exist.
    ValueError
        If required fields are missing or of incorrect types.
    yaml.YAMLError
        If the YAML file is invalid.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("YAML content must be a dictionary")

    if "orignial_scl_save_dir" not in config or not isinstance(config["orignial_scl_save_dir"], str):
        raise ValueError("Missing or invalid 'save_dir' in configuration")

    if "clipped_scl_save_dir" not in config or not isinstance(config["clipped_scl_save_dir"], str):
        raise ValueError("Missing or invalid 'save_dir' in configuration")


    return config

# ------------------------------------------------------------------------------
# Function: plot_scl_layer
# Purpose : Plot a single Sentinel-2 Scene Classification (SCL) layer.
# ------------------------------------------------------------------------------

def plot_scl_layer(
    scl: xr.DataArray,
    title: str = "Sentinel-2 Scene Classification Layer (SCL)",
    cmap: str = "tab20",
    figsize: tuple = (10, 8),
    show_axis: bool = False,
    save_path: str | None = None
) -> None:
    """
    Plot a single SCL layer using predefined Sentinel-2 class labels and colors.

    Parameters
    ----------
    scl : xr.DataArray
        A 2D xarray DataArray containing Sentinel-2 SCL values.

    title : str, optional
        Plot title (default: "Sentinel-2 Scene Classification Layer (SCL)").

    cmap : str, optional
        Colormap name to use (default: "tab20").

    figsize : tuple, optional
        Figure size in inches (default: (10, 8)).

    show_axis : bool, optional
        Whether to display axis ticks (default: False).

    save_path : str or None, optional
        File path to save the figure (e.g., "output.png").
        If None, the plot is shown interactively.
    """
    # Define standard SCL class labels
    scl_classes = {
        0: "No data",
        1: "Saturated / defective",
        2: "Dark area pixels",
        3: "Cloud shadows",
        4: "Vegetation",
        5: "Bare soils",
        6: "Water",
        7: "Clouds low probability",
        8: "Clouds medium probability",
        9: "Clouds high probability",
        10: "Cirrus",
        11: "Snow / ice"
    }

    # Build a discrete colormap
    cmap = plt.get_cmap(cmap, 12)
    bounds = np.arange(-0.5, 12.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create plot
    plt.figure(figsize=figsize)
    img = plt.imshow(scl, cmap=cmap, norm=norm)
    plt.title(title, fontsize=14)

    if not show_axis:
        plt.axis("off")

    # Add legend with class labels
    legend_handles = [
        mpatches.Patch(color=cmap(i), label=f"{i}: {label}")
        for i, label in scl_classes.items()
    ]
    plt.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=10
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


# ------------------------------------------------------------------------------
# Function: plot_all_scl_scenes
# Purpose : Plot each time step of the SCL band in a multi-temporal dataset.
# ------------------------------------------------------------------------------

def plot_all_scl_scenes(
    dataset: xr.Dataset,
    scl_band: str = "scl",
    cmap: str = "tab20",
    figsize: tuple = (10, 8),
    save_dir: str | None = None
) -> None:
    """
    Plot all scenes of the SCL band over time from a multi-temporal dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        An xarray Dataset containing a 3D SCL band with a 'time' dimension.

    scl_band : str, optional
        Name of the SCL band in the dataset (default: "scl").

    cmap : str, optional
        Colormap to use for the plots (default: "tab20").

    figsize : tuple, optional
        Size of each figure (default: (10, 8)).

    save_dir : str or None, optional
        Directory path to save all plots.
        If None, plots are shown interactively.
    """
    if scl_band not in dataset:
        raise ValueError(f"Dataset does not contain band '{scl_band}'")
    
    scl = dataset[scl_band]

    if "time" not in scl.dims:
        raise ValueError("SCL band must have a 'time' dimension")

    for scene_id in range(scl.sizes["time"]):
        scene = scl.isel(time=scene_id)
        timestamp = pd.to_datetime(scene.time.values).strftime("%Y-%m-%d")

        title = f"SCL Layer – {timestamp}"
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"scl_{timestamp}.png")

        plot_scl_layer(
            scl=scene,
            title=title,
            cmap=cmap,
            figsize=figsize,
            show_axis=False,
            save_path=save_path
        )
