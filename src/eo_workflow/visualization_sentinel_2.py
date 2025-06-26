# plot_vegetation_time_series.py
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd

def plot_scl_layer(
    scl: xr.DataArray,
    title: str = "Sentinel-2 Scene Classification Layer (SCL)",
    cmap: str = "tab20",
    figsize: tuple = (16, 12),
    show_axis: bool = False,
) -> None:
    """
    Plot a Sentinel-2 SCL layer with class-specific colors and labels.

    Parameters:
    - scl_layer: xarray.DataArray containing SCL values (integers).
    - title: Title of the plot.
    - show_axis: Whether to display axis ticks.
    - save_path: Optional path to save the figure. If None, the plot is shown.
    """
    # Define SCL classes and colors (Sentinel-2 convention)
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

    cmap = plt.get_cmap("tab20", 12)  # Discrete colormap with 12 classes
    bounds = np.arange(-0.5, 12.5, 1)
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Plotting
    plt.figure(figsize=(10, 8))
    img = plt.imshow(scl, cmap=cmap, norm=norm)
    plt.title(title, fontsize=14)
    
    if not show_axis:
        plt.axis("off")

    # Create a custom legend
    patches = [
        plt.matplotlib.patches.Patch(color=cmap(i), label=f"{i}: {label}")
        for i, label in scl_classes.items()
    ]
    plt.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize=10
    )

    plt.tight_layout()

    plt.savefig("test.png", dpi=300)
    plt.close()

# ------------------------------------------------------------------------------
# Function: plot_all_scl_scenes
# Purpose : Plot all time steps of the SCL band from an xarray.Dataset
# ------------------------------------------------------------------------------
def plot_all_scl_scenes(
    dataset: xr.Dataset,
    scl_band: str = "scl",
    cmap: str = "tab20",
    figsize: tuple = (16, 12)
) -> None:

    # Check that dataset and scl band are valid
    if scl_band not in dataset:
        raise ValueError(f"Dataset does not contain band '{scl_band}'")
    scl = dataset[scl_band]

    if "time" not in scl.dims:
        raise ValueError("SCL band must have a 'time' dimension")

    for scene_id in range(scl.sizes["time"]):
        scene = scl.isel(time=scene_id)
        plot_scl_layer(scene)


def plot_vegetation_time_series(table: pd.DataFrame):
    # Convert the 'Date' column to datetime type
    table['Date'] = pd.to_datetime(table['Date'])
    table.set_index('Date', inplace=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(table.index, table['Vegetation Surface Area'], marker='o', linestyle='-', color='b')

    plt.title('Vegetation Surface Area Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Vegetation Surface Area', fontsize=14)

    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("vegetation_time_series.png")