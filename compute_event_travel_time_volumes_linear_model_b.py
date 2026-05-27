"""
Compute the travel time volumes for a linear velocity model in Subarray B.

The script reads unscaled velocity-model parameters from linear_vel_model_params_a.csv.
It first scales both the surface velocity and gradient by scale_factor_a, then applies
each scale factor in scale_factors_b to the gradient and computes travel times.
"""

#-----------
# Imports
#-----------

from argparse import ArgumentParser
from pathlib import Path
from numpy import arange, asarray, arccosh, isclose, ndarray, sqrt, meshgrid
from matplotlib.pyplot import subplots
from h5py import File
from pandas import read_csv

from utils_plot import format_east_xlabels, format_north_ylabels, save_figure
from utils_basic import get_geophone_coords, VEL_MODEL_DIR as dirpath_vel
from utils_basic import (
    EASTMIN_B_LOC as min_east_b,
    EASTMAX_B_LOC as max_east_b,
    NORTHMIN_B_LOC as min_north_b,
    NORTHMAX_B_LOC as max_north_b,
    CORE_STATIONS_B as stations_b,
)


#-----------
# Helper functions
#-----------

def get_travel_time_3d(x, z_s, v0, alpha):
    """
    Travel time from a source at depth z_s to a surface receiver.

    v(z) = v0 + alpha*z  (if alpha = 0, v is constant v0)

    x   : horizontal epicentral distance, in m
    z_s : source depth, positive downward, in m
    v0  : surface velocity, in m/s
    alpha : velocity gradient, in 1/s
    """
    x = asarray(x, dtype=float)
    z_s = asarray(z_s, dtype=float)
    alpha_arr = asarray(alpha, dtype=float)

    if alpha_arr.size == 1 and isclose(float(alpha_arr), 0.0):
        return sqrt(x**2 + z_s**2) / v0

    alpha = alpha_arr
    v_s = v0 + alpha * z_s
    arg = 1.0 + alpha**2 * (x**2 + z_s**2) / (2.0 * v0 * v_s)
    travel_time = (1.0 / alpha) * arccosh(arg)

    return travel_time


def initialize_output_file(
    filepath: Path,
    scale_factors_b: list,
    easts_grid: ndarray,
    norths_grid: ndarray,
    depths_grid: ndarray,
):
    with File(filepath, "w") as f:
        f.create_dataset("scale_factors_b", data=scale_factors_b, dtype=float)
        f.create_dataset("easts_grid", data=easts_grid, dtype=float)
        f.create_dataset("norths_grid", data=norths_grid, dtype=float)
        f.create_dataset("depths_grid", data=depths_grid, dtype=float)


def save_travel_time_volumes(filepath: Path, i_scale_factor_b: int, travel_time_dict: dict):
    with File(filepath, "a") as f:
        group = f.require_group(f"scale_factor_b_{i_scale_factor_b:d}")
        for station, travel_time_vol in travel_time_dict.items():
            group.create_dataset(station, data=travel_time_vol, dtype=float)


def plot_travel_time_slice(
    travel_time_vol: ndarray,
    easts_grid: ndarray,
    norths_grid: ndarray,
    depths_grid: ndarray,
    i_depth_to_plot: int,
    east_station: float,
    north_station: float,
    station: str,
):
    fig, ax = subplots(1, 1, figsize=(10, 10))
    travel_time_vol_slice = travel_time_vol[:, :, i_depth_to_plot]
    im = ax.pcolormesh(easts_grid, norths_grid, travel_time_vol_slice, cmap="viridis")
    ax.scatter(east_station, north_station, marker="^", s=100, color="salmon", edgecolors="black")
    format_east_xlabels(ax)
    format_north_ylabels(ax)

    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + pos.width + 0.01, pos.y0, 0.02, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Travel time (s)")

    depth_to_plot = depths_grid[i_depth_to_plot]
    save_figure(fig, f"event_travel_time_slice_subarray_b_{station}_depth{depth_to_plot:.0f}m.png")


#-----------
# Main function
#-----------

parser = ArgumentParser()
parser.add_argument("--scale_factor_a", type=float, help="Primary scale factor", required=True)
parser.add_argument(
    "--scale_factors_b",
    type=float,
    nargs="+",
    help="Secondary scale factors for gradient scaling",
    default=[1 + 0.1 * i for i in range(1, 11)],
)
parser.add_argument("--east_int", type=float, help="East interval (m)", default=1.0)
parser.add_argument("--north_int", type=float, help="North interval (m)", default=1.0)
parser.add_argument("--depth_int", type=float, help="Depth interval (m)", default=1.0)
parser.add_argument("--max_depth", type=float, help="Maximum depth (m)", default=30.0)
parser.add_argument("--test", action="store_true", help="Run in test mode")

args = parser.parse_args()
scale_factor_a = args.scale_factor_a
scale_factors_b = args.scale_factors_b
east_int = args.east_int
north_int = args.north_int
depth_int = args.depth_int
max_depth = args.max_depth
test = args.test

print("Computing event travel time volumes for linear model (Subarray B)...")
if test:
    print("Running in test mode.")

# Load unscaled velocity model parameters
filename = "linear_vel_model_params_a.csv"
vel_path = Path(dirpath_vel) / filename
vel_params_df = read_csv(vel_path)
surface_vel_unscaled = float(vel_params_df.loc[0, "surface_vel"])
vel_gradient_unscaled = float(vel_params_df.loc[0, "vel_gradient"])

# Apply primary scaling
surface_vel_scaled_a = surface_vel_unscaled * scale_factor_a
vel_gradient_scaled_a = vel_gradient_unscaled * scale_factor_a

# Load geophone coordinates and Subarray B geometry
coords_df = get_geophone_coords()
stations = stations_b
min_east = min_east_b
max_east = max_east_b
min_north = min_north_b
max_north = max_north_b

# Define spatial grid
easts_grid = arange(min_east, max_east, east_int)
norths_grid = arange(min_north, max_north, north_int)
depths_grid = arange(0.0, max_depth, depth_int)

# Initialize output file
filename = f"event_travel_time_volumes_linear_model_b_scale_a{scale_factor_a:.1f}.h5"
filepath = Path(dirpath_vel) / filename
initialize_output_file(filepath, scale_factors_b, easts_grid, norths_grid, depths_grid)

# Iterate over secondary gradient scale factors
for i_scale_factor_b, scale_factor_b in enumerate(scale_factors_b):
    vel_gradient_scaled = vel_gradient_scaled_a * scale_factor_b
    print(
        f"Computing travel-time volumes for scale_factor_b={scale_factor_b:.3f} "
        f"(surface scale_factor_a={scale_factor_a:.3f})..."
    )

    travel_time_dict = {}
    for i_station, station in enumerate(stations):
        print(f"Computing travel-time volume for station {station}...")
        station_east = coords_df.loc[station, "east"]
        station_north = coords_df.loc[station, "north"]

        easts_mesh, norths_mesh, depths_mesh = meshgrid(easts_grid, norths_grid, depths_grid)
        distance_mesh = sqrt((easts_mesh - station_east) ** 2 + (norths_mesh - station_north) ** 2)

        travel_time_vol = get_travel_time_3d(
            distance_mesh, depths_mesh, surface_vel_scaled_a, vel_gradient_scaled
        )
        travel_time_dict[station] = travel_time_vol

        if i_station == 0 and test:
            plot_travel_time_slice(
                travel_time_vol,
                easts_grid,
                norths_grid,
                depths_grid,
                0,
                station_east,
                station_north,
                station,
            )

    save_travel_time_volumes(filepath, i_scale_factor_b, travel_time_dict)

    if test:
        print("Finished in test mode.")
        break

print(f"Saved travel-time volumes to {filepath}.")
