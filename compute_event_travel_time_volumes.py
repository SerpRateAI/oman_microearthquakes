"""
Compute the travel time volumes for a given combination of surface velocity and velocity gradient and a range of scaling factors.
"""

#-----------
# Imports
#-----------

from argparse import ArgumentParser
from pathlib import Path
from numpy import arange, asarray, arccosh, ndarray, sqrt, meshgrid
from matplotlib.pyplot import subplots
from h5py import File
from pandas import read_csv

from utils_plot import format_east_xlabels, format_north_ylabels, save_figure
from utils_basic import get_geophone_coords, VEL_MODEL_DIR as dirpath_vel
from utils_basic import (
    EASTMIN_A_LOC as min_east_a,
    EASTMAX_A_LOC as max_east_a,
    NORTHMIN_A_LOC as min_north_a,
    NORTHMAX_A_LOC as max_north_a,
    EASTMIN_B_LOC as min_east_b,
    EASTMAX_B_LOC as max_east_b,
    NORTHMIN_B_LOC as min_north_b,
    EASTMIN_A_LOC as min_east,
    NORTHMAX_B_LOC as max_north_b,
    CORE_STATIONS_A as stations_a,
    CORE_STATIONS_B as stations_b
)
#-----------
# Helper functions
#-----------

# Get the travel time for a given distance, source depth, and velocity-model parameters.
def get_travel_time_3d(x, z_s, v0, alpha):
    """
    Travel time from a source at depth z_s to a surface receiver.

    v(z) = v0 + alpha*z

    x   : horizontal epicentral distance, in m
    z_s : source depth, positive downward, in m
    v0  : surface velocity, in m/s
    alpha : velocity gradient, in 1/s
    """
    x = asarray(x, dtype=float)
    z_s = asarray(z_s, dtype=float)

    v_s = v0 + alpha * z_s

    arg = 1.0 + alpha**2 * (x**2 + z_s**2) / (2.0 * v0 * v_s)
    travel_time = (1.0 / alpha) * arccosh(arg)

    return travel_time

# Initialize the output file
def initialize_output_file(filepath: Path, scale_factors: list, easts_grid: ndarray, norths_grid: ndarray, depths_grid: ndarray):
    with File(filepath, "w") as f:
        f.create_dataset("scale_factors", data = scale_factors, dtype = float)
        f.create_dataset("easts_grid", data = easts_grid, dtype = float)
        f.create_dataset("norths_grid", data = norths_grid, dtype = float)
        f.create_dataset("depths_grid", data = depths_grid, dtype = float)

    return

# Save the travel time volumes for a given scale factor
def save_travel_time_volumes(filepath: Path, i_scale_factor: int, travel_time_dict: dict):
    with File(filepath, "a") as f:
        group = f.require_group(f"scale_factor_{i_scale_factor:d}")
        for station, travel_time_vol in travel_time_dict.items():
            group.create_dataset(station, data = travel_time_vol, dtype = float)

    return

# Plot a depth slice of the travel time volumes
def plot_travel_time_slice(travel_time_vol: ndarray, easts_grid: ndarray, norths_grid: ndarray, depths_grid: ndarray, i_depth_to_plot: int, east_station: float, north_station: float, station: str):
    fig, ax = subplots(1, 1, figsize = (10, 10))
    travel_time_vol_slice = travel_time_vol[:, :, i_depth_to_plot]
    print(travel_time_vol_slice.shape)
    im = ax.pcolormesh(easts_grid, norths_grid, travel_time_vol_slice, cmap = "viridis")
    ax.scatter(east_station, north_station, marker = "^", s = 100, color = "salmon", edgecolors = "black")
    format_east_xlabels(ax)
    format_north_ylabels(ax)

    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + pos.width + 0.01, pos.y0, 0.02, pos.height])
    cb = fig.colorbar(im, cax = cax)
    cb.set_label("Travel time (s)")

    depth_to_plot = depths_grid[i_depth_to_plot]
    save_figure(fig, f"event_travel_time_slice_{station}_depth{depth_to_plot:.0f}m.png")

    return

#-----------
# Main function
#-----------



# Command line arguments
parser = ArgumentParser()
parser.add_argument("--subarray", type = str, help = "Subarray", required = True)

parser.add_argument("--scale_factors", type = float, nargs = "+", help = "Scale factors", default = [0.1 * i for i in range(1, 11)])
parser.add_argument("--east_int", type = float, help = "East interval (m)", default = 1.0)
parser.add_argument("--north_int", type = float, help = "North interval (m)", default = 1.0)
parser.add_argument("--depth_int", type = float, help = "Depth interval (m)", default = 1.0)
parser.add_argument("--max_depth", type = float, help = "Maximum depth (m)", default = 30.0)
parser.add_argument("--test", action = "store_true", help = "Run in test mode")

# Parse the command line arguments
args = parser.parse_args()
subarray = args.subarray
scale_factors = args.scale_factors
east_int = args.east_int
north_int = args.north_int
depth_int = args.depth_int
max_depth = args.max_depth
test = args.test

print(f"Computing the event travel time volumes for subarray {subarray}...")
if test:
    print(f"Running in test mode.")

# Load the velocity model
filename = "vel_model_params_from_hammers.csv"
vel_path = Path(dirpath_vel) / filename
vel_params_df = read_csv(vel_path)
surface_vel = float(vel_params_df.loc[0, "surface_vel"])
vel_gradient = float(vel_params_df.loc[0, "vel_gradient"])

# Load the geophone coordinates
coords_df = get_geophone_coords()

if subarray == "A":
    stations = stations_a
    min_east = min_east_a
    max_east = max_east_a
    min_north = min_north_a
    max_north = max_north_a
elif subarray == "B":
    stations = stations_b
    min_east = min_east_b
    max_east = max_east_b
    min_north = min_north_b
    max_north = max_north_b
else:
    raise ValueError(f"Invalid subarray: {subarray}")

# Define the spatial grid
easts_grid = arange(min_east, max_east, east_int)
norths_grid = arange(min_north, max_north, north_int)
depths_grid = arange(0.0, max_depth, depth_int)

# Open the HDF5 file for writing
filename = f"event_travel_time_volumes_{subarray.lower()}.h5"
filepath = Path(dirpath_vel) / filename
initialize_output_file(filepath, scale_factors, easts_grid, norths_grid, depths_grid)

# Iterate over the scale factors
for i_scale_factor, scale_factor in enumerate(scale_factors):
    print(f"Computing the travel time volumes for scale factor {scale_factor:.1f}...")
    surface_vel_scaled = surface_vel * scale_factor
    vel_gradient_scaled = vel_gradient * scale_factor

    travel_time_dict = {}
    for i_station, station in enumerate(stations):
        print(f"Computing the travel time volume for station {station}...")
        station_east = coords_df.loc[station, "east"]
        station_north = coords_df.loc[station, "north"]

        # Define the 3D meshgrid
        easts_mesh, norths_mesh, depths_mesh = meshgrid(easts_grid, norths_grid, depths_grid)
        distance_mesh = sqrt((easts_mesh - station_east) ** 2 + (norths_mesh - station_north) ** 2)

        # Compute the travel time
        travel_time_vol = get_travel_time_3d(distance_mesh, depths_mesh, surface_vel_scaled, vel_gradient_scaled)
        travel_time_dict[station] = travel_time_vol

        # Plot the travel time slice if in test mode
        if i_station == 0 and test:
            plot_travel_time_slice(travel_time_vol, easts_grid, norths_grid, depths_grid, 0, station_east, station_north, station)

    # If test mode, break out of the loop
    if test:
        print(f"Finished in test mode.")
        exit()

    # Save the travel time volumes
    save_travel_time_volumes(filepath, i_scale_factor, travel_time_dict)

print(f"Saved the travel time volumes to {filepath}.")