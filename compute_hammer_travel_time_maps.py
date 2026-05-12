"""
Compute the hammer travel time maps for a range of velocity-model parameters.
"""

#-----------
# Imports
#-----------

from argparse import ArgumentParser
from pathlib import Path
from utils_basic import VEL_MODEL_DIR as dirpath_vel
from numpy import asarray, arcsinh, arange, meshgrid, sqrt, linspace, ndarray
from tqdm import tqdm
from h5py import File
from matplotlib.pyplot import subplots

from utils_basic import get_geophone_coords
from utils_basic import (
    EASTMIN_A_LOC as min_east,
    EASTMAX_A_LOC as max_east,
    NORTHMIN_A_LOC as min_north,
    NORTHMAX_A_LOC as max_north,
    CORE_STATIONS_A as stations
)
from utils_plot import format_east_xlabels, format_north_ylabels, save_figure

#-----------
# Helper functions
#-----------

# Get the travel time for a given distance and velocity-model parameters.
def get_travel_time(x, v0, alpha):
    x = asarray(x)

    travel_time = (2.0 / alpha) * arcsinh(alpha * x / (2.0 * v0))
    
    return travel_time

# Initialize the output file
def initialize_output_file(filepath: Path, surface_vel_grid: ndarray, vel_gradient_grid: ndarray, east_grid: ndarray, north_grid: ndarray):
    with File(filepath, "w") as f:
        f.create_dataset("surface_vel", data = surface_vel_grid, dtype = float)
        f.create_dataset("vel_gradient", data = vel_gradient_grid, dtype = float)
        f.create_dataset("east_grid", data = east_grid, dtype = float)
        f.create_dataset("north_grid", data = north_grid, dtype = float)

    return

# Save the travel time map
def save_travel_time_map(filepath: Path, i_surface_vel: int, i_vel_gradient: int, travel_time_dict: dict):
    with File(filepath, "a") as f:
        group = f.require_group(f"surface_vel_{i_surface_vel}_vel_gradient_{i_vel_gradient}")
        for station, travel_time_mat in travel_time_dict.items():
            group.create_dataset(station, data = travel_time_mat, dtype = float)

    return

# Plot the travel time map
def plot_travel_time_map(travel_time_mat: ndarray, east_grid: ndarray, north_grid: ndarray, east_station: float, north_station: float, station: str):

    fig, ax = subplots(1, 1, figsize = (10, 10))
    im = ax.pcolormesh(east_grid, north_grid, travel_time_mat, cmap = "viridis")
    ax.scatter(east_station, north_station, marker = "^", s = 100, color = "salmon", edgecolors = "black")
    format_east_xlabels(ax)
    format_north_ylabels(ax)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + pos.width + 0.01, pos.y0, 0.02, pos.height])
    cb = fig.colorbar(im, cax = cax)
    cb.set_label("Travel time (s)")
    save_figure(fig, f"hammer_travel_time_map_{station}.png")
    return

#-----------
# Main function
#-----------

parser = ArgumentParser()
parser.add_argument("--min_surface_vel", type = float, help = "Minimum surface velocity (m/s)", default = 200.0)
parser.add_argument("--max_surface_vel", type = float, help = "Maximum surface velocity (m/s)", default = 600.0)
parser.add_argument("--min_vel_gradient", type = float, help = "Minimum velocity gradient (1/s)", default = 100.0)
parser.add_argument("--max_vel_gradient", type = float, help = "Maximum velocity gradient (1/s)", default = 200.0)
parser.add_argument("--num_surface_vel", type = int, help = "Number of surface velocities", default = 20)
parser.add_argument("--num_vel_gradient", type = int, help = "Number of velocity gradients", default = 20)
parser.add_argument("--east_int", type = float, help = "East interval (m)", default = 1.0)
parser.add_argument("--north_int", type = float, help = "North interval (m)", default = 1.0)
parser.add_argument("--test", action = "store_true", help = "Run in test mode")


# Parse the command line arguments
args = parser.parse_args()
min_surface_vel = args.min_surface_vel
max_surface_vel = args.max_surface_vel
min_vel_gradient = args.min_vel_gradient
max_vel_gradient = args.max_vel_gradient
num_surface_vel = args.num_surface_vel
num_vel_gradient = args.num_vel_gradient
east_int = args.east_int
north_int = args.north_int
test = args.test


print(f"Computing the hammer travel time maps for different velocity-model parameters...")
print(f"Minimum surface velocity: {min_surface_vel:.1f} m/s")
print(f"Maximum surface velocity: {max_surface_vel:.1f} m/s")
print(f"Minimum velocity gradient: {min_vel_gradient:.1f} 1/s")
print(f"Maximum velocity gradient: {max_vel_gradient:.1f} 1/s")
print(f"Number of surface velocities: {num_surface_vel}")
print(f"Number of velocity gradients: {num_vel_gradient}")
print(f"East interval: {east_int} m")
print(f"North interval: {north_int} m")
print(f"Running in test mode: {test}")

# Load the geophone coordinates
station_df = get_geophone_coords()

# Define the grid
east_grid = arange(min_east, max_east, east_int)
north_grid = arange(min_north, max_north, north_int)

# Define the parameters space
surface_vel_grid = linspace(min_surface_vel, max_surface_vel, num_surface_vel)
vel_gradient_grid = linspace(min_vel_gradient, max_vel_gradient, num_vel_gradient)

# Open the HDF5 file for writing
filename = f"hammer_travel_time_maps.h5"
filepath = Path(dirpath_vel) / filename
initialize_output_file(filepath, surface_vel_grid, vel_gradient_grid, east_grid, north_grid)

# Iterate over the parameters space
for i_surface_vel, surface_vel in enumerate(tqdm(surface_vel_grid, total = len(surface_vel_grid), desc = "Surface velocity")):
    for i_vel_gradient, vel_gradient in enumerate(tqdm(vel_gradient_grid, total = len(vel_gradient_grid), desc = "Velocity gradient")):
        travel_time_dict = {}
        for i_station, station in enumerate(stations):
            station_east = station_df.loc[station, "east"]
            station_north = station_df.loc[station, "north"]

            # Define the meshgrid
            east_mesh, north_mesh = meshgrid(east_grid, north_grid)
            distance_mesh = sqrt((east_mesh - station_east) ** 2 + (north_mesh - station_north) ** 2)

            # Compute the travel time
            travel_time_mat = get_travel_time(distance_mesh, surface_vel, vel_gradient)

            travel_time_dict[station] = travel_time_mat

            # Plot the travel time map if in test mode
            if i_station == 0 and test:
                plot_travel_time_map(travel_time_mat, east_grid, north_grid, station_east, station_north, station)

        # If test mode, break out of the loop
        if test:
            print(f"Finished in test mode.")
            exit()

        # Save the travel time map
        save_travel_time_map(filepath, i_surface_vel, i_vel_gradient, travel_time_dict)

print(f"Saved the travel time maps to {filepath}.")