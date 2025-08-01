"""
Compute the body wave travel time volumes for all stations in a subarray.
"""

# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from time import time
from matplotlib import colormaps
from matplotlib.pyplot import subplots
from numpy import arange, sqrt, zeros, nan, where
from pathlib import Path
from tqdm import tqdm
from pyrocko.cake import load_model, m2d, d2m
from pyrocko.cake import PhaseDef
from h5py import File

from utils_basic import (
    EASTMIN_A_LOC as min_east_a,
    EASTMAX_A_LOC as max_east_a,
    NORTHMIN_A_LOC as min_north_a,
    NORTHMAX_A_LOC as max_north_a,
    EASTMIN_B_LOC as min_east_b,
    EASTMAX_B_LOC as max_east_b,
    NORTHMIN_B_LOC as min_north_b,
    NORTHMAX_B_LOC as max_north_b,
    CORE_STATIONS_A as core_stations_a,
    CORE_STATIONS_B as core_stations_b,
    get_geophone_coords,
    VEL_MODEL_DIR as dirpath_vel
)

from utils_loc import save_travel_time_volumes
from utils_plot import format_east_xlabels, format_north_ylabels, save_figure

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

"""
Compute the travel time between a station and a grid point.
The input are in meters.
"""
def compute_travel_time(east_station, north_station, east_grid, north_grid, depth_grid, vel_model, phases):
    distance_m = sqrt((east_station - east_grid)**2 + (north_station - north_grid)**2)
    distance_deg = distance_m * m2d

    arrivals = vel_model.arrivals([distance_deg], phases = phases, zstart = depth_grid)

    if len(arrivals) == 0:
        return None

    arrival = arrivals[0]
    travel_time = arrival.t

    return travel_time

"""
Plot depth slices of the travel time volumes.
"""
def plot_depth_slice(ax, travel_time_vol, easts_grid, norths_grid, depths_grid, depth_to_plot, east_station, north_station,
                     max_tt = 0.03):
    
    # Extract the depth slice
    i_depth = where(depths_grid == depth_to_plot)[0][0]
    travel_time_mat = travel_time_vol[i_depth, :, :]

    # Plot the depth slice
    cmap = colormaps["viridis"]
    cmap.set_bad("gray")
    im = ax.pcolormesh(easts_grid, norths_grid, travel_time_mat, cmap = cmap, vmin = 0.0, vmax = max_tt)

    # Plot the station
    ax.scatter(east_station, north_station, marker = "^", s = 100, color = "salmon", edgecolors = "black")

    # Set the axis labels
    format_east_xlabels(ax,
                        major_tick_spacing = 10.0,
                        num_minor_ticks = 5)
    
    format_north_ylabels(ax,
                        major_tick_spacing = 10.0,
                        num_minor_ticks = 5)
    
    # Set the title
    ax.set_title(f"Travel time at {depth_to_plot:.0f} m.", fontsize = 14, fontweight = "bold")

    # Set the aspect ratio
    ax.set_aspect('equal', 'box')

    # 
    fig = ax.get_figure()
    # get the bounding box of the main axes [x0, y0, width, height]
    pos = ax.get_position().bounds  
    # define colorbar width and padding (in figure coords)
    cbar_width = 0.02
    pad = 0.01
    # [left, bottom, width, height] for cax
    cax = fig.add_axes([
        pos[0] + pos[2] + pad,  # x0 + width + pad
        pos[1],                 # same bottom
        cbar_width,             # narrow width
        pos[3]                  # same height
    ])

    # Add the colorbar
    cb = fig.colorbar(im, cax = cax)
    cb.set_label("Travel time (s)", fontsize = 14, fontweight = "bold")

    return ax



# -----------------------------------------------------------------------------
# Input parameters
# -----------------------------------------------------------------------------

# Command line arguments
parser = ArgumentParser()
parser.add_argument("--phase", type = str, required = True, help = "Phase name")
parser.add_argument("--subarray", type = str, required = True, help = "Subarray name")


parser.add_argument("--scale_factor", type = float, default = 1.0, help = "Scale factor")
parser.add_argument("--delta", type = float, default = 1.0, help = "Grid spacing in meters")
parser.add_argument("--max_depth", type = float, default = 30.0, help = "Maximum depth in meters")
parser.add_argument("--test", action = "store_true", help = "Run in test mode")

# Parse the command line arguments
args = parser.parse_args()
phase = args.phase
subarray = args.subarray
scale_factor = args.scale_factor
delta = args.delta
max_depth = args.max_depth
test = args.test

if subarray == "A":
    min_east = min_east_a
    max_east = max_east_a
    min_north = min_north_a
    max_north = max_north_a

    stations = core_stations_a
elif subarray == "B":
    min_east = min_east_b
    max_east = max_east_b
    min_north = min_north_b
    max_north = max_north_b

    stations = core_stations_b
else:
    raise ValueError(f"Invalid subarray: {subarray}")

if phase == "P":
    filename_vel = f"vp_1d_scale{scale_factor:.1f}.nd"
    phases = [PhaseDef("P"), PhaseDef("p")]
elif phase == "S":
    filename_vel = f"vs_1d_scale{scale_factor:.1f}.nd"
    phases = [PhaseDef("S"), PhaseDef("s")]
else:
    raise ValueError(f"Invalid phase name: {phase}")

# -----------------------------------------------------------------------------
# Load the input data
# -----------------------------------------------------------------------------

# Get the geophone coordinates
print("Loading the geophone coordinates...")
coords_df = get_geophone_coords()

# Load the velocity model
print("Loading the velocity model...")
vel_path = Path(dirpath_vel) / filename_vel
vel_model = load_model(vel_path)

print(f"Loaded the velocity model from {vel_path}.")


# -----------------------------------------------------------------------------
# Compute the travel time volumes for each station
# -----------------------------------------------------------------------------

# Define the search grid
easts_grid = arange(min_east, max_east + delta, delta)
norths_grid = arange(min_north, max_north + delta, delta)
depths_grid = arange(0.0, max_depth + delta, delta)

# Compute the travel time volumes
travel_time_dict = {}
for i, station in enumerate(stations):
    print(f"Computing the travel time volumes for station {station}...")
    clock1 = time()
    # Initialize the travel time volumes
    travel_times = zeros((len(depths_grid), len(norths_grid), len(easts_grid)))

    # Get the station coordinates
    east_station = coords_df.loc[station, "east"]
    north_station = coords_df.loc[station, "north"]

    # Loop over the search grid
    for j, depth_source in tqdm(enumerate(depths_grid), total = len(depths_grid), desc = "North grid"):
        for k, north_source in enumerate(norths_grid):
            for l, east_source in enumerate(easts_grid):
                travel_time = compute_travel_time(east_station, north_station, east_source, north_source, depth_source, vel_model, phases)
                if travel_time is not None:
                    travel_times[j, k, l] = travel_time
                else:
                    travel_times[j, k, l] = nan

    clock2 = time()
    print(f"Finished computing the travel time volume for station {station}.")
    print(f"Time taken: {clock2 - clock1:.2f} seconds")

    travel_time_dict[station] = travel_times

    if test:
        depths = [0.0, 15.0, 30.0]
        for depth in depths: 
            print(f"Plotting the travel-time map at depth = {depth:.0f} m")
            fig, ax = subplots(1, 1, figsize = (10, 10))

            ax = plot_depth_slice(ax, travel_times, easts_grid, norths_grid, depths_grid, depth, east_station, north_station)
            save_figure(fig, f"travel_time_map_{phase.lower()}_{station}_depth{depth:.0f}m.png")

        break

# Save the travel time volumes
save_travel_time_volumes(phase, subarray, scale_factor, easts_grid, norths_grid, depths_grid, travel_time_dict)