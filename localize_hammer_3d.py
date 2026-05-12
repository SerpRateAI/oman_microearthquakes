"""
Localize a hammer shot in 3D using the precomputed travel time volumes
"""

#---------------------------------
# Import libraries
#---------------------------------
from numpy import unravel_index
from pandas import Timestamp, read_csv, DataFrame, concat
import argparse
from pathlib import Path

from utils_basic import (
    PICK_DIR  as dirpath_pick,
    DETECTION_DIR as dirpath_detection,
    LOC_DIR   as dirpath_loc,
    VEL_MODEL_DIR as dirpath_vel,
    get_geophone_coords
)
from utils_loc import (
    load_travel_time_volumes_combined, 
    plot_misfit_distribution, 
    process_arrival_info,
    get_misfit_and_origin_time_grids,
    save_hammer_location_to_hdf,
)
from utils_snuffler import read_time_windows
from utils_plot import save_figure

#---------------------------------
# Define functions
#---------------------------------

"""
Save the location to an CSV file
"""
def save_location_to_csv(hammer_id, arrival_type, phase, subarray, east, north, depth, origin_time, rms):
    filename = f"hammer_locations_3d.csv"
    filepath = Path(dirpath_loc) / filename    
    new_loc_dict = {
        "hammer_id": hammer_id,
        "arrival_type": arrival_type,
        "phase": phase,
        "subarray": subarray,
        "east": east,
        "north": north,
        "depth": depth,
        "origin_time": origin_time,
        "rms": rms,
    }
    
    # If the file exists, append the location to the file
    if filepath.exists():
        location_df = read_csv(filepath, dtype={"hammer_id": str, "phase": str, "subarray": str})

        if hammer_id in location_df["hammer_id"].values:
            print(f"Hammer {hammer_id} already exists in {filepath}. Updating the location.")
            location_df.loc[location_df["hammer_id"] == hammer_id, "east"] = east
            location_df.loc[location_df["hammer_id"] == hammer_id, "north"] = north
            location_df.loc[location_df["hammer_id"] == hammer_id, "depth"] = depth
            location_df.loc[location_df["hammer_id"] == hammer_id, "origin_time"] = origin_time
            location_df.loc[location_df["hammer_id"] == hammer_id, "rms"] = rms
        else:
            new_loc_df = DataFrame(new_loc_dict, index=[location_df.shape[0]])
            location_df = concat([location_df, new_loc_df])
    else:
        new_loc_df = DataFrame(new_loc_dict, index=[0])
        location_df = new_loc_df

    location_df.to_csv(filepath, index=False)
    print(f"Saved the location to {filepath}.")

#---------------------------------
# Define main function
#---------------------------------

# Command line arguments
parser = argparse.ArgumentParser(description="Localize an event in 3D using the precomputed travel time volumes")
parser.add_argument("--hammer_id", type=str, required=True, help="Hammer ID")
parser.add_argument("--subarray", type=str, required=True, help="Subarray")


hammer_id = parser.parse_args().hammer_id
subarray = parser.parse_args().subarray

print(f"Localizing hammer shot {hammer_id}...")

# Load travel time volumes
print(f"Loading the travel time volumes for P phase and Subarray {subarray}...")
filename = f"travel_time_volumes_p_{subarray.lower()}.h5"
filepath = Path(dirpath_vel) / filename
easts_grid, norths_grid, depths_grid, travel_time_p_dict = load_travel_time_volumes_combined(filepath, scale_factor = 1.0)

filename = f"travel_time_volumes_s_{subarray.lower()}.h5"
filepath = Path(dirpath_vel) / filename
easts_grid, norths_grid, depths_grid, travel_time_s_dict = load_travel_time_volumes_combined(filepath, scale_factor = 1.0)

# Load arrival information
print(f"Loading arrival information...")
filename = f"hammer_{hammer_id}.mkr"
filepath = Path(dirpath_pick) / filename
arrival_df = read_time_windows(filepath, phase_marker = True)

arrival_df = process_arrival_info(arrival_df, "manual_stack")

# Get the RMS and origin time grids
print(f"Computing RMS and origin time grids...")
misfit_vol, origin_times_grid = get_misfit_and_origin_time_grids(arrival_df, easts_grid, norths_grid, depths_grid, travel_time_p_dict, travel_time_s_dict)

# Get the location with the minimum RMS
print(f"Finding the location with the minimum RMS...")
i_depth_min_rms, i_north_min_rms, i_east_min_rms = unravel_index(misfit_vol.argmin(), misfit_vol.shape)
min_misfit = misfit_vol[i_depth_min_rms, i_north_min_rms, i_east_min_rms]
origin_time_min_rms = origin_times_grid[i_depth_min_rms, i_north_min_rms, i_east_min_rms]

east_min_rms = easts_grid[i_east_min_rms]
north_min_rms = norths_grid[i_north_min_rms]
depth_min_rms = depths_grid[i_depth_min_rms]

# Plot the misfit distribution
coord_df = get_geophone_coords()
coord_df = coord_df[coord_df.index.isin(arrival_df["station"])]
coord_df = coord_df.reset_index(drop=False)
fig, ax_map, ax_profile, cbar = plot_misfit_distribution(misfit_vol.transpose(1, 2, 0), easts_grid, norths_grid, depths_grid, i_east_min_rms, i_north_min_rms, i_depth_min_rms, coord_df,
                                                      title = f"Hammer {hammer_id}")
figname = f"misfit_distribution_hammer{hammer_id}_subarray_{subarray.lower()}.png"
save_figure(fig, figname)

# Compute the predicted arrival times
arrival_times_pred = {}
for station in arrival_df["station"].values:
    travel_time = travel_time_p_dict[station][i_depth_min_rms, i_north_min_rms, i_east_min_rms]
    arrival_times_pred[station] = Timestamp(origin_time_min_rms + travel_time, unit="s")

# Compute the misfit at each station
misfit_dict = {}
for station in arrival_df["station"].values:
    misfit = arrival_df.loc[arrival_df["station"] == station, "arrival_time"].values[0] - arrival_times_pred[station].timestamp()
    misfit_dict[station] = misfit

# Save the location information to an HDF5 file
param_dict = {
    "hammer_id": hammer_id,
    "weight": True,
}

origin_time_min_rms = Timestamp(origin_time_min_rms, unit="s")
location_dict = {
    "origin_time": origin_time_min_rms,
    "east": east_min_rms,
    "north": north_min_rms,
    "depth": depth_min_rms,
    "min_misfit": min_misfit,
}
grid_dict = {
    "easts_grid": easts_grid,
    "norths_grid": norths_grid,
    "depths_grid": depths_grid,
    "misfit_vol": misfit_vol,
}

filename = f"hammer_location_info_{hammer_id}.h5"
filepath = Path(dirpath_loc) / filename
save_hammer_location_to_hdf(filepath, location_dict, arrival_times_pred, misfit_dict, grid_dict)

# Print the location information
print(f"Minimum misfit: {min_misfit} s")
print(f"Origin time at minimum misfit: {origin_time_min_rms}")
print(f"Location: {east_min_rms} m E, {north_min_rms} m N, {depth_min_rms} m D")


