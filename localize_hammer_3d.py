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
    get_geophone_coords
)
from utils_loc import (
    load_travel_time_volumes, 
    plot_rms_distribution, 
    save_location_info,
    process_arrival_info,
    get_rms_and_origin_time_grids,
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
parser.add_argument("--phase", type=str, required=True, help="Phase")
parser.add_argument("--subarray", type=str, required=True, help="Subarray")

parser.add_argument("--arrival_type", type=str, default="sta_lta")

hammer_id = parser.parse_args().hammer_id
phase = parser.parse_args().phase
subarray = parser.parse_args().subarray
arrival_type = parser.parse_args().arrival_type

print(f"Localizing hammer shot {hammer_id}...")

# Load travel time volumes
print(f"Loading travel time volumes for {phase} phase and Subarray {subarray}...")
easts_grid, norths_grid, depths_grid, travel_time_dict = load_travel_time_volumes(phase, subarray)

# Load arrival information
print(f"Loading arrival information...")
if arrival_type == "sta_lta":
    filename = f"hammer_arrivals_sta_lta_{hammer_id}.csv"
    filepath = Path(dirpath_detection) / filename

    arrival_df = read_csv(filepath, parse_dates = ["arrival_time"])
else:
    filename = f"hammer_arrivals_{hammer_id}.txt"
    filepath = Path(dirpath_pick) / filename
    arrival_df = read_time_windows(filepath)

arrival_df = process_arrival_info(arrival_df, arrival_type)

# Get the RMS and origin time grids
print(f"Computing RMS and origin time grids...")
rms_vol, origin_times_grid = get_rms_and_origin_time_grids(arrival_df, easts_grid, norths_grid, depths_grid, travel_time_dict)

# Get the location with the minimum RMS
print(f"Finding the location with the minimum RMS...")
i_depth_min_rms, i_north_min_rms, i_east_min_rms = unravel_index(rms_vol.argmin(), rms_vol.shape)
min_rms = rms_vol[i_depth_min_rms, i_north_min_rms, i_east_min_rms]
origin_time_min_rms = origin_times_grid[i_depth_min_rms, i_north_min_rms, i_east_min_rms]

east_min_rms = easts_grid[i_east_min_rms]
north_min_rms = norths_grid[i_north_min_rms]
depth_min_rms = depths_grid[i_depth_min_rms]

# Plot the RMS distribution
coord_df = get_geophone_coords()
coord_df = coord_df[coord_df.index.isin(arrival_df["station"])]
coord_df = coord_df.reset_index(drop=False)
fig, ax_map, ax_profile, cbar = plot_rms_distribution(rms_vol, easts_grid, norths_grid, depths_grid, i_east_min_rms, i_north_min_rms, i_depth_min_rms, coord_df,
                                                      title = f"Hammer {hammer_id}, {phase.upper()} phase")
figname = f"rms_distribution_hammer{hammer_id}_{phase.lower()}_subarray_{subarray.lower()}.png"
save_figure(fig, figname)

# Compute the predicted arrival times
arrival_times_pred = {}
for station in arrival_df["station"].values:
    travel_time = travel_time_dict[station][i_depth_min_rms, i_north_min_rms, i_east_min_rms]
    arrival_times_pred[station] = Timestamp(origin_time_min_rms + travel_time, unit="s")

# Save the location information
origin_time_min_rms = Timestamp(origin_time_min_rms, unit="s")
location_dict = {
    "origin_time": origin_time_min_rms,
    "east": east_min_rms,
    "north": north_min_rms,
    "depth": depth_min_rms,
    "min_rms": min_rms
}
save_location_info("hammer", hammer_id, arrival_type, phase, location_dict, arrival_times_pred, easts_grid, norths_grid, depths_grid, rms_vol)

# Save the location to an CSV file
save_location_to_csv(hammer_id, arrival_type, phase, subarray, east_min_rms, north_min_rms, depth_min_rms, origin_time_min_rms, min_rms)

print(f"Minimum RMS: {min_rms} s")
print(f"Origin time: {origin_time_min_rms}")
print(f"Location: {east_min_rms} m E, {north_min_rms} m N, {depth_min_rms} m D")



