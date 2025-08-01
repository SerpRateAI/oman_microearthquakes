"""
Localize a template event in 3D using the precomputed travel time volumes
"""

#---------------------------------
# Import libraries
#---------------------------------
from numpy import array, zeros, sqrt, nan, mean, unravel_index, inf, isnan
from pandas import Timestamp, read_csv, DataFrame, concat
import argparse
from tqdm import tqdm
from pathlib import Path

from utils_basic import (
    PICK_DIR  as dirpath_pick,
    LOC_DIR   as dirpath_loc,
    DETECTION_DIR as dirpath_detection,
    get_geophone_coords
)
from utils_loc import (
    load_travel_time_volumes, 
    plot_rms_distribution, 
    save_location_info,
    process_arrival_info,
)
from utils_snuffler import read_time_windows
from utils_plot import save_figure

#---------------------------------
# Define functions
#---------------------------------

"""
Get the RMS and origin time grids
"""
def get_rms_and_origin_time_grids(arrival_df, easts_grid, norths_grid, depths_grid, travel_time_dict):
    # Loop through the grid points
    rms_vol = zeros((len(depths_grid), len(norths_grid), len(easts_grid)))
    origin_times_grid = zeros((len(depths_grid), len(norths_grid), len(easts_grid)))
    print(f"Computing RMS and origin time grids...")
    for i_depth, _ in tqdm(enumerate(depths_grid), total=len(depths_grid), desc="Depth"):
        for i_north, _ in enumerate(norths_grid):
            for i_east, _ in enumerate(easts_grid):
                rms, origin_time = get_rms_and_origin_time_for_grid_point(arrival_df, i_east, i_north, i_depth, travel_time_dict)
                rms_vol[i_depth, i_north, i_east] = rms
                origin_times_grid[i_depth, i_north, i_east] = origin_time

    # Replace the nan values with inf
    rms_vol[isnan(rms_vol)] = inf

    return rms_vol, origin_times_grid

"""
Get the RMS and origin time for one grid point
"""
def get_rms_and_origin_time_for_grid_point(arrival_df, i_east, i_north, i_depth, travel_time_dict):
    # Loop through the arrival times
    origin_times = zeros(len(arrival_df))
    travel_times = zeros(len(arrival_df))
    arrival_times = zeros(len(arrival_df))
    for i, row in arrival_df.iterrows():
        station = row["station"]
        travel_time = travel_time_dict[station][i_depth, i_north, i_east]
        arrival_time = row["arrival_time"]

        origin_time = arrival_time - travel_time
        origin_times[i] = origin_time
        travel_times[i] = travel_time
        arrival_times[i] = arrival_time

    origin_time = mean(origin_times)
    rms = sqrt( mean( (arrival_times - travel_times - origin_time) ** 2))

    return rms, origin_time

"""
Save the location to an CSV file
"""
def save_location_to_csv(filepath, template_id, arrival_type, phase, scale_factor, subarray, east, north, depth, origin_time, rms):
    filename = f"template_event_locations.csv"
    filepath = Path(dirpath_loc) / filename    
    new_loc_dict = {
        "template_id": template_id,
        "arrival_type": arrival_type,
        "phase": phase,
        "scale_factor": scale_factor,
        "subarray": subarray,
        "east": east,
        "north": north,
        "depth": depth,
        "origin_time": origin_time,
        "rms": rms,
    }
    
    # If the file exists, append the location to the file
    if filepath.exists():
        location_df = read_csv(filepath, dtype={"template_id": str, "arrival_type": str, "phase": str, "scale_factor": float, "subarray": str})

        if template_id in location_df["template_id"].values:
            print(f"Template {template_id} already exists in {filepath}. Updating the location.")
            location_df.loc[location_df["template_id"] == template_id, "east"] = east
            location_df.loc[location_df["template_id"] == template_id, "north"] = north
            location_df.loc[location_df["template_id"] == template_id, "depth"] = depth
            location_df.loc[location_df["template_id"] == template_id, "arrival_type"] = arrival_type
            location_df.loc[location_df["template_id"] == template_id, "scale_factor"] = scale_factor
            location_df.loc[location_df["template_id"] == template_id, "origin_time"] = origin_time
            location_df.loc[location_df["template_id"] == template_id, "rms"] = rms
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
parser.add_argument("--template_id", type=str, required=True, help="Template ID")
parser.add_argument("--phase", type=str, required=True, help="Phase")
parser.add_argument("--subarray", type=str, required=True, help="Subarray")

parser.add_argument("--arrival_type", type=str, default="sta_lta_stack", help="Arrival type")

template_id = parser.parse_args().template_id
phase = parser.parse_args().phase
subarray = parser.parse_args().subarray
arrival_type = parser.parse_args().arrival_type

print(f"Localizing template event {template_id}...")

# Load travel time volumes
print(f"Loading travel time volumes for {phase} phase and Subarray {subarray}...")
easts_grid, norths_grid, depths_grid, travel_time_dict = load_travel_time_volumes(phase, subarray)

# Load arrival information
print(f"Loading arrival information...")
if arrival_type == "kurtosis_stack":
    filename = f"template_arrivals_kurtosis_stack_{template_id}.csv"
    filepath = Path(dirpath_pick) / filename
    arrival_df = read_csv(filepath, parse_dates = ["arrival_time"])
elif arrival_type == "sta_lta_stack":
    filename = f"template_arrivals_sta_lta_stack_{template_id}.csv"
    filepath = Path(dirpath_detection) / filename
    arrival_df = read_csv(filepath, parse_dates = ["arrival_time"])
else:
    filename = f"template_arrivals_{template_id}.txt"
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
                                                      title = f"Template {template_id}, {phase.upper()} phase")
figname = f"rms_distribution_template{template_id}_{arrival_type}_{phase.lower()}_subarray_{subarray.lower()}.png"
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
save_location_info("template", template_id, arrival_type, phase, 1.0, location_dict, arrival_times_pred, easts_grid, norths_grid, depths_grid, rms_vol)

# Save the location to an CSV file
save_location_to_csv(filepath, template_id, arrival_type, phase, 1.0, subarray, east_min_rms, north_min_rms, depth_min_rms, origin_time_min_rms, min_rms)

print(f"Minimum RMS: {min_rms} s")
print(f"Origin time: {origin_time_min_rms}")
print(f"Location: {east_min_rms} m E, {north_min_rms} m N, {depth_min_rms} m D")



