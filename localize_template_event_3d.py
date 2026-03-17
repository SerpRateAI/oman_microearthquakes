"""
Localize a template event in 3D using the precomputed travel time volumes
"""

#---------------------------------
# Import libraries
#---------------------------------
from numpy import array, zeros, nan, mean, unravel_index, inf, isnan, sum, ones
from pandas import Timestamp, read_csv, DataFrame, concat
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Union
from utils_basic import (
    PICK_DIR  as dirpath_pick,
    LOC_DIR   as dirpath_loc,
    DETECTION_DIR as dirpath_detection,
    VEL_MODEL_DIR as dirpath_vel,
    get_geophone_coords,
    get_freq_limits_string,
    get_template_subarray
)
from utils_loc import (
    load_travel_time_volumes_combined,
    plot_misfit_distribution, 
    save_location_to_hdf_individual,
    save_location_to_hdf_combined,
    process_arrival_info,
    save_location_to_csv_individual,
)
from utils_snuffler import read_time_windows
from utils_plot import save_figure

#---------------------------------
# Define functions
#---------------------------------

"""
Get the misfit and origin time grids
The misfit is computed as the weighted mean of the L1 misfit.
"""
def get_misfit_and_origin_time_grids(arrival_df, easts_grid, norths_grid, depths_grid, travel_time_dict, weight = False):
    # Loop through the grid points
    misfit_vol = zeros((len(depths_grid), len(norths_grid), len(easts_grid)))
    origin_times_grid = zeros((len(depths_grid), len(norths_grid), len(easts_grid)))
    print(f"Computing misfit and origin time grids...")
    for i_depth, _ in tqdm(enumerate(depths_grid), total=len(depths_grid), desc="Depth"):
        for i_north, _ in enumerate(norths_grid):
            for i_east, _ in enumerate(easts_grid):
                misfit, origin_time = get_misfit_and_origin_time_for_grid_point(arrival_df, i_east, i_north, i_depth, travel_time_dict,
                                                                          weight = weight)
                misfit_vol[i_depth, i_north, i_east] = misfit
                origin_times_grid[i_depth, i_north, i_east] = origin_time

    # Replace the nan values with inf
    misfit_vol[isnan(misfit_vol)] = inf

    return misfit_vol, origin_times_grid

"""
Get the misfit and origin time for one grid point
The misfit is computed as the weighted mean of the L1 misfit.
"""
def get_misfit_and_origin_time_for_grid_point(arrival_df, i_east, i_north, i_depth, travel_time_dict, weight = False):
    if weight:
        weights = arrival_df["uncertainty"]
    else:
        weights = ones(len(arrival_df))

    # Loop through the arrival times
    origin_times = zeros(len(arrival_df))
    travel_times = zeros(len(arrival_df))
    arrival_times = zeros(len(arrival_df))
    for i, row in arrival_df.iterrows():
        station = row["station"]
        travel_time = travel_time_dict[station][i_depth, i_north, i_east]
        if isnan(travel_time):
            return nan, nan
        
        arrival_time = row["arrival_time"]

        origin_time = arrival_time - travel_time
        origin_times[i] = origin_time
        travel_times[i] = travel_time
        arrival_times[i] = arrival_time

    origin_time = mean(origin_times)
    misfit = sum(abs(arrival_times - travel_times - origin_time) / weights) / sum(1 / weights)
    # print(misfit)

    return misfit, origin_time



#---------------------------------
# Define main function
#---------------------------------

# Command line arguments
parser = argparse.ArgumentParser(description="Localize an event in 3D using the precomputed travel time volumes")
parser.add_argument("--template_id", type=str, required=True, help="Template ID")
parser.add_argument("--min_freq_filter", type=float, default=20.0)
parser.add_argument("--max_freq_filter", type=float, default=None)
parser.add_argument("--subarray", type=str, required=True, help="Subarray")
parser.add_argument("--phase", type=str, required=True, help="Phase")
parser.add_argument("--weight", help="Weight the RMS by the uncertainties", default=True)
parser.add_argument("--scale_factor", type=float, required=True, help="Scale factor")
parser.add_argument("--arrival_type", type=str, default="sta_lta_stack", help="Arrival type")

template_id = parser.parse_args().template_id
min_freq_filter = parser.parse_args().min_freq_filter
max_freq_filter = parser.parse_args().max_freq_filter
subarray = parser.parse_args().subarray
phase = parser.parse_args().phase
weight = parser.parse_args().weight
scale_factor = parser.parse_args().scale_factor
arrival_type = parser.parse_args().arrival_type

print(f"Localizing template event {template_id}...")
# Load travel time volumes
subarray = get_template_subarray(template_id)
print(f"Loading travel time volumes for {phase} phase, subarray {subarray}, scale factor {scale_factor}...")
filename = f"travel_time_volumes_{phase.lower()}_{subarray.lower()}.h5"
filepath = Path(dirpath_vel) / filename
easts_grid, norths_grid, depths_grid, travel_time_dict = load_travel_time_volumes_combined(filepath, scale_factor)

# Load arrival information
freq_string = get_freq_limits_string(min_freq_filter, max_freq_filter)
print(f"Loading arrival information...")
print(arrival_type)
if arrival_type == "kurtosis_stack":
    filename = f"template_arrivals_kurtosis_stack_{template_id}_{freq_string}_{phase.lower()}.csv"
    filepath = Path(dirpath_pick) / filename
    arrival_df = read_csv(filepath, parse_dates = ["arrival_time"])
elif arrival_type == "sta_lta_stack":
    filename = f"template_arrivals_sta_lta_stack_{template_id}_{freq_string}_{phase.lower()}.csv"
    filepath = Path(dirpath_detection) / filename
    arrival_df = read_csv(filepath, parse_dates = ["arrival_time"])
elif arrival_type == "manual_stack":
    filename = f"template_arrivals_{template_id}_{arrival_type}_{freq_string}_{phase.lower()}.txt"
    filepath = Path(dirpath_pick) / filename
    arrival_df = read_time_windows(filepath)

arrival_df = process_arrival_info(arrival_df, arrival_type)

# Get the misfit and origin time grids
print(f"Computing misfit and origin time grids...")
misfit_vol, origin_times_grid = get_misfit_and_origin_time_grids(arrival_df, easts_grid, norths_grid, depths_grid, travel_time_dict, weight = weight)

# Get the location with the minimum misfit
print(f"Finding the location with the minimum misfit...")
i_depth_min_rms, i_north_min_rms, i_east_min_rms = unravel_index(misfit_vol.argmin(), misfit_vol.shape)
min_misfit = misfit_vol[i_depth_min_rms, i_north_min_rms, i_east_min_rms]
origin_time_min_misfit = origin_times_grid[i_depth_min_rms, i_north_min_rms, i_east_min_rms]

east_min_rms = easts_grid[i_east_min_rms]
north_min_rms = norths_grid[i_north_min_rms]
depth_min_rms = depths_grid[i_depth_min_rms]

# Plot the misfit distribution
coord_df = get_geophone_coords()
coord_df = coord_df[coord_df.index.isin(arrival_df["station"])]
coord_df = coord_df.reset_index(drop=False)
fig, ax_map, ax_profile, cbar = plot_misfit_distribution(misfit_vol, easts_grid, norths_grid, depths_grid, i_east_min_rms, i_north_min_rms, i_depth_min_rms, coord_df,
                                                      title = f"Template {template_id}, {phase.upper()} phase, scale factor {scale_factor:.1f}")
if weight:
    figname = f"misfit_distribution_template{template_id}_{freq_string}_{arrival_type}_{phase.lower()}_subarray_{subarray.lower()}_scale{scale_factor:.1f}_weight.png"
else:
    figname = f"misfit_distribution_template{template_id}_{freq_string}_{arrival_type}_{phase.lower()}_subarray_{subarray.lower()}_scale{scale_factor:.1f}.png"
save_figure(fig, figname)


# Compute the predicted arrival times
arrival_times_pred_dict = {}
for station in arrival_df["station"].values:
    travel_time = travel_time_dict[station][i_depth_min_rms, i_north_min_rms, i_east_min_rms]
    arrival_times_pred_dict[station] = Timestamp(origin_time_min_misfit + travel_time, unit="s")

# Compute the misfit at each station
misfit_dict = {}
for station in arrival_df["station"].values:
    misfit = arrival_df.loc[arrival_df["station"] == station, "arrival_time"].values[0] - arrival_times_pred_dict[station].timestamp()
    misfit_dict[station] = misfit
    # print(f"Misfit at {station}: {misfit} s")

# Save the location information
origin_time_min_misfit = Timestamp(origin_time_min_misfit, unit="s")
location_dict = {
    "origin_time": origin_time_min_misfit,
    "east": east_min_rms,
    "north": north_min_rms,
    "depth": depth_min_rms,
    "min_misfit": min_misfit
}

# if weight:
#     filename = f"location_info_template_{template_id}_{freq_string}_{arrival_type}_{phase.lower()}_{scale_factor:.1f}_{subarray.lower()}_weight.h5"
# else:
#     filename = f"location_info_template_{template_id}_{freq_string}_{arrival_type}_{phase.lower()}_{scale_factor:.1f}_{subarray.lower()}.h5"

filename = f"location_info_template_{template_id}_{freq_string}.h5"
filepath = Path(dirpath_loc) / filename
print(f"Saving location information to {filepath}...")
param_dict = {
    "arrival_type": arrival_type,
    "phase": phase,
    "subarray": subarray,
    "scale_factor": scale_factor,
    "weight": weight
}
grid_dict = {
    "easts_grid": easts_grid,
    "norths_grid": norths_grid,
    "depths_grid": depths_grid,
    "misfit_vol": misfit_vol
}
arrival_time_dict = arrival_times_pred_dict
station_misfit_dict = misfit_dict
save_location_to_hdf_combined(filepath, param_dict, location_dict, arrival_time_dict, station_misfit_dict, grid_dict)

# Save the location to an CSV file
if weight:
    filename = f"location_info_template_{template_id}_{freq_string}_{arrival_type}_{phase.lower()}_{scale_factor:.1f}_{subarray.lower()}_weight.csv"
else:
    filename = f"location_info_template_{template_id}_{freq_string}_{arrival_type}_{phase.lower()}_{scale_factor:.1f}_{subarray.lower()}.csv"

print(f"Saving location information to {filename}...")
filepath = Path(dirpath_loc) / filename
save_location_to_csv_individual(filepath, template_id, arrival_type, phase, scale_factor, weight, east_min_rms, north_min_rms, depth_min_rms, origin_time_min_misfit, min_misfit)

print(f"Minimum misfit: {min_misfit} s")
print(f"Origin time at minimum misfit: {origin_time_min_misfit}")
print(f"Location: {east_min_rms} m E, {north_min_rms} m N, {depth_min_rms} m D")



