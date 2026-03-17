"""
Localize a template event in 3D using both P and S arrival times
"""

#---------------------------------
# Import libraries
#---------------------------------
from numpy import array, zeros, nan, mean, unravel_index, inf, isnan, sum, ones, concatenate
from pandas import Timestamp, read_csv, DataFrame, concat
import argparse
from tqdm import tqdm
from pathlib import Path

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
    load_travel_time_volumes_individual,
    load_travel_time_volumes_combined,
    plot_misfit_distribution, 
    save_location_to_hdf_individual,
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
def get_misfit_and_origin_time_grids(arrival_p_df, arrival_s_df, easts_grid, norths_grid, depths_grid, travel_time_p_dict, travel_time_s_dict, weight = True):
    # Loop through the grid points
    misfit_vol = zeros((len(depths_grid), len(norths_grid), len(easts_grid)))
    origin_times_grid = zeros((len(depths_grid), len(norths_grid), len(easts_grid)))
    print(f"Computing misfit and origin time grids...")
    for i_depth, _ in tqdm(enumerate(depths_grid), total=len(depths_grid), desc="Depth"):
        for i_north, _ in enumerate(norths_grid):
            for i_east, _ in enumerate(easts_grid):
                misfit, origin_time = get_misfit_and_origin_time_for_grid_point(arrival_p_df, arrival_s_df, i_east, i_north, i_depth, travel_time_p_dict, travel_time_s_dict,
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
def get_misfit_and_origin_time_for_grid_point(arrival_p_df, arrival_s_df, i_east, i_north, i_depth, travel_time_p_dict, travel_time_s_dict, weight = True):
    if weight:
        uncertainties_p = arrival_p_df["uncertainty"].values
        uncertainties_s = arrival_s_df["uncertainty"].values
    else:
        uncertainties_p = ones(len(arrival_p_df))
        uncertainties_s = ones(len(arrival_s_df))

    # Loop through the P arrival times
    origin_times_p = zeros(len(arrival_p_df))
    travel_times_p = zeros(len(arrival_p_df))
    arrival_times_p = zeros(len(arrival_p_df))
    for i, row in arrival_p_df.iterrows():
        station = row["station"]
        travel_time = travel_time_p_dict[station][i_depth, i_north, i_east]
        if isnan(travel_time):
            return nan, nan
        
        arrival_time = row["arrival_time"]

        origin_time = arrival_time - travel_time
        origin_times_p[i] = origin_time
        travel_times_p[i] = travel_time
        arrival_times_p[i] = arrival_time

    # Loop through the S arrival times
    origin_times_s = zeros(len(arrival_s_df))
    travel_times_s = zeros(len(arrival_s_df))
    arrival_times_s = zeros(len(arrival_s_df))
    for i, row in arrival_s_df.iterrows():
        station = row["station"]
        travel_time = travel_time_s_dict[station][i_depth, i_north, i_east]
        if isnan(travel_time):
            return nan, nan
        
        arrival_time = row["arrival_time"]
        origin_time = arrival_time - travel_time
        origin_times_s[i] = origin_time
        travel_times_s[i] = travel_time
        arrival_times_s[i] = arrival_time

    # Compute the origin time
    origin_times = concatenate((origin_times_p, origin_times_s))
    uncertainties = concatenate((uncertainties_p, uncertainties_s))
    origin_time = sum(origin_times / uncertainties) / sum(1 / uncertainties)
    travel_times = concatenate((travel_times_p, travel_times_s))
    arrival_times = concatenate((arrival_times_p, arrival_times_s))

    misfit = sum(abs(arrival_times - travel_times - origin_time) / uncertainties) / sum(1 / uncertainties)
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
parser.add_argument("--weight", help="Weight the RMS by the uncertainties", default=True)
parser.add_argument("--arrival_type", type=str, default="manual_stack", help="Arrival type")

template_id = parser.parse_args().template_id
min_freq_filter = parser.parse_args().min_freq_filter
max_freq_filter = parser.parse_args().max_freq_filter
weight = parser.parse_args().weight
arrival_type = parser.parse_args().arrival_type

print(f"Localizing template event {template_id}...")
# Load the travel time volumes for P and S phases
subarray = get_template_subarray(template_id)
print(f"Loading travel time volumes for P and S phases, subarray {subarray}...")
filename = f"travel_time_volumes_p_{subarray.lower()}_scale1.0.h5"
filepath = Path(dirpath_vel) / filename
easts_grid, norths_grid, depths_grid, travel_time_p_dict = load_travel_time_volumes_individual(filepath)
filename = f"travel_time_volumes_s_{subarray.lower()}_scale1.0.h5"
filepath = Path(dirpath_vel) / filename
easts_grid, norths_grid, depths_grid, travel_time_s_dict = load_travel_time_volumes_individual(filepath)

# Load arrival information for P and S phases
freq_string = get_freq_limits_string(min_freq_filter, max_freq_filter)
print(f"Loading arrival information...")

# Load arrival information for P phase
print(f"Loading arrival information for P phase...")
filename = f"template_arrivals_{template_id}_{arrival_type}_{freq_string}_p.txt"
filepath = Path(dirpath_pick) / filename
arrival_p_df = read_time_windows(filepath)
arrival_p_df = process_arrival_info(arrival_p_df, arrival_type)

# Load arrival information for S phase
filename = f"template_arrivals_{template_id}_{arrival_type}_{freq_string}_s.txt"
filepath = Path(dirpath_pick) / filename
arrival_s_df = read_time_windows(filepath)
arrival_s_df = process_arrival_info(arrival_s_df, arrival_type)


# Get the misfit and origin time grids
print(f"Computing misfit and origin time grids...")
misfit_vol, origin_times_grid = get_misfit_and_origin_time_grids(arrival_p_df, arrival_s_df, easts_grid, norths_grid, depths_grid, travel_time_p_dict, travel_time_s_dict, weight = weight)

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
coord_df = coord_df[coord_df.index.isin(arrival_p_df["station"])]
coord_df = coord_df.reset_index(drop=False)
fig, ax_map, ax_profile, cbar = plot_misfit_distribution(misfit_vol, easts_grid, norths_grid, depths_grid, i_east_min_rms, i_north_min_rms, i_depth_min_rms, coord_df,
                                                      title = f"Template {template_id}, P and S phases, subarray {subarray.lower()}")
if weight:
    figname = f"misfit_distribution_template{template_id}_{freq_string}_{arrival_type}_p_and_s_subarray_{subarray.lower()}_weight.png"
else:
    figname = f"misfit_distribution_template{template_id}_{freq_string}_{arrival_type}_p_and_s_subarray_{subarray.lower()}.png"
save_figure(fig, figname)


# # Compute the predicted arrival times
# arrival_times_pred_p_dict = {}
# arrival_times_pred_s_dict = {}
# for station in arrival_p_df["station"].values:
#     travel_time = travel_time_p_dict[station][i_depth_min_rms, i_north_min_rms, i_east_min_rms]
#     arrival_times_pred_p_dict[station] = Timestamp(origin_time_min_misfit + travel_time, unit="s")

# for station in arrival_s_df["station"].values:
#     travel_time = travel_time_s_dict[station][i_depth_min_rms, i_north_min_rms, i_east_min_rms]
#     arrival_times_pred_s_dict[station] = Timestamp(origin_time_min_misfit + travel_time, unit="s")

# # Compute the misfit at each station
# misfit_dict = {}
# for station in arrival_p_df["station"].values:
#     misfit = arrival_p_df.loc[arrival_p_df["station"] == station, "arrival_time"].values[0] - arrival_times_pred_p_dict[station].timestamp()
#     misfit_dict[station] = misfit
#     # print(f"Misfit at {station}: {misfit} s")
# for station in arrival_s_df["station"].values:
#     misfit = arrival_s_df.loc[arrival_s_df["station"] == station, "arrival_time"].values[0] - arrival_times_pred_s_dict[station].timestamp()
#     misfit_dict[station] = misfit
#     # print(f"Misfit at {station}: {misfit} s")

# # Save the location information
# origin_time_min_misfit = Timestamp(origin_time_min_misfit, unit="s")
# location_dict = {
#     "origin_time": origin_time_min_misfit,
#     "east": east_min_rms,
#     "north": north_min_rms,
#     "depth": depth_min_rms,
#     "min_misfit": min_misfit
# }

# if weight:
#     filename = f"location_info_template_{template_id}_{freq_string}_{arrival_type}_{phase.lower()}_{scale_factor:.1f}_{subarray.lower()}_weight.h5"
# else:
#     filename = f"location_info_template_{template_id}_{freq_string}_{arrival_type}_{phase.lower()}_{scale_factor:.1f}_{subarray.lower()}.h5"

# print(f"Saving location information to {filename}...")
# filepath = Path(dirpath_loc) / filename
# save_location_to_hdf_individual(filepath, "template", template_id, arrival_type, phase, subarray, scale_factor, weight, location_dict, arrival_times_pred_dict, misfit_dict, easts_grid, norths_grid, depths_grid, misfit_vol)

# # Save the location to an CSV file
# if weight:
#     filename = f"location_info_template_{template_id}_{freq_string}_{arrival_type}_{phase.lower()}_{scale_factor:.1f}_{subarray.lower()}_weight.csv"
# else:
#     filename = f"location_info_template_{template_id}_{freq_string}_{arrival_type}_{phase.lower()}_{scale_factor:.1f}_{subarray.lower()}.csv"

# print(f"Saving location information to {filename}...")
# filepath = Path(dirpath_loc) / filename
# save_location_to_csv_individual(filepath, template_id, arrival_type, phase, scale_factor, weight, east_min_rms, north_min_rms, depth_min_rms, origin_time_min_misfit, min_misfit)

print(f"Minimum misfit: {min_misfit} s")
print(f"Origin time at minimum misfit: {origin_time_min_misfit}")
print(f"Location: {east_min_rms} m E, {north_min_rms} m N, {depth_min_rms} m D")



