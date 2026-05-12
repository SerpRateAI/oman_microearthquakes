"""
Localize a hub event using the manual arrival-time picks and the precomputed event travel time
volumes produced by `compute_event_travel_time_volumes.py`.

The travel time volumes are stored in a single HDF5 file per subarray
(`event_travel_time_volumes_{subarray}.h5`) and indexed by a velocity scale factor. The manual
picks do not distinguish between P and S phases, so all arrivals share the same travel-time
volume. Arrays follow the compute script's native ``(north, east, depth)`` layout end to end.
"""

#---------------------------------
# Import libraries
#---------------------------------
from argparse import ArgumentParser
from numpy import unravel_index
from pandas import Timestamp, read_csv
from pathlib import Path

from utils_basic import (
    DETECTION_DIR as dirpath_event,
    PICK_DIR as dirpath_pick,
    VEL_MODEL_DIR as dirpath_vel,
    LOC_DIR as dirpath_loc,
    get_geophone_coords,
    get_freq_limits_string,
)
from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix
from utils_snuffler import read_time_windows
from utils_loc import (
    process_arrival_info,
    load_event_travel_time_volumes,
    plot_misfit_distribution,
    save_hub_event_location_to_hdf,
    get_misfit_and_origin_time_grids_no_phase,
)
from utils_plot import save_figure


#---------------------------------
# Define main function
#---------------------------------

# Command line arguments
parser = ArgumentParser(description="Localize a hub event in 3D using the manual arrival-time picks and the precomputed event travel time volumes")
parser.add_argument("--group_label", type=int, required=True, help="Group label")

parser.add_argument("--min_freq_filter", type=float, default=20.0)
parser.add_argument("--max_freq_filter", type=float, default=None)
parser.add_argument("--min_cc", type=float, default=0.85)
parser.add_argument("--min_num_similar_snippet", type=int, default=10)
parser.add_argument("--min_num_similar_station", type=int, default=3)
parser.add_argument("--sta_window_sec", type=float, default=0.005)
parser.add_argument("--lta_window_sec", type=float, default=0.05)
parser.add_argument("--thr_on", type=float, default=4.0)
parser.add_argument("--thr_off", type=float, default=1.0)

parser.add_argument("--weight", help="Weight the RMS by the uncertainties", action="store_true", default=True)
parser.add_argument("--scale_factor", type=float, default=1.0, help="Velocity-model scale factor used for the travel time volumes")

args = parser.parse_args()
group_label = args.group_label

min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
min_cc = args.min_cc
min_num_similar_snippet = args.min_num_similar_snippet
min_num_similar_station = args.min_num_similar_station
sta_window_sec = args.sta_window_sec
lta_window_sec = args.lta_window_sec
thr_on = args.thr_on
thr_off = args.thr_off
weight = args.weight
scale_factor = args.scale_factor

print("--------------------------------")
print(f"Localizing the hub event for Group {group_label:d}...")
print("--------------------------------")


# Build the suffix
suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
suffix_repeating_snippet = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar_snippet)
suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

# Get the hub event information
print("Getting the hub event information...")
filename = f"event_group_info_{suffix_group}.csv"
filepath = Path(dirpath_event) / filename
group_df = read_csv(filepath)
id_hub = group_df.loc[group_df["label"] == group_label, "id_hub"].values[0]
print(f"Hub event ID: {id_hub}")

# Read the manual arrival-time picks (no phase distinction)
filename = f"hub_event_picks_group{group_label:d}_{suffix_group}.mkr"
filepath = Path(dirpath_pick) / filename
arrival_df = read_time_windows(filepath, phase_marker = False)

arrival_df = process_arrival_info(arrival_df, "manual_stack")

print(arrival_df)

# Get the subarray
subarray = arrival_df["station"][0][0]
print(f"Subarray: {subarray}")

# Load the event travel time volumes for the requested scale factor
filename = f"event_travel_time_volumes_{subarray.lower()}.h5"
filepath = Path(dirpath_vel) / filename

print(f"Loading the event travel time volumes (scale factor {scale_factor:.2f})...")
easts_grid, norths_grid, depths_grid, travel_time_dict = load_event_travel_time_volumes(filepath, scale_factor)

# Get the misfit and origin time grids in native (north, east, depth) layout
print(f"Computing misfit and origin time grids...")
misfit_vol, origin_times_grid = get_misfit_and_origin_time_grids_no_phase(
    arrival_df, easts_grid, norths_grid, depths_grid, travel_time_dict, weight = weight,
)

# Get the location with the minimum misfit
print(f"Finding the location with the minimum misfit...")
i_north_min_rms, i_east_min_rms, i_depth_min_rms = unravel_index(misfit_vol.argmin(), misfit_vol.shape)
min_misfit = misfit_vol[i_north_min_rms, i_east_min_rms, i_depth_min_rms]
origin_time_min_misfit = origin_times_grid[i_north_min_rms, i_east_min_rms, i_depth_min_rms]

east_min_rms = easts_grid[i_east_min_rms]
north_min_rms = norths_grid[i_north_min_rms]
depth_min_rms = depths_grid[i_depth_min_rms]

# Plot the misfit distribution (native (north, east, depth) layout throughout)
coord_df = get_geophone_coords()
coord_df = coord_df[coord_df.index.isin(arrival_df["station"])]
coord_df = coord_df.reset_index(drop=False)

title = f"Group {group_label:d}, Hub event {id_hub}, scale factor {scale_factor:.2f}"
fig, ax_map, ax_profile, cbar = plot_misfit_distribution(
    misfit_vol, easts_grid, norths_grid, depths_grid,
    i_east_min_rms, i_north_min_rms, i_depth_min_rms, coord_df,
    title = title,
)

scale_suffix = f"scale{scale_factor:.2f}"
if weight:
    figname = f"misfit_distribution_group{group_label:d}_{scale_suffix}_weight.png"
else:
    figname = f"misfit_distribution_group{group_label:d}_{scale_suffix}.png"
save_figure(fig, figname)

# Compute the predicted arrival times
arrival_times_pred_dict = {}
for _, row in arrival_df.iterrows():
    station = row["station"]
    travel_time = travel_time_dict[station][i_north_min_rms, i_east_min_rms, i_depth_min_rms]
    arrival_times_pred_dict[station] = Timestamp(origin_time_min_misfit + travel_time, unit="s")

# Compute the misfit at each station
misfit_dict = {}
for station in arrival_df["station"].values:
    misfit = arrival_df.loc[arrival_df["station"] == station, "arrival_time"].values[0] - arrival_times_pred_dict[station].timestamp()
    misfit_dict[station] = misfit


# Save the location information to an HDF5 file
param_dict = {
    "scale_factor": scale_factor,
    "weight": weight,
}

origin_time_min_misfit = Timestamp(origin_time_min_misfit, unit="s")
location_dict = {
    "origin_time": origin_time_min_misfit,
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

filename = f"hub_event_location_info_group{group_label:d}.h5"
filepath = Path(dirpath_loc) / filename
save_hub_event_location_to_hdf(filepath, param_dict, location_dict, arrival_times_pred_dict, misfit_dict, grid_dict)

print(f"Minimum misfit: {min_misfit} s")
print(f"Origin time at minimum misfit: {origin_time_min_misfit}")
print(f"Location: {east_min_rms} m E, {north_min_rms} m N, {depth_min_rms} m D")
