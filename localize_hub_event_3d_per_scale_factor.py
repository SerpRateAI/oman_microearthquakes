"""
Localize a hub event using manual arrival-time picks and precomputed linear-model travel-time
volumes.

The event subarray (A or B) is inferred from station names. Scale factors are read from the
``scale_factors`` dataset in ``event_travel_time_volumes_subarray_{a|b}.h5`` (see
``compute_event_travel_time_volumes_linear_model.py``). Localization is run for each stored scale.

Output HDF5 stores root attributes ``scale_factors`` and ``weight``, with one group per index
named ``scale_factor_{index}``. A CSV with columns ``scale_factor``, ``min_misfit``, and ``depth``
is written to ``LOC_DIR``.
"""

from argparse import ArgumentParser
from numpy import array, unravel_index
from pandas import DataFrame, Timestamp, read_csv
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
    infer_hub_event_subarray,
    load_hub_event_travel_time_scale_factors,
    load_hub_event_travel_times_for_scale,
    plot_misfit_distribution,
    plot_hub_event_min_misfit_and_depth_vs_scale_factor_arrays,
    append_hub_event_location_one_scale_to_multi_scale_hdf,
    get_misfit_and_origin_time_grids_no_phase,
)
from utils_plot import save_figure


parser = ArgumentParser(
    description=(
        "Localize a hub event in 3D using manual picks and linear-model travel-time volumes "
        "(Subarray A or B, single scale-factor sweep)"
    ),
)
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

print("--------------------------------")
print(f"Localizing the hub event for Group {group_label:d}...")
print("--------------------------------")

suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
suffix_repeating_snippet = get_repeating_snippet_suffix(min_cc=min_cc, min_num_similar=min_num_similar_snippet)
suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

print("Getting the hub event information...")
filename = f"event_group_info_{suffix_group}.csv"
filepath = Path(dirpath_event) / filename
group_df = read_csv(filepath)
id_hub = group_df.loc[group_df["label"] == group_label, "id_hub"].values[0]
print(f"Hub event ID: {id_hub}")

filename = f"hub_event_picks_group{group_label:d}_{suffix_group}.mkr"
filepath = Path(dirpath_pick) / filename
arrival_df = read_time_windows(filepath, phase_marker=False)

arrival_df = process_arrival_info(arrival_df, "manual_stack")

subarray = infer_hub_event_subarray(arrival_df)
print(f"Subarray: {subarray}")

scale_factors = load_hub_event_travel_time_scale_factors(subarray, dirpath_vel)
print(f"Velocity scale factors from travel-time file: {scale_factors}")

coord_df = get_geophone_coords()
coord_df = coord_df[coord_df.index.isin(arrival_df["station"])]
coord_df = coord_df.reset_index(drop=False)

filepath_h5 = Path(dirpath_loc) / f"hub_event_location_info_group{group_label:d}.h5"
scale_factors_arr = array(scale_factors, dtype=float)
min_misfits_series = []
depths_series = []

for idx, scale_factor in enumerate(scale_factors):
    print(f"--- Scale factor [{idx:d}] = {scale_factor:.4f} (Subarray {subarray}) ---")

    print(f"Loading the event travel time volumes (scale factor {scale_factor:.2f})...")
    easts_grid, norths_grid, depths_grid, travel_time_dict = load_hub_event_travel_times_for_scale(
        subarray, scale_factor, dirpath_vel,
    )

    print("Computing misfit and origin time grids...")
    misfit_vol, origin_times_grid = get_misfit_and_origin_time_grids_no_phase(
        arrival_df, easts_grid, norths_grid, depths_grid, travel_time_dict, weight=weight,
    )

    print("Finding the location with the minimum misfit...")
    i_north_min_rms, i_east_min_rms, i_depth_min_rms = unravel_index(misfit_vol.argmin(), misfit_vol.shape)
    min_misfit = misfit_vol[i_north_min_rms, i_east_min_rms, i_depth_min_rms]
    origin_time_min_misfit = origin_times_grid[i_north_min_rms, i_east_min_rms, i_depth_min_rms]

    east_min_rms = easts_grid[i_east_min_rms]
    north_min_rms = norths_grid[i_north_min_rms]
    depth_min_rms = depths_grid[i_depth_min_rms]

    title = (
        f"Group {group_label:d}, Hub event {id_hub}, "
        f"Subarray {subarray}, scale factor {scale_factor:.2f}"
    )
    fig, ax_map, ax_profile, cbar = plot_misfit_distribution(
        misfit_vol, easts_grid, norths_grid, depths_grid,
        i_east_min_rms, i_north_min_rms, i_depth_min_rms, coord_df,
        title=title,
    )

    scale_suffix = f"scale{scale_factor:.2f}"
    if weight:
        figname = f"misfit_distribution_group{group_label:d}_{scale_suffix}_weight.png"
    else:
        figname = f"misfit_distribution_group{group_label:d}_{scale_suffix}.png"
    save_figure(fig, figname)

    arrival_times_pred_dict = {}
    for _, row in arrival_df.iterrows():
        station = row["station"]
        travel_time = travel_time_dict[station][i_north_min_rms, i_east_min_rms, i_depth_min_rms]
        arrival_times_pred_dict[station] = Timestamp(origin_time_min_misfit + travel_time, unit="s")

    misfit_dict = {}
    for station in arrival_df["station"].values:
        misfit = (
            arrival_df.loc[arrival_df["station"] == station, "arrival_time"].values[0]
            - arrival_times_pred_dict[station].timestamp()
        )
        misfit_dict[station] = misfit

    origin_time_ts = Timestamp(origin_time_min_misfit, unit="s")
    location_dict = {
        "origin_time": origin_time_ts,
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

    payload = {
        "location_dict": location_dict,
        "arrival_time_dict": arrival_times_pred_dict,
        "station_misfit_dict": misfit_dict,
        "grid_dict": grid_dict,
    }

    append_hub_event_location_one_scale_to_multi_scale_hdf(
        filepath_h5, idx, scale_factors_arr, weight, payload,
    )

    min_misfits_series.append(min_misfit)
    depths_series.append(depth_min_rms)

    print(f"Minimum misfit: {min_misfit} s")
    print(f"Origin time at minimum misfit: {origin_time_ts}")
    print(f"Location: {east_min_rms} m E, {north_min_rms} m N, {depth_min_rms} m D")

scale_factor_sweep_df = DataFrame({
    "scale_factor": scale_factors,
    "min_misfit": min_misfits_series,
    "depth": depths_series,
})
csv_name = f"hub_event_min_misfit_depth_vs_scale_factor_group{group_label:d}.csv"
filepath = Path(dirpath_loc) / csv_name
scale_factor_sweep_df.to_csv(filepath, index=False)
print(f"Wrote scale-factor sweep CSV: {filepath}")

if len(scale_factors) >= 2:
    title_sf = f"Group {group_label:d}, Hub event {id_hub}, Subarray {subarray}"
    fig_sf, _, _ = plot_hub_event_min_misfit_and_depth_vs_scale_factor_arrays(
        scale_factors, min_misfits_series, depths_series, title=title_sf,
    )
    save_figure(fig_sf, f"hub_event_min_misfit_depth_vs_scale_factor_group{group_label:d}.png")
