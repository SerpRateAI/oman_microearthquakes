"""
This script bins the associated events by hour.
"""

# Import libraries
from os.path import join
from argparse import ArgumentParser
from pandas import read_json, to_datetime, date_range, cut, read_csv
from matplotlib.pyplot import figure
from matplotlib import colormaps

# Import modules
from utils_basic import DETECTION_DIR as dirpath
from utils_basic import STARTTIME_GEO as starttime_bin, ENDTIME_GEO as endtime_bin
from utils_basic import get_geophone_coords, get_freq_limits_string
from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix   
from utils_plot import save_figure, format_datetime_xlabels, add_day_night_shading

# Read the information of the associated events
parser = ArgumentParser()   
parser.add_argument("--min_cc", type=float, default=0.85)
parser.add_argument("--min_num_similar_snippet", type=int, default=10)
parser.add_argument("--min_num_similar_station", type=int, default=3)
parser.add_argument("--sta_window_sec", type=float, default=0.005)
parser.add_argument("--lta_window_sec", type=float, default=0.05)
parser.add_argument("--thr_on", type=float, default=4.0)
parser.add_argument("--thr_off", type=float, default=1.0)
parser.add_argument("--min_freq_filter", type=float, default=20.0)
parser.add_argument("--max_freq_filter", type=float, default=None)
parser.add_argument("--assoc_window_sec", type=float, default=0.1)
parser.add_argument("--min_stations", type=int, default=3)

args = parser.parse_args()
min_cc = args.min_cc
min_num_similar_snippet = args.min_num_similar_snippet
min_num_similar_station = args.min_num_similar_station
sta_window_sec = args.sta_window_sec
lta_window_sec = args.lta_window_sec
thr_on = args.thr_on
thr_off = args.thr_off
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter

# Get the suffices
suffix = freq_limits_string = get_freq_limits_string(min_freq_filter, max_freq_filter)
sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
suffix += f"_{sta_lta_suffix}"
repeating_snippet_suffix = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar_snippet)
suffix += f"_{repeating_snippet_suffix}"

# Read the information of all associated events
print("Reading all associated events...")
filename = f"associated_events_repeating_{suffix}.jsonl"
filepath = join(dirpath, filename)
event_df = read_json(filepath, lines = True)

# Define the bin edges
bin_edges = date_range(starttime_bin, endtime_bin, freq="h")
bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2

# Bin the events by hour
event_df["hour"] = cut(event_df["first_onset"], bins=bin_edges, labels=bin_centers)
hourly_counts = event_df.groupby("hour", observed=False).size().reindex(bin_centers, fill_value=0)

# Save the results
print("Saving the results...")
filename = f"hourly_counts_associated_events_repeating_{suffix}.csv"
filepath = join(dirpath, filename)
hourly_counts.index.name = "hour"
hourly_counts.name = "count"
hourly_counts.to_csv(filepath, index=True)
print(f"Saved hourly counts of all events to {filepath}")

# Read the event group information
suffix_group = suffix + f"_num_sim_sta{min_num_similar_station:d}"
filename = f"event_group_info_{suffix_group}.csv"
filepath = join(dirpath, filename)
group_info_df = read_csv(filepath)

for i_group, group_label in enumerate(group_info_df["label"]):
    filename = f"grouped_events_group{group_label}_{suffix_group}.jsonl"
    filepath = join(dirpath, filename)
    event_df = read_json(filepath, lines = True)

    # Bin the events by hour
    event_df["hour"] = cut(event_df["first_onset"], bins=bin_edges, labels=bin_centers)
    hourly_counts = event_df.groupby("hour", observed=False).size().reindex(bin_centers, fill_value=0)

    # Save the results
    print("Saving the results...")
    filename = f"hourly_counts_grouped_events_group{group_label}_{suffix_group}.csv"
    filepath = join(dirpath, filename)
    hourly_counts.index.name = "hour"
    hourly_counts.name = "count"
    hourly_counts.to_csv(filepath, index=True)
    print(f"Saved hourly counts of events of group {group_label} to {filepath}")