
"""
This script is used to associate matched events for one manual template.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict
from numpy import asarray, arange
from pandas import DataFrame
from matplotlib.pyplot import Axes, figure
from matplotlib.gridspec import GridSpec

from utils_basic import GEO_CHANNELS as channels, GEO_COMPONENTS as components, SAMPLING_RATE as sampling_rate, ROOTDIR_GEO as dirpath_geo, PICK_DIR as dirpath_pick, DETECTION_DIR as dirpath_det
from utils_cc import TemplateMatches, associate_matched_events, get_normalized_time_lags, plot_station_lag_time_histogram
from utils_plot import save_figure

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("--template_id", type=str, help="Template ID")
parser.add_argument("--cc_threshold", type=float, default=0.85, help="Cross-correlation threshold")
parser.add_argument("--max_num_unmatched_sta", type=int, default=0, help="Maximum number of unmatched stations to consider a matched event")

parser.add_argument("--stations_to_plot", type=str, nargs="+", default=["A01", "A02", "A03", "A04", "A05", "A06"], help="Stations to plot")
parser.add_argument("--subplot_height", type=float, default=3.0, help="Height of each subplot")
parser.add_argument("--subplot_width", type=float, default=4.0, help="Width of each subplot")

parser.add_argument("--min_lag", type=float, default=-1e-2, help="Minimum lag time (s)")
parser.add_argument("--max_lag", type=float, default=1e-2, help="Maximum lag time (s)")
parser.add_argument("--bin_width", type=float, default=2 * 1 / sampling_rate, help=f"Bin width (s), default to 2 * 1 / {sampling_rate:.0f} = {2 * 1 / sampling_rate:.3f}")

parser.add_argument("--min_freq_filter", type=float, help="The low corner frequency for filtering the data", default=20.0)

parser.add_argument("--dirpath_det", type=str, help="The directory path for the detection results", default="/vortexfs1/home/tianze.liu/slurm_oman/detection_results")

args = parser.parse_args()
template_id = args.template_id
stations_to_plot = args.stations_to_plot
cc_threshold = args.cc_threshold
max_num_unmatched_sta = args.max_num_unmatched_sta
min_lag = args.min_lag
max_lag = args.max_lag
bin_width = args.bin_width
subplot_height = args.subplot_height
subplot_width = args.subplot_width
min_freq_filter = args.min_freq_filter

# Load the match information
print(f"Loading the data for {template_id}...")
filepath = Path(dirpath_det) / f"template_matches_manual_templates_freq{min_freq_filter:.0f}hz_cc{cc_threshold:.2f}.h5"
tm_dict = {}
for station in stations_to_plot:
    tm = TemplateMatches.from_hdf(filepath, id=template_id, station=station)
    tm_dict[station] = tm

for station, tm in tm_dict.items():
    print(station)
    print(tm.template.starttime)

# Associate the events
print(f"Associating the events for {template_id}...")
num_sta = len(tm_dict)
min_matched_sta = num_sta - max_num_unmatched_sta
record_df = associate_matched_events(tm_dict, min_matched_sta)

num_matched_events = len(record_df.index.unique(level=0))
print(f"Number of matched events: {num_matched_events}")

# Compute the normalized time lags
print(f"Computing the normalized time lags for {template_id}...")
record_df = get_normalized_time_lags(tm_dict, record_df)

# Save the record dataframe
filename = f"matched_events_manual_templates_freq{min_freq_filter:.0f}hz_cc{cc_threshold:.2f}_num_unmatch{max_num_unmatched_sta:d}.h5"
outpath = Path(dirpath_det) / filename

record_out_df = record_df.reset_index() # Reset the index to make it a regular dataframe
print(record_out_df.head())
record_out_df.to_hdf(outpath, key=f"template_{template_id}", mode="w")
print(f"Saved the record dataframe to {outpath}")

# Plot the histograms of the normalized time lags
print(f"Plotting the histograms of the normalized time lags for {template_id}...")
## Compute the number of rows
num_rows = len(stations_to_plot) // 3 + 1

## Generate the subplots
fig = figure(figsize=(subplot_width * 3, subplot_height * num_rows))
gs = GridSpec(num_rows, 3, figure=fig, hspace=0.5)

for i, station in enumerate(stations_to_plot):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    plot_station_lag_time_histogram(ax, record_df, station, min_lag, max_lag, bin_width)

    ax.set_ylim(0, num_matched_events + 2)

## Set the subplot titles
fig.suptitle(f"Template {template_id}, num. unmatched sta. = {max_num_unmatched_sta:d}", fontsize=14, fontweight="bold", y = 0.95)

## Save the figure
save_figure(fig, f"template_match_lag_time_histograms_{template_id}_cc{cc_threshold:.2f}_num_unmatch{max_num_unmatched_sta:d}.png")





