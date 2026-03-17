"""
This script plots the bin count image of the associated STA/LTA events.
"""

# Import libraries
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv
from numpy import histogram2d, linspace
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure

# Import modules
from utils_basic import DETECTION_DIR as dirpath
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, get_freq_limits_string
from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix   
from utils_plot import save_figure, format_east_xlabels, format_north_ylabels

# Read the information of the associated events
parser = ArgumentParser()
parser.add_argument("--bin_size", type=float, default=2.0)
parser.add_argument("--repeating", action="store_true", help="If set, process the repeating snippets only.")
parser.add_argument("--min_cc", type=float, default=0.85)
parser.add_argument("--min_num_similar", type=int, default=10)
parser.add_argument("--sta_window_sec", type=float, default=0.005)
parser.add_argument("--lta_window_sec", type=float, default=0.05)
parser.add_argument("--thr_on", type=float, default=4.0)
parser.add_argument("--thr_off", type=float, default=1.0)
parser.add_argument("--min_freq_filter", type=float, default=20.0)
parser.add_argument("--max_freq_filter", type=float, default=None)
parser.add_argument("--assoc_window_sec", type=float, default=0.1)
parser.add_argument("--min_stations", type=int, default=3)

args = parser.parse_args()
repeating = args.repeating
min_cc = args.min_cc
min_num_similar = args.min_num_similar
sta_window_sec = args.sta_window_sec
lta_window_sec = args.lta_window_sec
thr_on = args.thr_on
thr_off = args.thr_off
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
assoc_window_sec = args.assoc_window_sec
min_stations = args.min_stations
bin_size = args.bin_size

# Get the STA/LTA suffix
suffix = freq_limits_string = get_freq_limits_string(min_freq_filter, max_freq_filter)
sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
suffix += f"_{sta_lta_suffix}"
if repeating:
    suffix = f"repeating_{suffix}"
    repeating_snippet_suffix = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar)
    suffix += f"_{repeating_snippet_suffix}"

# Read the associated events
print("Reading the associated events...")
filename = f"associated_events_{suffix}_window{assoc_window_sec:.3f}s_min_sta{min_stations:d}.csv"
filepath = join(dirpath, filename)
event_df = read_csv(filepath, parse_dates=["first_onset", "event_start", "event_end"])

# Compute the bin count image
print("Computing the bin count image...")
east_bins = linspace(min_east, max_east, int((max_east - min_east) / bin_size) + 1)
north_bins = linspace(min_north, max_north, int((max_north - min_north) / bin_size) + 1)
bin_counts, east_edges, north_edges = histogram2d(event_df["east"], event_df["north"], bins=(east_bins, north_bins))
east_centers = (east_edges[:-1] + east_edges[1:]) / 2
north_centers = (north_edges[:-1] + north_edges[1:]) / 2
bin_counts = bin_counts.T

# Plot the bin count image
print("Plotting the bin count image...")
fig = figure(figsize=(6, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
cmap = colormaps["hot"]


norm = LogNorm(vmin=1, vmax=bin_counts.max(), clip=True)
mappable = ax.pcolormesh(
    east_centers,
    north_centers,
    bin_counts,
    shading="auto",
    cmap=cmap,
    norm=norm,
)

ax.set_xlim(east_centers[0], east_centers[-1])
ax.set_ylim(north_centers[0], north_centers[-1])
ax.set_aspect("equal")

format_east_xlabels(ax)
format_north_ylabels(ax)
ax.set_title("Associated STA/LTA event bin count", fontsize=14, fontweight="bold")

# Plot the station locations
print("Plotting the station locations...")
station_df = get_geophone_coords()
ax.scatter(station_df["east"], station_df["north"], s=30.0, c="lightgray", marker="^", edgecolors="black", linewidths=0.5)

# Add the colorbar
position = ax.get_position()
cbar_width = 0.02
cbar_height = position.height
cbar_position = [position.x1 + 0.02, position.y0, cbar_width, cbar_height]
cax = fig.add_axes(cbar_position)
cbar = fig.colorbar(mappable, cax=cax, orientation="vertical", label="Count", norm=norm, cmap=cmap)

# Save the plot
figname = f"associated_sta_lta_event_bin_count_image_{suffix}_window{assoc_window_sec:.3f}s_min_sta{min_stations:d}.png"
save_figure(fig, figname)