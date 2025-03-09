"""
Plot the phase differences between the three components of a pair of stations measured using the multitaper method
"""

### Imports ###
from os.path import join
from json import loads
from argparse import ArgumentParser
from numpy import pi
from pandas import read_csv
from matplotlib.pyplot import subplots
from matplotlib.gridspec import GridSpec

from utils_basic import GEO_COMPONENTS as components, MT_DIR as dirname_mt, SPECTROGRAM_DIR as dirname_spec, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords
from utils_plot import component2label, format_datetime_xlabels, format_east_xlabels, format_north_ylabels, format_phase_diff_ylabels, get_geo_component_color, save_figure

### Input parameters ###
# Command line arguments
parser = ArgumentParser(description="Plot the inter-station 3C phase differences between a pair of geophone stations")
parser.add_argument("--station1", type=str, help="Station 1")
parser.add_argument("--station2", type=str, help="Station 2")
parser.add_argument("--window_length_mt", type=float, help="MT window length in second")
parser.add_argument("--min_cohe", type=float, help="Minimum coherence")

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")

# Parse the command line arguments
args = parser.parse_args()
station1 = args.station1
station2 = args.station2
window_length_mt = args.window_length_mt
min_cohe = args.min_cohe

mode_name = args.mode_name

# Constants
figwidth = 12
figheight = 4

ax_gap = 0.07

markersize_sta = 15.0
markersize_pha = 3.0

fontsize_legend = 10.0
fontsize_title = 12.0

linewidth_sta = 0.2
linewidth_conn = 1.0
linewidth_pha = 0.5

major_date_tick_spacing = "5d"
num_minor_date_ticks = 5

print(f"Plotting the 3C phase differences between {station1} and {station2}...")

### Read the station coordinates and station pair informations ###
filename = "delaunay_station_pairs.csv"
filepath = join(dirname_mt, filename)

sta_pair_df = read_csv(filepath)
distance = sta_pair_df.loc[(sta_pair_df["station1"] == station1) & (sta_pair_df["station2"] == station2)]["distance"].values[0]

coord_df = get_geophone_coords()

### Read the phase differences ###
filename = f"multitaper_inter_sta_phase_diffs_{mode_name}_{station1}_{station2}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
filepath = join(dirname_mt, filename)
phase_diff_df = read_csv(filepath, parse_dates=["time"])

### Plot the station map on the left ###
fig, ax_map = subplots(1, 1, figsize = (figwidth, figheight))

# Plot the stations
ax_map.scatter(coord_df["east"], coord_df["north"], markersize_sta, marker = "^", facecolor = "lightgray", edgecolor="black", linewidth = linewidth_sta, zorder = 2)

# Plot all connectors while highlighting the one whose phase differences are shown on the right
station1_pha = station1
station2_pha = station2

for _, row in sta_pair_df.iterrows():
    station1 = row["station1"]
    station2 = row["station2"]

    east1 = coord_df.loc[station1, "east"]
    north1 = coord_df.loc[station1, "north"]

    east2 = coord_df.loc[station2, "east"]
    north2 = coord_df.loc[station2, "north"]

    if station1 == station1_pha and station2 == station2_pha:
        ax_map.plot([east1, east2], [north1, north2], linewidth = linewidth_conn, color = "crimson", zorder = 1)
    else:
        ax_map.plot([east1, east2], [north1, north2], linewidth = linewidth_conn, color = "lightgray", zorder = 1)

ax_map.set_xlim((min_east, max_east))
ax_map.set_ylim((min_north, max_north))

format_east_xlabels(ax_map)
format_north_ylabels(ax_map)

ax_map.set_aspect("equal")

### Plot the phase differences on the right ###
bbox = ax_map.get_position()  # Get the current axis's position
pos_new = [bbox.x1 + ax_gap, bbox.y0, 1-bbox.width-ax_gap, bbox.height] 

ax_pha = fig.add_axes(pos_new)

for component in components:
    # Get the color of the component
    color = get_geo_component_color(component)

    # Plot the data
    ax_pha.errorbar(phase_diff_df["time"], phase_diff_df[f"phase_diff_{component.lower()}"], yerr = phase_diff_df[f"phase_diff_uncer_{component.lower()}"], 
                               fmt = "o", markerfacecolor="none", markeredgecolor=color, label = component2label(component), markersize = markersize_pha,
                               markeredgewidth = linewidth_pha, elinewidth = linewidth_pha, capsize=2, zorder=2)

# Set the axis limits
ax_pha.set_xlim(starttime, endtime)
ax_pha.set_ylim(-pi, pi)

# Set the axis labels
format_datetime_xlabels(ax_pha,
                        plot_axis_label=True, plot_tick_label=True,
                        date_format = "%Y-%m-%d",
                        major_tick_spacing = major_date_tick_spacing, num_minor_ticks=num_minor_date_ticks)
    
# Set the y-axis labels
format_phase_diff_ylabels(ax_pha,
                          plot_axis_label=True, plot_tick_label=True)

# Add the legend
ax_pha.legend(loc="lower right", fontsize=fontsize_legend, frameon=True, edgecolor="black", facecolor="white", framealpha=1.0)

# Add the station names
ax_pha.set_title(f"{station1_pha}-{station2_pha}, {distance:.1f} m", fontsize = fontsize_title, fontweight = "bold")

### Save the figure ###
figname = f"multitaper_inter_sta_phase_diffs_{mode_name}_{station1_pha}_{station2_pha}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.png"
save_figure(fig, figname)
