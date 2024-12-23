'''
Plot the map of the stations whose phase differences are measured and the respective observations for the AGU 2024 iPoster
'''

### Importing modules ###
from os.path import join
from json import loads
from argparse import ArgumentParser
from numpy import pi
from pandas import read_csv
from matplotlib.pyplot import subplots

from utils_basic import INNER_STATIONS_A as stations, EASTMIN_A_INNER as min_east, EASTMAX_A_INNER as max_east, NORTHMIN_A_INNER as min_north, NORTHMAX_A_INNER as max_north
from utils_basic import GEO_COMPONENTS as components, MT_DIR as dirname_mt, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_basic import get_borehole_coords, get_geophone_coords
from utils_plot import component2label, format_datetime_xlabels, format_east_xlabels, format_north_ylabels, format_phase_diff_ylabels, get_geo_component_color, save_figure

### Input parameters ###
# Command line arguments
parser = ArgumentParser(description="Plot the map of the stations whose phase differences are derived and the respective observations for the AGU 2024 iPoster")
parser.add_argument("--station_pairs", type=str, help="Station pairs whose phase differences are measured")
parser.add_argument("--color_geo", type=str, default="gold", help="Color of the geophone markers.")
parser.add_argument("--color_highlight", type=str, default="crimson", help="Color of the highlighted geophone markers.")
parser.add_argument("--color_borehole", type=str, default="mediumpurple", help="Color of the borehole markers.")

# Parse the command line arguments
args = parser.parse_args()
station_pairs = loads(args.station_pairs)
color_geo = args.color_geo
color_borehole = args.color_borehole
color_highlight = args.color_highlight

# Constants
mode_name = "PR02549"
mode_order = 2
borehole_name = "BA1A"

figwidth = 20.0
figheight = 6.0

station_label_offsets = (2.0, 2.0)

station_fontsize = 12
legend_fontsize = 10
pair_fontsize = 12
title_fontsize = 12

linewidth_station = 1.0
linewidth_connection = 1.0
linewidth_phase = 0.5

station_size = 100.0
borehole_size = 100.0
phase_diff_size = 2.0

major_dist_tick_spacing = 10.0
num_minor_dist_ticks = 5

major_date_tick_spacing = "5d"
num_minor_date_ticks = 5

gap_hori = 0.05
gap_vert = 0.05

### Plot the map of the inner stations ###
# Get the station coordinates
sta_coord_df = get_geophone_coords()
sta_coord_df = sta_coord_df.loc[stations]

# Get the borehole locations
bor_coord_df = get_borehole_coords()
bor_coord_df = bor_coord_df.loc[borehole_name]

# Get the station to highlight
stations_highlight = set()
for pair in station_pairs:
    stations_highlight.add(pair[0])
    stations_highlight.add(pair[1])

# Create the figure
fig, ax_map = subplots(1, 1, figsize=(figwidth, figheight))

# Plot the stations
for station in stations:
    x = sta_coord_df.loc[station, "east"]
    y = sta_coord_df.loc[station, "north"]
    if station in stations_highlight:
        ax_map.scatter(x, y, s=station_size, c=color_geo, marker="^", facecolors=color_highlight, edgecolors=color_highlight, linewidths=linewidth_station, zorder=3)
        ax_map.annotate(station, (x, y), xytext=(station_label_offsets[0], station_label_offsets[1]), 
                        textcoords="offset points", fontsize=station_fontsize, fontweight="bold", color=color_highlight,
                        va = "bottom", ha = "left", zorder=4)
    else:
        ax_map.scatter(x, y, s=station_size, c=color_geo, marker="^", facecolors=color_geo, edgecolors="black", zorder=3, linewidths=linewidth_station)

# Plot the borehole
x = bor_coord_df["east"]
y = bor_coord_df["north"]
ax_map.scatter(x, y, s=borehole_size, c=color_borehole, marker="o", facecolors=color_borehole, edgecolors="black", zorder=3)
        
# Plot the lines connecting the station pairs
for pair in station_pairs:
    station1 = pair[0]
    station2 = pair[1]
    x1 = sta_coord_df.loc[station1, "east"]
    y1 = sta_coord_df.loc[station1, "north"]
    x2 = sta_coord_df.loc[station2, "east"]
    y2 = sta_coord_df.loc[station2, "north"]
    ax_map.plot([x1, x2], [y1, y2], color=color_highlight, linewidth=linewidth_connection, zorder=2)

# Set the axis limits
ax_map.set_xlim(min_east, max_east)
ax_map.set_ylim(min_north, max_north)

# Set the aspect ratio to be equal
ax_map.set_aspect("equal")

# Set the axis labels
format_east_xlabels(ax_map,
                    major_tick_spacing = major_dist_tick_spacing, num_minor_ticks=num_minor_dist_ticks)

format_north_ylabels(ax_map,
                    major_tick_spacing = major_dist_tick_spacing, num_minor_ticks=num_minor_dist_ticks)

### Plot the phase differences ###
# Compute the panel dimensions
num_pair = len(station_pairs)
bbox = ax_map.get_position()
phase_diff_panel_height = (bbox.height - (num_pair - 1) * gap_vert) / num_pair
phase_diff_panel_width = 1.0 - bbox.x1 - gap_hori

# Plot each station pair
for i, (station1, station2) in enumerate(station_pairs):
    # Generate the subplot
    ax_phase_diff = fig.add_axes([bbox.x1 + gap_hori, bbox.y0 + i * (phase_diff_panel_height + gap_vert), 
                                  phase_diff_panel_width, phase_diff_panel_height])

    # Plot each component
    for component in components:
        # Get the color of the component
        color = get_geo_component_color(component)

        # Read the phase-difference data
        filename = f"mt_cross_station_phase_diffs_{mode_name}_{station1}_{station2}_{component.lower()}.csv"
        inpath = join(dirname_mt, filename)
        phase_diff_df = read_csv(inpath, parse_dates=["time"])

        # Plot the data
        ax_phase_diff.errorbar(phase_diff_df["time"], phase_diff_df["phase_diff"], yerr = phase_diff_df["uncertainty"], 
                               fmt = "o", markerfacecolor="none", markeredgecolor=color, label = component2label(component), markersize = phase_diff_size,
                               markeredgewidth = linewidth_phase, elinewidth = linewidth_phase, capsize=2, zorder=2)

    # Set the axis limits
    ax_phase_diff.set_xlim(starttime, endtime)
    ax_phase_diff.set_ylim(-pi, pi)

    # Set the axis labels
    if i == 0:
        format_datetime_xlabels(ax_phase_diff,
                                plot_axis_label=True, plot_tick_label=True,
                                date_format = "%Y-%m-%d",
                                major_tick_spacing = major_date_tick_spacing, num_minor_ticks=num_minor_date_ticks)
    else:
        format_datetime_xlabels(ax_phase_diff,
                                plot_axis_label=False, plot_tick_label=False,
                                major_tick_spacing = major_date_tick_spacing, num_minor_ticks=num_minor_date_ticks)
        
    # Set the y-axis labels
    if i == 0:
        format_phase_diff_ylabels(ax_phase_diff,
                                  plot_axis_label=True, plot_tick_label=True)
    else:
        format_phase_diff_ylabels(ax_phase_diff,
                                  plot_axis_label=False, plot_tick_label=True)

    # Add the legend
    if i == 0:
        ax_phase_diff.legend(loc="lower right", fontsize=legend_fontsize, frameon=True, edgecolor="black", facecolor="white", framealpha=1.0)

    # Add the station names
    ax_phase_diff.text(0.01, 0.98, f"{station1} - {station2}", fontsize=pair_fontsize, fontweight="bold", 
                       ha = "left", va = "top", transform = ax_phase_diff.transAxes)

    # Add the title
    if i == num_pair - 1:
        ax_phase_diff.set_title("Inter-station phase differences", fontsize=title_fontsize, fontweight="bold")

# Save the figure
figname = "agu_2024_map_n_phase_diffs.png"
save_figure(fig, figname)




