"""
Plot the inter-station phase differences as a function of time for three selected station pairs as an example
"""

import numpy as np
from json import loads
from numpy import pi
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from os.path import join
from pandas import read_csv
from matplotlib.pyplot import figure

from utils_basic import MT_DIR as indir, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime, GEO_COMPONENTS as components
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, get_mode_order
from utils_plot import add_day_night_shading, component2label, format_east_xlabels, format_north_ylabels, format_datetime_xlabels, format_phase_diff_ylabels, save_figure, get_geo_component_color

### Inputs ###
# Command line arguments
parser = ArgumentParser()
parser.add_argument("--pair1", type=str, help = "The first station pair")
parser.add_argument("--pair2", type=str, help = "The second station pair")
parser.add_argument("--pair3", type=str, help = "The third station pair")

parser.add_argument("--min_cohe", type=float, default=0.85, help="Minimum coherence")
parser.add_argument("--mode_name", type=str, default="PR02549", help="Mode name")
parser.add_argument("--window_length_mt", type=float, default=900.0, help="Window length for the multitaper analysis")

parser.add_argument("--figwidth", type=float, default=12.0, help="Figure width")
parser.add_argument("--widthfrac_map", type=float, default=0.25, help="Fraction of the figure width for the map")
parser.add_argument("--gap_x", type=float, default=0.02, help="Gap between the map and the phase differences")
parser.add_argument("--gap_y", type=float, default=0.02, help="Gap between the phase differences")
parser.add_argument("--margin_x", type=float, default=0.02, help="Margin on the left and right of the figure")
parser.add_argument("--margin_y", type=float, default=0.02, help="Margin on the top and bottom of the figure")

parser.add_argument("--markersize", type=float, default=2.0, help="Marker size for the stations and velocities")

parser.add_argument("--linewidth_sta", type=float, default=0.2, help="Line width for the stations")
parser.add_argument("--linewidth_conn", type=float, default=1.0, help="Line width for the connections")
parser.add_argument("--linewidth_vel", type=float, default=0.5, help="Line width for the velocities")

parser.add_argument("--fontsize_pair_label", type=float, default=12.0, help="Font size for the pair label")
parser.add_argument("--fontsize_legend", type=float, default=10.0, help="Font size for the legend")
parser.add_argument("--fontsize_title", type=float, default=14.0, help="Font size for the title")

parser.add_argument("--major_date_tick_spacing", type=str, default="5d", help="Major date tick spacing")
parser.add_argument("--num_minor_date_ticks", type=int, default=5, help="Number of minor date ticks")

parser.add_argument("--pair_label_x", type=float, default=0.01, help="X coordinate for the pair label")
parser.add_argument("--pair_label_y", type=float, default=0.97, help="Y coordinate for the pair label")

args = parser.parse_args()

pair1 = loads(args.pair1)
pair2 =loads(args.pair2)
pair3 = loads(args.pair3)

window_length_mt = args.window_length_mt
min_cohe = args.min_cohe
mode_name = args.mode_name

figwidth = args.figwidth
widthfrac_map = args.widthfrac_map
gap_x = args.gap_x
gap_y = args.gap_y
margin_x = args.margin_x
margin_y = args.margin_y

markersize = args.markersize

linewidth_sta = args.linewidth_sta
linewidth_conn = args.linewidth_conn
linewidth_vel = args.linewidth_vel

fontsize_pair_label = args.fontsize_pair_label
fontsize_legend = args.fontsize_legend
fontsize_title = args.fontsize_title

major_date_tick_spacing = args.major_date_tick_spacing
num_minor_date_ticks = args.num_minor_date_ticks

pair_label_x = args.pair_label_x
pair_label_y = args.pair_label_y

### Read the station coordinates and station pair informations ###
filename = "delaunay_station_pairs.csv"
filepath = join(indir, filename)

pairs_df = read_csv(filepath)

coord_df = get_geophone_coords()

### Plotting ###
pairs = [pair1, pair2, pair3]

# Compute the plot dimensions
aspect_ratio_map = (max_north - min_north) / (max_east - min_east)
aspect_ratio_fig = aspect_ratio_map * 3 * widthfrac_map / (1 - 2 * margin_y - 2 * margin_y)
figheight = figwidth * aspect_ratio_fig

map_height = (1 - 2 * margin_y - 2 * margin_y) / 3
map_width = map_height * figheight / aspect_ratio_map / figwidth
phase_height = map_height
phase_width = 1 - 2 * margin_x - gap_x - map_width

# Generate the figure
fig = figure(figsize = (figwidth, figheight))

# Plot each pair, with the map on the left and the phase differences on the right
for i, pair in enumerate(pairs):
    station1_pha = pair[0]
    station2_pha = pair[1]

    print(f"Plotting the map for the pair {pair}")
    # Add the subplot for the map

    ax_map = fig.add_axes([margin_x, margin_y + i * (map_height + gap_y), map_width, map_height])

    # Read the phase differences
    filename = f"multitaper_inter_sta_phase_diffs_{mode_name}_{station1_pha}_{station2_pha}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
    filepath = join(indir, filename)
    phase_diff_df = read_csv(filepath, parse_dates=["time"])

    # Plot each station pair while highlighting the one whose phase differences are shown on the right
    for _, row in pairs_df.iterrows():
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

        ax_map.set_aspect("equal")



    # Format the x and y axis labels
    if i == 0:
        format_east_xlabels(ax_map, 
                            plot_axis_label=True, plot_tick_label=True)
    else:
        format_east_xlabels(ax_map, 
                            plot_axis_label=False, plot_tick_label=False)

    format_north_ylabels(ax_map, 
                        plot_axis_label=True, plot_tick_label=True)

    # Add the subplot for the phase differences
    bbox = ax_map.get_position()  # Get the current axis's position
    pos_new = [bbox.x1 + gap_x, bbox.y0, 1-bbox.width-gap_x, bbox.height] 

    ax_pha = fig.add_axes(pos_new)

    legend_dict = {}
    for component in components:
        # Get the color of the component
        color = get_geo_component_color(component)

        # Plot the data
        marker = ax_pha.errorbar(phase_diff_df["time"], phase_diff_df[f"phase_diff_{component.lower()}"], yerr = phase_diff_df[f"phase_diff_uncer_{component.lower()}"], 
                        fmt = "o", markerfacecolor="none", markeredgecolor=color, label = component2label(component), markersize = markersize_vel,
                        markeredgewidth = linewidth_vel, elinewidth = linewidth_vel, capsize=2, zorder=2)

        label = component2label(component)
        legend_dict[label] = marker

    # Add the day-night shading
    add_day_night_shading(ax_pha)

    # Add the pair label
    ax_pha.text(pair_label_x, pair_label_y, f"{station1_pha} - {station2_pha}",
                transform = ax_pha.transAxes,
                fontsize = fontsize_pair_label, fontweight = "bold", ha = "left", va = "top")

    # Set the axis limits
    ax_pha.set_xlim(starttime, endtime)
    ax_pha.set_ylim(-pi, pi)

    # Set the axis labels
    if i == 0:
        format_datetime_xlabels(ax_pha,
                                plot_axis_label=True, plot_tick_label=True,
                                date_format = "%Y-%m-%d",
                                major_tick_spacing = major_date_tick_spacing, num_minor_ticks=num_minor_date_ticks)
    else:
        format_datetime_xlabels(ax_pha,
                                plot_axis_label=False, plot_tick_label=False,
                                date_format = "%Y-%m-%d",
                                major_tick_spacing = major_date_tick_spacing, num_minor_ticks=num_minor_date_ticks)
    
    # Set the y-axis labels
    format_phase_diff_ylabels(ax_pha,
                          plot_axis_label=True, plot_tick_label=True)

    # Add the legend for the phase differences
    if i == 0:
        ax_pha.legend(list(legend_dict.values()), list(legend_dict.keys()), loc="lower right", fontsize=fontsize_legend, frameon=True, edgecolor="black", facecolor="white", framealpha=1.0)

# Add the super title
mode_order = get_mode_order(mode_name)
fig.suptitle(f"Mode {mode_order:d}, 3C phase differences between example geophone pairs", fontsize = fontsize_title, fontweight = "bold", y = 1.01)

### Save the figure ###
figname = f"stationary_resonance_example_mt_inter_sta_phase_diffs_{mode_name}.png"
save_figure(fig, figname)
