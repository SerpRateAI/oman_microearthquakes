"""
Plot the apparent velocity as a function of time for 3 selected station triads as an example
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
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import add_day_night_shading, component2label, format_east_xlabels, format_north_ylabels, format_datetime_xlabels, format_app_vel_ylabels, format_back_azi_ylabels, save_figure, get_geo_component_color

### Inputs ###
# Command line arguments
parser = ArgumentParser()
parser.add_argument("--triad1", type=str, help = "The first station triad")
parser.add_argument("--triad2", type=str, help = "The second station triad")
parser.add_argument("--triad3", type=str, help = "The third station triad")

parser.add_argument("--min_cohe", type=float, default=0.85, help="Minimum coherence")
parser.add_argument("--mode_name", type=str, default="PR02549", help="Mode name")
parser.add_argument("--window_length_mt", type=float, default=900.0, help="Window length for the multitaper analysis")

parser.add_argument("--figwidth", type=float, default=12.0, help="Figure width")
parser.add_argument("--widthfrac_map", type=float, default=0.25, help="Fraction of the figure width for the map")
parser.add_argument("--gap_x", type=float, default=0.02, help="Gap between the map and the scatter plots")
parser.add_argument("--gap_y_major", type=float, default=0.03, help="Gap between station triads")
parser.add_argument("--gap_y_minor", type=float, default=0.02, help="Gap between apparent velocities and back azimuths")
parser.add_argument("--margin_x", type=float, default=0.02, help="Margin on the left and right of the figure")
parser.add_argument("--margin_y", type=float, default=0.02, help="Margin on the top and bottom of the figure")

parser.add_argument("--markersize", type=float, default=2.0, help="Marker size for the stations and velocities")
parser.add_argument("--linewidth_conn", type=float, default=1.0, help="Line width for the connections")
parser.add_argument("--linewidth_vel", type=float, default=0.5, help="Line width for the velocities")

parser.add_argument("--fontsize_pair_label", type=float, default=12.0, help="Font size for the pair label")
parser.add_argument("--fontsize_legend", type=float, default=10.0, help="Font size for the legend")
parser.add_argument("--fontsize_title", type=float, default=14.0, help="Font size for the title")

parser.add_argument("--major_date_tick_spacing", type=str, default="5d", help="Major date tick spacing")
parser.add_argument("--num_minor_date_ticks", type=int, default=5, help="Number of minor date ticks")

parser.add_argument("--major_vel_tick_spacing", type=float, default=1000.0, help="Major velocity tick spacing")
parser.add_argument("--num_minor_vel_ticks", type=int, default=5, help="Number of minor velocity ticks")

parser.add_argument("--major_azi_tick_spacing", type=float, default=90.0, help="Major azimuth tick spacing")
parser.add_argument("--num_minor_azi_ticks", type=int, default=3, help="Number of minor azimuth ticks")

parser.add_argument("--min_vel", type=float, default=0.0, help="Minimum velocity")
parser.add_argument("--max_vel", type=float, default=4000.0, help="Maximum velocity")

parser.add_argument("--triad_label_x", type=float, default=0.02, help="X coordinate for the triad label")
parser.add_argument("--triad_label_y", type=float, default=0.98, help="Y coordinate for the triad label")

# Parse the command line arguments
args = parser.parse_args()

triad1 = loads(args.triad1)
triad2 = loads(args.triad2)
triad3 = loads(args.triad3)

window_length_mt = args.window_length_mt
min_cohe = args.min_cohe
mode_name = args.mode_name

figwidth = args.figwidth
widthfrac_map = args.widthfrac_map
gap_x = args.gap_x
gap_y_major = args.gap_y_major
gap_y_minor = args.gap_y_minor
margin_x = args.margin_x
margin_y = args.margin_y

markersize = args.markersize
linewidth_conn = args.linewidth_conn
linewidth_vel = args.linewidth_vel

fontsize_pair_label = args.fontsize_pair_label
fontsize_legend = args.fontsize_legend
fontsize_title = args.fontsize_title

major_date_tick_spacing = args.major_date_tick_spacing
num_minor_date_ticks = args.num_minor_date_ticks

major_vel_tick_spacing = args.major_vel_tick_spacing
num_minor_vel_ticks = args.num_minor_vel_ticks

major_azi_tick_spacing = args.major_azi_tick_spacing
num_minor_azi_ticks = args.num_minor_azi_ticks

min_vel = args.min_vel
max_vel = args.max_vel

triad_label_x = args.triad_label_x
triad_label_y = args.triad_label_y

### Read the station coordinates and station pair informations ###
filename = "delaunay_station_triads.csv"
filepath = join(indir, filename)

triads_df = read_csv(filepath)

coord_df = get_geophone_coords()

### Plotting ###
triads = [triad1, triad2, triad3]

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
for i, triad in enumerate(triads):
    station1_vel = triad[0]
    station2_vel = triad[1]
    station3_vel = triad[2]

    print(f"Plotting the map for the triad {triad}")
    # Add the subplot for the map

    ax_map = fig.add_axes([margin_x, margin_y + i * (map_height + gap_y_major), map_width, map_height])

    # Read the apparent velocities
    filename = f"stationary_resonance_station_triad_app_vels_{station1_vel}_{station2_vel}_{station3_vel}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
    filepath = join(indir, filename)
    app_vel_df = read_csv(filepath, parse_dates=["time"])

    # Plot the edges of the highlighted triad first
    east1 = coord_df.loc[station1_vel, "east"]
    north1 = coord_df.loc[station1_vel, "north"]

    east2 = coord_df.loc[station2_vel, "east"]
    north2 = coord_df.loc[station2_vel, "north"]

    east3 = coord_df.loc[station3_vel, "east"]
    north3 = coord_df.loc[station3_vel, "north"]

    ax_map.plot([east1, east2], [north1, north2], linewidth = linewidth_conn, color = "crimson", zorder = 1)
    ax_map.plot([east2, east3], [north2, north3], linewidth = linewidth_conn, color = "crimson", zorder = 1)
    ax_map.plot([east3, east1], [north3, north1], linewidth = linewidth_conn, color = "crimson", zorder = 1)

    plotted_edges = set([tuple(sorted([station1_vel, station2_vel])), tuple(sorted([station2_vel, station3_vel])), tuple(sorted([station3_vel, station1_vel]))])

    # Plot each station pair while highlighting the one whose phase differences are shown on the right
    for _, row in triads_df.iterrows():
        station1 = row["station1"]
        station2 = row["station2"]
        station3 = row["station3"]

        east1 = coord_df.loc[station1, "east"]
        north1 = coord_df.loc[station1, "north"]

        east2 = coord_df.loc[station2, "east"]
        north2 = coord_df.loc[station2, "north"]

        east3 = coord_df.loc[station3, "east"]
        north3 = coord_df.loc[station3, "north"]

        edges = set([tuple(sorted([station1, station2])), tuple(sorted([station2, station3])), tuple(sorted([station3, station1]))])

        for edge in edges:
            if edge not in plotted_edges:
                station1, station2 = edge
                east1 = coord_df.loc[station1, "east"]
                north1 = coord_df.loc[station1, "north"]
                east2 = coord_df.loc[station2, "east"]
                north2 = coord_df.loc[station2, "north"]
                ax_map.plot([east1, east2], [north1, north2], linewidth = linewidth_conn, color = "lightgray", zorder = 1)
                plotted_edges.add(edge)

    # Set the map limits
    ax_map.set_xlim((min_east, max_east))
    ax_map.set_ylim((min_north, max_north))

    # Set the map aspect ratio
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
    
    # Add the station triad label
    ax_map.text(triad_label_x, triad_label_y, f"{station1_vel} - {station2_vel} - {station3_vel}",
                transform = ax_map.transAxes,
                fontsize = fontsize_pair_label, fontweight = "bold", ha = "left", va = "top")

    # Add the subplot for the back azimuths
    bbox = ax_map.get_position()  # Get the current axis's position
    scatter_height = (bbox.height - gap_y_minor) / 2
    scatter_width = 1 - 2 * margin_x - gap_x
    pos_new = [bbox.x1 + gap_x, bbox.y0, scatter_width, scatter_height] 

    ax_baz = fig.add_axes(pos_new)

    legend_dict = {}
    for component in components:
        # Get the color of the component
        color = get_geo_component_color(component)

        # Plot the data
        marker = ax_baz.errorbar(app_vel_df["time"], app_vel_df[f"back_azi_{component.lower()}"], 
                        yerr = app_vel_df[f"back_azi_uncer_{component.lower()}"],
                        fmt = "o", markerfacecolor="none", markeredgecolor=color, label = component2label(component), markersize = markersize,
                        markeredgewidth = linewidth_vel, elinewidth = linewidth_vel, capsize=2, zorder=2)

        label = component2label(component)
        legend_dict[label] = marker

    # Add the day-night shading
    add_day_night_shading(ax_baz)

    # Set the axis limits
    ax_baz.set_xlim(starttime, endtime)
    ax_baz.set_ylim(-180.0, 180.0)

    # Set the axis labels
    if i == 0:
        format_datetime_xlabels(ax_baz,
                                plot_axis_label=True, plot_tick_label=True,
                                date_format = "%Y-%m-%d",
                                major_tick_spacing = major_date_tick_spacing, num_minor_ticks=num_minor_date_ticks)
    else:
        format_datetime_xlabels(ax_baz,
                                plot_axis_label=False, plot_tick_label=False,
                                date_format = "%Y-%m-%d",
                                major_tick_spacing = major_date_tick_spacing, num_minor_ticks=num_minor_date_ticks)
    
    # Set the y-axis labels
    format_back_azi_ylabels(ax_baz,
                            abbreviation=True,
                            plot_axis_label=True, plot_tick_label=True,
                            major_tick_spacing = major_azi_tick_spacing, num_minor_ticks=num_minor_azi_ticks)

    # Add the legend for the apparent velocities
    if i == 2:
        ax_baz.legend(list(legend_dict.values()), list(legend_dict.keys()), loc="upper left", fontsize=fontsize_legend, frameon=True, edgecolor="black", facecolor="white", framealpha=1.0)

    # Add the subplot for the apparent velocities
    bbox = ax_baz.get_position()  # Get the current axis's position
    pos_new = [bbox.x0, bbox.y1 + gap_y_minor, scatter_width, scatter_height] 

    ax_vel = fig.add_axes(pos_new)

    # Plot the apparent velocities
    for component in components:
        color = get_geo_component_color(component)
        marker = ax_vel.errorbar(app_vel_df["time"], app_vel_df[f"vel_app_{component.lower()}"], 
                        yerr = app_vel_df[f"vel_app_uncer_{component.lower()}"],
                        fmt = "o", markerfacecolor="none", markeredgecolor=color, label = component2label(component), markersize = markersize,
                        markeredgewidth = linewidth_vel, elinewidth = linewidth_vel, capsize=2, zorder=2)
    
    # Add the day-night shading
    add_day_night_shading(ax_vel)

    # Set the axis limits
    ax_vel.set_xlim(starttime, endtime)
    ax_vel.set_ylim(min_vel, max_vel)

    # Set the axis labels
    format_datetime_xlabels(ax_vel,
                            plot_axis_label=False, plot_tick_label=False,
                            date_format = "%Y-%m-%d",
                            major_tick_spacing = major_date_tick_spacing, num_minor_ticks=num_minor_date_ticks)
    
    # Set the y-axis labels
    format_app_vel_ylabels(ax_vel,
                            abbreviation=True,
                            plot_axis_label=True, plot_tick_label=True,
                            major_tick_spacing = major_vel_tick_spacing, num_minor_ticks=num_minor_vel_ticks)
    
# Add the super title
mode_order = get_mode_order(mode_name)
fig.suptitle(f"Mode {mode_order:d}, horizontal apparent velocities and back azimuths for example geophone triads", fontsize = fontsize_title, fontweight = "bold", x = 0.6, y = 1.04)

### Save the figure ###
figname = f"stationary_resonance_example_station_triad_app_vels_{mode_name}.png"
save_figure(fig, figname)
