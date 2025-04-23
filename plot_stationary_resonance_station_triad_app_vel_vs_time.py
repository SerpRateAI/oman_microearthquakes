"""
Plot the apparent velocity as a function of time for a station triad
"""
###
# Import the necessary libraries
###

import numpy as np
from json import loads
from numpy import pi
from argparse import ArgumentParser
from matplotlib.pyplot import figure
from os.path import join
from pandas import read_csv

from utils_basic import MT_DIR as indir, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime, GEO_COMPONENTS as components
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, get_mode_order
from utils_plot import add_day_night_shading, component2label, format_east_xlabels, format_north_ylabels, format_datetime_xlabels, format_app_vel_ylabels, format_back_azi_ylabels, save_figure, get_geo_component_color
from utils_plot import plot_station_triads

### 
# Define the command line arguments
###

parser = ArgumentParser()
parser.add_argument("--triad", type=str, help = "The station triad")

parser.add_argument("--min_cohe", type=float, default=0.85, help="Minimum coherence")
parser.add_argument("--mode_name", type=str, default="PR02549", help="Mode name")
parser.add_argument("--window_length_mt", type=float, default=900.0, help="Window length for the multitaper analysis")

parser.add_argument("--figwidth", type=float, default=12.0, help="Figure width")
parser.add_argument("--widthfrac_map", type=float, default=0.25, help="Fraction of the figure width for the map")
parser.add_argument("--wspace_map_vel", type=float, default=0.02, help="Width space between the map and the velocities")
parser.add_argument("--hspace_phase_vel", type=float, default=0.02, help="Height space between the phase differences and the velocities")
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

triad_to_plot = loads(args.triad)

window_length_mt = args.window_length_mt
min_cohe = args.min_cohe
mode_name = args.mode_name

figwidth = args.figwidth
widthfrac_map = args.widthfrac_map
wspace_map_vel = args.wspace_map_vel
hspace_phase_vel = args.hspace_phase_vel
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

###
# Read the input files
###

# Read the station coordinates and station pair informations
filename = "delaunay_station_triads.csv"
filepath = join(indir, filename)

triad_df = read_csv(filepath)
coord_df = get_geophone_coords()

# Read the apparent velocities
station1 = triad_to_plot[0]
station2 = triad_to_plot[1]
station3 = triad_to_plot[2]

# Read the apparent velocities
filename = f"stationary_resonance_station_triad_app_vels_{mode_name}_{station1}_{station2}_{station3}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
filepath = join(indir, filename)
app_vel_df = read_csv(filepath, parse_dates=["time"])

###
# Plotting
###

### Generate the figure ###
# Compute the plot dimensions
aspect_ratio_map = (max_north - min_north) / (max_east - min_east)
aspect_ratio_fig = aspect_ratio_map * widthfrac_map / (1 - 2 * margin_y)
figheight = figwidth * aspect_ratio_fig

map_height = 1 - 2 * margin_y
map_width = widthfrac_map


### Plot the map ###
# Generate the figure
fig = figure(figsize = (figwidth, figheight))

# Add the subplot for the map
print(f"Plotting the map for the triads while highlighting the one whose phase differences are shown on the right..")

# Add the subplot for the map
ax_map = fig.add_axes([margin_x, margin_y, map_width, map_height])

# Plot each station pair while highlighting the one whose phase differences are shown on the right
triad_to_highlight_df = triad_df.loc[(triad_df["station1"] == station1) & (triad_df["station2"] == station2) & (triad_df["station3"] == station3)]
plot_station_triads(ax_map, coord_df, triad_df, linewidth = linewidth_conn, highlight_color = "crimson", zorder = 1, triads_to_highlight = triad_to_highlight_df)

# Set the map limits
ax_map.set_xlim((min_east, max_east))
ax_map.set_ylim((min_north, max_north))

# Set the map aspect ratio
ax_map.set_aspect("equal")

# Format the x and y axis labels
format_east_xlabels(ax_map, 
                    plot_axis_label=True, plot_tick_label=True)

format_north_ylabels(ax_map, 
                    plot_axis_label=True, plot_tick_label=True)

# Add the station triad label
ax_map.text(triad_label_x, triad_label_y, f"{station1} - {station2} - {station3}",
            transform = ax_map.transAxes,
            fontsize = fontsize_pair_label, fontweight = "bold", ha = "left", va = "top")

### Plot the back azimuths and apparent velocities ###
# Add the subplot for the back azimuths
bbox = ax_map.get_position()  # Get the current axis's position
scatter_height = (bbox.height - hspace_phase_vel) / 2
scatter_width = 1 - 2 * margin_x - wspace_map_vel
pos_new = [bbox.x1 + wspace_map_vel, bbox.y0, scatter_width, scatter_height] 

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
format_datetime_xlabels(ax_baz,
                        plot_axis_label=True, plot_tick_label=True,
                        date_format = "%Y-%m-%d",
                        major_tick_spacing = major_date_tick_spacing, num_minor_ticks=num_minor_date_ticks)

# Set the y-axis labels
format_back_azi_ylabels(ax_baz,
                        abbreviation=True,
                        plot_axis_label=True, plot_tick_label=True,
                        major_tick_spacing = major_azi_tick_spacing, num_minor_ticks=num_minor_azi_ticks)

# Add the legend for the apparent velocities
ax_baz.legend(list(legend_dict.values()), list(legend_dict.keys()), loc="upper left", fontsize=fontsize_legend, frameon=True, edgecolor="black", facecolor="white", framealpha=1.0)

# Add the subplot for the apparent velocities
bbox = ax_baz.get_position()  # Get the current axis's position
pos_new = [bbox.x0, bbox.y1 + hspace_phase_vel, scatter_width, scatter_height] 

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
fig.suptitle(f"Mode {mode_order:d}, horizontal apparent velocities and back azimuths for {station1} - {station2} - {station3}", fontsize = fontsize_title, fontweight = "bold", x = 0.6, y = 1.06)

### Save the figure ###
figname = f"stationary_resonance_station_triad_app_vel_vs_time_{mode_name}_{station1}_{station2}_{station3}.png"
save_figure(fig, figname)
