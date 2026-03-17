"""
Plot the propagation direction vs time for a station triad
"""
###
# Import the necessary libraries
###

import numpy as np
from json import loads
from numpy import pi, deg2rad, pi, linspace, histogram
from argparse import ArgumentParser
from matplotlib.pyplot import figure
from os.path import join
from pandas import read_csv

from utils_basic import MT_DIR as dirpath_mt, LOC_DIR as dirpath_loc, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime, GEO_COMPONENTS as components
from utils_basic import get_geophone_coords, get_mode_order
from utils_plot import add_day_night_shading, component2label, format_east_xlabels, format_north_ylabels, format_datetime_xlabels, format_app_vel_ylabels, format_back_azi_ylabels, save_figure, get_geo_component_color
from utils_plot import plot_station_triads

###
# Helper functions
###

def plot_rose_diagram(ax, app_vel_df, component):
    # Get the data
    back_azis = app_vel_df[f"back_azi_{component.lower()}"].values
    angles = deg2rad(back_azis)

    # Define bins (e.g., 10-degree intervals)
    num_bins = 72  # 5-degree bins
    bins = linspace(-pi, pi, num_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2

    # Compute histogram
    counts, _ = histogram(angles, bins=bins)

    # Plot the rose diagram in the inset
    ax.bar(centers, counts, width=(2 * pi / num_bins), align='center', color=get_geo_component_color(component), edgecolor='k', alpha = 0.5)
    ax.grid(True, linestyle=":")

    ax.set_theta_zero_location("N")  # North at the top
    ax.set_theta_direction(-1)  # Clockwise

    ax.set_thetagrids([0, 90, 180, 270], labels=["N", "E", "S", "W"], fontsize = 12, fontweight = "bold")

    ax.set_yticklabels([])
    ax.set_rlabel_position(90)

    return ax
### 
# Define the command line arguments
###

parser = ArgumentParser()
parser.add_argument("--triad1", type=str, help = "The first station triad")
parser.add_argument("--triad2", type=str, help = "The second station triad")

parser.add_argument("--min_cohe", type=float, default=0.85, help="Minimum coherence")
parser.add_argument("--mode_name", type=str, default="PR02549", help="Mode name")
parser.add_argument("--window_length_mt", type=float, default=900.0, help="Window length for the multitaper analysis")

parser.add_argument("--figheight", type=float, default=5.0, help="Figure height")
parser.add_argument("--aspect_ratio", type=float, default=0.25, help="Aspect ratio of the baz vs time plot")
parser.add_argument("--hspace", type=float, default=0.1, help="Space between the two axes")
parser.add_argument("--wspace", type=float, default=0.05, help="Space between the two plots")

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

window_length_mt = args.window_length_mt
min_cohe = args.min_cohe
mode_name = args.mode_name

figheight = args.figheight
aspect_ratio = args.aspect_ratio
wspace = args.wspace
hspace = args.hspace
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
filepath = join(dirpath_loc, filename)

triad_df = read_csv(filepath)
coord_df = get_geophone_coords()

# Read the apparent velocities
station1_triad1 = triad1[0]
station2_triad1 = triad1[1]
station3_triad1 = triad1[2]

station1_triad2 = triad2[0]
station2_triad2 = triad2[1]
station3_triad2 = triad2[2]

# Read the apparent velocities of the first triad
filename = f"stationary_resonance_station_triad_app_vels_{mode_name}_{station1_triad1}_{station2_triad1}_{station3_triad1}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
filepath = join(dirpath_loc, filename)
app_vel1_df = read_csv(filepath, parse_dates=["time"])

# Read the apparent velocities of the second triad
filename = f"stationary_resonance_station_triad_app_vels_{mode_name}_{station1_triad2}_{station2_triad2}_{station3_triad2}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
filepath = join(dirpath_loc, filename)
app_vel2_df = read_csv(filepath, parse_dates=["time"])

###
# Plotting
###

### Generate the figure ###
# Compute the plot dimensions
heightfrac_time = (1 - 2 * margin_y - hspace) / 2
width_time = figheight * heightfrac_time / aspect_ratio
width_polar = figheight * heightfrac_time
figwidth = (width_time + width_polar) / (1 -  2 * margin_x - wspace) 
widthfrac_time = width_time / figwidth
widthfrac_polar = width_polar / figwidth

# Generate the figure
fig = figure(figsize = (figwidth, figheight))

### Make the first Baz vs Time plot ###
# Generate the axes for the first baz vs time plot
ax_time1 = fig.add_axes([margin_x, margin_y, widthfrac_time, heightfrac_time])

# Add the subplot for the first Baz vs Time plot
print(f"Plotting the first Baz vs Time plot for the triad {station1_triad1} - {station2_triad1} - {station3_triad1}..")

legend_dict = {}
for component in components:
    # Get the color of the component
    color = get_geo_component_color(component)

    # Plot the data
    marker = ax_time1.errorbar(app_vel1_df["time"], app_vel1_df[f"back_azi_{component.lower()}"], 
                    yerr = app_vel1_df[f"back_azi_uncer_{component.lower()}"],
                    fmt = "o", markerfacecolor="none", markeredgecolor=color, label = component2label(component), markersize = markersize,
                    markeredgewidth = linewidth_vel, elinewidth = linewidth_vel, capsize=2, zorder=2)

    label = component2label(component)
    legend_dict[label] = marker

# Add the day-night shading
add_day_night_shading(ax_time1)

# Set the axis limits
ax_time1.set_xlim(starttime, endtime)
ax_time1.set_ylim(-180.0, 180.0)

# Set the axis labels
format_datetime_xlabels(ax_time1,
                        plot_axis_label=True, plot_tick_label=True,
                        date_format = "%Y-%m-%d",
                        major_tick_spacing = major_date_tick_spacing, num_minor_ticks=num_minor_date_ticks)

# Set the y-axis labels
format_back_azi_ylabels(ax_time1,
                        abbreviation=True,
                        plot_axis_label=True, plot_tick_label=True,
                        major_tick_spacing = major_azi_tick_spacing, num_minor_ticks=num_minor_azi_ticks)

# Add the legend for the apparent velocities

ax_time1.text(-0.07, 1.1, "(b)", transform = ax_time1.transAxes, fontsize = 14, fontweight = "bold", ha = "left", va = "top")

# Add the super title
mode_order = get_mode_order(mode_name)
ax_time1.set_title(f"Triad {station1_triad1} - {station2_triad1} - {station3_triad1}", fontsize = fontsize_title, fontweight = "bold", x = 0.5, y = 1.04)

### Make the first polar plot ###
position_time = ax_time1.get_position()
ax_polar1 = fig.add_axes([position_time.x1 + wspace, position_time.y0, widthfrac_polar, heightfrac_time], projection = "polar")

for component in components:
    ax_polar1 = plot_rose_diagram(ax_polar1, app_vel1_df, component)

### Make the second Baz vs Time plot ###
# Generate the axes for the second baz vs time plot
ax_time2 = fig.add_axes([margin_x, margin_y + hspace + heightfrac_time, widthfrac_time, heightfrac_time])

print(f"Plotting the second Baz vs Time plot for the triad {station1_triad2} - {station2_triad2} - {station3_triad2}..")

legend_dict = {}
for component in components:
    # Get the color of the component
    color = get_geo_component_color(component)

    # Plot the data
    marker = ax_time2.errorbar(app_vel2_df["time"], app_vel2_df[f"back_azi_{component.lower()}"], 
                    yerr = app_vel2_df[f"back_azi_uncer_{component.lower()}"],
                    fmt = "o", markerfacecolor="none", markeredgecolor=color, label = component2label(component), markersize = markersize,
                    markeredgewidth = linewidth_vel, elinewidth = linewidth_vel, capsize=2, zorder=2)

    label = component2label(component)
    legend_dict[label] = marker

# Add the day-night shading
add_day_night_shading(ax_time2)

# Set the axis limits
ax_time2.set_xlim(starttime, endtime)
ax_time2.set_ylim(-180.0, 180.0)

# Set the axis labels
format_datetime_xlabels(ax_time2,
                        plot_axis_label=False, plot_tick_label=False,
                        date_format = "%Y-%m-%d",
                        major_tick_spacing = major_date_tick_spacing, num_minor_ticks=num_minor_date_ticks)

# Set the y-axis labels
format_back_azi_ylabels(ax_time2,
                        abbreviation=True,
                        plot_axis_label=True, plot_tick_label=True,
                        major_tick_spacing = major_azi_tick_spacing, num_minor_ticks=num_minor_azi_ticks)

# Add the legend for the apparent velocities
ax_time2.legend(list(legend_dict.values()), list(legend_dict.keys()), loc="upper left", fontsize=fontsize_legend, frameon=True, edgecolor="black", facecolor="white", framealpha=1.0)

ax_time2.text(-0.07, 1.1, "(a)", transform = ax_time2.transAxes, fontsize = 14, fontweight = "bold", ha = "left", va = "top")

# Add the super title
mode_order = get_mode_order(mode_name)
ax_time2.set_title(f"Triad {station1_triad2} - {station2_triad2} - {station3_triad2}", fontsize = fontsize_title, fontweight = "bold", x = 0.5, y = 1.04)

### Make the second polar plot ###
position_time = ax_time2.get_position()
ax_polar2 = fig.add_axes([position_time.x1 + wspace, position_time.y0, widthfrac_polar, heightfrac_time], projection = "polar")

for component in components:
    ax_polar2 = plot_rose_diagram(ax_polar2, app_vel2_df, component)


### Save the figure ###
figname = f"jgr_station_triad_back_azi_vs_time.png"
save_figure(fig, figname)