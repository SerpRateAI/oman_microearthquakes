"""
Plot the histograms of the apparent velocities for the stationary resonance and the hammer shots at a geophone triad
"""

###
# Import the necessary libraries
###

import numpy as np
from json import loads
from numpy import pi
from argparse import ArgumentParser
from os.path import join, exists
from pandas import read_csv, DataFrame
from matplotlib.pyplot import subplots, figure
from matplotlib.patches import Rectangle

from utils_basic import LOC_DIR as dirpath_loc, GEO_COMPONENTS as components, MT_DIR as dirpath_mt
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_triads
from utils_plot import get_geo_component_color, save_figure
from utils_plot import format_app_vel_xlabels, format_east_xlabels, format_north_ylabels, component2label, plot_station_triads

###
# Define the command line arguments
###

parser = ArgumentParser()
parser.add_argument("--triad", type=str, help="The station triad")

parser.add_argument("--mode_name", type=str, default="PR02549", help="The mode name")
parser.add_argument("--window_length_mt", type=float, default=900.0, help="The window length for the multitaper")
parser.add_argument("--min_cohe_resonance", type=float, default=0.85, help="The minimum coherence for computing the apparent velocities of the stationary resonance")
parser.add_argument("--min_cohe_hammer", type=float, default=0.50, help="The minimum coherence for computing the apparent velocities of the hammer shots")
parser.add_argument("--freq_target", type=float, default=25.0, help="The target frequency for the apparent velocities")

parser.add_argument("--min_vel_app", type=float, default=0.0, help="The minimum apparent velocity for the histogram")
parser.add_argument("--max_vel_app", type=float, default=4000.0, help="The maximum apparent velocity for the histogram")
parser.add_argument("--num_bins", type=int, default=100, help="The number of bins for the histogram")
parser.add_argument("--max_count", type=int, default=100, help="The maximum number of counts for the histogram")

parser.add_argument("--alpha_rect", type=float, default=0.2, help="The alpha value for the rectangle")
parser.add_argument("--linewidth_mean", type=float, default=2.0, help="The linewidth for the mean")
parser.add_argument("--linewidth_hist", type=float, default=1.0, help="The linewidth for the histogram")
parser.add_argument("--linewidth_triad", type=float, default=1.0, help="The linewidth for the station triads")
parser.add_argument("--linewidth_hammer", type=float, default=0.5, help="The linewidth for the hammer shots")

parser.add_argument("--markersize_hammer", type=float, default=100, help="The markersize for the hammer shots")

parser.add_argument("--fontsize_component_label", type=int, default=14, help="The fontsize for the component labels")
parser.add_argument("--fontsize_title", type=int, default=16, help="The fontsize for the title")
parser.add_argument("--fontsize_axis_label", type=int, default=12, help="The fontsize for the axis labels")

parser.add_argument("--component_label_x", type=float, default=0.01, help="The x-coordinate for the component labels")
parser.add_argument("--component_label_y", type=float, default=0.97, help="The y-coordinate for the component labels")

parser.add_argument("--figwidth", type=float, default=15, help="The width of the figure")
parser.add_argument("--map_width_frac", type=float, default=0.4, help="The width of the map as a fraction of the figure width")
parser.add_argument("--wspace", type=float, default=0.05, help="The width of the space between the map and the histograms")
parser.add_argument("--hspace", type=float, default=0.05, help="The height of the space between the histograms")
parser.add_argument("--margin_x", type=float, default=0.02, help="The margin of the x-axis")
parser.add_argument("--margin_y", type=float, default=0.02, help="The margin of the y-axis")

parser.add_argument("--color_hammer", type=str, default="salmon", help="The color for the hammer shots")    

args = parser.parse_args()

triad = loads(args.triad)

mode_name = args.mode_name
window_length_mt = args.window_length_mt
min_cohe_resonance = args.min_cohe_resonance
min_cohe_hammer = args.min_cohe_hammer
freq_target = args.freq_target

min_vel_app = args.min_vel_app
max_vel_app = args.max_vel_app
num_bins = args.num_bins
max_count = args.max_count

alpha_rect = args.alpha_rect
linewidth_mean = args.linewidth_mean
linewidth_hist = args.linewidth_hist

fontsize_component_label = args.fontsize_component_label
fontsize_title = args.fontsize_title
fontsize_axis_label = args.fontsize_axis_label

component_label_x = args.component_label_x
component_label_y = args.component_label_y

figwidth = args.figwidth
map_width_frac = args.map_width_frac
wspace = args.wspace
hspace = args.hspace
margin_x = args.margin_x
margin_y = args.margin_y

color_hammer = args.color_hammer
linewidth_triad = args.linewidth_triad
linewidth_hammer = args.linewidth_hammer
markersize_hammer = args.markersize_hammer

###
# Read the input files
###
print(f"Reading the input files...")

station1 = triad[0]
station2 = triad[1]
station3 = triad[2]

# Read the station triads
triad_df = get_geophone_triads()

# Read the apparent velocities for the stationary resonance
filename = f"stationary_resonance_station_triad_app_vels_{mode_name}_{station1}_{station2}_{station3}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe_resonance:.2f}.csv"
filepath = join(dirpath_mt, filename)
app_vel_reson_df = read_csv(filepath, parse_dates=["time"])

# Read the hammer-shot locations
filename = "hammer_locations.csv"
filepath = join(dirpath_loc, filename)
hammer_df = read_csv(filepath, dtype={"hammer_id": str})

###
# Compute the standard deviation of the hammer apparent velocities at the station triad
###
print(f"Computing the standard deviation of the hammer apparent velocities at the station triad...")
# Get the apparent velocities of the hammer shots at the station triad
app_vel_dicts = []
for _, row in hammer_df.iterrows():
    hammer_id = row["hammer_id"]

    # Read the hammer-shot apparent velocities
    filename = f"hammer_triad_app_vels_{hammer_id}_{freq_target:.0f}hz_min_cohe_{min_cohe_hammer:.2f}.csv"
    filepath = join(dirpath_mt, filename)

    if not exists(filepath):
        print(f"File {filepath} does not exist. Skipping the hammer shot {hammer_id}.")
        continue

    app_vel_df = read_csv(filepath)

    app_vel_dict = {}
    for component in components:
        app_vel = app_vel_df.loc[(app_vel_df["station1"] == station1) & (app_vel_df["station2"] == station2) & (app_vel_df["station3"] == station3), f"vel_app_{component.lower()}"].values[0]
        app_vel_dict[component] = app_vel
    
    app_vel_dicts.append(app_vel_dict)

app_vel_df = DataFrame(app_vel_dicts)

# Compute the standard deviation for each component
mean_dict = {}
std_dict = {}
for component in components:
    mean_dict[component] = app_vel_df[component].mean()
    std_dict[component] = app_vel_df[component].std()

###
# Compute the figure dimensions
###
map_aspect_ratio = (max_north - min_north) / (max_east - min_east)
figheight = figwidth * map_width_frac * map_aspect_ratio / (1 - margin_y)

###
# Plot the map  
###
print(f"Plotting the map...")

# Generate the figure and the axes
fig = figure(figsize=(figwidth, figheight))
ax_map = fig.add_axes([margin_x, margin_y, map_width_frac, 1 - 2 * margin_y])

# Plot the station triads
triad_highlight_df = triad_df[(triad_df["station1"] == station1) & (triad_df["station2"] == station2) & (triad_df["station3"] == station3)]
plot_station_triads(ax_map, triads_to_highlight = triad_highlight_df, linewidth = linewidth_triad)

ax_map.set_xlim(min_east, max_east)
ax_map.set_ylim(min_north, max_north)
ax_map.set_aspect("equal")

# Set the label axis labels
format_east_xlabels(ax_map, plot_axis_label=True, plot_tick_label=True)
format_north_ylabels(ax_map, plot_axis_label=True, plot_tick_label=True)

# Plot the hammer shots
for _, row in hammer_df.iterrows():
    east = row["east"]
    north = row["north"]
    ax_map.scatter(east, north, marker="*", s=markersize_hammer, color=color_hammer, edgecolor="black", linewidth=linewidth_hammer)

###
# Plot the histograms of the apparent velocities
###
print(f"Plotting the histograms of the apparent velocities...")

# Compute the height of the histograms
hist_height_frac = (1 - 2 * margin_y - 2 * hspace) / 3
hist_width_frac = 1 - 2 * margin_x - wspace - map_width_frac

# Plot the histograms of the apparent velocities for each component
for i, component in enumerate(components):
    # Plot the stationary resonance
    ax_hist = fig.add_axes([margin_x + map_width_frac + wspace, margin_y + i * (hist_height_frac + hspace), hist_width_frac, hist_height_frac])
    color = get_geo_component_color(component)
    ax_hist.hist(app_vel_reson_df[f"vel_app_{component.lower()}"], 
            bins=num_bins, 
            range=(min_vel_app, max_vel_app), 
            density=False,
            color=color, 
            edgecolor="black",
            linewidth=linewidth_hist)
    
    # Set the y-axis limits
    ax_hist.set_ylim(0, max_count)

    # Plot the hammer shots
    mean_vel = mean_dict[component]
    std_vel = std_dict[component]
    ax_hist.axvline(mean_vel, color=color, label="Mean", linewidth=linewidth_mean)
    ax_hist.add_patch(Rectangle((mean_vel - std_vel, 0), 2 * std_vel, max_count, color=color, alpha=alpha_rect))

    # Set the x-axis limits
    ax_hist.set_xlim(min_vel_app, max_vel_app)

    # Set the x axis labels
    if i == 0:
        format_app_vel_xlabels(ax_hist, plot_axis_label=True, plot_tick_label=True)
    else:
        format_app_vel_xlabels(ax_hist, plot_axis_label=False, plot_tick_label=False)

    # Set the y-axis labels
    ax_hist.set_ylabel("Count", fontsize=fontsize_axis_label)

    # Plot the component label
    ax_hist.text(component_label_x, component_label_y, component2label(component),
            transform=ax_hist.transAxes,
            fontsize=fontsize_component_label, fontweight="bold", ha="left", va="top")

# Set the title
fig.suptitle(f"Station triad: {station1}-{station2}-{station3}", fontsize=fontsize_title, fontweight="bold", y=1.05)

###
# Save the figure
###

filename = f"stationary_resonance_and_hammer_app_vel_histograms_{mode_name}_{station1}_{station2}_{station3}.png"
save_figure(fig, filename)






