"""
Plot the average apparent velocities of the vehicle and the resonance signals for each station triad in a portrait layout.
Only the horizontal component with the smallest apparent velocity variance at each triad is plotted.
"""

###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from json import loads
from numpy import isnan, array, inf
from pandas import DataFrame
from pandas import read_csv
from matplotlib.pyplot import figure
from matplotlib.colors import Normalize
from matplotlib import colormaps

from utils_basic import IMAGE_DIR as dirname_img, LOC_DIR as dirname_loc
from utils_basic import GEO_COMPONENTS as components, INNER_STATIONS as inner_stations, MIDDLE_STATIONS as middle_stations, OUTER_STATIONS as outer_stations
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, get_geophone_triads    
from utils_basic import get_mode_order
from utils_satellite import load_maxar_image
from utils_plot import APPARENT_VELOCITY_LABEL as cbar_label
from utils_plot import add_colorbar, save_figure, format_east_xlabels, format_north_ylabels, plot_station_triads   

### Inputs ###
# Command-line arguments
parser = ArgumentParser(description = "Plot the apparent velocities of station triads of a hammer shot and a stationary resonance in a time window.")

parser.add_argument("--occurrence", type=str, help="The occurrence of the vehicle signal", default="approaching")
parser.add_argument("--mode_name", type=str, help="The name of the mode", default="PR02549")

parser.add_argument("--window_length_reson", type=float, help="The length of the time window of the resonance signal in seconds", default=900.0)
parser.add_argument("--min_num_obs_reson", type=int, help="The minimum number of observations of the resonance signal", default=100)
parser.add_argument("--min_num_obs_vehicle", type=int, help="The minimum number of observations of the vehicle signal", default=3)
parser.add_argument("--min_cohe_reson", type=float, help="The minimum coherence of the resonance signal", default=0.85)
parser.add_argument("--max_back_azi_std_vehicle", type=float, help="The maximum standard deviation of the back azimuth of the vehicle signal", default=10.0)

parser.add_argument("--scale_factor", type=float, help="The scale factor of the quiver", default=30.0)   
parser.add_argument("--quiver_width", type=float, help="The width of the quiver", default=0.003)
parser.add_argument("--quiver_head_width", type=float, help="The width of the quiver head", default=6.0)
parser.add_argument("--quiver_head_length", type=float, help="The length of the quiver head", default=7.0)
parser.add_argument("--quiver_linewidth", type=float, help="The linewidth of the quiver", default=0.5)

parser.add_argument("--linewidth_triad", type=float, help="The linewidth of the station triads", default=1.0)

parser.add_argument("--figwidth", type=float, help="The width of the figure", default=10.0)
parser.add_argument("--margin_x", type=float, help="The margin of the figure", default=0.02)
parser.add_argument("--margin_y", type=float, help="The margin of the figure", default=0.02)
parser.add_argument("--hspace", type=float, help="The height of the space between the subplots", default=0.02)

parser.add_argument("--min_vel_app", type=float, help="The minimum velocity of the apparent velocities", default=0.0)
parser.add_argument("--max_vel_app", type=float, help="The maximum velocity of the apparent velocities", default=4000.0)

parser.add_argument("--fontsize_title", type=float, help="The fontsize of the title", default=14)

parser.add_argument("--cmap_name", type=str, help="The name of the colormap", default="plasma")
parser.add_argument("--image_alpha", type=float, help="The alpha of the image", default=0.2)

parser.add_argument("--cbar_width", type=float, help="The width of the colorbar", default=0.01)
parser.add_argument("--cbar_height", type=float, help="The height of the colorbar", default=0.1)
parser.add_argument("--cbar_offset_x", type=float, help="The offset of the colorbar", default=0.02)
parser.add_argument("--cbar_offset_y", type=float, help="The offset of the colorbar", default=0.01)

parser.add_argument("--pannel_label_x", type=float, help="The x coordinate of the pannel label", default=-0.03)
parser.add_argument("--pannel_label_y", type=float, help="The y coordinate of the pannel label", default=1.03)
parser.add_argument("--pannel_label_fontsize", type=float, help="The fontsize of the pannel label", default=14)

# Parse the arguments
args = parser.parse_args()
occurrence = args.occurrence
mode_name = args.mode_name
window_length_reson = args.window_length_reson
min_num_obs_reson = args.min_num_obs_reson
min_num_obs_vehicle = args.min_num_obs_vehicle
min_cohe_reson = args.min_cohe_reson
max_back_azi_std_vehicle = args.max_back_azi_std_vehicle

figwidth = args.figwidth
margin_x = args.margin_x
margin_y = args.margin_y
hspace = args.hspace
fontsize_title = args.fontsize_title

cmap_name = args.cmap_name
scale_factor = args.scale_factor
quiver_width = args.quiver_width
quiver_head_width = args.quiver_head_width
quiver_head_length = args.quiver_head_length
quiver_linewidth = args.quiver_linewidth
min_vel_app = args.min_vel_app
max_vel_app = args.max_vel_app
linewidth_triad = args.linewidth_triad
image_alpha = args.image_alpha
cbar_width = args.cbar_width
cbar_height = args.cbar_height
cbar_offset_x = args.cbar_offset_x
cbar_offset_y = args.cbar_offset_y

pannel_label_x = args.pannel_label_x
pannel_label_y = args.pannel_label_y
pannel_label_fontsize = args.pannel_label_fontsize

###
# Constants
###

###
# Read the input files
###

# Load the station information
print("Loading the station information...")
sta_df = get_geophone_coords()

# Load the station triad information
print("Loading the station triad information...")
triad_df = get_geophone_triads()

# Keep only the triads consisting of inner, middle, and outer stations
stations_to_plot = inner_stations + middle_stations + outer_stations
triad_df = triad_df[triad_df["station1"].isin(stations_to_plot) &
                    triad_df["station2"].isin(stations_to_plot) &
                    triad_df["station3"].isin(stations_to_plot)]

# Load the vehicle apparent velocities
print("Loading the vehicle apparent velocities...")
filename = f"vehicle_station_triad_avg_app_vels_{occurrence}_min_num_obs{min_num_obs_vehicle:d}_max_back_azi_std{max_back_azi_std_vehicle:.0f}.csv"
filepath = join(dirname_loc, filename)
vel_vehicle_df = read_csv(filepath)

vel_vehicle_df = vel_vehicle_df[vel_vehicle_df["station1"].isin(stations_to_plot) &
                              vel_vehicle_df["station2"].isin(stations_to_plot) &
                              vel_vehicle_df["station3"].isin(stations_to_plot)]

for component in components:
    vel_vehicle_df[f"app_vel_cov_mat_{component.lower()}"] = vel_vehicle_df[f"app_vel_cov_mat_{component.lower()}"].apply(lambda x: array(loads(x)))

# Load the resonance apparent velocities
print("Loading the resonance apparent velocities...")
filename = f"stationary_resonance_station_triad_avg_app_vels_{mode_name}_mt_win{window_length_reson:.0f}s_min_cohe{min_cohe_reson:.2f}_min_num_obs{min_num_obs_reson:d}.csv"
filepath = join(dirname_loc, filename)
vel_reson_df = read_csv(filepath)

vel_reson_df = vel_reson_df[vel_reson_df["station1"].isin(stations_to_plot) &
                            vel_reson_df["station2"].isin(stations_to_plot) &
                            vel_reson_df["station3"].isin(stations_to_plot)]

for component in components:
    vel_reson_df[f"app_vel_cov_mat_{component.lower()}"] = vel_reson_df[f"app_vel_cov_mat_{component.lower()}"].apply(lambda x: array(loads(x)))

# Load the satellite image
rgb_image, extent_img = load_maxar_image()

###
# Generate the figure
###

# Compute the figure height
print("Computing the figure height...")
aspect_map = (max_north - min_north) / (max_east - min_east)
figheight = figwidth * (1 - 2 * margin_x) * aspect_map * 2 / (1 - 2 * margin_y - hspace)

frac_map_width = 1 - 2 * margin_x
frac_map_height = (1 - 2 * margin_y - hspace) / 2

# Create the figure
print("Creating the figure...")
fig = figure(figsize = (figwidth, figheight))

# Define the colormap for the velocities
norm = Normalize(vmin = min_vel_app, vmax = max_vel_app)
cmap = colormaps[cmap_name]

###
# Plot the vehicle apparent velocities on the bottom row
###

# Plot the vehicle apparent velocities
print("Plotting the vehicle apparent velocities...")

left = margin_x
bottom = margin_y
width = frac_map_width
height = frac_map_height

ax = fig.add_axes([left, bottom, width, height])

# Plot the satellite image
ax.imshow(rgb_image, extent = extent_img, alpha = image_alpha)

# Plot the vehicle apparent velocities
triads_to_plot_dicts = []
for _, row in vel_vehicle_df.iterrows():
    station1 = row["station1"]
    station2 = row["station2"]
    station3 = row["station3"]

    east = triad_df.loc[(triad_df["station1"] == station1) &
                        (triad_df["station2"] == station2) &
                        (triad_df["station3"] == station3), "east"].values[0]
    north = triad_df.loc[(triad_df["station1"] == station1) &
                            (triad_df["station2"] == station2) &
                            (triad_df["station3"] == station3), "north"].values[0]

    # Find the horizontal component with the smallest apparent velocity variance
    for i, component in enumerate(["1", "2"]):
        vel_app = row[f"avg_app_vel_{component.lower()}"]
        vel_app_east = row[f"avg_app_vel_east_{component.lower()}"]
        vel_app_north = row[f"avg_app_vel_north_{component.lower()}"]
        back_azi = row[f"avg_back_azi_{component.lower()}"]
        cov_mat = row[f"app_vel_cov_mat_{component.lower()}"]

        if isnan(vel_app_east):
            min_app_vel_var = inf
            continue

        # Calculate apparent velocity variance from covariance matrix
        app_vel_var = cov_mat[0,0] + cov_mat[1,1]

        if i == 0:
            min_app_vel_var = app_vel_var
            vec_east = vel_app_east / vel_app
            vec_north = vel_app_north / vel_app
            vec_amp = vel_app
            back_azi_plot = back_azi
        else:
            if app_vel_var < min_app_vel_var:
                min_app_vel_var = app_vel_var
                vec_east = vel_app_east / vel_app
                vec_north = vel_app_north / vel_app
                vec_amp = vel_app
                back_azi_plot = back_azi

    triads_to_plot_dicts.append({"station1": station1, "station2": station2, "station3": station3})

    ax.quiver(east, north, vec_east, vec_north, vec_amp,
                cmap = cmap, norm = norm,
                scale = scale_factor, width = quiver_width,
                headwidth = quiver_head_width, headlength = quiver_head_length,
                linewidth = quiver_linewidth,
                zorder = 2)
    
# Plot the station triads
print("Plotting the station triads...")
triads_to_plot_df = DataFrame(triads_to_plot_dicts)
plot_station_triads(ax, linewidth = linewidth_triad, linecolor = "gray", triads_to_plot = triads_to_plot_df, zorder = 1)

# Add the pannel label
ax.text(pannel_label_x, pannel_label_y, "(b)",
        transform = ax.transAxes,
        fontsize = pannel_label_fontsize, fontweight = "bold", va = "top", ha = "right")

# Set the x and y limits
ax.set_xlim(min_east, max_east)
ax.set_ylim(min_north, max_north)

# Set the x and y labels
format_east_xlabels(ax)
format_north_ylabels(ax)

# Add the title
title = f"{occurrence.capitalize()} vehicle"
ax.set_title(title, fontsize = fontsize_title, fontweight = "bold")

# Add the colorbar to the bottom left corner
bbox = ax.get_position()
pos = [bbox.x0 + cbar_offset_x, bbox.y0 + cbar_offset_y, cbar_width, cbar_height]
add_colorbar(fig, pos, cbar_label,
                cmap = cmap, norm = norm)

###
# Plot the resonance apparent velocities on the top row
###

# Plot the apparent velocities of the resonance
print("Plotting the apparent velocities of the resonance...")

left = margin_x
bottom = margin_y + frac_map_height + hspace
width = frac_map_width
height = frac_map_height

ax = fig.add_axes([left, bottom, width, height])

# Plot the satellite image
ax.imshow(rgb_image, extent = extent_img, alpha = image_alpha)

# Plot the apparent velocities of the resonance
triads_to_plot_dicts = []
for _, row in vel_reson_df.iterrows():
    station1 = row["station1"]
    station2 = row["station2"]
    station3 = row["station3"]

    east = triad_df.loc[(triad_df["station1"] == station1) &
                        (triad_df["station2"] == station2) &
                        (triad_df["station3"] == station3), "east"].values[0]
    north = triad_df.loc[(triad_df["station1"] == station1) &
                            (triad_df["station2"] == station2) &
                            (triad_df["station3"] == station3), "north"].values[0]
    
    # Find the horizontal component with the smallest apparent velocity variance
    for i, component in enumerate(["1", "2"]):
        vel_app = row[f"avg_app_vel_{component.lower()}"]
        vel_app_east = row[f"avg_app_vel_east_{component.lower()}"]
        vel_app_north = row[f"avg_app_vel_north_{component.lower()}"]
        back_azi = row[f"avg_back_azi_{component.lower()}"]
        cov_mat = row[f"app_vel_cov_mat_{component.lower()}"]

        if isnan(vel_app_east):
            min_app_vel_var = inf
            continue

        # Calculate apparent velocity variance from covariance matrix
        app_vel_var = cov_mat[0,0] + cov_mat[1,1]

        if i == 0:
            min_app_vel_var = app_vel_var
            vec_east = vel_app_east / vel_app
            vec_north = vel_app_north / vel_app
            vec_amp = vel_app
            back_azi_plot = back_azi
        else:
            if app_vel_var < min_app_vel_var:
                min_app_vel_var = app_vel_var
                vec_east = vel_app_east / vel_app
                vec_north = vel_app_north / vel_app
                vec_amp = vel_app
                back_azi_plot = back_azi

    triads_to_plot_dicts.append({"station1": station1, "station2": station2, "station3": station3})

    # Plot the apparent velocity
    ax.quiver(east, north, vec_east, vec_north, vec_amp,
                cmap = cmap, norm = norm,
                scale = scale_factor, width = quiver_width,
                headwidth = quiver_head_width, headlength = quiver_head_length,
                linewidth = quiver_linewidth,
                zorder = 2)

# Plot the station triads
print("Plotting the station triads...")
triads_to_plot_df = DataFrame(triads_to_plot_dicts)
plot_station_triads(ax, linewidth = linewidth_triad, linecolor = "gray", triads_to_plot = triads_to_plot_df, zorder = 1)

# Add the pannel label
ax.text(pannel_label_x, pannel_label_y, "(a)",
        transform = ax.transAxes,
        fontsize = pannel_label_fontsize, fontweight = "bold", va = "top", ha = "right")

# Set the x and y limits
ax.set_xlim(min_east, max_east)
ax.set_ylim(min_north, max_north)

# Set the x and y labels
format_east_xlabels(ax, plot_axis_label = False, plot_tick_label = False)
format_north_ylabels(ax)

# Add the title
mode_order = get_mode_order(mode_name)
title = f"Mode {mode_order}"
ax.set_title(title, fontsize = fontsize_title, fontweight = "bold")

###
# Save the figure
###

# Save the figure
print("Saving the figure...")
save_figure(fig, f"vehicle_and_resonance_triad_app_vels_{occurrence}_{mode_name}_portrait.png")
