"""
Plot the apparent velocities of a hammer shot on geophone triads computed using the P arrival times and multitaper method.
"""

###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from numpy import interp, nan, sin, cos, pi, deg2rad, linspace, histogram, deg2rad, isnan
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure
from rasterio import open
from rasterio.plot import reshape_as_image
from matplotlib.colors import Normalize
from matplotlib import colormaps
from matplotlib import colormaps

from utils_basic import IMAGE_DIR as dirname_img, MT_DIR as dirname_mt, LOC_DIR as dirname_loc
from utils_basic import GEO_COMPONENTS as components, INNER_STATIONS as inner_stations, MIDDLE_STATIONS as middle_stations, OUTER_STATIONS as outer_stations
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, get_borehole_coords
from utils_plot import add_colorbar, save_figure, format_east_xlabels, format_north_ylabels, plot_station_triads, component2label   

### Inputs ###
# Command-line arguments
parser = ArgumentParser(description = "Plot the source properties, including the localization results and the inferred source dimensions.")

parser.add_argument("--hammer_id", type=str, help="The ID of the hammer signal")
parser.add_argument("--freq_target", type=float, help="The target frequency in Hz", default=25.0)
parser.add_argument("--min_cohe", type=float, help="The minimum coherence", default=0.50)
parser.add_argument("--max_vel_app", type=float, help="The maximum apparent velocity", default=3500.0)
parser.add_argument("--scale_factor", type=float, help="The scale factor for the vectors", default=30.0)
parser.add_argument("--colormap_name", type=str, help="The name of the colormap", default="cool")   
parser.add_argument("--markercolor_hammer", type=str, help="The color of the hammer", default="salmon")

# Parse the arguments
args = parser.parse_args()
hammer_id = args.hammer_id
freq_target = args.freq_target
min_cohe = args.min_cohe
max_vel_app = args.max_vel_app
scale_factor = args.scale_factor
colormap_name = args.colormap_name
markercolor_hammer = args.markercolor_hammer

# Constants
filename_image = "maxar_2019-09-17_local.tif"
figwidth = 15.0

wspace = 0.02
margin_x = 0.03
margin_y = 0.03

min_vel_app = 0.0

linewidth_triad = 1.0
linewidth_loc = 0.5

markersize = 150.0

quiver_width = 0.003
quiver_head_width = 6.0
quiver_head_length = 7.0
quiver_linewidth = 0.5

cbar_offset = 0.02

fontsize_title = 12
fontsize_suptitle = 14
suptitle_y = 1.08

###
# Read the input files
###

# Load the station information
print("Loading the station information...")
sta_df = get_geophone_coords()

# Load the station triad information
print("Loading the station triad information...")
inpath = join(dirname_mt, "delaunay_station_triads.csv")
triad_df = read_csv(inpath)

# Load the hammer location
print("Loading the hammer location...")
inpath = join(dirname_loc, "hammer_locations.csv")
loc_df = read_csv(inpath, dtype={"hammer_id": str}, parse_dates=["origin_time"])
east_loc = loc_df[ loc_df["hammer_id"] == hammer_id ]["east"].values[0]
north_loc = loc_df[ loc_df["hammer_id"] == hammer_id ]["north"].values[0]

# Define the stations to plot
stations_to_plot = inner_stations + middle_stations + outer_stations

# Load the apparent velocities computed using the P arrival times
print("Loading the apparent velocities computed using the P arrival times...")
filename = f"hammer_station_triad_app_vels_p_{hammer_id}.csv"
inpath = join(dirname_loc, filename)
vel_p_df = read_csv(inpath)

# Load the apparent velocities computed using the multitaper method
print("Loading the apparent velocities computed using the multitaper method...")
filename = f"hammer_triad_app_vels_{hammer_id}_{freq_target:.0f}hz_min_cohe_{min_cohe:.2f}.csv"
inpath = join(dirname_mt, filename)
vel_mt_df = read_csv(inpath)

# Keep only the triads consisting of inner, middle, and outer stations
vel_p_df = vel_p_df[vel_p_df["station1"].isin(stations_to_plot) &
                    vel_p_df["station2"].isin(stations_to_plot) &
                    vel_p_df["station3"].isin(stations_to_plot)]

vel_mt_df = vel_mt_df[vel_mt_df["station1"].isin(stations_to_plot) &
                      vel_mt_df["station2"].isin(stations_to_plot) &
                      vel_mt_df["station3"].isin(stations_to_plot)]

# Load the satellite image
inpath = join(dirname_img, filename_image)
with open(inpath) as src:
    # Read the image in RGB format
    rgb_band = src.read([1, 2, 3])

    # Reshape the image
    rgb_image = reshape_as_image(rgb_band)

    # Extract the extent of the image
    extent_img = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

###
# Plot the apparent velocities
###

# Compute the figure height
print("Computing the figure height...")
aspect_map = (max_north - min_north) / (max_east - min_east)
figheight = figwidth * aspect_map * (1 - 2 * margin_x - 3 * wspace) / 4 / (1 - 2 * margin_y)

map_width_frac = (1 - 2 * margin_x - 3 * wspace) / 4
map_height_frac = (1 - 2 * margin_y)

# Create the figure
print("Creating the figure...")
fig = figure(figsize = (figwidth, figheight))

# Define the colormap for the velocities
norm = Normalize(vmin = min_vel_app, vmax = max_vel_app)
cmap = colormaps[colormap_name]

# Plot the vectors for each component of the MT apparent velocities
print("Plotting the vectors for each component of the MT apparent velocities...")
for i, component in enumerate(components):
    print(f"Plotting the vectors for the {component} component of the MT apparent velocities...")

    left = margin_x + i * (map_width_frac + wspace)
    bottom = margin_y
    width = map_width_frac
    height = map_height_frac

    ax = fig.add_axes([left, bottom, width, height])

    # Plot the satellite image
    ax.imshow(rgb_image, extent = extent_img, alpha = 0.5)

    # Plot the vector for each station triad
    print("Plotting the vectors for each station triad...")
    triads_to_plot_dicts = []
    for _, row in vel_mt_df.iterrows():
        east = row["east"]
        north = row["north"]
        station1 = row["station1"]
        station2 = row["station2"]
        station3 = row["station3"]

        vel_app = row[f"vel_app_{component.lower()}"]
        vel_app_east = row[f"vel_app_east_{component.lower()}"]
        vel_app_north = row[f"vel_app_north_{component.lower()}"]
        back_azi_std = row[f"back_azi_std_{component.lower()}"]

        if isnan(vel_app):
            continue

        triad_dict = {
            "station1": station1,
            "station2": station2,
            "station3": station3,
        }
        triads_to_plot_dicts.append(triad_dict)

        # Plot the vector
        vec_east = vel_app_east / vel_app
        vec_north = vel_app_north / vel_app
        ax.quiver(east, north, vec_east, vec_north, vel_app,
                  cmap = cmap, norm = norm,
                  scale = scale_factor, width = quiver_width,
                  headwidth = quiver_head_width, headlength = quiver_head_length,
                  linewidth = quiver_linewidth,
                  zorder = 2)
    triads_to_plot_df = DataFrame(triads_to_plot_dicts)

    # Plot the station triads
    print("Plotting the station triads...")

    plot_station_triads(ax, linewidth = linewidth_triad, linecolor = "gray", zorder = 1,
                        triads_to_plot = triads_to_plot_df)

    # Plot the hammer location
    ax.scatter(east_loc, north_loc, color = markercolor_hammer, marker = "*", s = markersize, edgecolor = "black", linewidth = linewidth_loc, zorder = 3)

    # Set the x and y limits
    ax.set_xlim(min_east, max_east)
    ax.set_ylim(min_north, max_north)

    # Set the x and y labels
    format_east_xlabels(ax)

    if i == 0:
        format_north_ylabels(ax)
    else:
        format_north_ylabels(ax, plot_axis_label = False, plot_tick_label = False)

    # Add a title
    title = component2label(component)
    ax.set_title(title, fontsize = fontsize_title, fontweight = "bold")

# Plot the vectors for the P apparent velocities
print("Plotting the vectors for the P apparent velocities...")

left = margin_x + 3 * (map_width_frac + wspace)
bottom = margin_y
width = map_width_frac
height = map_height_frac

ax = fig.add_axes([left, bottom, width, height])

# Plot the satellite image
ax.imshow(rgb_image, extent = extent_img, alpha = 0.5)

# Plot the vector for each station triad
print("Plotting the vectors for each station triad...")
for _, row in vel_p_df.iterrows():
    east = row["east"]
    north = row["north"]
    station1 = row["station1"]
    station2 = row["station2"]
    station3 = row["station3"]

    vel_app = row["vel_app"]
    vel_app_east = row["vel_app_east"]
    vel_app_north = row["vel_app_north"]

    if isnan(vel_app):
        continue

    triad_dict = {
        "station1": station1,
        "station2": station2,
        "station3": station3,
    }

    # Plot the vector
    vec_east = vel_app_east / vel_app
    vec_north = vel_app_north / vel_app
    ax.quiver(east, north, vec_east, vec_north, vel_app,
                cmap = cmap, norm = norm,
                scale = scale_factor, width = quiver_width,
                headwidth = quiver_head_width, headlength = quiver_head_length,
                linewidth = quiver_linewidth,
                zorder = 2)

# Plot the station triads
print("Plotting the station triads...")

plot_station_triads(ax, linewidth = linewidth_triad, linecolor = "gray", zorder = 1,
                    triads_to_plot = vel_p_df)

# Plot the hammer location
ax.scatter(east_loc, north_loc, color = markercolor_hammer, marker = "*", s = markersize, edgecolor = "black", linewidth = linewidth_loc, zorder = 3)

# Set the x and y limits
ax.set_xlim(min_east, max_east)
ax.set_ylim(min_north, max_north)

# Set the x and y labels
format_east_xlabels(ax)
format_north_ylabels(ax, plot_axis_label = False, plot_tick_label = False)

# Add a colorbar
bbox = ax.get_position()
cbar_width = cbar_offset / 2
cax = fig.add_axes([bbox.x1 + cbar_offset, bbox.y0, cbar_width, bbox.height])
add_colorbar(fig, cax, "Velocity (m s$^{-1}$)", cmap = cmap, norm = norm)

# Add a title
ax.set_title("P", fontsize = fontsize_title, fontweight = "bold")

# Add a super title
fig.suptitle(f"Hammer {hammer_id}", fontsize = fontsize_suptitle, fontweight = "bold", y = suptitle_y)

# Save the figure
save_figure(fig, f"hammer_station_triad_app_vels_p_and_mt_{hammer_id}_{freq_target:.0f}hz_min_cohe_{min_cohe:.2f}.png")


