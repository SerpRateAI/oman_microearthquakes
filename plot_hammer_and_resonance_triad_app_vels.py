"""
Plot the apparent velocities of station triads of a hammer shot and a stationary resonance in a time window.
"""

###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from numpy import interp, nan, sin, cos, pi, linspace, histogram, deg2rad, isnan
from pandas import DataFrame, Timedelta
from pandas import read_csv, concat
from matplotlib.pyplot import figure
from matplotlib.cm import ScalarMappable
from rasterio import open
from rasterio.plot import reshape_as_image
from matplotlib.colors import Normalize
from matplotlib import colormaps

from utils_basic import IMAGE_DIR as dirname_img, MT_DIR as dirname_mt, LOC_DIR as dirname_loc
from utils_basic import GEO_COMPONENTS as components, INNER_STATIONS as inner_stations, MIDDLE_STATIONS as middle_stations, OUTER_STATIONS as outer_stations
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, str2timestamp
from utils_basic import get_mode_order
from utils_plot import APPARENT_VELOCITY_LABEL as cbar_label
from utils_plot import add_colorbar, save_figure, format_east_xlabels, format_north_ylabels, plot_station_triads, component2label   

### Inputs ###
# Command-line arguments
parser = ArgumentParser(description = "Plot the apparent velocities of station triads of a hammer shot and a stationary resonance in a time window.")

parser.add_argument("--hammer_id", type=str, help="The ID of the hammer signal")
parser.add_argument("--mode_name", type=str, help="The name of the mode")
parser.add_argument("--center_time", type=str, help="The center time of the time window")
parser.add_argument("--window_length", type=float, help="The length of the time window in seconds")

parser.add_argument("--freq_target", type=float, help="The target frequency of the hammer signal", default=25.0)

parser.add_argument("--min_cohe_hammer", type=float, help="The minimum coherence of the hammer signal", default=0.50)
parser.add_argument("--min_cohe_reson", type=float, help="The minimum coherence of the resonance signal", default=0.85)

parser.add_argument("--scale_factor", type=float, help="The scale factor of the quiver", default=30.0)   
parser.add_argument("--quiver_width", type=float, help="The width of the quiver", default=0.003)
parser.add_argument("--quiver_head_width", type=float, help="The width of the quiver head", default=6.0)
parser.add_argument("--quiver_head_length", type=float, help="The length of the quiver head", default=7.0)
parser.add_argument("--quiver_linewidth", type=float, help="The linewidth of the quiver", default=0.5)

parser.add_argument("--linewidth_triad", type=float, help="The linewidth of the station triads", default=1.0)
parser.add_argument("--linewidth_loc", type=float, help="The linewidth of the hammer location", default=0.5)
parser.add_argument("--markersize", type=float, help="The markersize of the hammer location", default=150.0)

parser.add_argument("--figwidth", type=float, help="The width of the figure", default=10.0)
parser.add_argument("--margin_x", type=float, help="The margin of the figure", default=0.02)
parser.add_argument("--margin_y", type=float, help="The margin of the figure", default=0.02)
parser.add_argument("--wspace", type=float, help="The width of the space between the subplots", default=0.03)
parser.add_argument("--hspace", type=float, help="The height of the space between the subplots", default=0.02)
parser.add_argument("--component_label_x", type=float, help="The x coordinate of the component label", default=0.02)
parser.add_argument("--component_label_y", type=float, help="The y coordinate of the component label", default=0.98)

parser.add_argument("--min_vel_app", type=float, help="The minimum velocity of the apparent velocities", default=0.0)
parser.add_argument("--max_vel_app", type=float, help="The maximum velocity of the apparent velocities", default=3000.0)

parser.add_argument("--fontsize_title", type=float, help="The fontsize of the title", default=14)
parser.add_argument("--fontsize_component", type=float, help="The fontsize of the component label", default=12)
parser.add_argument("--fontsize_suptitle", type=float, help="The fontsize of the suptitle", default=14)
parser.add_argument("--suptitle_y", type=float, help="The y-coordinate of the suptitle", default=1.08)

parser.add_argument("--cmap_name", type=str, help="The name of the colormap", default="plasma")
parser.add_argument("--color_hammer", type=str, help="The color of the hammer", default="lightcoral")
parser.add_argument("--image_alpha", type=float, help="The alpha of the image", default=0.5)

parser.add_argument("--cbar_width", type=float, help="The width of the colorbar", default=0.01)
parser.add_argument("--cbar_height", type=float, help="The height of the colorbar", default=0.1)
parser.add_argument("--cbar_offset_x", type=float, help="The offset of the colorbar", default=0.02)
parser.add_argument("--cbar_offset_y", type=float, help="The offset of the colorbar", default=0.01)

# Parse the arguments
args = parser.parse_args()
hammer_id = args.hammer_id
mode_name = args.mode_name
freq_target = args.freq_target
center_time = str2timestamp(args.center_time)
window_length = args.window_length
min_cohe_hammer = args.min_cohe_hammer
min_cohe_reson = args.min_cohe_reson

figwidth = args.figwidth
margin_x = args.margin_x
margin_y = args.margin_y
wspace = args.wspace
hspace = args.hspace
fontsize_title = args.fontsize_title
fontsize_suptitle = args.fontsize_suptitle
suptitle_y = args.suptitle_y
cmap_name = args.cmap_name
scale_factor = args.scale_factor
quiver_width = args.quiver_width
quiver_head_width = args.quiver_head_width
quiver_head_length = args.quiver_head_length
quiver_linewidth = args.quiver_linewidth
min_vel_app = args.min_vel_app
max_vel_app = args.max_vel_app
linewidth_triad = args.linewidth_triad
linewidth_loc = args.linewidth_loc
markersize = args.markersize
component_label_x = args.component_label_x
component_label_y = args.component_label_y
fontsize_component = args.fontsize_component
image_alpha = args.image_alpha
color_hammer = args.color_hammer
cbar_width = args.cbar_width
cbar_height = args.cbar_height
cbar_offset_x = args.cbar_offset_x
cbar_offset_y = args.cbar_offset_y

###
# Constants
###

filename_image = "maxar_2019-09-17_local.tif"

###
# Read the input files
###

# Load the station information
print("Loading the station information...")
sta_df = get_geophone_coords()

# Load the station triad information
print("Loading the station triad information...")
triad_df = read_csv(join(dirname_mt, "delaunay_station_triads.csv"))

# Load the hammer location
print("Loading the hammer location...")
filename = f"hammer_locations.csv"
filepath = join(dirname_loc, filename)
loc_df = read_csv(filepath, dtype={"hammer_id": str}, parse_dates=["origin_time"])
east_loc = loc_df[ loc_df["hammer_id"] == hammer_id ]["east"].values[0]
north_loc = loc_df[ loc_df["hammer_id"] == hammer_id ]["north"].values[0]

# Keep only the triads consisting of inner, middle, and outer stations
stations_to_plot = inner_stations + middle_stations + outer_stations
triad_df = triad_df[triad_df["station1"].isin(stations_to_plot) &
                    triad_df["station2"].isin(stations_to_plot) &
                    triad_df["station3"].isin(stations_to_plot)]

# Load the apparent velocities
print("Loading the apparent velocities...")
filename = f"hammer_triad_app_vels_{hammer_id}_{freq_target:.0f}hz_min_cohe_{min_cohe_hammer:.2f}.csv"
filepath = join(dirname_mt, filename)
vel_hammer_df = read_csv(filepath)

# Keep only the triads consisting of inner, middle, and outer stations
vel_hammer_df = vel_hammer_df[vel_hammer_df["station1"].isin(stations_to_plot) &
                              vel_hammer_df["station2"].isin(stations_to_plot) &
                              vel_hammer_df["station3"].isin(stations_to_plot)]

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
# Generate the figure
###

# Compute the figure height
print("Computing the figure height...")
aspect_map = (max_north - min_north) / (max_east - min_east)
figheight = figwidth * aspect_map * (1 - 2 * margin_x - wspace) * 3 / 2 / (1 - 2 * margin_y - 2 * hspace)

map_width = (1 - 2 * margin_x - wspace) / 2
map_height = (1 - 2 * margin_y - 2 * hspace) / 3

# Create the figure
print("Creating the figure...")
fig = figure(figsize = (figwidth, figheight))

# Define the colormap for the velocities
norm = Normalize(vmin = min_vel_app, vmax = max_vel_app)
cmap = colormaps[cmap_name]

###
# Plot the hammer apparent velocities on the first column
###

# Plot the hammer apparent velocities
print("Plotting the hammer apparent velocities...")

for i, component in enumerate(components):
    print(f"Plotting the {component} component...")

    left = margin_x
    bottom = 1 - margin_y - map_height - i * (map_height + hspace)
    width = map_width
    height = map_height

    ax = fig.add_axes([left, bottom, width, height])

    # Plot the satellite image
    ax.imshow(rgb_image, extent = extent_img, alpha = image_alpha)


    # Plot the hammer apparent velocities
    print("Plotting the hammer apparent velocities...")
    triads_to_plot_dicts = []
    for _, row in vel_hammer_df.iterrows():
        east = row["east"]
        north = row["north"]
        vel_app = row[f"vel_app_{component.lower()}"]
        vel_app_east = row[f"vel_app_east_{component.lower()}"]
        vel_app_north = row[f"vel_app_north_{component.lower()}"]

        if isnan(vel_app):
            continue

        triads_to_plot_dicts.append(dict(station1 = row["station1"], station2 = row["station2"], station3 = row["station3"]))

        # Plot the hammer apparent velocity
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
    triads_to_plot_df = DataFrame(triads_to_plot_dicts)
    plot_station_triads(ax, sta_df, triad_df, linewidth = linewidth_triad, linecolor = "gray", triads_to_plot = triads_to_plot_df, zorder = 1)

        
    # Add the hammer location
    ax.scatter(east_loc, north_loc, color = color_hammer, marker = "*", s = markersize, edgecolor = "black", linewidth = linewidth_loc, zorder = 3)

    # Set the x and y limits
    ax.set_xlim(min_east, max_east)
    ax.set_ylim(min_north, max_north)
    
    # Set the x and y labels
    if i == 2:
        format_east_xlabels(ax)
    else:
        format_east_xlabels(ax, plot_axis_label = False, plot_tick_label = False)

    format_north_ylabels(ax)

    # Add the component label
    label = component2label(component)
    ax.text(component_label_x, component_label_y, label,
            transform = ax.transAxes,
            fontsize = fontsize_component, fontweight = "bold", va = "top", ha = "left", bbox = dict(facecolor = "white", alpha = 1.0))

    # Add the title
    if i == 0:
        title = f"Hammer {hammer_id}, {freq_target:.0f} Hz"
        ax.set_title(title, fontsize = fontsize_title, fontweight = "bold")

    # Add the colorbar to the bottom left corner
    if i == 2:
        bbox = ax.get_position()
        pos = [bbox.x0 + cbar_offset_x, bbox.y0 + cbar_offset_y, cbar_width, cbar_height]
        add_colorbar(fig, pos, cbar_label,
                     cmap = cmap, norm = norm)

###
# Plot the resonance apparent velocities on the second column
###

# Assemble the apparent velocities of the time window
print("Assembling the apparent velocities of the time window...")
vel_reson_dfs = []
for _, row in triad_df.iterrows():
    station1 = row["station1"]
    station2 = row["station2"]
    station3 = row["station3"]
    east = row["east"]
    north = row["north"]

    # Load the apparent velocities of the triad
    filename = f"station_triad_app_vels_{mode_name}_{station1}_{station2}_{station3}_mt_win{window_length:.0f}s_min_cohe{min_cohe_reson:.2f}.csv"
    filepath = join(dirname_mt, filename)
    vel_reson_df = read_csv(filepath, parse_dates=["time"])

    # Extract the apparent velocities of the time window
    vel_reson_df = vel_reson_df[vel_reson_df["time"] == center_time]

    if len(vel_reson_df) == 0:
        continue
    
    vel_reson_df["east"] = east
    vel_reson_df["north"] = north
    vel_reson_df["station1"] = station1
    vel_reson_df["station2"] = station2
    vel_reson_df["station3"] = station3

    vel_reson_dfs.append(vel_reson_df)

vel_reson_df = concat(vel_reson_dfs)
vel_reson_df.reset_index(drop = True, inplace = True)

# Plot the apparent velocities of the resonance
print("Plotting the apparent velocities of the resonance...")

for i, component in enumerate(components):
    print(f"Plotting the {component} component...")

    left = margin_x + map_width + wspace
    bottom = 1 - margin_y - map_height - i * (map_height + hspace)
    width = map_width
    height = map_height

    ax = fig.add_axes([left, bottom, width, height])

    # Plot the satellite image
    ax.imshow(rgb_image, extent = extent_img, alpha = image_alpha)



    # Plot the apparent velocities of the resonance
    print("Plotting the apparent velocities of the resonance...")
    triads_to_plot_dicts = []
    for _, row in vel_reson_df.iterrows():
        east = row["east"]
        north = row["north"]
        vel_app = row[f"vel_app_{component.lower()}"]
        back_azi = row[f"back_azi_{component.lower()}"]
        station1 = row["station1"]
        station2 = row["station2"]
        station3 = row["station3"]

        if isnan(vel_app):  
            continue

        triads_to_plot_dicts.append(dict(station1 = station1, station2 = station2, station3 = station3))

        vec_east = sin(deg2rad(back_azi))
        vec_north = cos(deg2rad(back_azi))

        # Plot the apparent velocity
        ax.quiver(east, north, vec_east, vec_north, vel_app,
                  cmap = cmap, norm = norm,
                  scale = scale_factor, width = quiver_width,
                  headwidth = quiver_head_width, headlength = quiver_head_length,
                  linewidth = quiver_linewidth,
                  zorder = 2)

    # Plot the station triads
    print("Plotting the station triads...")
    triads_to_plot_df = DataFrame(triads_to_plot_dicts)
    plot_station_triads(ax, sta_df, triad_df, linewidth = linewidth_triad, linecolor = "gray", triads_to_plot = triads_to_plot_df, zorder = 1)

    # Set the x and y limits
    ax.set_xlim(min_east, max_east)
    ax.set_ylim(min_north, max_north)
    
    # Set the x and y labels
    if i == 2:
        format_east_xlabels(ax)
    else:
        format_east_xlabels(ax, plot_axis_label = False, plot_tick_label = False)

    format_north_ylabels(ax, plot_axis_label = False, plot_tick_label = False)

    # Add the component label
    label = component2label(component)
    ax.text(component_label_x, component_label_y, label,
            transform = ax.transAxes,
            fontsize = fontsize_component, fontweight = "bold", va = "top", ha = "left", bbox = dict(facecolor = "white", alpha = 1.0))

    # Add the title
    if i == 0:
        mode_order = get_mode_order(mode_name)
        start_time = center_time - Timedelta(seconds = window_length / 2)
        end_time = center_time + Timedelta(seconds = window_length / 2)
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_time.strftime(" %H:%M:%S")

        title = f"Mode {mode_order}, {start_str}–{end_str}"
        ax.set_title(title, fontsize = fontsize_title, fontweight = "bold")

###
# Save the figure
###

# Save the figure
print("Saving the figure...")
time_str = center_time.strftime("%Y%m%d%H%M%S")
save_figure(fig, f"hammer_and_resonance_triad_app_vels_{hammer_id}_{mode_name}_{time_str}.png")
