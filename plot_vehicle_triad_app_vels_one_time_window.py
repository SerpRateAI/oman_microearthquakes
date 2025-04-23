"""
Plot the apparent velocities of the vehicle signal on station triads for a time window.
"""

###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from numpy import interp, nan, sin, cos, pi, linspace, histogram, deg2rad, isnan
from pandas import DataFrame, Timestamp
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
from utils_basic import get_geophone_coords, get_geophone_triads
from utils_plot import APPARENT_VELOCITY_LABEL as cbar_label
from utils_plot import add_colorbar, save_figure, format_east_xlabels, format_north_ylabels, plot_station_triads, component2label   

### Inputs ###
# Command-line arguments
parser = ArgumentParser(description = "Plot the apparent velocities of station triads of a hammer shot and a stationary resonance in a time window.")

parser.add_argument("--window_id", type=int, help="The ID of the time window", default=1)
parser.add_argument("--occurrence", type=str, help="The occurrence of the vehicle signal", default="approaching")

parser.add_argument("--scale_factor", type=float, help="The scale factor of the quiver", default=30.0)   
parser.add_argument("--quiver_width", type=float, help="The width of the quiver", default=0.003)
parser.add_argument("--quiver_head_width", type=float, help="The width of the quiver head", default=6.0)
parser.add_argument("--quiver_head_length", type=float, help="The length of the quiver head", default=7.0)
parser.add_argument("--quiver_linewidth", type=float, help="The linewidth of the quiver", default=0.5)

parser.add_argument("--linewidth_triad", type=float, help="The linewidth of the station triads", default=1.0)
parser.add_argument("--linewidth_loc", type=float, help="The linewidth of the hammer location", default=0.5)
parser.add_argument("--markersize", type=float, help="The markersize of the hammer location", default=150.0)

parser.add_argument("--figwidth", type=float, help="The width of the figure", default=15.0)
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
parser.add_argument("--suptitle_y", type=float, help="The y-coordinate of the suptitle", default=1.2)

parser.add_argument("--cmap_name", type=str, help="The name of the colormap", default="plasma")
parser.add_argument("--image_alpha", type=float, help="The alpha of the image", default=0.2)

parser.add_argument("--cbar_width", type=float, help="The width of the colorbar", default=0.01)
parser.add_argument("--cbar_height", type=float, help="The height of the colorbar", default=0.3)
parser.add_argument("--cbar_offset_x", type=float, help="The offset of the colorbar", default=0.01)
parser.add_argument("--cbar_offset_y", type=float, help="The offset of the colorbar", default=0.04)

# Parse the arguments
args = parser.parse_args()
occurrence = args.occurrence
window_id = args.window_id
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
triad_df = get_geophone_triads()

# Keep only the triads consisting of inner, middle, and outer stations
stations_to_plot = inner_stations + middle_stations + outer_stations
triad_df = triad_df[triad_df["station1"].isin(stations_to_plot) &
                    triad_df["station2"].isin(stations_to_plot) &
                    triad_df["station3"].isin(stations_to_plot)]

# Load the time windows
print("Loading the time windows...")
filename = f"vehicle_time_windows_{occurrence}.csv"
filepath = join(dirname_loc, filename)
window_df = read_csv(filepath, parse_dates = ["start_time", "end_time"])

# Load the satellite image
print("Loading the satellite image...")
inpath = join(dirname_img, filename_image)
with open(inpath) as src:
    # Read the image in RGB format
    rgb_band = src.read([1, 2, 3])

    # Reshape the image
    rgb_image = reshape_as_image(rgb_band)

    # Extract the extent of the image
    extent_img = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

# Read the vehicle apparent velocities of the time window
print("Loading the vehicle apparent velocities of the time window...")
filename = f"vehicle_station_triad_app_vels_{occurrence}_window{window_id:d}.csv"
filepath = join(dirname_loc, filename)
vel_df = read_csv(filepath)

###
# Plotting the apparent velocities of the vehicle signal on station triads for all time windows
###

# Compute the figure height
print("Computing the figure height...")
aspect_map = (max_north - min_north) / (max_east - min_east)
figheight = figwidth * aspect_map * (1 - 2 * margin_x - 2 *wspace) / 3 / (1 - 2 * margin_y - hspace)

map_width = (1 - 2 * margin_x - 2 * wspace) / 3
map_height = (1 - 2 * margin_y - hspace)

# Define the colormap for the velocities
norm = Normalize(vmin = min_vel_app, vmax = max_vel_app)
cmap = colormaps[cmap_name]

# Create the figure
print("Creating the figure...")
fig = figure(figsize = (figwidth, figheight))

for i, component in enumerate(components):
    # Create the subplot
    ax = fig.add_axes([margin_x + i * (map_width + wspace), margin_y, map_width, map_height])

    # Plot the satellite image
    ax.imshow(rgb_image, alpha = image_alpha, extent = extent_img)

    # Plot each station triad
    triads_to_plot_dicts = []
    for _, row in vel_df.iterrows():
        station1 = row["station1"]
        station2 = row["station2"]
        station3 = row["station3"]

        # Check if the stations are in stations_to_plot
        if station1 not in stations_to_plot or station2 not in stations_to_plot or station3 not in stations_to_plot:
            continue

        vel_triad_df = vel_df.loc[(vel_df["station1"] == station1) &
                            (vel_df["station2"] == station2) &
                            (vel_df["station3"] == station3)]
        

        # Get the east and north coordinates of the station triad
        east = triad_df.loc[(triad_df["station1"] == station1) &
                            (triad_df["station2"] == station2) &
                            (triad_df["station3"] == station3), "east"].values[0]
        north = triad_df.loc[(triad_df["station1"] == station1) &
                             (triad_df["station2"] == station2) &
                             (triad_df["station3"] == station3), "north"].values[0]

        # Get the average apparent velocity east and north components
        app_vel = vel_triad_df[f"app_vel_{component.lower()}"].values[0]

        if isnan(app_vel):
            continue

        app_vel_east = vel_triad_df[f"app_vel_east_{component.lower()}"].values[0]
        app_vel_north = vel_triad_df[f"app_vel_north_{component.lower()}"].values[0]

        vec_east = app_vel_east / app_vel
        vec_north = app_vel_north / app_vel

        # Plot the vector
        ax.quiver(east, north, vec_east, vec_north, app_vel,
                    cmap = cmap, norm = norm,
                    scale = scale_factor, width = quiver_width,
                    headwidth = quiver_head_width, headlength = quiver_head_length,
                    linewidth = quiver_linewidth,
                    zorder = 2)

        triads_to_plot_dicts.append({"station1": station1, "station2": station2, "station3": station3})

    # Plot the station triads
    print("Plotting the station triads...")
    triads_to_plot_df = DataFrame(triads_to_plot_dicts)
    print(len(triads_to_plot_df))
    plot_station_triads(ax, linewidth = linewidth_triad, linecolor = "gray", triads_to_plot = triads_to_plot_df, zorder = 1)

    # Set the x and y limits
    ax.set_xlim(min_east, max_east)
    ax.set_ylim(min_north, max_north)
    
    # Set the x and y labels
    format_east_xlabels(ax)
    if i == 0:
        format_north_ylabels(ax)
    else:
        format_north_ylabels(ax,
                             plot_axis_label = False,
                             plot_tick_label = False)

    # Add the title
    title = component2label(component)
    ax.set_title(title, fontsize = fontsize_title, fontweight = "bold")

    # Add the colorbar to the bottom left corner
    if i == 0:
        bbox = ax.get_position()
        pos = [bbox.x0 + cbar_offset_x, bbox.y0 + cbar_offset_y, cbar_width, cbar_height]
        add_colorbar(fig, pos, cbar_label,
                    cmap = cmap, norm = norm)
        
# Add the suptitle
start_time = Timestamp(window_df.loc[window_df["window_id"] == window_id, "start_time"].values[0])  
end_time = Timestamp(window_df.loc[window_df["window_id"] == window_id, "end_time"].values[0])

suptitle = f"{occurrence.capitalize()} vehicle, Window {window_id:d}, {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%H:%M:%S')}"
fig.suptitle(suptitle, fontsize = fontsize_suptitle, fontweight = "bold", y = suptitle_y)

###
# Save the figure
###

# Save the figure
print("Saving the figure...")
figname = f"vehicle_triad_app_vels_{occurrence}_window{window_id:d}.png"
save_figure(fig, figname)
