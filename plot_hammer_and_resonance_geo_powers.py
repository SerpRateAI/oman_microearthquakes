"""
Plot the spectral powers of a hammer shot at a target frequency and a stationary resonance at a target time window.
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

from utils_basic import SPECTROGRAM_DIR as dirname_spec, IMAGE_DIR as dirname_img, MT_DIR as dirname_mt, LOC_DIR as dirname_loc
from utils_basic import GEO_COMPONENTS as components, GEO_STATIONS as stations
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, str2timestamp
from utils_basic import get_mode_order
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import add_colorbar, save_figure, format_east_xlabels, format_north_ylabels, plot_station_triads, component2label   

###
# Input arguments
###

parser = ArgumentParser(description = "Plot the apparent velocities of station triads of a hammer shot and a stationary resonance in a time window.")

parser.add_argument("--hammer_id", type=str, help="The ID of the hammer signal")
parser.add_argument("--mode_name", type=str, help="The name of the mode")
parser.add_argument("--center_time_reson", type=str, help="The center time of the resonance")

parser.add_argument("--min_db_plot", type=float, help="The minimum relative power to plot in decibels", default=-30.0)
parser.add_argument("--window_length_reson", type=float, help="The length of the time window in seconds for the resonance", default=300.0)
parser.add_argument("--freq_target_hammer", type=float, help="The target frequency of the hammer signal", default=25.0)
parser.add_argument("--overlap_reson", type=float, help="The overlap for computing the spectrogram of the resonance", default=0.0)
parser.add_argument("--min_prom_reson", type=float, help="The minimum prominence of the resonance", default=15.0)
parser.add_argument("--min_rbw_reson", type=float, help="The minimum relative bandwidth of the resonance", default=15.0)
parser.add_argument("--max_mean_db_reson", type=float, help="The maximum mean power of the resonance", default=10.0)

parser.add_argument("--linewidth", type=float, help="The linewidth of the markers", default=0.5)
parser.add_argument("--markersize_station", type=float, help="The markersize of the stations", default=75.0)
parser.add_argument("--markersize_hammer", type=float, help="The markersize of the hammer", default=200.0)

parser.add_argument("--figwidth", type=float, help="The width of the figure", default=10.0)
parser.add_argument("--margin_x", type=float, help="The margin of the figure", default=0.02)
parser.add_argument("--margin_y", type=float, help="The margin of the figure", default=0.02)
parser.add_argument("--wspace", type=float, help="The width of the space between the subplots", default=0.03)
parser.add_argument("--hspace", type=float, help="The height of the space between the subplots", default=0.02)
parser.add_argument("--component_label_x", type=float, help="The x coordinate of the component label", default=0.02)
parser.add_argument("--component_label_y", type=float, help="The y coordinate of the component label", default=0.98)

parser.add_argument("--cmap_name", type=str, help="The name of the colormap", default="plasma")
parser.add_argument("--color_hammer", type=str, help="The color of the hammer", default="salmon")
parser.add_argument("--image_alpha", type=float, help="The alpha of the image", default=0.5)

parser.add_argument("--cbar_width", type=float, help="The width of the colorbar", default=0.01)
parser.add_argument("--cbar_height", type=float, help="The height of the colorbar", default=0.3)
parser.add_argument("--cbar_offset_x", type=float, help="The offset of the colorbar", default=0.02)
parser.add_argument("--cbar_offset_y", type=float, help="The offset of the colorbar", default=0.01)

parser.add_argument("--fontsize_title", type=float, help="The fontsize of the title", default=12)

# Parse the arguments
args = parser.parse_args()

hammer_id = args.hammer_id
mode_name = args.mode_name
freq_target_hammer = args.freq_target_hammer
center_time_reson = str2timestamp(args.center_time_reson)
min_db_plot = args.min_db_plot

window_length_reson = args.window_length_reson
overlap_reson = args.overlap_reson
min_prom_reson = args.min_prom_reson
min_rbw_reson = args.min_rbw_reson
max_mean_db_reson = args.max_mean_db_reson
linewidth = args.linewidth
markersize_station = args.markersize_station
markersize_hammer = args.markersize_hammer
figwidth = args.figwidth
margin_x = args.margin_x
margin_y = args.margin_y
wspace = args.wspace
hspace = args.hspace
component_label_x = args.component_label_x
component_label_y = args.component_label_y
cmap_name = args.cmap_name
color_hammer = args.color_hammer
image_alpha = args.image_alpha
cbar_width = args.cbar_width
cbar_height = args.cbar_height
cbar_offset_x = args.cbar_offset_x
cbar_offset_y = args.cbar_offset_y
fontsize_title = args.fontsize_title

###
# Constants
###

filename_image = "maxar_2019-09-17_local.tif"

###
# Load the data
###

# Load the station locations
loc_df = get_geophone_coords()

# Load the hammer location
print("Loading the hammer location...")
inpath = join(dirname_loc, "hammer_locations.csv")
hammer_df = read_csv(inpath, parse_dates=["origin_time"], dtype={"hammer_id": str})
x_hammer = hammer_df.loc[hammer_df["hammer_id"] == hammer_id, "east"].values[0]
y_hammer = hammer_df.loc[hammer_df["hammer_id"] == hammer_id, "north"].values[0]

# Load the satellite image
print("Loading the satellite image...")
inpath = join(dirname_img, filename_image)
with open(inpath) as f:
    # Read the image in RGB format
    rgb_band = f.read([1, 2, 3])

    # Reshape the image
    image = reshape_as_image(rgb_band)

    # Extract the extent of the image
    extent_img = [f.bounds.left, f.bounds.right, f.bounds.bottom, f.bounds.top]

# Extract the power at the target frequency for the hammer signal at all stations
print("Extracting the power at the target frequency for the hammer signal at all stations...")
power_hammer_dicts = []
for station in stations:
    print(f"Extracting the power at the target frequency for the hammer signal at {station}...")
    east = loc_df.loc[station, "east"]
    north = loc_df.loc[station, "north"]

    filename = f"hammer_mt_aspecs_{hammer_id}_{station}.csv"
    inpath = join(dirname_mt, filename)
    spec_df = read_csv(inpath)

    # Find the frequency closest to the target frequency
    ind = spec_df["frequency"].sub(freq_target_hammer).abs().idxmin()
    power = spec_df.loc[ind, f"aspec_total"]

    # Store the power in a dictionary
    power_hammer_dict = {
        "station": station,
        "east": east,
        "north": north,
        "power": power
    }

    power_hammer_dicts.append(power_hammer_dict)

# Convert the list of dictionaries to a DataFrame
power_hammer_df = DataFrame(power_hammer_dicts)

# Normalize the power
power_hammer_df["power"] = power_hammer_df["power"] - power_hammer_df["power"].max()

# Read the resonance properties
print("Reading the resonance properties...")
suffix_spec = get_spectrogram_file_suffix(window_length_reson, overlap_reson)
suffix_peak = get_spec_peak_file_suffix(min_prom_reson, min_rbw_reson, max_mean_db_reson)

filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.csv"
inpath = join(dirname_spec, filename)
reson_df = read_csv(inpath, parse_dates=["time"])
reson_df = reson_df[reson_df["time"] == center_time_reson]

# Extract the power of the resonance signal for the time window at all stations
print("Extracting the power of the resonance signal for the time window at all stations...")

power_reson_dicts = []
for _, row in reson_df.iterrows():
    station = row["station"]
    power = row["total_power"]
    east = loc_df.loc[station, "east"]
    north = loc_df.loc[station, "north"]

    # Store the power in a dictionary
    power_reson_dict = {
        "station": station,
        "east": east,
        "north": north,
        "power": power
    }

    power_reson_dicts.append(power_reson_dict)

# Convert the list of dictionaries to a DataFrame
power_reson_df = DataFrame(power_reson_dicts)

# Normalize the power
power_reson_df["power"] = power_reson_df["power"] - power_reson_df["power"].max()
print(power_reson_df["power"].max())

###
# Plot the hammer and resonance powers
###

print("Plotting...")

# Compute the figure height
print("Computing the figure height...")
aspect_map = (max_north - min_north) / (max_east - min_east)
figheight = figwidth * aspect_map * (1 - 2 * margin_x - wspace) / 2 / (1 - 2 * margin_y)

map_width = (1 - 2 * margin_x - wspace) / 2
map_height = 1 - 2 * margin_y

# Create the figure
print("Creating the figure...")
fig = figure(figsize = (figwidth, figheight))

# Define the colormap for the velocities
norm = Normalize(vmin = min_db_plot, vmax = 0.0)
cmap = colormaps[cmap_name]

###
# Plot the hammer powers
###

left = margin_x
bottom = margin_y
width = map_width
height = map_height

ax = fig.add_axes([left, bottom, width, height])

# Plot the satellite image
ax.imshow(image, alpha = image_alpha, extent = extent_img)

# Plot the hammer powers on all stations
ax.scatter(power_hammer_df["east"], power_hammer_df["north"], c = power_hammer_df["power"], marker = "^", cmap = cmap, norm = norm,
           s = markersize_station, edgecolors = "black", linewidth = linewidth)

# Plot the hammer location
ax.scatter(x_hammer, y_hammer, marker = "*", s = markersize_hammer, facecolor = color_hammer, edgecolors = "black", linewidth = linewidth)

# Plot the colorbar at the bottom left corner
bbox = ax.get_position()
pos = [bbox.x0 + cbar_offset_x, bbox.y0 + cbar_offset_y, cbar_width, cbar_height]
add_colorbar(fig, pos, "Normalized Power (dB)",
             cmap = cmap, norm = norm)

# Set the x and y limits
ax.set_xlim(min_east, max_east)
ax.set_ylim(min_north, max_north)

# Set the axis labels
format_east_xlabels(ax)
format_north_ylabels(ax)

# Set the title
ax.set_title(f"Hammer {hammer_id}, {freq_target_hammer} Hz", fontsize = fontsize_title, fontweight = "bold")
###
# Plot the resonance powers
###

left = margin_x + map_width + wspace

ax = fig.add_axes([left, bottom, width, height])

# Plot the satellite image
ax.imshow(image, alpha = image_alpha, extent = extent_img)

# Plot the resonance powers on all stations
ax.scatter(power_reson_df["east"], power_reson_df["north"], c = power_reson_df["power"], marker = "^", cmap = cmap, norm = norm,
           s = markersize_station, edgecolors = "black", linewidth = linewidth)

# Set the x and y limits
ax.set_xlim(min_east, max_east)
ax.set_ylim(min_north, max_north)

# Set the axis labels
format_east_xlabels(ax)
format_north_ylabels(ax,
                     plot_tick_label = False,
                     plot_axis_label = False)

# Set the title
mode_order = get_mode_order(mode_name)
start_time = center_time_reson - Timedelta(seconds = window_length_reson / 2)
end_time = center_time_reson + Timedelta(seconds = window_length_reson / 2)
start_str = start_time.strftime("%H:%M:%S")
end_str = end_time.strftime("%H:%M:%S")
ax.set_title(f"Mode {mode_order}, {start_str} - {end_str}", fontsize = fontsize_title, fontweight = "bold")

###
# Save the figure
###

# Save the figure
print("Saving the figure...")
time_str = center_time_reson.strftime("%Y%m%d%H%M%S")
save_figure(fig, f"hammer_and_resonance_geo_powers_{hammer_id}_{mode_name}_{time_str}.png")















