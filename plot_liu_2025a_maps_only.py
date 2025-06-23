# Plot only the map and hydrophone depth profiles in Fig. 1 in Liu et al. (2025a) for presentation purposes
from os.path import join
from numpy import cos, pi, linspace, isnan, nan


from argparse import ArgumentParser
from json import loads
from scipy.interpolate import interp1d
from pandas import DataFrame, Timedelta
from pandas import concat, read_csv, read_hdf
from matplotlib.pyplot import figure
from matplotlib import colormaps
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
from colorcet import cm

from rasterio import open
from rasterio.plot import reshape_as_image
from cartopy.crs import Orthographic, Geodetic
import cartopy.feature as cfeature

from utils_basic import EASTMIN_WHOLE as min_east_array, EASTMAX_WHOLE as max_east_array, NORTHMIN_WHOLE as min_north_array, NORTHMAX_WHOLE as max_north_array
from utils_basic import HYDRO_DEPTHS as depth_dict, GEO_COMPONENTS as components
from utils_basic import SPECTROGRAM_DIR as dir_spec, MT_DIR as dir_mt
from utils_basic import CENTER_LONGITUDE as lon, CENTER_LATITUDE as lat
from utils_basic import IMAGE_DIR as dir_img
from utils_basic import get_geophone_coords, get_borehole_coords, str2timestamp
from utils_basic import power2db
from utils_satellite import load_maxar_image
from utils_plot import format_east_xlabels, format_db_ylabels, format_freq_xlabels, format_north_ylabels, format_depth_ylabels, get_geo_component_color, save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser(description="Input parameters for plotting the station maps, hydrophone depth profiles, and 3C spectra of a geophone station for Liu et al. (2025a).")
parser.add_argument("--stations_highlight", type=str, nargs="+", help="List of highlighted geophone stations.")

parser.add_argument("--station_size", type=float, default=150.0, help="Size of the geophone markers.")
parser.add_argument("--borehole_size", type=float, default=150.0, help="Size of the borehole markers.")
parser.add_argument("--hydro_size", type=float, default=150.0, help="Size of the hydrophone markers.")

parser.add_argument("--min_db", type=float, default=-20.0, help="Minimum dB value for the color scale.")
parser.add_argument("--max_db", type=float, default=80.0, help="Maximum dB value for the color scale.")
parser.add_argument("--min_arrow_db", type=float, default=0.0, help="Minimum dB value for the arrow color scale.")
parser.add_argument("--subarray_label_size", type=float, default=15.0, help="Font size of the subarray labels.")
parser.add_argument("--axis_label_size", type=float, default=12.0, help="Font size of the axis labels.")
parser.add_argument("--tick_label_size", type=float, default=12.0, help="Font size of the tick labels.")
parser.add_argument("--title_size", type=float, default=15.0, help="Font size of the title.")
parser.add_argument("--legend_size", type=float, default=12.0, help="Font size of the legend.")
parser.add_argument("--location_font_size", type=float, default=15.0, help="Font size of the location labels.")
parser.add_argument("--arrow_gap", type=float, default=5.0, help="Gap between the arrow and the text.")
parser.add_argument("--arrow_length", type=float, default=10.0, help="Length of the arrow.")
parser.add_argument("--arrow_width", type=float, default=0.01, help="Width of the arrow.")
parser.add_argument("--arrow_head_width", type=float, default=5.0, help="Width of the arrow head.")
parser.add_argument("--arrow_head_length", type=float, default=5.0, help="Length of the arrow head.")

parser.add_argument("--linewidth_marker", type=float, default=1.0, help="Line width of the markers.")
parser.add_argument("--linewidth_highlight", type=float, default=2.0, help="Line width of the highlighted markers.")
parser.add_argument("--linewidth_star", type=float, default=0.5, help="Line width of the star.")

parser.add_argument("--image_alpha", type=float, default=0.5, help="Opacity of the satellite image.")

parser.add_argument("--color_geo", type=str, default="gold", help="Color of the geophone markers.")
parser.add_argument("--color_borehole", type=str, default="violet", help="Color of the borehole markers.")
parser.add_argument("--color_hydro", type=str, default="violet", help="Color of the hydrophone markers.")
parser.add_argument("--color_highlight", type=str, default="crimson", help="Color of the highlighted geophone markers.")
parser.add_argument("--color_missing", type=str, default="gray", help="Color of the missing resonance frequencies.")
parser.add_argument("--color_water", type=str, default="deepskyblue", help="Color of the water.")

parser.add_argument("--subarray_a_label_x", type=float, default=-5.0, help="X-coordinate of the subarray A label.")
parser.add_argument("--subarray_a_label_y", type=float, default=-40.0, help="Y-coordinate of the subarray A label.")
parser.add_argument("--subarray_b_label_x", type=float, default=-60.0, help="X-coordinate of the subarray B label.")
parser.add_argument("--subarray_b_label_y", type=float, default=70.0, help="Y-coordinate of the subarray B label.")

# Parse the command line arguments
args = parser.parse_args()

stations_highlight = args.stations_highlight

image_alpha = args.image_alpha

station_size = args.station_size
borehole_size = args.borehole_size
hydro_size = args.hydro_size

subarray_label_size = args.subarray_label_size
axis_label_size = args.axis_label_size
tick_label_size = args.tick_label_size
title_size = args.title_size
location_font_size = args.location_font_size
legend_size = args.legend_size

arrow_gap = args.arrow_gap
arrow_length = args.arrow_length
arrow_width = args.arrow_width
arrow_head_width = args.arrow_head_width
arrow_head_length = args.arrow_head_length

linewidth_marker = args.linewidth_marker
linewidth_highlight = args.linewidth_highlight
linewidth_star = args.linewidth_star

min_db = args.min_db
max_db = args.max_db
min_arrow_db = args.min_arrow_db

color_geo = args.color_geo
color_borehole = args.color_borehole
color_hydro = args.color_hydro
color_highlight = args.color_highlight
color_missing = args.color_missing
color_water = args.color_water

subarray_a_label_x = args.subarray_a_label_x
subarray_a_label_y = args.subarray_a_label_y
subarray_b_label_x = args.subarray_b_label_x
subarray_b_label_y = args.subarray_b_label_y

# Constants
fig_width = 15.0

margin_x = 0.05
margin_y = 0.05
wspace = 0.08

width_ratio = 6 # Width ratio between the station map and the hydrophone depth profile

globe_x = 0.1
globe_y = 0.4
globe_width = 0.2
globe_height = 0.2

scale_bar_length = 25.0

min_depth = 0.0
max_depth = 400.0

hydro_min = -0.5
hydro_max = 0.5

freq_min = 0.0
freq_max = 200.0

water_level = 15.0
water_amp = 2.5
water_period = 0.2

linewidth_coast = 0.2
linewidth_water = 2.0
linewidth_spec = 1.0
linewidth_arrow = 1.0

min_vel_app = 0.0
max_vel_app = 2000.0

station_font_size = 14.0
station_label_x = 7.0
station_label_y = 7.0

borehole_font_size = 14.0
borehole_label_x = 40.0
borehole_label_y = -40.0


location_label_x = 0.25

water_font_size = 14.0

major_dist_spacing = 25.0
major_depth_spacing = 50.0
major_freq_spacing = 50.0
major_db_spacing = 20.0

major_tick_length = 5.0
minor_tick_length = 2.0
tick_width = 1.0

frame_width = 1.0

subplot_label_size = 18.0
subplot_offset_x = -0.02
subplot_offset_y = 0.02

# Load the geophone and borehole coordinates
geo_df = get_geophone_coords()
boho_df = get_borehole_coords()

# Load the satellite image
rgb_image, extent_img = load_maxar_image()

### Generate the figure and axes ###
# Compute the aspect ratio of the figure and generate the figure
aspect_ratio_map = (max_north_array - min_north_array) / (max_east_array - min_east_array)
aspect_ratio_fig = aspect_ratio_map * (1 - 2 * margin_x - wspace) / (1 - 2 * margin_y) * width_ratio / (width_ratio + 1)
fig_height = fig_width * aspect_ratio_fig

fig = figure(figsize = (fig_width, fig_height))

# Compute the dimensions of the subplots
width_map = (1.0 - 2 * margin_x - wspace) * width_ratio / (width_ratio + 1)
width_hydro = width_map / width_ratio

# Add the axes
ax_map = fig.add_axes([margin_x, margin_y, width_map, 1.0 - 2 * margin_y])
ax_hydro = fig.add_axes([margin_x + width_map + wspace, margin_y, width_hydro, 1.0 - 2 * margin_y])

### Plot the station map ###
print("Plotting the station map...")
# Plot the satellite image as the background
ax_map.imshow(rgb_image, extent = extent_img, zorder = 0, alpha = image_alpha)

# Plot the geophone locations
for station, coords in geo_df.iterrows():
    east = coords["east"]
    north = coords["north"]

    if station in stations_highlight:
        ax_map.scatter(east, north, marker = "^", s = station_size, color = color_geo, edgecolors = color_highlight, linewidths = linewidth_highlight)
        ax_map.annotate(station, (east, north), 
                        textcoords = "offset points", xytext = (station_label_x, station_label_y), ha = "left", va = "bottom", fontsize = station_font_size, 
                        color = "black", fontweight = "bold", bbox = dict(facecolor = "white", edgecolor = "none", alpha = 0.5))
    else:
        ax_map.scatter(east, north, marker = "^", s = station_size, color = color_geo, edgecolors = "black", linewidths = linewidth_marker, label = "Geophone")

# Plot the borehole locations
for borehole, coords in boho_df.iterrows():
    east = coords["east"]
    north = coords["north"]

    ax_map.scatter(east, north, marker = "o", s = borehole_size, color = color_hydro, edgecolors = "black", linewidths = linewidth_marker, label = "Borehole/Hydrophones")
    ax_map.annotate(borehole, (east, north), 
                    textcoords = "offset points", xytext = (borehole_label_x, borehole_label_y), ha = "left", va = "top", fontsize = borehole_font_size, fontweight = "bold", 
                    color = "black", arrowprops=dict(arrowstyle = "-", color = "black"), bbox = dict(facecolor = "white", edgecolor = "none", alpha = 0.5))

# Add the subarray labels
ax_map.text(subarray_a_label_x, subarray_a_label_y, "Subarray A", color = "black", fontsize = subarray_label_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "center")
ax_map.text(subarray_b_label_x, subarray_b_label_y, "Subarray B", color = "black", fontsize = subarray_label_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "center")

# Add the scale bar
scale_bar = AnchoredSizeBar(ax_map.transData, scale_bar_length, 
                            f"{scale_bar_length:.0f} m", loc = "lower left", bbox_transform = ax_map.transAxes, frameon = True, size_vertical = 1.0, pad = 0.5, fontproperties = FontProperties(size = axis_label_size))
ax_map.add_artist(scale_bar)

# Plot the legend
handles, labels = ax_map.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
legend = ax_map.legend(unique_labels.values(), unique_labels.keys(), loc = "upper left", frameon = True, fancybox = False, fontsize = legend_size, bbox_transform = ax_map.transAxes)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_alpha(1.0)

# Set the axis limits
ax_map.set_xlim(min_east_array, max_east_array)
ax_map.set_ylim(min_north_array, max_north_array)

ax_map.set_aspect("equal")

# Set the axis ticks
format_east_xlabels(ax_map, 
                    plot_axis_label = True, 
                    major_tick_spacing = major_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)

format_north_ylabels(ax_map, 
                    plot_axis_label = True, 
                    major_tick_spacing = major_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)

# Set the axis labels
ax_map.set_xlabel("East (m)")
ax_map.set_ylabel("North (m)")

### Plot the large-scale map with coastlines ###
# Define the orthographic projection centered at the given longitude and latitude
ax_coast = fig.add_axes([globe_x, globe_y, globe_width, globe_height], projection = Orthographic(central_longitude=lon, central_latitude=lat))
ax_coast.set_global()

# Add features
ax_coast.add_feature(cfeature.LAND, color='lightgray')
ax_coast.add_feature(cfeature.OCEAN, color='skyblue')

# Add coastlines
ax_coast.coastlines(linewidth = linewidth_coast)

# Plot a star at the given longitude and latitude
ax_coast.scatter(lon, lat, marker = '*', s = 100, color=color_hydro, edgecolor = "black", linewidths = linewidth_star, transform = Geodetic(), zorder = 10)

### Plot the hydrophone depth profiles ###
print(f"Plotting the hydrophone depth profiles...")

# Plot the water level
water_line_x = linspace(hydro_min, hydro_max, 100)
water_line_y = water_level + water_amp * cos(2 * pi * water_line_x / water_period)

ax_hydro.plot(water_line_x, water_line_y, color = color_water, linewidth = linewidth_water)
ax_hydro.fill_between(water_line_x, water_line_y, max_depth, color = color_water, alpha = 0.2)
ax_hydro.text(hydro_min, water_level, "Water table", color = color_water, fontsize = water_font_size, va = "top", ha = "right", rotation = 60)

# Plot the hydrophones
for location in depth_dict.keys():
    depth = depth_dict[location]
    ax_hydro.scatter(0.0, depth, marker = "o", color = color_hydro, edgecolors = "black", s = hydro_size, linewidths = linewidth_marker, label = "Hydrophone")

    ax_hydro.text(location_label_x, depth, location, color = "black", fontsize = location_font_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "center")

ax_hydro.text(0.0, -15.0, "BA1A & BA1B\n(A00 & B00)", color = "black", fontsize = location_font_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "center")

max_hydro_depth = max(depth_dict.values())
ax_hydro.plot([0.0, 0.0], [min_depth, max_hydro_depth], color = "black", linewidth = linewidth_marker, zorder = 0)

# # Plot the legend
# handles, labels = ax_hydro.get_legend_handles_labels()
# unique_labels = dict(zip(labels, handles))
# legend = ax_hydro.legend(unique_labels.values(), unique_labels.keys(), loc = "lower left", frameon = False, fontsize = legend_size)


# Set the axis limits
ax_hydro.set_xlim(hydro_min, hydro_max)
ax_hydro.set_ylim(min_depth, max_depth)

ax_hydro.invert_yaxis()

# Set the axis ticks
ax_hydro.set_xticks([])
format_depth_ylabels(ax_hydro, 
                    plot_axis_label = True, 
                    major_tick_spacing = major_depth_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)

### Save the figure ###
print("Saving the figure...")
figname = "liu_2025a_maps_only.png"
save_figure(fig, figname)