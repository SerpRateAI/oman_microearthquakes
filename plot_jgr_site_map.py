# Plot station maps, hydrophone depth profiles, and 3C spectra of a geophone station for Liu et al. (2025a)
from os.path import join
from numpy import cos, pi, linspace, isnan, nan, amax


from argparse import ArgumentParser
from json import loads
from scipy.interpolate import interp1d
from pandas import DataFrame, Timedelta
from pandas import concat, read_csv, read_hdf
from geopandas import read_file, datasets as gpd_datasets
from fiona import listlayers
from matplotlib.pyplot import figure
from matplotlib import colormaps
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
from colorcet import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch

from rasterio import open
from rasterio.plot import reshape_as_image
from cartopy.crs import Orthographic, Geodetic
import cartopy.feature as cfeature
from cartopy.io import shapereader 
from pyproj import Transformer




from utils_basic import EASTMIN_WHOLE as min_east_array, EASTMAX_WHOLE as max_east_array, NORTHMIN_WHOLE as min_north_array, NORTHMAX_WHOLE as max_north_array
from utils_basic import HYDRO_DEPTHS as depth_dict, GEO_COMPONENTS as components
from utils_basic import SPECTROGRAM_DIR as dir_spec, MT_DIR as dir_mt
from utils_basic import CENTER_LONGITUDE as lon, CENTER_LATITUDE as lat
from utils_basic import IMAGE_DIR as dir_img
from utils_basic import get_geophone_coords, get_borehole_coords, str2timestamp
from utils_basic import INNER_STATIONS as inner_stations, MIDDLE_STATIONS as middle_stations, OUTER_STATIONS as outer_stations
from utils_basic import PLOTTING_DIR as dir_plot
from utils_basic import power2db
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_time_slice_from_geo_stft
from utils_satellite import load_maxar_image
from utils_plot import format_east_xlabels, format_db_ylabels, format_freq_xlabels, format_north_ylabels, format_depth_ylabels, get_geo_component_color, save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser(description="Input parameters for plotting the station maps, hydrophone depth profiles, and 3C spectra of a geophone station for Liu et al. (2025a).")
parser.add_argument("--stations_highlight", type=str, nargs="+", help="List of highlighted geophone stations.")
parser.add_argument("--station_spec", type=str, help="Station whose 3C spectra will be plotted.")
parser.add_argument("--time_window", type=str, help="Time window for the 3C spectra.")
parser.add_argument("--window_length_stft", type=float, default=300.0, help="Window length in seconds for computing the STFT.")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap between consecutive windows for computing the STFT.")
parser.add_argument("--min_prom", type=float, default=15.0, help="Minimum prominence of a spectral peak.")
parser.add_argument("--min_rbw", type=float, default=15.0, help="Minimum reverse bandwidth of a spectral peak.")
parser.add_argument("--max_mean_db", type=float, default=15.0, help="Maximum mean dB value for excluding noise windows.")

parser.add_argument("--station_size", type=float, default=150.0, help="Size of the geophone markers.")
parser.add_argument("--borehole_size", type=float, default=150.0, help="Size of the borehole markers.")
parser.add_argument("--hydro_size", type=float, default=150.0, help="Size of the hydrophone markers.")

parser.add_argument("--min_db", type=float, default=-20.0, help="Minimum dB value for the color scale.")
parser.add_argument("--max_db", type=float, default=80.0, help="Maximum dB value for the color scale.")
parser.add_argument("--min_arrow_db", type=float, default=0.0, help="Minimum dB value for the arrow color scale.")
parser.add_argument("--subarray_label_size", type=float, default=15.0, help="Font size of the subarray labels.")
parser.add_argument("--freq_label_size", type=float, default=15.0, help="Font size of the frequency labels.")
parser.add_argument("--axis_label_size", type=float, default=12.0, help="Font size of the axis labels.")
parser.add_argument("--tick_label_size", type=float, default=12.0, help="Font size of the tick labels.")
parser.add_argument("--title_size", type=float, default=15.0, help="Font size of the title.")
parser.add_argument("--legend_size", type=float, default=12.0, help="Font size of the legend.")
parser.add_argument("--location_font_size", type=float, default=15.0, help="Font size of the location labels.")
parser.add_argument("--arrow_gap", type=float, default=5.0, help="Gap between the arrow and the text.")
parser.add_argument("--arrow_length", type=float, default=10.0, help="Length of the arrow.")
parser.add_argument("--arrow_width_fancy", type=float, default=0.01, help="Width of the fancy arrow.")
parser.add_argument("--arrow_width_simple", type=float, default=1.5, help="Width of the simple arrow.")
parser.add_argument("--arrow_head_width", type=float, default=5.0, help="Width of the arrow head.")
parser.add_argument("--arrow_head_length", type=float, default=5.0, help="Length of the arrow head.")
parser.add_argument("--delta_f", type=float, default=12.75, help="Mode spacing in Hz.")

parser.add_argument("--linewidth_marker", type=float, default=1.0, help="Line width of the markers.")
parser.add_argument("--linewidth_highlight", type=float, default=2.0, help="Line width of the highlighted markers.")
parser.add_argument("--linewidth_star", type=float, default=0.5, help="Line width of the star.")

parser.add_argument("--image_alpha", type=float, default=0.5, help="Opacity of the satellite image.")

parser.add_argument("--color_geo", type=str, default="gold", help="Color of the geophone markers.")
parser.add_argument("--color_borehole", type=str, default="violet", help="Color of the borehole markers.")
parser.add_argument("--color_hydro", type=str, default="violet", help="Color of the hydrophone markers.")
parser.add_argument("--color_highlight", type=str, default="crimson", help="Color of the highlighted geophone markers.")
parser.add_argument("--color_missing", type=str, default="gray", help="Color of the missing resonance frequencies.")
parser.add_argument("--color_water", type=str, default="deepskyblue", help="Color of borehole water.")
parser.add_argument("--color_land", type=str, default="lightgray", help="Color of the land.")


parser.add_argument("--subarray_a_label_x", type=float, default=-5.0, help="X-coordinate of the subarray A label.")
parser.add_argument("--subarray_a_label_y", type=float, default=-40.0, help="Y-coordinate of the subarray A label.")
parser.add_argument("--subarray_b_label_x", type=float, default=-60.0, help="X-coordinate of the subarray B label.")
parser.add_argument("--subarray_b_label_y", type=float, default=70.0, help="Y-coordinate of the subarray B label.")

parser.add_argument("--path_gpkg", type=str, default="/proj/mazu/tianze.liu/oman/geology/Nicolas_ea_5_lith_4.gpkg",
                    help="Path to the simplified Samail Ophiolite GeoPackage (.gpkg).")
parser.add_argument("--color_mafic", type=str, default="lightskyblue",
                    help="Fill color for mafic units in the inset.")
parser.add_argument("--color_ultramafic", type=str, default="palegreen",
                    help="Fill color for ultramafic units in the inset.")
parser.add_argument("--inset_width_frac", type=float, default=0.40,
                    help="Inset width as a fraction of the site-map width.")
parser.add_argument("--inset_height_frac", type=float, default=0.40,
                    help="Inset height as a fraction of the site-map height.")
parser.add_argument("--inset_offset_x_frac", type=float, default=0.1,
                    help="Inset offset in x (fraction of parent axes width).")
parser.add_argument("--inset_offset_y_frac", type=float, default=0.1,
                    help="Inset offset in y (fraction of parent axes height).")

# Parse the command line arguments
args = parser.parse_args()

stations_highlight = args.stations_highlight
station_spec = args.station_spec
time_window = args.time_window
window_length_stft = args.window_length_stft
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

image_alpha = args.image_alpha

station_size = args.station_size
borehole_size = args.borehole_size
hydro_size = args.hydro_size

subarray_label_size = args.subarray_label_size
freq_label_size = args.freq_label_size
axis_label_size = args.axis_label_size
tick_label_size = args.tick_label_size
title_size = args.title_size
location_font_size = args.location_font_size
legend_size = args.legend_size

arrow_gap = args.arrow_gap
arrow_length = args.arrow_length
arrow_width_fancy = args.arrow_width_fancy
arrow_width_simple = args.arrow_width_simple
arrow_head_width = args.arrow_head_width
arrow_head_length = args.arrow_head_length
delta_f = args.delta_f

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
color_land = args.color_land

subarray_a_label_x = args.subarray_a_label_x
subarray_a_label_y = args.subarray_a_label_y
subarray_b_label_x = args.subarray_b_label_x
subarray_b_label_y = args.subarray_b_label_y

path_gpkg = args.path_gpkg
color_mafic = args.color_mafic
color_ultramafic = args.color_ultramafic
inset_width_frac = args.inset_width_frac
inset_height_frac = args.inset_height_frac
inset_offset_x_frac = args.inset_offset_x_frac
inset_offset_y_frac = args.inset_offset_y_frac

# Constants
fig_width = 15.0

margin_x = 0.05
margin_y = 0.05
hspace = 0.05
vspace = 0.08

width_ratio = 6 # Width ratio between the station map and the hydrophone depth profile

globe_x = 0.15
globe_y = 0.5
globe_width = 0.1
globe_height = 0.1

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
station_label_x = 0.0
station_label_y = 15.0

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
aspect_ratio_fig = aspect_ratio_map * (1 - 2 * margin_x - vspace) / (1 - 2 * margin_y - hspace) * width_ratio / (width_ratio + 1)
fig_height = fig_width * aspect_ratio_fig

fig = figure(figsize = (fig_width, fig_height))

# Compute the dimensions of the subplots
width_map = (1.0 - 2 * margin_x - vspace) * width_ratio / (width_ratio + 1)
height_map = (1.0 - 2 * margin_y - hspace)
width_hydro = width_map / width_ratio
height_hydro = height_map

# Add the axes
ax_map = fig.add_axes([margin_x, margin_y + hspace, width_map, height_map])
ax_hydro = fig.add_axes([margin_x + width_map + vspace, margin_y + hspace, width_hydro, height_hydro])

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
                            f"{scale_bar_length:.0f} m", loc = "lower right", bbox_transform = ax_map.transAxes, frameon = True, size_vertical = 1.0, pad = 0.5, fontproperties = FontProperties(size = axis_label_size))
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

# Add the subplot label
bbox_map = ax_map.get_position()
top_left_x = bbox_map.x0
top_left_y = bbox_map.y1
# print(f"Top left: ({top_left_x}, {top_left_y})")
fig.text(top_left_x + subplot_offset_x, top_left_y + subplot_offset_y, "(a)", fontsize = subplot_label_size, fontweight = "bold", va = "bottom", ha = "right")

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

# Set the axis limits
ax_hydro.set_xlim(hydro_min, hydro_max)
ax_hydro.set_ylim(min_depth, max_depth)

ax_hydro.invert_yaxis()

# Set the axis ticks
ax_hydro.set_xticks([])
format_depth_ylabels(ax_hydro, 
                    plot_axis_label = True, 
                    major_tick_spacing = major_depth_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)

# Add the subplot label
bbox = ax_hydro.get_position()
top_left_x = bbox.x0
top_left_y = bbox.y1
fig.text(top_left_x + subplot_offset_x, top_left_y + subplot_offset_y, "(b)", fontsize = subplot_label_size, fontweight = "bold", va = "bottom", ha = "right")

### Save the figure ###
print("Saving the figure...")
figname = "jgr_site_map.png"
save_figure(fig, figname)