# Plot station maps and hydrophone depth profiles in the AGU 2024 iPoster

from os.path import join
from numpy import cos, pi, linspace
from argparse import ArgumentParser
from json import loads
from pandas import Timedelta
from pandas import concat, read_csv, read_hdf
from matplotlib.pyplot import subplots
from matplotlib.patches import Rectangle
from rasterio import open
from rasterio.plot import reshape_as_image
# from cartopy.io.img_tiles import Stamen
# from cartopy.io.shapereader import Reader
# from cartopy.io.raster import RasterSource
# from cartopy.mpl.geoaxes import GeoAxes
from cartopy.crs import Orthographic, Geodetic
import cartopy.feature as cfeature

from utils_basic import EASTMIN_WHOLE as min_east_array, EASTMAX_WHOLE as max_east_array, NORTHMIN_WHOLE as min_north_array, NORTHMAX_WHOLE as max_north_array
from utils_basic import EASTMAX_A_INNER as max_east_inset, NORTHMAX_A_INNER as max_north_inset, EASTMIN_A_INNER as min_east_inset, NORTHMIN_A_INNER as min_north_inset
from utils_basic import HYDRO_DEPTHS as depth_dict
from utils_basic import CENTER_LONGITUDE as lon, CENTER_LATITUDE as lat
from utils_basic import IMAGE_DIR as dir_img
from utils_basic import get_geophone_coords, get_borehole_coords, str2timestamp
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_time_slice_from_geo_stft
from utils_plot import component2label, format_east_xlabels, format_db_ylabels, format_freq_xlabels, format_north_ylabels, format_depth_ylabels, get_geo_component_color, save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser(description="Input parameters for plotting the station maps and hydrophone depth profiles in the AGU 2024 iPoster.")
parser.add_argument("--stations_highlight", type=str, help="List of highlighted geophone stations.")
parser.add_argument("--station_spec", type=str, help="Station whose 3C spectra will be plotted.")

parser.add_argument("--color_geo", type=str, default="gold", help="Color of the geophone markers.")
parser.add_argument("--color_borehole", type=str, default="violet", help="Color of the borehole markers.")
parser.add_argument("--color_hydro", type=str, default="violet", help="Color of the hydrophone markers.")
parser.add_argument("--color_highlight", type=str, default="crimson", help="Color of the highlighted geophone markers.")

# Parse the command line arguments
args = parser.parse_args()

stations_highlight = loads(args.stations_highlight)
station_spec = args.station_spec

color_geo = args.color_geo
color_borehole = args.color_borehole
color_hydro = args.color_hydro
color_highlight = args.color_highlight

# Constants
fig_width = 15.0
fig_height = 12.0
spec_gap = 0.05

axis_offset = 0.1

min_depth = 0.0
max_depth = 450.0

hydro_min = -0.5
hydro_max = 1.5

freq_min = 0.0
freq_max = 200.0

water_level = 15.0
water_amp = 2.5
water_period = 0.2

linewidth_marker = 1.0
linewidth_box = 1.5
linewidth_coast = 0.2
linewidth_water = 2.0
linewidth_spec = 1.0
linewidth_arrow = 1.0

station_size = 100.0
borehole_size = 100.0
hydro_size = 100.0

station_font_size = 12.0
station_label_x = 7.0
station_label_y = 7.0

borehole_font_size = 12.0
borehole_label_x = 40.0
borehole_label_y = -40.0

location_font_size = 12.0

water_font_size = 12.0

major_dist_spacing = 25.0
major_depth_spacing = 50.0
major_freq_spacing = 50.0
major_db_spacing = 20.0

axis_label_size = 12.0
tick_label_size = 12.0
title_size = 14.0
component_label_size = 14.0
freq_label_size = 12.0

legend_size = 12.0

major_tick_length = 5.0
minor_tick_length = 2.0
tick_width = 1.0

frame_width = 1.0

arrow_gap = 5.0
arrow_length = 10.0
arrow_width = 1.0
arrow_headwidth = 5.0
arrow_headlength = 5.0

subplot_label_size = 18.0
subplot_offset_x = -0.04
subplot_offset_y = 0.02

filename_image = "spot_2019-12-06_local.tif"

# Load the geophone and borehole coordinates
geo_df = get_geophone_coords()
boho_df = get_borehole_coords()

# Load the satellite image
inpath = join(dir_img, filename_image)
with open(inpath) as src:
    # Read the image in RGB format
    rgb_band = src.read([1, 2, 3])

    # Reshape the image
    rgb_image = reshape_as_image(rgb_band)

    # Extract the extent of the image
    extent_img = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

# Generate the figure and axes
# Compute the aspect ratio
east_range = max_east_array - min_east_array
north_range = max_north_array - min_north_array
aspect_ratio = north_range / east_range

fig, ax_sta = subplots(1, 1, figsize = (fig_width, fig_height))

### Plot the station map ###
# Plot the satellite image as the background
ax_sta.imshow(rgb_image, extent = extent_img, zorder = 0)

# Plot the geophone locations
for station, coords in geo_df.iterrows():
    east = coords["east"]
    north = coords["north"]

    if station in stations_highlight:
        ax_sta.scatter(east, north, marker = "^", s = station_size, color = color_geo, edgecolors = color_highlight, linewidths = linewidth_marker)
        ax_sta.annotate(station, (east, north), 
                        textcoords = "offset points", xytext = (station_label_x, station_label_y), ha = "left", va = "bottom", fontsize = station_font_size, 
                        color = color_highlight, fontweight = "bold", bbox = dict(facecolor = "white", edgecolor = "none", alpha = 0.5))
    else:
        ax_sta.scatter(east, north, marker = "^", s = station_size, color = color_geo, edgecolors = "black", linewidths = linewidth_marker, label = "Geophone")

# Plot the borehole locations
for borehole, coords in boho_df.iterrows():
    east = coords["east"]
    north = coords["north"]

    ax_sta.scatter(east, north, marker = "o", s = borehole_size, color = color_hydro, edgecolors = "black", linewidths = linewidth_marker, label = "Borehole/Hydrophones")
    ax_sta.annotate(borehole, (east, north), 
                    textcoords = "offset points", xytext = (borehole_label_x, borehole_label_y), ha = "left", va = "top", fontsize = borehole_font_size, fontweight = "bold", 
                    color = color_hydro, arrowprops=dict(arrowstyle = "-", color = "black"), bbox = dict(facecolor = "white", edgecolor = "none", alpha = 0.5))

# # Add the box around the inset
# inset = Rectangle((min_east_inset, min_north_inset), max_east_inset - min_east_inset, max_north_inset - min_north_inset, edgecolor = color_highlight, facecolor = "none", linewidth = linewidth_box, zorder = 1)
# ax_sta.add_patch(inset)

# Plot the legend
handles, labels = ax_sta.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
legend = ax_sta.legend(unique_labels.values(), unique_labels.keys(), loc = "upper left", frameon = True, fancybox = False, fontsize = legend_size, bbox_to_anchor = (0.0, 1.0), bbox_transform = ax_sta.transAxes)
legend.get_frame().set_facecolor("white")

# Set the axis limits
ax_sta.set_xlim(min_east_array, max_east_array)
ax_sta.set_ylim(min_north_array, max_north_array)

ax_sta.set_aspect("equal")

# Set the axis ticks
format_east_xlabels(ax_sta, 
                    major_tick_spacing = major_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)
format_north_ylabels(ax_sta,
                    major_tick_spacing = major_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)

# Set the axis labels
ax_sta.set_xlabel("East (m)")
ax_sta.set_ylabel("North (m)")

# # Adjust the frame width
# for spine in ax_sta.spines.values():
#     spine.set_linewidth(frame_width)

# Plot the large-scale map with coastlines
# Define the orthographic projection centered at the given longitude and latitude
ax_coast = fig.add_axes([0.25, 0.15, 0.2, 0.2], projection = Orthographic(central_longitude=lon, central_latitude=lat))
ax_coast.set_global()

# Add features
ax_coast.add_feature(cfeature.LAND, color='lightgray')
ax_coast.add_feature(cfeature.OCEAN, color='skyblue')

# Add coastlines
ax_coast.coastlines(linewidth = linewidth_coast)

# Plot a star at the given longitude and latitude
ax_coast.scatter(lon, lat, marker = '*', s = 100, color=color_hydro, edgecolor = "black", linewidths = linewidth_marker, transform = Geodetic(), zorder = 10)

### Plot the hydrophone depth profiles ###
# Add the axis 
bbox = ax_sta.get_position()
map_height = bbox.height
map_width = bbox.width
profile_height = map_height
profile_width = map_width / 3
ax_hydro = fig.add_axes([bbox.x1 + axis_offset, bbox.y0, profile_width, profile_height])

# Plot the hydrophones
for offset in [0, 1]:
    for location in depth_dict.keys():
        depth = depth_dict[location]

        if offset == 0 and location in ["01", "02"]:
            ax_hydro.scatter(offset, depth, marker = "o", color = "lightgray", edgecolors = "black", s = hydro_size, linewidths = linewidth_marker, label = "Broken")
        else:
            ax_hydro.scatter(offset, depth, marker = "o", color = color_hydro, edgecolors = "black", s = hydro_size, linewidths = linewidth_marker, label = "Functional")

        ax_hydro.text(0.5, depth, location, color = "black", fontsize = location_font_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "center")

ax_hydro.text(0.0, -15.0, "BA1A\n(A00)", color = "black", fontsize = location_font_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "center")
ax_hydro.text(1.0, -15.0, "BA1B\n(B00)", color = "black", fontsize = location_font_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "center")

max_hydro_depth = max(depth_dict.values())
ax_hydro.plot([0.0, 0.0], [min_depth, max_hydro_depth], color = "black", linewidth = linewidth_marker, zorder = 0)
ax_hydro.plot([1.0, 1.0], [min_depth, max_hydro_depth], color = "black", linewidth = linewidth_marker, zorder = 0)

# Plot the legend
handles, labels = ax_hydro.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax_hydro.legend(unique_labels.values(), unique_labels.keys(), loc = "lower left", frameon = False, fontsize = legend_size)

# Plot the water level
water_line_x = linspace(hydro_min, hydro_max, 100)
water_line_y = water_level + water_amp * cos(2 * pi * water_line_x / water_period)

ax_hydro.plot(water_line_x, water_line_y, color = "dodgerblue", linewidth = linewidth_water)
ax_hydro.text(-0.6, water_level, "Water table", color = "dodgerblue", fontsize = water_font_size, verticalalignment = "center", horizontalalignment = "right")

# Set the axis limits
ax_hydro.set_xlim(hydro_min, hydro_max)
ax_hydro.set_ylim(min_depth, max_depth)

ax_hydro.invert_yaxis()

# Set the axis ticks
ax_hydro.set_xticks([])
format_depth_ylabels(ax_hydro, label = True, major_tick_spacing = major_depth_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)

# # Adjust the frame width
# for spine in ax_hydro.spines.values():
#     spine.set_linewidth(frame_width)

# Save the figure
save_figure(fig, "agu_2024_map_n_profile.png")