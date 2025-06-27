"""
Plot the power decay of the hammer signal at a certain frequency
"""

from os.path import join
from pandas import read_csv
from argparse import ArgumentParser
from matplotlib.pyplot import subplots
from matplotlib import colormaps

from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import LOC_DIR as dirpath_loc, MT_DIR as dirpath_mt, SPECTROGRAM_DIR as dirpath_spec
from utils_basic import get_geophone_coords, get_borehole_coords
from utils_satellite import load_maxar_image
from utils_plot import GEO_PSD_UNIT as psd_unit, GEO_PSD_LABEL as psd_label
from utils_plot import figure, save_figure, format_east_xlabels, format_north_ylabels


###
# Input arguments
### 

parser = ArgumentParser()   
parser.add_argument("--station_highlight", type=str, help="The station to highlight")

parser.add_argument("--freq_target", type=float, help="The frequency of the PSD to plot", default=150.0)
parser.add_argument("--alpha", type=float, help="The transparency of the image", default=0.2)
parser.add_argument("--figwidth", type=float, help="The width of the figure", default=12)
parser.add_argument("--margin_x", type=float, help="The margin of the figure on the x-axis", default=0.02)
parser.add_argument("--margin_y", type=float, help="The margin of the figure on the y-axis", default=0.02)
parser.add_argument("--power_decay_x", type=float, help="The x offset of the power decay", default=0.06)
parser.add_argument("--power_decay_y", type=float, help="The y offset of the power decay", default=0.05)
parser.add_argument("--power_decay_width", type=float, help="The width of the power decay", default=0.4)
parser.add_argument("--power_decay_height", type=float, help="The height of the power decay", default=0.4)

parser.add_argument("--panel_label_x1", type=float, help="The x offset of the panel label 1", default=-0.02)
parser.add_argument("--panel_label_y1", type=float, help="The y offset of the panel label 1", default=1.02)
parser.add_argument("--panel_label_x2", type=float, help="The x offset of the panel label 2", default=-0.05)
parser.add_argument("--panel_label_y2", type=float, help="The y offset of the panel label 2", default=1.05)
parser.add_argument("--panel_label_size", type=float, help="The size of the panel labels", default=16)

parser.add_argument("--psd_ref", type=float, help="The PSD in dB of the reference line at distance 0", default=70.0)
parser.add_argument("--psd_slope", type=float, help="The slope in dB/m of the power decay", default=-1.0)
parser.add_argument("--ref_label_x", type=float, help="The x offset of the reference label", default=30.0)
parser.add_argument("--ref_label_y", type=float, help="The y offset of the reference label", default=50.0)
parser.add_argument("--ref_label_size", type=float, help="The size of the reference label", default=12)

parser.add_argument("--markersize_hammer", type=float, help="The size of the markers", default=200)
parser.add_argument("--markersize_station", type=float, help="The size of the markers", default=100)
parser.add_argument("--markersize_borehole", type=float, help="The size of the markers", default=100)
parser.add_argument("--markersize_psd", type=float, help="The size of the markers", default=100)
parser.add_argument("--linewidth_marker_map", type=float, help="The width of the lines", default=0.75)
parser.add_argument("--linewidth_marker_psd", type=float, help="The width of the lines", default=0.75)
parser.add_argument("--linewidth_psd", type=float, help="The width of the lines", default=1.5)

parser.add_argument("--color_hammer_marker", type=str, help="The color of the markers", default="salmon")
parser.add_argument("--color_hammer_psd", type=str, help="The color of the markers", default="salmon")
parser.add_argument("--color_reference", type=str, help="The color of the reference line", default="crimson")
parser.add_argument("--color_station", type=str, help="The color of the markers", default="orange")

parser.add_argument("--station_label_size", type=float, help="The size of the station label", default=14)
parser.add_argument("--station_label_x", type=float, help="The x offset of the station label", default=0)
parser.add_argument("--station_label_y", type=float, help="The y offset of the station label", default=10)

parser.add_argument("--axis_label_size", type=float, help="The size of the axis labels", default=12)
parser.add_argument("--title_size", type=float, help="The size of the title", default=14)
parser.add_argument("--legend_size", type=float, help="The size of the legend", default=12)

args = parser.parse_args()

station_highlight = args.station_highlight

freq_target = args.freq_target
figwidth = args.figwidth
margin_x = args.margin_x
margin_y = args.margin_y
power_decay_x = args.power_decay_x
power_decay_y = args.power_decay_y
power_decay_width = args.power_decay_width
power_decay_height = args.power_decay_height
psd_ref = args.psd_ref
psd_slope = args.psd_slope
ref_label_x = args.ref_label_x
ref_label_y = args.ref_label_y
ref_label_size = args.ref_label_size
panel_label_x1 = args.panel_label_x1
panel_label_y1 = args.panel_label_y1
panel_label_x2 = args.panel_label_x2
panel_label_y2 = args.panel_label_y2
panel_label_size = args.panel_label_size

alpha = args.alpha
markersize_hammer = args.markersize_hammer
markersize_station = args.markersize_station
markersize_borehole = args.markersize_borehole
markersize_psd = args.markersize_psd
linewidth_psd = args.linewidth_psd
linewidth_marker_map = args.linewidth_marker_map
linewidth_marker_psd = args.linewidth_marker_psd
color_hammer_marker = args.color_hammer_marker
color_station = args.color_station
color_hammer_psd = args.color_hammer_psd
color_reference = args.color_reference

station_label_size = args.station_label_size
station_label_x = args.station_label_x
station_label_y = args.station_label_y

axis_label_size = args.axis_label_size
title_size = args.title_size
legend_size = args.legend_size

###
# Read the data
###

# Load the satellite image
image, extent = load_maxar_image()

# Read the geophone coordinates
geophone_df = get_geophone_coords()

# Read the borehole coordinates
borehole_df = get_borehole_coords()

# Read the hammer coordinates
filepath = join(dirpath_loc, "hammer_locations.csv")
hammer_loc_df = read_csv(filepath, dtype = {"hammer_id": str})

# Read the PSD vs distance
filepath = join(dirpath_mt, f"hammer_mt_psd_vs_distance_{freq_target:.0f}hz_{station_highlight}.csv")
psd_dist_df = read_csv(filepath, dtype = {"hammer_id": str})

###
# Plot the map
###

# Generate the figure
map_aspect = (max_north - min_north) / (max_east - min_east)
figheight = figwidth * (1 - 2 * margin_x) * map_aspect / (1 - 2 * margin_y)
fig = figure(figsize = (figwidth, figheight))

# Plot the map
ax = fig.add_axes([margin_x, margin_y, 1 - 2 * margin_x, 1 - 2 * margin_y])
ax.imshow(image, extent = extent, alpha = alpha)

# Plot the geophones
for station, row in geophone_df.iterrows():
    east = row["east"]
    north = row["north"]

    if station == station_highlight:
        zorder = 2
        color = color_station
    else:
        zorder = 1
        color = "lightgray"

    ax.scatter(east, north,
               marker = "^",
               s = markersize_station,
               c = color,
               edgecolor = "black",
               linewidth = linewidth_marker_map,
               zorder = zorder)

    if station == station_highlight:
        ax.annotate(station,
                    (east, north),
                    textcoords = "offset points",
                    xytext = (station_label_x, station_label_y),
                    fontsize = station_label_size, fontweight = "bold",
                    ha = "right", va = "bottom", zorder = 2)

# Plot the boreholes
ax.scatter(borehole_df["east"], borehole_df["north"],
           marker = "o",
           s = markersize_borehole,
           c = "lightgray",
           edgecolor = "black",
           linewidth = linewidth_marker_map,
           zorder = 1)

# Plot the hammers
for _, row in hammer_loc_df.iterrows():
    east = row["east"]
    north = row["north"]
    hammer_id = row["hammer_id"]

    zorder = 1
    ax.scatter(east, north,
               marker = "*",
               s = markersize_hammer,
               c = color_hammer_marker,
               edgecolor = "black",
               linewidth = linewidth_marker_map,
               zorder = zorder)

# Set the limits of the axes
ax.set_xlim(min_east, max_east)
ax.set_ylim(min_north, max_north)

# Format the axes
format_east_xlabels(ax)
format_north_ylabels(ax)

# Add the panel labels
ax.text(panel_label_x1, panel_label_y1, "(a)",
        transform = ax.transAxes,
        fontsize = panel_label_size, fontweight = "bold",
        ha = "right", va = "bottom", zorder = 1)

###
# Plot the PSD vs distance
###

# Add the axis
ax = ax.inset_axes([power_decay_x, power_decay_y, power_decay_width, power_decay_height])

# Plot the hammer PSD vs distance
for _, row in psd_dist_df.iterrows():
    hammer_id = row["hammer_id"]
    distance = row["distance"]
    psd = row["psd"]

    ax.scatter(distance, psd,
                marker = "o",
                s = markersize_psd,
                color = color_hammer_psd,
                linewidth = linewidth_marker_psd, 
                edgecolor = "black",
                zorder = 2)

# Plot the reference line
min_distance = psd_dist_df["distance"].min()
max_distance = psd_dist_df["distance"].max()
ax.plot([min_distance, max_distance],
        [psd_ref + psd_slope * min_distance, psd_ref + psd_slope * max_distance],
        color = color_reference,
        linewidth = linewidth_psd, linestyle = "--",
        zorder = 1)

# Add the reference label
ax.text(ref_label_x, ref_label_y, f"{psd_slope:.1f} dB m$^{{-1}}$",
        fontsize = ref_label_size, fontweight = "bold", color = color_reference,
        ha = "left", va = "bottom", zorder = 1)

# Format the labels
ax.set_xlabel(f"Distance to {station_highlight} (m)", fontsize = axis_label_size)
ax.set_ylabel(f"{psd_label}", fontsize = axis_label_size)

# Set the title
ax.set_title(f"Hammer PSD at {freq_target:.0f} Hz vs distance to {station_highlight}",
             fontsize = title_size,
             fontweight = "bold")

# Add the panel labels
ax.text(panel_label_x2, panel_label_y2, "(b)",
        transform = ax.transAxes,
        fontsize = panel_label_size, fontweight = "bold",
        ha = "right", va = "bottom", zorder = 1)

# Save the figure
save_figure(fig, "hammer_power_decay.png")
