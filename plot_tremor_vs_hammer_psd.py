"""
Plot the comparison between the PSD of the tremor in a time window and a hammer
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
parser.add_argument("--hammer_highlight", type=str, help="The ID of the hammer to highlight")
parser.add_argument("--station_highlight", type=str, help="The station to highlight")
parser.add_argument("--starttime", type=str, help="The starttime of the tremor PSD")
parser.add_argument("--duration", type=float, help="The duration of the tremor PSD")

parser.add_argument("--base_name", type=str, help="The base mode name", default="PR02549")
parser.add_argument("--base_order", type=int, help="The base mode order", default=2)
parser.add_argument("--mode_order", type=int, help="The order of the tremor mode to plot", default=12)
parser.add_argument("--freq_target", type=float, help="The frequency of the PSD to plot", default=150.0)
parser.add_argument("--alpha", type=float, help="The transparency of the image", default=0.2)
parser.add_argument("--figwidth", type=float, help="The width of the figure", default=15)
parser.add_argument("--map_width_frac", type=float, help="The width of the map as a fraction of the figure width", default=0.6)
parser.add_argument("--margin_x", type=float, help="The margin of the figure on the x-axis", default=0.02)
parser.add_argument("--margin_y", type=float, help="The margin of the figure on the y-axis", default=0.02)
parser.add_argument("--wspace", type=float, help="The width of the space between the map and the subplots", default=0.05)
parser.add_argument("--hspace", type=float, help="The height of the space between the map and the subplots", default=0.1)

parser.add_argument("--markersize_hammer", type=float, help="The size of the markers", default=150)
parser.add_argument("--markersize_station", type=float, help="The size of the markers", default=100)
parser.add_argument("--markersize_borehole", type=float, help="The size of the markers", default=100)
parser.add_argument("--markersize_psd", type=float, help="The size of the markers", default=100)
parser.add_argument("--linewidth_marker_map", type=float, help="The width of the lines", default=0.75)
parser.add_argument("--linewidth_marker_psd", type=float, help="The width of the lines", default=0.75)
parser.add_argument("--linewidth_psd", type=float, help="The width of the lines", default=1.5)

parser.add_argument("--color_hammer_marker", type=str, help="The color of the markers", default="salmon")
parser.add_argument("--color_station", type=str, help="The color of the markers", default="orange")
parser.add_argument("--color_borehole", type=str, help="The color of the markers", default="violet")
parser.add_argument("--color_highlight", type=str, help="The color of the highlighted marker", default="crimson")
parser.add_argument("--cmap_name", type=str, help="The name of the colormap", default="Accent")
parser.add_argument("--i_color_hammer", type=int, help="The index of the color of the hammer", default=0)
parser.add_argument("--i_color_tremor", type=int, help="The index of the color of the tremor", default=1)

parser.add_argument("--station_label_size", type=float, help="The size of the station label", default=12)
parser.add_argument("--station_label_x", type=float, help="The x offset of the station label", default=0)
parser.add_argument("--station_label_y", type=float, help="The y offset of the station label", default=10)

parser.add_argument("--hammer_label_size", type=float, help="The size of the hammer label", default=12)
parser.add_argument("--hammer_label_x", type=float, help="The x offset of the hammer label", default=0)
parser.add_argument("--hammer_label_y", type=float, help="The y offset of the hammer label", default=10)

parser.add_argument("--psd_label_size", type=float, help="The size of the PSD label", default=12)
parser.add_argument("--psd_label_x", type=float, help="The x offset of the PSD label", default=0)
parser.add_argument("--psd_label_y", type=float, help="The y offset of the PSD label", default=5)

parser.add_argument("--mode_label_size", type=float, help="The size of the mode label", default=12)
parser.add_argument("--mode_label_y", type=float, help="The y offset of the mode label", default=37)

parser.add_argument("--axis_label_size", type=float, help="The size of the axis labels", default=12)
parser.add_argument("--title_size", type=float, help="The size of the title", default=14)
parser.add_argument("--legend_size", type=float, help="The size of the legend", default=12)

parser.add_argument("--min_freq", type=float, help="The minimum frequency to plot", default=0)
parser.add_argument("--max_freq", type=float, help="The maximum frequency to plot", default=200)
parser.add_argument("--min_psd", type=float, help="The minimum PSD to plot", default=-10)
parser.add_argument("--max_psd", type=float, help="The maximum PSD to plot", default=75)


args = parser.parse_args()

hammer_highlight = args.hammer_highlight
station_highlight = args.station_highlight
starttime = args.starttime
duration = args.duration

base_name = args.base_name
base_order = args.base_order
mode_order = args.mode_order
freq_target = args.freq_target
figwidth = args.figwidth
map_width_frac = args.map_width_frac
margin_x = args.margin_x
margin_y = args.margin_y
wspace = args.wspace
hspace = args.hspace

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
color_borehole = args.color_borehole
color_highlight = args.color_highlight
cmap_name = args.cmap_name
i_color_hammer = args.i_color_hammer
i_color_tremor = args.i_color_tremor

station_label_size = args.station_label_size
station_label_y = args.station_label_y
station_label_x = args.station_label_x

hammer_label_size = args.hammer_label_size
hammer_label_y = args.hammer_label_y
hammer_label_x = args.hammer_label_x

psd_label_size = args.psd_label_size
psd_label_y = args.psd_label_y
psd_label_x = args.psd_label_x

mode_label_size = args.mode_label_size
mode_label_y = args.mode_label_y

axis_label_size = args.axis_label_size
title_size = args.title_size
legend_size = args.legend_size
min_freq = args.min_freq
max_freq = args.max_freq
min_psd = args.min_psd
max_psd = args.max_psd

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

# Read the tremor PSD
filename = f"tremor_mt_aspec_{station_highlight}_{starttime}_{duration:.0f}s.csv"
filepath = join(dirpath_mt, filename)
tremor_df = read_csv(filepath)

# Read the hammer PSD
filename = f"hammer_mt_aspecs_{hammer_highlight}_{station_highlight}.csv"
filepath = join(dirpath_mt, filename)
hammer_psd_df = read_csv(filepath)

# Read the harmonic series
filename = f"stationary_harmonic_series_{base_name}_base{base_order}.csv"
filepath = join(dirpath_spec, filename)
harmonic_df = read_csv(filepath)

###
# Plot the map
###

# Generate the figure
map_aspect = (max_north - min_north) / (max_east - min_east)
figheight = figwidth * map_width_frac * map_aspect * (1 + 2 * margin_y)
fig = figure(figsize = (figwidth, figheight))

# Plot the map
map_height = 1 - 2 * margin_y
ax = fig.add_axes([margin_x, margin_y, map_width_frac, map_height])
ax.imshow(image, extent = extent, alpha = alpha)

# Plot the geophones
for station, row in geophone_df.iterrows():
    east = row["east"]
    north = row["north"]

    if station == station_highlight:
        edgecolor = color_highlight 
        zorder = 2
    else:
        edgecolor = "black"
        zorder = 1

    ax.scatter(east, north,
               marker = "^",
               s = markersize_station,
               c = color_station,
               edgecolor = edgecolor,
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
           c = color_borehole,
           edgecolor = "black",
           linewidth = linewidth_marker_map,
           zorder = 1)

# Plot the hammers
for _, row in hammer_loc_df.iterrows():
    east = row["east"]
    north = row["north"]
    hammer_id = row["hammer_id"]

    if hammer_id == hammer_highlight:
        edgecolor = color_highlight
        zorder = 2
    else:
        edgecolor = "black"
        zorder = 1

    ax.scatter(east, north,
               marker = "*",
               s = markersize_hammer,
               c = color_hammer_marker,
               edgecolor = edgecolor,
               linewidth = linewidth_marker_map,
               zorder = zorder)
    
    if hammer_id == hammer_highlight:
        ax.annotate(f"H{hammer_id}",
                    (east, north),
                    textcoords = "offset points",
                    xytext = (hammer_label_x, hammer_label_y),
                    fontsize = hammer_label_size, fontweight = "bold",
                    ha = "right", va = "bottom", zorder = 2)

# Set the limits of the axes
ax.set_xlim(min_east, max_east)
ax.set_ylim(min_north, max_north)

# Format the axes
format_east_xlabels(ax)
format_north_ylabels(ax)

###
# Plot the PSD vs distance
###

# Add the axis
width_frac = 1 - map_width_frac - wspace - 2 * margin_x
height_frac = (1 - 2 * margin_y - hspace) / 2
ax = fig.add_axes([margin_x + map_width_frac + wspace, margin_y, width_frac, height_frac])

# Get the colormap
cmap = colormaps[cmap_name]
color_hammer = cmap(i_color_hammer)
color_tremor = cmap(i_color_tremor)

# Plot the hammer PSD vs distance
for _, row in psd_dist_df.iterrows():
    hammer_id = row["hammer_id"]
    distance = row["distance"]
    psd = row["psd"]

    if hammer_id == hammer_highlight:
        ax.scatter(distance, psd,
                   marker = "o",
                   s = markersize_psd,
                   color = color_hammer,
                   linewidth = linewidth_marker_psd, 
                   edgecolor = "black",
                   zorder = 2)
        
        ax.annotate(f"H{hammer_id}",
                    (distance, psd),
                    textcoords = "offset points",
                    xytext = (psd_label_x, psd_label_y),
                    fontsize = psd_label_size, fontweight = "bold",
                    ha = "left", va = "bottom", zorder = 2)
    else:
        ax.scatter(distance, psd,
                   marker = "o",
                   s = markersize_psd,
                   color = "lightgray",
                   linewidth = linewidth_marker_psd, 
                   edgecolor = "black",
                   zorder = 1)
        

# Format the labels
ax.set_xlabel(f"Distance to {station_highlight} (m)", fontsize = axis_label_size)
ax.set_ylabel(f"{psd_label}", fontsize = axis_label_size)

# Set the title
ax.set_title(f"Hammer PSD at {freq_target:.0f} Hz vs distance to {station_highlight}",
             fontsize = title_size,
             fontweight = "bold")

###
# Plot the tremor and the hammer PSDs
###

# Add the axis
ax = fig.add_axes([margin_x + map_width_frac + wspace, margin_y + height_frac + hspace, width_frac, height_frac])

# Plot the tremor PSD
ax.plot(tremor_df["frequency"], tremor_df["aspec_total_mt"],
        color = color_tremor,
        linewidth = linewidth_psd,
        label = "Tremor")

# Plot the mode label
freq_mode = harmonic_df.loc[harmonic_df["mode_order"] == mode_order, "observed_freq"].values[0]
ax.text(freq_mode, mode_label_y, f"Mode {mode_order}",
        fontsize = mode_label_size, fontweight = "bold",
        ha = "center", va = "bottom")

# Plot the hammer PSD
ax.plot(hammer_psd_df["frequency"], hammer_psd_df["aspec_total"],
        color = color_hammer,
        linewidth = linewidth_psd,
        label = f"H{hammer_highlight}")

# Set the limits of the axes
ax.set_xlim(min_freq, max_freq)
ax.set_ylim(min_psd, max_psd)

# Format the labels
ax.set_xlabel(f"Frequency (Hz)", fontsize = axis_label_size)
ax.set_ylabel(f"{psd_label}", fontsize = axis_label_size)

# Add the legend
ax.legend(fontsize = legend_size, loc = "upper right")

# Set the title
ax.set_title(f"Tremor and hammer PSDs at {station_highlight}",
             fontsize = title_size,
             fontweight = "bold")
# Save the figure
save_figure(fig, "tremor_vs_hammer_psd.png")
