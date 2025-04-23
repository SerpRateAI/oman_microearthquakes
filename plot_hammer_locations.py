"""
Plot the locations of the hammers
"""

from os.path import join
from utils_basic import LOC_DIR as dirpath_loc
from pandas import read_csv
from argparse import ArgumentParser
from matplotlib.pyplot import subplots

from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, get_borehole_coords
from utils_satellite import load_maxar_image
from utils_plot import save_figure, format_east_xlabels, format_north_ylabels

# Parse the command line arguments
parser = ArgumentParser()
parser.add_argument("--alpha", type=float, help="The transparency of the image", default=0.5)
parser.add_argument("--figwidth", type=float, help="The width of the figure", default=10)
parser.add_argument("--figheight", type=float, help="The height of the figure", default=10)
parser.add_argument("--markersize_hammer", type=float, help="The size of the markers", default=70)
parser.add_argument("--markersize_station", type=float, help="The size of the markers", default=70)
parser.add_argument("--markersize_borehole", type=float, help="The size of the markers", default=70)
parser.add_argument("--linewidth", type=float, help="The width of the lines", default=1)
parser.add_argument("--color_hammer", type=str, help="The color of the markers", default="salmon")
parser.add_argument("--color_station", type=str, help="The color of the markers", default="orange")
parser.add_argument("--color_borehole", type=str, help="The color of the markers", default="violet")

args = parser.parse_args()
figwidth = args.figwidth
figheight = args.figheight
alpha = args.alpha
markersize_hammer = args.markersize_hammer
markersize_station = args.markersize_station
markersize_borehole = args.markersize_borehole
linewidth = args.linewidth
color_hammer = args.color_hammer
color_station = args.color_station
color_borehole = args.color_borehole

# Read the hammers
filepath = join(dirpath_loc, "hammer_locations.csv")
hammers_df = read_csv(filepath, dtype = {"hammer_id": str})

# Load the geophone coordinates
station_df = get_geophone_coords()

# Load the borehole coordinates
borehole_df = get_borehole_coords()

# Load the satellite image
image, extent = load_maxar_image()

# Plot the image
fig, ax = subplots(1, 1, figsize = (figwidth, figheight))
ax.imshow(image, extent = extent, alpha = alpha)

# Plot the stations
ax.scatter(station_df["east"], station_df["north"], c = color_station, s = markersize_station, marker = "^", edgecolors = "black", linewidth = linewidth)

# Plot the boreholes
ax.scatter(borehole_df["east"], borehole_df["north"], c = color_borehole, s = markersize_borehole, marker = "o", edgecolors = "black", linewidth = linewidth)

# Plot the hammers
ax.scatter(hammers_df["east"], hammers_df["north"], c = color_hammer, s = markersize_hammer, marker = "*", edgecolors = "black", linewidth = linewidth)

# Set the x- and y-limits
ax.set_xlim(min_east, max_east)
ax.set_ylim(min_north, max_north)

# Format the x-labels
format_east_xlabels(ax)

# Format the y-labels
format_north_ylabels(ax)

# Save the figure
figname = "hammer_locations.png"
save_figure(fig, figname)
