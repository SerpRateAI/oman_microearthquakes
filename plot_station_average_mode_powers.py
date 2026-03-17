"""
Plot the average power of a mode at each station.
"""

from argparse import ArgumentParser
from pandas import read_csv, read_hdf
from os.path import join
from matplotlib.pyplot import subplots, figure, colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from utils_basic import SPECTROGRAM_DIR as indir, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north, EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east
from utils_basic import get_geophone_coords
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix 
from utils_plot import save_figure, format_east_xlabels, format_north_ylabels, add_colorbar

# Input arguments
parser = ArgumentParser(description = "Plot the average power of a mode at each station")
parser.add_argument("--mode_order", type = int, help = "Mode order", default = 2)
parser.add_argument("--base_mode_order", type = int, help = "Base mode order", default = 2)
parser.add_argument("--base_mode_name", type = str, help = "Base mode name", default = "PR02549")
parser.add_argument("--window_length", type = float, help = "Window length in seconds", default = 300.0)
parser.add_argument("--overlap", type = float, help = "Overlap in seconds", default = 0.0)
parser.add_argument("--min_prom", type = float, help = "Minimum prominence", default = 15.0)
parser.add_argument("--min_rbw", type = float, help = "Minimum reverse bandwidth", default = 15.0)
parser.add_argument("--max_mean_db", type = float, help = "Maximum mean dB", default = 15.0)
parser.add_argument("--figwidth", type = float, help = "Figure width", default = 5.0)

# Parse the arguments
args = parser.parse_args()

mode_order = args.mode_order
base_mode_order = args.base_mode_order
base_mode_name = args.base_mode_name
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db
figwidth = args.figwidth

margin_x = 0.02
margin_y = 0.02

axis_label_size = 12
tick_label_size = 10

fontsize_title = 14

cbar_offset_x = 0.07
cbar_offset_y = 0.07
cbar_width = 0.02
cbar_height = 0.3

# Print the inputs
print("###")
print("Plotting the average power of a mode at each station.")
print(f"Mode order: {mode_order:d}.")
print(f"Base mode order: {base_mode_order:d}.")
print(f"Base mode name: {base_mode_name}.")
print(f"Window length: {window_length:f} seconds.")
print(f"Overlap: {overlap:f} seconds.")
print(f"Minimum prominence: {min_prom:f} Hz.")
print(f"Minimum reverse bandwidth: {min_rbw:f} Hz.")
print(f"Maximum mean dB: {max_mean_db:f} dB.")
print(f"Figure width: {figwidth:f} inches.")

# Read the harmonic series
print("Reading the harmonic series...")
filename = f"stationary_harmonic_series_{base_mode_name}_base{base_mode_order}.csv"
filepath = join(indir, filename)
harmonic_df = read_csv(filepath)

mode_name = harmonic_df.loc[harmonic_df["mode_order"] == mode_order, "mode_name"].values[0]
freq = harmonic_df.loc[harmonic_df["mode_order"] == mode_order, "observed_freq"].values[0]

# Read the properties of the mode
print("Reading the properties of the mode...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(indir, filename)
resonance_df = read_hdf(filepath, key = "properties")

# Get the station coordinates
coord_df = get_geophone_coords()

# Plot the data
## Compute the figure height
aspect_ratio = (max_north - min_north) / (max_east - min_east)
figheight = figwidth * (1 - 2 * margin_x) * aspect_ratio / (1 - 2 * margin_y)
map_width = 1 - 2 * margin_x
map_height = 1 - 2 * margin_y

# Create the figure
fig = figure(figsize = (figwidth, figheight))

# Add the subplot for the map
ax_map = fig.add_axes([margin_x, margin_y, map_width, map_height], facecolor = "lightgray")

# Compute the mean power for each station
mean_power_dict = {}
for station, coords in coord_df.iterrows():
    power = resonance_df.loc[resonance_df["station"] == station, "total_power"].mean()
    mean_power_dict[station] = power

min_power = min(mean_power_dict.values())
max_power = max(mean_power_dict.values())

# Plot the data
cmap = colormaps["inferno"]
for station, coords in coord_df.iterrows():
    power = mean_power_dict[station]
    ax_map.scatter(coords["east"], coords["north"], marker = "^", c = power, cmap = cmap, norm = Normalize(vmin = min_power, vmax = max_power), edgecolors = "black", linewidth = 0.5)

ax_map.set_aspect("equal")
ax_map.set_xlim(min_east, max_east)
ax_map.set_ylim(min_north, max_north)

# Format the axis labels
format_east_xlabels(ax_map, axis_label_size = axis_label_size, tick_label_size = tick_label_size)
format_north_ylabels(ax_map, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Add a colorbar
bbox = ax_map.get_position()
position = [bbox.x0 + cbar_offset_x, bbox.y0 + cbar_offset_y, cbar_width, cbar_height]
add_colorbar(fig, position, "Average power (dB)",
             mappable = ScalarMappable(cmap = cmap, norm = Normalize(vmin = min_power, vmax = max_power)),
             orientation = "vertical",
             axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Set the title
ax_map.set_title(f"Mode {mode_order:d}, {freq:.2f} Hz", fontsize = fontsize_title, fontweight = "bold")

# Save the figure
print("Saving the figure...")
figname = f"station_average_mode_powers_{mode_name}.png"
save_figure(fig, figname)