"""
Plot the pow
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
from utils_plot import save_figure

# Input arguments
parser = ArgumentParser(description = "Plot the station average mode powers for the JGR paper")
parser.add_argument("--mode_order1", type = int, help = "Mode order", default = 2)
parser.add_argument("--mode_order2", type = int, help = "Mode order", default = 3)
parser.add_argument("--mode_order3", type = int, help = "Mode order", default = 12)

parser.add_argument("--base_mode_order", type = int, help = "Base mode order", default = 2)
parser.add_argument("--base_mode_name", type = str, help = "Base mode name", default = "PR02549")
parser.add_argument("--window_length", type = float, help = "Window length in seconds", default = 300.0)
parser.add_argument("--overlap", type = float, help = "Overlap in seconds", default = 0.0)
parser.add_argument("--min_prom", type = float, help = "Minimum prominence", default = 15.0)
parser.add_argument("--min_rbw", type = float, help = "Minimum reverse bandwidth", default = 15.0)
parser.add_argument("--max_mean_db", type = float, help = "Maximum mean dB", default = 15.0)
parser.add_argument("--figwidth", type = float, help = "Figure width", default = 15.0)

# Parse the arguments
args = parser.parse_args()

mode_order1 = args.mode_order1
mode_order2 = args.mode_order2
mode_order3 = args.mode_order3
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db
base_mode_order = args.base_mode_order
base_mode_name = args.base_mode_name
figwidth = args.figwidth

margin_x = 0.02
margin_y = 0.02

hspace = 0.01

# Print the inputs
print("###")
print("Plotting the station average mode powers for the JGR paper.")
print(f"Base mode name: {base_mode_name}.")
print(f"Base mode order: {base_mode_order:d}.")
print(f"Mode order 1: {mode_order1:d}.")
print(f"Mode order 2: {mode_order2:d}.")
print(f"Mode order 3: {mode_order3:d}.")
print(f"Window length: {window_length:f} seconds.")
print(f"Overlap: {overlap:f} seconds.")
print(f"Minimum prominence: {min_prom:f} Hz.")
print(f"Minimum reverse bandwidth: {min_rbw:f} Hz.")
print(f"Maximum mean dB: {max_mean_db:f} dB.")

# Read the data
print("Reading the harmonic series...")
filename = f"stationary_harmonic_series_{base_mode_name}_base{base_mode_order}.csv"
filepath = join(indir, filename)
harmonic_df = read_csv(filepath)

mode_name1 = harmonic_df.loc[harmonic_df["mode_order"] == mode_order1, "mode_name"].values[0]
mode_name2 = harmonic_df.loc[harmonic_df["mode_order"] == mode_order2, "mode_name"].values[0]
mode_name3 = harmonic_df.loc[harmonic_df["mode_order"] == mode_order3, "mode_name"].values[0]

# Read the data
print("Reading the properties of individual modes...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

filename = f"stationary_resonance_properties_geo_{mode_name1}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(indir, filename)
resonance1_df = read_hdf(filepath, key = "properties")

filename = f"stationary_resonance_properties_geo_{mode_name2}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(indir, filename)
resonance2_df = read_hdf(filepath, key = "properties")

filename = f"stationary_resonance_properties_geo_{mode_name3}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(indir, filename)
resonance3_df = read_hdf(filepath, key = "properties")

# Get the station coordinates
coord_df = get_geophone_coords()

# Plot the data
## Compute the figure height
## Create the figure
aspect_ratio = (max_north - min_north) / (max_east - min_east)
figheight = figwidth * (1 - 2 * margin_x - 2 * hspace) / 3 / (1 - 2 * margin_y) * aspect_ratio
map_width = (1 - 2 * margin_x - 2 * hspace) / 3
map_height = 1 - 2 * margin_y

# Create the figure
fig = figure(figsize = (figwidth, figheight))

# Add the subplots for the maps
ax_map1 = fig.add_axes([margin_x, margin_y, map_width, map_height], facecolor = "lightgray")
ax_map2 = fig.add_axes([margin_x + map_width + hspace, margin_y, map_width, map_height], facecolor = "lightgray")
ax_map3 = fig.add_axes([margin_x + 2 * map_width + 2 * hspace, margin_y, map_width, map_height], facecolor = "lightgray")

# Plot the first mode
cmap = colormaps["inferno"]
min_power = resonance1_df["total_power"].min()
max_power = resonance1_df["total_power"].max()
for station, coords in coord_df.iterrows():
    power = resonance1_df.loc[resonance1_df["station"] == station, "total_power"].mean()
    ax_map1.scatter(coords["east"], coords["north"], marker = "^", c = power, cmap = cmap, norm = Normalize(vmin = min_power, vmax = max_power), edgecolors = "black", linewidth = 0.5)

ax_map1.set_aspect("equal")

# cbar = fig.colorbar(ScalarMappable(cmap = cmap, norm = Normalize(vmin = min_power, vmax = max_power)), ax = ax_map1)


# Plot the second mode
## Compute the mean power for each station
mean_power_dict = {}
for station, coords in coord_df.iterrows():
    power = resonance2_df.loc[resonance2_df["station"] == station, "total_power"].mean()
    mean_power_dict[station] = power

min_power = min(mean_power_dict.values())
max_power = max(mean_power_dict.values())
for station, coords in coord_df.iterrows():
    power = mean_power_dict[station]
    ax_map2.scatter(coords["east"], coords["north"], marker = "^", c = power, cmap = cmap, norm = Normalize(vmin = min_power, vmax = max_power), edgecolors = "black", linewidth = 0.5)

ax_map2.set_aspect("equal")

# cbar = fig.colorbar(ScalarMappable(cmap = cmap, norm = Normalize(vmin = min_power, vmax = max_power)), ax = ax_map2)

# Plot the third mode
## Compute the mean power for each station
mean_power_dict = {}
for station, coords in coord_df.iterrows():
    power = resonance3_df.loc[resonance3_df["station"] == station, "total_power"].mean()
    mean_power_dict[station] = power

min_power = min(mean_power_dict.values())
max_power = max(mean_power_dict.values())
for station, coords in coord_df.iterrows():
    power = mean_power_dict[station]
    ax_map3.scatter(coords["east"], coords["north"], marker = "^", c = power, cmap = cmap, norm = Normalize(vmin = min_power, vmax = max_power), edgecolors = "black", linewidth = 0.5)

ax_map3.set_aspect("equal")

# cbar = fig.colorbar(ScalarMappable(cmap = cmap, norm = Normalize(vmin = min_power, vmax = max_power)), ax = ax_map3)

# Save the figure
print("Saving the figure...")
figname = f"jgr_station_average_mode_powers_modes_{mode_order1}_{mode_order2}_{mode_order3}.png"
save_figure(fig, figname)