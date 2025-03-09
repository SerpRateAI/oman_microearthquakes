"""
Plot the frequency variation as a function of  
"""

### Import the necessary librariess ###
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, read_hdf
from matplotlib.pyplot import subplots
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import colorcet as cc

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import add_colorbar, add_vertical_scalebar, add_day_night_shading, format_datetime_xlabels, get_cmap_segment, save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser()
parser.add_argument("--base_mode", type = str, default = "PR02549", help = "The name of the base mode")
parser.add_argument("--base_order", type = int, default = 2, help = "The order of the base mode")
parser.add_argument("--window_length", type = float, default = 300.0, help = "The STFT window length (s)")
parser.add_argument("--overlap", type = float, default = 0.5, help = "The STFT overlap (0-1)")
parser.add_argument("--min_prom", type = float, default = 15.0, help = "The minimum prominence for peak detection")
parser.add_argument("--min_rbw", type = float, default = 1.0, help = "The minimum reverse bandwidth for peak detection")
parser.add_argument("--max_mean_db", type = float, default = 10.0, help = "The maximum mean power for ruling out a time window")

args = parser.parse_args()

base_mode = args.base_mode
base_order = args.base_order
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Constants
cmap_name = "isolum"
min_mean_freq = 0.0
max_mean_freq = 200.0

figwidth = 15.0
figheight = 15.0

axis_label_size = 12
tick_label_size = 10

mode_label_x = 0.005
mode_label_y = 0.95
mode_label_size = 12

marker_size = 3

cbar_offset_x = 0.0
cbar_offset_y = -0.08
cbar_width = 1.0
cbar_height = 0.01

### Read the input data ###
# Read the harmonic series information
print(f"Reading the information of the harmonic series with {base_mode} as the base mode {base_order}...")
filename = f"stationary_harmonic_series_{base_mode}_base{base_order:d}.csv"
inpath = join(indir, filename)
harmo_df = read_csv(inpath)

# Determine the number of modes to plot
harmo_df = harmo_df[ harmo_df["detected"] ]
harmo_df.reset_index(drop = True, inplace = True)
num_plot = len(harmo_df)
print(f"Number of modes to plot: {num_plot}")

### Plotting ###
# Generate the subplots
print("Generating the subplots...")
fig, axes = subplots(nrows = num_plot, ncols = 1, figsize = (figwidth, figheight), sharex = True)

# Generate the color map
cmap = cc.cm.isolum
norm = Normalize(vmin = min_mean_freq, vmax = max_mean_freq)

# Plot each mode
print("Plotting each mode...")
for i, row in harmo_df.iterrows():
    mode_name = row["mode_name"]
    mode_order = row["mode_order"]
    mean_freq = row["observed_freq"]

    # Read the frequencies of the mode
    print(f"Plotting the mode {mode_name}...")

    suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
    suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

    filename = f"stationary_resonance_profile_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
    inpath = join(indir, filename)
    current_mode_df = read_hdf(inpath, key = "profile")

    # Plot the frequency variation as a function of time
    freqs = current_mode_df["frequency"].values
    times = current_mode_df["time"].values

    ax = axes[num_plot - i - 1]
    color = cmap(norm(mean_freq))
    ax.scatter(times, freqs, color = color, s = marker_size)

    # Add the day-night shading
    print("Adding the day-night shading...")
    add_day_night_shading(ax)

    # Add the mode label
    if i == 0:
        ax.text(mode_label_x, mode_label_y, f"Mode {mode_order}", transform = ax.transAxes, ha = "left", va = "top", fontsize = mode_label_size, fontweight = "bold")
    else:
        ax.text(mode_label_x, mode_label_y, f"{mode_order}", transform = ax.transAxes, ha = "left", va = "top", fontsize = mode_label_size, fontweight = "bold")

# Set the axis limits
# axes[num_plot - 1].set_xlim(starttime, endtime)
# Format the x-axis labels
print("Formatting the x-axis labels...")

format_datetime_xlabels(axes[num_plot - 1],
                        axis_label_size = axis_label_size,
                        tick_label_size = tick_label_size,
                        major_tick_spacing="1d", num_minor_ticks=4,
                        date_format="%Y-%m-%d",
                        rotation=30, ha="right", va="top")

# Format the y-axis labels
axes[num_plot - 1].set_ylabel("Freq. (Hz)", fontsize = axis_label_size)

# Add the colorbar
print("Adding the colorbar...")
# Add colorbar axes
bbox = axes[num_plot - 1].get_position()
cbar_pos = [bbox.x0 + cbar_offset_x, bbox.y0 + cbar_offset_y, bbox.width, cbar_height]
cax = fig.add_axes(cbar_pos)

# Add the colorbar
mappable = ScalarMappable(norm = norm, cmap = cmap)
fig.colorbar(mappable, cax = cax, orientation = "horizontal")
# Set the colorbar label fontsize
cax.set_xlabel("Average frequency (Hz)", fontsize=axis_label_size)


# # Plot the colorbar
# print("Plotting the colorbar...")
# bbox = ax_freq.get_position()
# cbar_pos = [bbox.x1 + 0.02, bbox.y0, 0.01, bbox.height]
# add_colorbar(fig, cbar_pos,  "Mean frequency (Hz)", cmap = cmap, norm = norm)

# # Set the title
# print("Setting the title...")
# title = f"Harmonic relations between the stationary resonances with {base_mode} as Mode {base_number}"
# fig.suptitle(title, fontsize = 14, fontweight = "bold", y = 0.92)

# Save the figure
print("Saving the figure...")
filename = f"stationary_resonance_all_modes_vs_time_{base_mode}_base{base_order:d}.png"
save_figure(fig, filename)

    


    