"""
Plot the power and frequency variations of two modes as a function of time for a geophone station
"""

# Imports
from os.path import join
from argparse import ArgumentParser
from json import loads
from numpy import zeros, fill_diagonal
from pandas import Timedelta
from pandas import read_csv, read_hdf
from matplotlib import colormaps
from matplotlib.pyplot import figure, subplots, get_cmap
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import get_geophone_coords, get_sunrise_sunset_times
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import format_datetime_xlabels, format_db_ylabels, add_day_night_shading, add_colorbar, save_figure

# Inputs
# Command-line arguments
parser = ArgumentParser(description = "Plot the stationary resonance properties of a mode vs time for all geophone stations.")
parser.add_argument("--station", type = str, help = "Station to plot")

parser.add_argument("--base_mode_name", type = str, default = "PR02549", help = "Base mode name.")
parser.add_argument("--base_mode_order", type = int, default = 2, help = "Base mode order.")
parser.add_argument("--mode_order1", type = int, default = 2, help = "Order of the first mode")
parser.add_argument("--mode_order2", type = int, default = 3, help = "Order of the second mode")

parser.add_argument("--window_length", type = float, default = 300.0, help = "Window length in seconds.")
parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap in seconds.")
parser.add_argument("--min_prom", type = float, default = 15.0, help = "Minimum prominence for plotting the power variation.")
parser.add_argument("--min_rbw", type = float, default = 15.0, help = "Minimum relative bandwidth for plotting the power variation.")
parser.add_argument("--max_mean_db", type = float, default = 15.0, help = "Maximum mean power for plotting the power variation.")


parser.add_argument("--figwidth", type = float, default = 15, help = "Figure width")
parser.add_argument("--figheight", type = float, default = 5, help = "Figure height")
parser.add_argument("--max_power", type = float, default = 45.0, help = "Maximum power for the power axis.")
parser.add_argument("--min_power", type = float, default = 5.0, help = "Maximum power for the power axis.")
parser.add_argument("--marker_size", type = float, default = 10.0, help = "Marker size.")
parser.add_argument("--min_corr", type=float, default=0.0, help="Minimum correlation value.")
parser.add_argument("--max_corr", type=float, default=1.0, help="Maximum correlation value.")

# Constants
axis_label_size = 10.0
tick_label_size = 8.0
title_size = 12.0

# Parse the arguments
args = parser.parse_args()

station = args.station

base_mode_name = args.base_mode_name
base_mode_order = args.base_mode_order
mode_order1 = args.mode_order1
mode_order2 = args.mode_order2
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db
figwidth = args.figwidth
figheight = args.figheight
min_power = args.min_power
max_power = args.max_power
min_corr = args.min_corr
max_corr = args.max_corr
marker_size = args.marker_size  

# Reading the input
## Read the resonance frequencies
filename = f"stationary_harmonic_series_{base_mode_name}_base{base_mode_order}.csv"
filepath = join(indir, filename)

harmonic_df = read_csv(filepath)

mode_name1 = harmonic_df.loc[harmonic_df["mode_order"] == mode_order1, "mode_name"].values[0]
mode_name2 = harmonic_df.loc[harmonic_df["mode_order"] == mode_order2, "mode_name"].values[0]

## Read the stationary resonance properties
print("Reading the stationary resonance properties...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

filename_in = f"stationary_resonance_properties_geo_{mode_name1}_{suffix_spec}_{suffix_peak}.h5"
inpath = join(indir, filename_in)
resonance1_df = read_hdf(inpath, key = "properties")

filename_in = f"stationary_resonance_properties_geo_{mode_name2}_{suffix_spec}_{suffix_peak}.h5"
inpath = join(indir, filename_in)
resonance2_df = read_hdf(inpath, key = "properties")

## Get the properties for the station
resonance1_df = resonance1_df[ resonance1_df["station"] == station]
resonance2_df = resonance2_df[ resonance2_df["station"] == station]

## Read the cross-correlation results
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

filename = f"stationary_harmonic_avg_power_corr_{base_mode_name}_base{base_mode_order}_{suffix_spec}_{suffix_peak}.csv"
filepath = join(indir, filename)
corr_df = read_csv(filepath)

## Assemble the correlation matrix
num_harmonics = corr_df["mode_i_order"].unique().shape[0] + 1
corr_mat = zeros((num_harmonics, num_harmonics))

mode_order_dict = {}
for _, row in corr_df.iterrows():
    mode_i_index = int(row["mode_i_index"])
    mode_j_index = int(row["mode_j_index"])

    corr = row["correlation"]

    corr_mat[mode_i_index, mode_j_index] = corr
    corr_mat[mode_j_index, mode_i_index] = corr

    mode_order_dict[mode_i_index] = int(row["mode_i_order"])
    mode_order_dict[mode_j_index] = int(row["mode_j_order"])

fill_diagonal(corr_mat, 1.0)

# Plotting
## Plot the properties
fig, axs = subplots(2, 1, figsize = (figwidth, figheight))
fig.subplots_adjust(hspace = 0.15)

times1 = resonance1_df["time"]
powers1 = resonance1_df["total_power"]
times2 = resonance2_df["time"]
powers2 = resonance2_df["total_power"]

cmap_power = "inferno"

## Mode 2
axs[0].scatter(times1, powers1, marker = "o", c = powers1, cmap = cmap_power, edgecolors = "black",
            label = mode_name1, vmin = min_power, vmax = max_power,
            zorder = 2, s = marker_size)

add_day_night_shading(axs[0])
axs[0].set_ylim([min_power, max_power])
axs[0] = format_db_ylabels(axs[0],
                            plot_axis_label = True, plot_tick_label = True,
                            major_tick_spacing = 10, num_minor_ticks = 5,
                            axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                            major_tick_length = 5, minor_tick_length = 2.5, tick_width = 1)

format_datetime_xlabels(axs[0],
                        plot_axis_label = False, plot_tick_label = False,
                        major_tick_spacing = "5d", num_minor_ticks = 5,
                        date_format = "%Y-%m-%d", 
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                        va = "top", ha = "right", rotation = 30)

axs[0].set_title(f"Mode {mode_order1}, {station}", fontsize = title_size, fontweight = "bold")

## Plot the correlation matrix in an inset axis
ax_corr = axs[0].inset_axes([-0.03, 0.45, 0.5, 0.5])
cmap = colormaps["viridis"]

mappable = ax_corr.matshow(corr_mat, cmap=cmap, vmin=min_corr, vmax=max_corr)
ax_corr.set_xticklabels([])

tick_labels = []
for i, mode_order in mode_order_dict.items():
    if i == 0:
        tick_labels.append(f"Mode {mode_order:d}")
    else:
        tick_labels.append(f"{mode_order:d}")

print(tick_labels)
ax_corr.set_yticks(range(num_harmonics))
ax_corr.set_yticklabels(tick_labels, rotation=0, ha="right", va="center", fontweight="bold", fontsize=10)

for i, label in enumerate(ax_corr.get_yticklabels()):
    if i == 0:
        label.set_color("crimson")
    else:
        label.set_color("black")

# Add a colorbar
bbox = ax_corr.get_position()
position = [bbox.x1 - 0.05, bbox.y0, 0.01, bbox.height]
cax = fig.add_axes(position)
cbar = fig.colorbar(mappable, cax=cax, orientation="vertical")
cbar.set_label("Power correlation", fontsize=axis_label_size)
cbar.ax.tick_params(labelsize=tick_label_size)

# Add the subplot label
axs[0].text(-0.05, 1.05, "(a)", fontsize = title_size, fontweight = "bold", transform = axs[0].transAxes)

## Mode 3
axs[1].scatter(times2, powers2, marker = "o", c = powers2, cmap = cmap_power, edgecolors = "black",
            label = mode_name2, vmin = min_power, vmax = max_power,
            zorder = 1, s = marker_size)

add_day_night_shading(axs[1])
axs[1].set_ylim([min_power, max_power])

axs[1] = format_db_ylabels(axs[1],
                            plot_axis_label = True, plot_tick_label = True,
                            major_tick_spacing = 10, num_minor_ticks = 5,
                            axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                            major_tick_length = 5, minor_tick_length = 2.5, tick_width = 1)

format_datetime_xlabels(axs[1], 
                        major_tick_spacing = "5d", num_minor_ticks = 5,
                        date_format = "%Y-%m-%d", 
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size)

axs[1].set_title(f"Mode {mode_order2}, {station}", fontsize = title_size, fontweight = "bold")

# Add the subplot label
axs[1].text(-0.05, 1.05, "(b)", fontsize = title_size, fontweight = "bold", transform = axs[1].transAxes)

## Save the figure
save_figure(fig, f"jgr_mode_power_n_frequency_variations_{mode_name1}_{mode_name2}.png")