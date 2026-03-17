"""
Plot the mode quality factor vs. time for a given mode.
"""

from argparse import ArgumentParser
from os.path import join
from pandas import read_csv, read_hdf
from matplotlib.pyplot import figure, subplots
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
from pandas import Timedelta, Timestamp

from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import get_geophone_coords
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import format_datetime_xlabels, format_db_ylabels, add_day_night_shading, add_colorbar, save_figure

# Inputs
# Command-line arguments
parser = ArgumentParser(description = "Plot the stationary resonance properties of a mode vs time for all geophone stations.")
parser.add_argument("--station", type = str, help = "Station to plot")

parser.add_argument("--base_mode_name", type = str, default = "PR02549", help = "Base mode name.")
parser.add_argument("--base_mode_order", type = int, default = 2, help = "Base mode order.")
parser.add_argument("--mode_order", type = int, default = 2, help = "Order of the mode")

parser.add_argument("--window_length", type = float, default = 300.0, help = "Window length in seconds.")
parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap in seconds.")
parser.add_argument("--min_prom", type = float, default = 15.0, help = "Minimum prominence for plotting the power variation.")
parser.add_argument("--min_rbw", type = float, default = 15.0, help = "Minimum relative bandwidth for plotting the power variation.")
parser.add_argument("--max_mean_db", type = float, default = 15.0, help = "Maximum mean power for plotting the power variation.")
parser.add_argument("--linewidth_plot", type = float, default = 1.0, help = "Linewidth for plotting the quality factor.")

parser.add_argument("--figwidth", type = float, default = 15, help = "Figure width")
parser.add_argument("--figheight", type = float, default = 5, help = "Figure height")
parser.add_argument("--min_qf", type = float, default = 100.0, help = "Minimum quality factor for the quality factor axis.")
parser.add_argument("--max_qf", type = float, default = 5000.0, help = "Maximum quality factor for the quality factor axis.")

# Parse the arguments
args = parser.parse_args()

station = args.station
base_mode_name = args.base_mode_name
base_mode_order = args.base_mode_order
mode_order = args.mode_order
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db
linewidth_plot = args.linewidth_plot
figwidth = args.figwidth
figheight = args.figheight
max_qf = args.max_qf
min_qf = args.min_qf

# Constants
title_size = 14.0
axis_label_size = 12.0
tick_label_size = 10.0

# Reading the input data
filename = f"stationary_harmonic_series_{base_mode_name}_base{base_mode_order}.csv"
filepath = join(indir, filename)
harmonic_df = read_csv(filepath)

mode_name = harmonic_df.loc[harmonic_df["mode_order"] == mode_order, "mode_name"].values[0]

suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

filename_in = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
inpath = join(indir, filename_in)
resonance_df = read_hdf(inpath, key = "properties")
resonance_df = resonance_df.sort_values(by = "time")
print(f"Number of time windows with resonance detected: {len(resonance_df)}")

# Plotting the mode quality factor vs. time
fig, ax = subplots(figsize = (figwidth, figheight))
ax.plot(resonance_df["time"], resonance_df["quality_factor_z"], label = "Quality factor", linewidth = linewidth_plot)
ax.set_title(f"Mode {mode_order} quality factor vs. time", fontsize = title_size, fontweight = "bold", y = 1.02)


# Add the day-night shading
add_day_night_shading(ax)

# Format the x-axis labels
format_datetime_xlabels(ax, major_tick_spacing = "1d", num_minor_ticks = 4, date_format = "%Y-%m-%d", rotation = 30, ha = "right", va = "top")

# Format the y-axis labels
ax.set_ylabel("Quality factor", fontsize = axis_label_size)
ax.set_yscale("log")
ax.set_xlim(Timestamp(resonance_df["time"].min()), Timestamp(resonance_df["time"].min()) + Timedelta(days = 1))
ax.set_ylim(min_qf, max_qf)

# # Add the colorbar
# cbar = add_colorbar(fig, ax, "Quality factor", fontsize = axis_label_size)

# Save the figure
save_figure(fig, "jgr_mode_quality_factor_vs_time.png")