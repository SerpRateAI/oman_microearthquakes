# Plot the variation of stationary harmonics as a function of time for all geophone stations
# The stations are plotted in the order of increasing north coordinate
# Imports
from os.path import join
from argparse import ArgumentParser
from json import loads
from pandas import Timedelta
from pandas import read_csv, read_hdf
from matplotlib.pyplot import figure, subplots, get_cmap
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import get_geophone_coords, get_geo_sunrise_sunset_times
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import plot_stationary_resonance_properties_vs_time, save_figure

# Inputs
# Command-line arguments
parser = ArgumentParser(description = "Plot the stationary resonance properties of a mode vs time for all geophone stations.")
parser.add_argument("--mode_name", type = str, help = "Mode name.")
parser.add_argument("--max_power", type = float, default = 15.0, help = "Maximum power for plotting the power variation.")
parser.add_argument("--min_power", type = float, default = -10.0, help = "Maximum quality factor for plotting the quality factor variation.")

parser.add_argument("--base_mode_name", type = str, default = "PR02549", help = "Base mode name.")
parser.add_argument("--base_mode_order", type = int, default = 2, help = "Base mode order.")

parser.add_argument("--window_length", type = float, default = 300.0, help = "Window length in seconds.")
parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap in seconds.")
parser.add_argument("--min_prom", type = float, default = 15.0, help = "Minimum prominence for plotting the power variation.")
parser.add_argument("--min_rbw", type = float, default = 15.0, help = "Minimum relative bandwidth for plotting the power variation.")
parser.add_argument("--max_mean_db", type = float, default = 15.0, help = "Maximum mean power for plotting the power variation.")

# Parse the arguments
args = parser.parse_args()
mode_name = args.mode_name
min_power = args.min_power
max_power = args.max_power
base_mode_name = args.base_mode_name
base_mode_order = args.base_mode_order
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Constants
axis_label_size = 8.0
tick_label_size = 6.0

# Print the inputs
print(f"### Plotting the stationary resonance properties of {mode_name} vs time for all geophone stations ###")
print("")

# Read the resonance frequencies
filename = f"stationary_harmonic_series_{base_mode_name}_base{base_mode_order}.csv"
filepath = join(indir, filename)

harmonic_df = read_csv(filepath)

mode_order = harmonic_df.loc[harmonic_df["mode_name"] == mode_name, "mode_order"].values[0]

# Read the sunrise and sunset times
print("Reading the sunrise and sunset times...")
sun_df = get_geo_sunrise_sunset_times()

# Read the stationary resonance properties
print("Reading the stationary resonance properties...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

filename_in = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
inpath = join(indir, filename_in)
resonance_df = read_hdf(inpath, key = "properties")

# Get the station coordinates
coord_df = get_geophone_coords()
coord_df.sort_values(by = "north", inplace = True)

# Plot the frequency variation
print("Plotting the frequency variation...")
fig_freq, ax_freq, cbar_freq = plot_stationary_resonance_properties_vs_time("frequency", resonance_df,
                                                                            title = f"Mode {mode_order}, frequency vs time",
                                                                            axis_label_size = axis_label_size, tick_label_size = tick_label_size)
                     
# Save the figure
print("Saving the figure...")
figname = f"stationary_resonance_freq_vs_time_{mode_name}_geo.png"
save_figure(fig_freq, figname, dpi = 600)

# Plot the power variation
print("Plotting the power variation...")
fig_power, ax_power, cbar_power = plot_stationary_resonance_properties_vs_time("total_power", resonance_df,
                                                                                title = f"Mode {mode_order}, power vs time",
                                                                                min_power = min_power, max_power = max_power,
                                                                                axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Save the figure
print("Saving the figure...")
figname = f"stationary_resonance_power_vs_time_{mode_name}_geo.png"
save_figure(fig_power, figname, dpi = 600)


