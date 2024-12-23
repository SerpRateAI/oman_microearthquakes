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

# Parse the arguments
args = parser.parse_args()
mode_name = args.mode_name
min_power = args.min_power
max_power = args.max_power


# Print the inputs
print(f"### Plotting the stationary resonance properties of {mode_name} vs time for all geophone stations ###")
print("")

# Read the sunrise and sunset times
print("Reading the sunrise and sunset times...")
sun_df = get_geo_sunrise_sunset_times()

# Read the stationary resonance properties
print("Reading the stationary resonance properties...")
filename_in = f"stationary_resonance_properties_{mode_name}_geo.h5"
inpath = join(indir, filename_in)
resonance_df = read_hdf(inpath, key = "properties")

# Get the station coordinates
coord_df = get_geophone_coords()
coord_df.sort_values(by = "north", inplace = True)

# Plot the frequency variation
print("Plotting the frequency variation...")
fig_freq, ax_freq, cbar_freq = plot_stationary_resonance_properties_vs_time("frequency", resonance_df)
                     
# Save the figure
print("Saving the figure...")
figname = f"stationary_resonance_freq_vs_time_{mode_name}_geo.png"
save_figure(fig_freq, figname, dpi = 600)

# Plot the power variation
print("Plotting the power variation...")
fig_power, ax_power, cbar_power = plot_stationary_resonance_properties_vs_time("total_power", resonance_df,
                                                                                min_power = min_power, max_power = max_power)

# Save the figure
print("Saving the figure...")
figname = f"stationary_resonance_power_vs_time_{mode_name}_geo.png"
save_figure(fig_power, figname, dpi = 600)

# Plot the quality factor variation
print("Plotting the quality factor variation...")
fig_qf, ax_qf, cbar_qf = plot_stationary_resonance_properties_vs_time("quality_factor", resonance_df)

# Save the figure
print("Saving the figure...")
figname = f"stationary_resonance_qf_vs_time_{mode_name}_geo.png"
save_figure(fig_qf, figname, dpi = 600)


