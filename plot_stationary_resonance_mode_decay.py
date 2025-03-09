"""
Plot the decay of a stationary resonance mode
"""
###
# Imports
###

from argparse import ArgumentParser
from pandas import Timestamp, Timedelta
from pandas import read_hdf
from os.path import join
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as dirname, GEO_COMPONENTS as components, str2timestamp
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_spec import get_spec_peak_file_suffix, get_spectrogram_file_suffix
from utils_plot import get_geo_component_color, save_figure

###
# Input arguments
###

# Command-line arguments

parser = ArgumentParser(description="Plot the decay of a stationary resonance mode")

parser.add_argument("--mode_name", type=str, help="Name of the mode to plot")
parser.add_argument("--station", type=str, help="Station name")
parser.add_argument("--time_window", type=str, help="Time window to plot")

parser.add_argument("--bandwidth", type=float, help="Bandwidth in Hz for filtering the signal around the mode frequency", default=1.0)
parser.add_argument("--window_length", type=float, help="Window length in seconds", default=300.0)
parser.add_argument("--overlap", type=float, help="Overlap percentage", default=0.0)
parser.add_argument("--min_prom", type=float, help="Minimum prominence for peak detection", default=15.0)
parser.add_argument("--min_rbw", type=float, help="Minimum reverse bandwidth for peak detection", default=15.0)
parser.add_argument("--max_mean_db", type=float, help="Maximum mean db for peak detection", default=10.0)

args = parser.parse_args()

mode_name = args.mode_name
station = args.station
time_window = str2timestamp(args.time_window)

bandwidth = args.bandwidth

window_length = args.window_length
overlap = args.overlap

min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Constants
figwidth = 10
figheight = 10

linewidth = 0.1

###
# Read and process the data
###

# Read the mode data
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(dirname, filename)

mode_df = read_hdf(filepath, key="properties")

# Get the mode frequency
freq_mode = mode_df.loc[(mode_df["time"] == time_window) & (mode_df["station"] == station), "frequency"].values[0]
print(freq_mode)

# Read the waveforms
starttime = time_window - Timedelta(seconds=window_length / 2)
stream = read_and_process_windowed_geo_waveforms(starttime, 
                                                 dur = 2 *window_length, stations = station,
                                                filter = True, zerophase = True, filter_type = "butter", min_freq = freq_mode - bandwidth / 2, max_freq = freq_mode + bandwidth / 2, corners = 4,
                                                normalize = True)
# Plot the waveforms
fig, axs = subplots(nrows=3, ncols=1, figsize=(figwidth, figheight), sharex=True)

for i, component in enumerate(components):
    ax = axs[i]
    color = get_geo_component_color(component)

    trace = stream.select(component=component)[0]
    time_ax = trace.times()
    waveform = trace.data

    ax.plot(time_ax, waveform, color=color, linewidth=linewidth)

    ax.set_xlim(time_ax[0], time_ax[-1])
    ax.set_ylim([-1.0, 1.0])

    if i == 2:
        ax.set_xlabel("Time (s)")

# Save the figure
figname = f"stationary_resonance_mode_decay_{mode_name}_{station}.png"
save_figure(fig, figname)