"""
Plot the spectra of a template event.
"""
#------------------------------------------------------------------------------
# Import packages
#------------------------------------------------------------------------------

import argparse
from pathlib import Path
from numpy import log10, logspace
from scipy.signal.windows import dpss
from obspy import Stream
from matplotlib.pyplot import subplots

from utils_basic import (
    SAMPLING_RATE as sampling_rate,
    PICK_DIR as dirpath_pick,
    GEO_COMPONENTS as components,
    power2db,
)

from utils_snuffler import read_time_windows
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_mt import mt_autospec
from utils_plot import get_geo_component_color, format_freq_xlabels, format_db_ylabels, save_figure

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------

# Get the frequency scaling for converting velocity spectra to displacement spectra
def get_freq_scaling(min_freq = 50.0, max_freq = 100.0, db0 = -35.0):
    freqax = logspace(log10(min_freq), log10(max_freq), 10)
    scalings = power2db(freqax ** 2) + db0

    return freqax, scalings

#------------------------------------------------------------------------------
# Parse command-line arguments
#------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--template_id", type=str, help="Template ID")
parser.add_argument("--window_length", type=float, help="Window length in seconds", default=0.1)

parser.add_argument("--nw", type=int, help="Number of tapers", default=3)

parser.add_argument("--figwidth", type=float, help="Figure width", default=15.0)
parser.add_argument("--figheight", type=float, help="Figure height", default=7.5)
parser.add_argument("--linewidth", type=float, help="Line width", default=1.0)
parser.add_argument("--max_db", type=float, help="Maximum dB", default=45.0)
parser.add_argument("--min_db", type=float, help="Minimum dB", default=-20.0)
parser.add_argument("--fontsize_axis_label", type=float, help="Font size of axis label", default=12.0)
parser.add_argument("--fontsize_annotation", type=float, help="Font size of annotation", default=14.0)

args = parser.parse_args()

template_id = args.template_id

window_length = args.window_length
nw = args.nw

figwidth = args.figwidth
figheight = args.figheight
linewidth = args.linewidth

min_db = args.min_db
max_db = args.max_db
fontsize_axis_label = args.fontsize_axis_label
fontsize_annotation = args.fontsize_annotation

#------------------------------------------------------------------------------
# Read the input data
#------------------------------------------------------------------------------
# Load the matched events
filename = f"template_windows_{template_id}.txt"
filepath = Path(dirpath_pick) / filename
template_window_df = read_time_windows(filepath)
stations = template_window_df["station"].unique()

stream = Stream()
for _, row in template_window_df.iterrows():
    station = row["station"]
    starttime = row["starttime"]

    print(f"Reading waveform for {station} at {starttime}")
    stream_station = read_and_process_windowed_geo_waveforms(starttime, dur = window_length, stations = [station])
    stream.extend(stream_station)

#------------------------------------------------------------------------------
# Compute the spectra using the multitaper method
#------------------------------------------------------------------------------
print(f"Computing the spectra using the multitaper method")
aspec_param_dict = {}
for station in stations:
    print(f"Computing the spectra for {station}")

    aspec_param_dict[station] = {}
    for component in components:
        trace = stream.select(station=station, component=component)[0]
        waveform = trace.data
        num_pts = len(waveform)

        taper_mat, ratio_vec = dpss(num_pts, nw, 2 * nw - 1, return_ratios=True)
        aspec_param = mt_autospec(waveform, taper_mat, ratio_vec, sampling_rate)
        aspec_param_dict[station][component] = aspec_param

#------------------------------------------------------------------------------
# Plot the spectra
#------------------------------------------------------------------------------
print(f"Plotting the spectra")

# Generate the figure
fig, axs = subplots(1, 3, figsize=(figwidth, figheight), sharex=True)

# Plot the spectra
for station in stations:
    for i, component in enumerate(components):
        aspec_param = aspec_param_dict[station][component]
        freqax = aspec_param.freqax
        aspec = aspec_param.aspec

        aspec_db = power2db(aspec)
        color = get_geo_component_color(component)

        #axs[i].plot(freqax, aspec_db, color=color, linewidth=linewidth)
        axs[i].plot(freqax[1:], aspec_db[1:], color=color, linewidth=linewidth)

# Plot the frequency scaling
freqax, scalings = get_freq_scaling()
axs[0].plot(freqax, scalings, color="crimson", linewidth=2 * linewidth, linestyle="--")
axs[0].text(0.6, 0.4, r"$\propto f^2$", fontsize=fontsize_annotation, transform=axs[0].transAxes, color="crimson", ha="left", va="top")

# Set the y-axis labels
for i, component in enumerate(components):
    ax = axs[i]

    ax.set_xlim(1 / window_length, sampling_rate / 2)
    ax.set_ylim(min_db, max_db)

    #format_freq_xlabels(ax)

    ax.set_xscale("log")
    ax.set_xlabel(f"Frequency (Hz)", fontsize=fontsize_axis_label)
    ax.grid(True, linestyle="--", alpha=0.5, which="major")
    ax.grid(True, linestyle=":", alpha=0.5, which="minor")

    if i == 0:
        format_db_ylabels(ax)
    else:
        format_db_ylabels(ax, plot_tick_label=False, plot_axis_label=False)

# Set the title
fig.suptitle(f"Template {template_id}", fontsize=16, fontweight="bold", y=0.95)

# Save the figure
figname = f"template_mt_spectra_{template_id}.png"
save_figure(fig, figname)