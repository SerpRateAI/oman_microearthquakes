# Plot the 3C waveforms of a stationary resonance narrow-bandpass filtered at the peak frequency

# Imports
from os.path import join
from argparse import ArgumentParser
from scipy.signal import hilbert
from pandas import read_hdf
from matplotlib.pyplot import subplots

from utils_basic import GEO_COMPONENTS as components, SPECTROGRAM_DIR as indir
from utils_basic import get_datetime_axis_from_trace, get_geophone_coords, str2timestamp, time2suffix
from utils_spec import get_start_n_end_from_center
from utils_preproc import get_envelope, read_and_process_windowed_geo_waveforms
from utils_plot import format_datetime_xlabels, get_geo_component_color, save_figure

# Inputs
# Command line arguments
parser = ArgumentParser(description = "Plot the 3C waveforms of a stationary resonance narrow-bandpass filtered at the peak frequency")

parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--center_time", type = str, help = "Center time of the time window")
parser.add_argument("--window_length", type = float, help = "Length of the time window in seconds")
parser.add_argument("--decimate_factor", type = int, help = "Decimation factor")
parser.add_argument("--scale_factor", type = float, help = "Scale factor")
parser.add_argument("--panel_width", type = float, help = "Panel width in inches")
parser.add_argument("--panel_height", type = float, help = "Panel height in inches")
parser.add_argument("--linewidth_wf", type = float, help = "Line width of the waveforms")
parser.add_argument("--linewidth_env", type = float, help = "Line width of the envelopes")

# Parse the command line arguments
args = parser.parse_args()

mode_name = args.mode_name
center_time = str2timestamp(args.center_time)
window_length = args.window_length
decimate_factor = args.decimate_factor
scale_factor = args.scale_factor
panel_width = args.panel_width
panel_height = args.panel_height
linewidth_wf = args.linewidth_wf
linewidth_env = args.linewidth_env

# Read the station coordinates
print("Reading the station coordinates...")
coord_df = get_geophone_coords()
coord_df.sort_values(by = "north", inplace = True)

# Read the stationary resonance properties
print(f"Reading the properties of {mode_name}...")
filename = f"stationary_resonance_geo_mean_properties_{mode_name}.h5"
inpath = join(indir, filename)
properties_df = read_hdf(inpath, key = "properties")

# Get the center frequency and bandwidth
freq_reson = properties_df.loc[center_time, "frequency"]
qf_reson = properties_df.loc[center_time, "mean_quality_factor"]
qf_filter = qf_reson / 2

# Read the 3C waveforms
print(f"Reading and processing the 3C waveforms of the stationary resonance with a peak filter at {freq_reson:.3f} Hz with a quality factor of {qf_filter:.1f}...")
starttime, endtime = get_start_n_end_from_center(center_time, window_length)

stream = read_and_process_windowed_geo_waveforms(starttime, 
                                                 endtime = endtime, 
                                                 filter = True, filter_type = "peak",
                                                 decimate = True, decimate_factor = decimate_factor,
                                                 freq = freq_reson, qf = qf_filter)

# Plot the 3C waveforms and the envelopes
print("Plotting the 3C waveforms and the envelopes...")
fig, axes = subplots(nrows = 1, ncols = 3, figsize = (3 * panel_width, panel_height), sharey = True)

for i, component in enumerate(components):
    ax = axes[i]
    color = get_geo_component_color(component)

    for j, station in enumerate(coord_df.index):
        trace = stream.select(station = station, component = component)[0]
        timeax = get_datetime_axis_from_trace(trace)
        waveform = trace.data
        envelope = get_envelope(waveform)

        ax.plot(timeax, waveform * scale_factor + j, color = color, linewidth = linewidth_wf, alpha = 0.5)
        ax.plot(timeax, envelope * scale_factor + j, color = color, linewidth = linewidth_env)

        ax.text(timeax[0], j, station, fontsize = 12, fontweight = "bold", va = "center", ha = "right")

    ax.set_xlim(timeax[0], timeax[-1])
    ax.set_ylim(-1, j + 1)

    ax.set_yticks([])

    format_datetime_xlabels(ax,
                            date_format="%Y-%m-%d %H:%M:%S",
                            major_tick_spacing = "30s", num_minor_ticks = 6,
                            va = "top", ha = "right", rotation = 15)

fig.suptitle(f"Frequency: {freq_reson:.3f} Hz, quality factor: {qf_filter:.1f}", fontsize = 14, fontweight = "bold", y = 0.93)

# Save the figure
time_suffix = time2suffix(center_time)
filename = f"stationary_resonance_waveforms_peak_filtered_{mode_name}.png"
save_figure(fig, filename)
