"""
Sonify a segment of the tremor signal and save the audio file.
"""

#---------------------------------------------------------------------------------------------------
# Import the necessary libraries
#---------------------------------------------------------------------------------------------------

from os.path import join
from argparse import ArgumentParser
from numpy import amax, abs, linspace, float32, count_nonzero
from soundfile import write, read
from pandas import Timestamp
from scipy.signal import resample_poly
from scipy.signal.windows import dpss
from math import gcd

from matplotlib.pyplot import subplots

from utils_basic import (
    ROOTDIR_GEO as dirpath_in,
    AUDIO_DIR as dirpath_out,
    SAMPLING_RATE as sampling_rate_in,
    SAMPLING_RATE_AUDIO as sampling_rate_out,
    get_freq_limits_string,
    power2db
)
from utils_cont_waveform import load_waveform_slice
from utils_mt import mt_autospec
from utils_plot import component2label, save_figure, format_datetime_xlabels, format_freq_xlabels


#---------------------------------------------------------------------------------------------------
# Define the input parameters
#---------------------------------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--station", type=str, help="Station name.")
parser.add_argument("--starttime", type=Timestamp, help="Start time of the time window.")
parser.add_argument("--duration", type=float, help="Duration of the time window in seconds.")
parser.add_argument("--nw", help="Bandwidth factor.", type = int, default = 3)
parser.add_argument("--component", type=str, help="Component to sonify.", default = "Z")
parser.add_argument("--min_freq_filter", type=float, help="Minimum frequency of the filter.", default = 20.0)
parser.add_argument("--max_freq_filter", type=float, help="Maximum frequency of the filter.", default = 200.0)
parser.add_argument("--axis_label_size", type=float, help="Size of the axis labels.", default = 12)
parser.add_argument("--tick_label_size", type=float, help="Size of the tick labels.", default = 10)

args = parser.parse_args()
station = args.station
starttime = args.starttime
duration = args.duration
nw = args.nw
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
component = args.component
axis_label_size = args.axis_label_size
tick_label_size = args.tick_label_size

#---------------------------------------------------------------------------------------------------
# Load the waveform slice
#---------------------------------------------------------------------------------------------------
freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
num_pts = int(duration * sampling_rate_in)
filename = f"preprocessed_data_{freq_str}.h5"
filepath_in = join(dirpath_in, filename)
waveform_dict, time_axis = load_waveform_slice(filepath_in, station, starttime, 
                                    num_pts = num_pts)
waveform = waveform_dict[component]

print(f"Sonifying the {component} component of the {station} station...")

#---------------------------------------------------------------------------------------------------
# Sonify the waveform
#---------------------------------------------------------------------------------------------------

# Normalize to avoid clipping
waveform = waveform / amax(abs(waveform)) * 0.98



# Compute the spectrum
num_taper = 2 * nw - 1
num_pts = len(waveform)
taper_mat, ratio_vec = dpss(num_pts, nw, num_taper, return_ratios=True)
param = mt_autospec(waveform, taper_mat, ratio_vec, sampling_rate_in, normalize=True)
spec = param.aspec
freq_axis = param.freqax


# Plot the waveform and spectrum
fig, axs = subplots(2, 1, figsize=(10, 5))
fig.subplots_adjust(hspace=0.3)
color = "black"
ax = axs[0]
ax.plot(time_axis, waveform, color=color, linewidth=0.1)
format_datetime_xlabels(ax, major_tick_spacing = "15s", axis_label_size = axis_label_size, tick_label_size = tick_label_size)
ax.set_ylabel("Normalized amplitude", fontsize = axis_label_size)
ax.set_title(f"{station}, {component2label(component)}", fontsize=14, fontweight="bold")
ax.set_xlim(time_axis[0], time_axis[-1])
ax.set_ylim(-1, 1)
ax.yaxis.set_tick_params(labelsize = tick_label_size)

ax = axs[1]
spec = power2db(spec)
ax.plot(freq_axis, spec, color = color, linewidth=1.0)
format_freq_xlabels(ax, major_tick_spacing = 20.0, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

ax.set_ylabel("Normalized power (dB)", fontsize = axis_label_size)
ax.set_xlim(min_freq_filter, max_freq_filter)
ax.set_ylim(-10, 40)
ax.yaxis.set_tick_params(labelsize = tick_label_size)


figname = f"sonified_tremor_waveform_n_spectrum_{freq_str}_{station}_{component.lower()}_{starttime.strftime('%Y%m%d%H%M%S')}_{duration:.0f}s.png"
save_figure(fig, figname)

# Resample the data to the output sampling rate that is compatible with the audio format
g = gcd(sampling_rate_out, int(sampling_rate_in))
up = sampling_rate_out // g
down = sampling_rate_in // g
waveform = resample_poly(waveform, up, down)

# Write as 32-bit float WAV (safe, no extra scaling needed)
filename = f"sonified_tremor_audio_{freq_str}_{station}_{component.lower()}_{starttime.strftime('%Y%m%d%H%M%S')}_{duration:.0f}s.wav"
filepath_out = join(dirpath_out, filename)
write(filepath_out, waveform.astype(float32), sampling_rate_out, subtype="FLOAT")
print(f"Sonified waveform saved to {filepath_out}")

# Read back the waveform
y, sr = read(filepath_out, dtype="float64", always_2d=False)
print("file tail max abs:", max(abs(y[-3*sr:])))



