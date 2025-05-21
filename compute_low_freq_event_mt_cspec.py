"""
Conduct multitaper cross-spectral analysis of the low-frequency events recorded on two hydrophone stations
"""

# Load the required modules
from os.path import join
from argparse import ArgumentParser
from numpy import amax, array, pi
from scipy.signal.windows import dpss
from pandas import Timestamp
from matplotlib.pyplot import subplots

from utils_basic import HYDRO_STATIONS as loc_dict, HYDRO_DEPTHS as depth_dict, SAMPLING_RATE as sampling_rate
from utils_basic import power2db
from utils_preproc import read_and_process_windowed_hydro_waveforms
from utils_mt import mt_cspec, mt_autospec

from utils_plot import format_phase_diff_ylabels, save_figure, add_zoom_effect

# Parse the command line arguments
parser = ArgumentParser(description="Conduct multitaper cross-spectral analysis of the low-frequency events recorded on two hydrophone stations")
parser.add_argument("--station", type=str, help="Station name", required=True)
parser.add_argument("--location1", type=str, help="Location 1", required=True)
parser.add_argument("--location2", type=str, help="Location 2", required=True)
parser.add_argument("--starttime", type=Timestamp, help="Start time of the event", required=True)
parser.add_argument("--duration", type=float, help="Duration of the event", required=True)

parser.add_argument("--nw", type=int, help="Time-bandwidth product", default = 3)
parser.add_argument("--num_tapers", type=int, help="Number of tapers", default = 5)

parser.add_argument("--figwidth", type=float, help="Width of the figure", default = 10)
parser.add_argument("--figheight", type=float, help="Height of the figure", default = 10)
parser.add_argument("--hspace", type=float, help="Width of the space between the subplots", default = 0.25)

parser.add_argument("--min_db", type=float, help="Minimum decibel to plot", default = -20)
parser.add_argument("--max_db", type=float, help="Maximum decibel to plot", default = 40)

parser.add_argument("--line_width_obs", type=float, help="Line width of the observation", default = 2.0)
parser.add_argument("--line_width_box", type=float, help="Line width of the box", default = 3.0)
parser.add_argument("--line_width_ref", type=float, help="Line width of the reference", default = 1.0)

parser.add_argument("--max_freq_plot_low", type=float, help="Maximum frequency to plot in the low-frequency range", default = 30.0)

parser.add_argument("--vels_ref", type=float, nargs = "+", help="Reference velocities", default = [1500.0, 6000.0])
parser.add_argument("--freq0", type=float, help="Reference frequency", default = 17.0)
parser.add_argument("--phase_diff0", type=float, help="Reference phase difference", default = pi / 4 * 3)
parser.add_argument("--freq_inc", type=float, help="Frequency increment", default = 5.0)
parser.add_argument("--vel_label_gap", type=float, help="Gap between the velocity labels and the line", default = 0.5)

args = parser.parse_args()

station = args.station
location1 = args.location1
location2 = args.location2
starttime = args.starttime
duration = args.duration

nw = args.nw
num_tapers = args.num_tapers

figwidth = args.figwidth
figheight = args.figheight
hspace = args.hspace

min_db = args.min_db
max_db = args.max_db

max_freq_plot_low = args.max_freq_plot_low

line_width_obs = args.line_width_obs
line_width_box = args.line_width_box
line_width_ref = args.line_width_ref

vels_ref = args.vels_ref
freq0 = args.freq0
phase_diff0 = args.phase_diff0
freq_inc = args.freq_inc
vel_label_gap = args.vel_label_gap

# Read and process the waveforms
stream = read_and_process_windowed_hydro_waveforms(starttime, stations = station, locations = [location1, location2], dur = duration)

# Compute the auto-spectral density
trace1 = stream.select(location = location1)[0]
trace2 = stream.select(location = location2)[0]

signal1 = trace1.data
signal2 = trace2.data

num_pts = len(signal1)

taper_mat, ratio_vec = dpss(num_pts, nw, num_tapers, return_ratios = True)

aspec_params1 = mt_autospec(signal1, taper_mat, ratio_vec, sampling_rate)
aspec_params2 = mt_autospec(signal2, taper_mat, ratio_vec, sampling_rate)

# Compute the cross-spectral density
cspec_params = mt_cspec(signal1, signal2, taper_mat, ratio_vec, sampling_rate)

# Plot the coherence and phase difference
fig, ax = subplots(5, 1, figsize = (figwidth, figheight))
fig.subplots_adjust(hspace = hspace)
## Plot the waveforms
ax_waveform = ax[0]
timeax = trace1.times()

signal1 = signal1 / amax(abs(signal1))
signal2 = signal2 / amax(abs(signal2))

ax_waveform.plot(timeax, signal1, label = f"{location1}", color = "mediumpurple", linewidth = line_width_obs)
ax_waveform.plot(timeax, signal2, label = f"{location2}", color = "orange", linewidth = line_width_obs)
ax_waveform.legend()

ax_waveform.set_xlim(timeax[0], timeax[-1])
ax_waveform.set_yticks([-1, 0, 1])
ax_waveform.set_ylim(-1, 1)

ax_waveform.set_xlabel("Time (s)", fontsize = 12)
ax_waveform.set_ylabel("Normalized amplitude", fontsize = 12)

## Plot the auto-spectral density
ax_aspec = ax[1]
freqax = aspec_params1.freqax
aspec1 = power2db(aspec_params1.aspec)
aspec2 = power2db(aspec_params2.aspec)

ax_aspec.plot(freqax, aspec1, label = f"{location1}", color = "mediumpurple", linewidth = line_width_obs)
ax_aspec.plot(freqax, aspec2, label = f"{location2}", color = "orange", linewidth = line_width_obs)

min_freq_plot_full = freqax[1]
max_freq_plot_full = freqax[-1]

ax_aspec.set_xscale("log")
ax_aspec.set_xlim(min_freq_plot_full, max_freq_plot_full)

ax_aspec.set_ylim(min_db, max_db)
ax_aspec.set_ylabel("Normalized PSD (dB)", fontsize = 12)

## Plot the coherence
ax_cohe = ax[2]
cohes = cspec_params.cohe
freqax = cspec_params.freqax

ax_cohe.set_ylim(0, 1)

ax_cohe.plot(freqax, cohes, color = "black", linewidth = line_width_obs)
ax_cohe.set_ylabel("Coherence", fontsize = 12)

ax_cohe.set_xscale("log")
ax_cohe.set_xlim(min_freq_plot_full, max_freq_plot_full)


## Plot the phase difference in the full frequency range
ax_phase_diff_full = ax[3]
phase_diffs = cspec_params.phase_diff
phase_diff_uncers = cspec_params.phase_diff_uncer

ax_phase_diff_full.plot(freqax, phase_diffs, color = "black", linewidth = line_width_obs)
ax_phase_diff_full.fill_between(freqax, phase_diffs - phase_diff_uncers, phase_diffs + phase_diff_uncers, color = "gray", alpha = 0.5)

ax_phase_diff_full.set_xscale("log")
ax_phase_diff_full.set_xlim(min_freq_plot_full, max_freq_plot_full)

format_phase_diff_ylabels(ax_phase_diff_full)

## Plot the phase difference in the low-frequency range
### Plot the phase differences
ax_phase_diff_low = ax[4]

ax_phase_diff_low.plot(freqax, phase_diffs, color = "black", linewidth = line_width_obs)
ax_phase_diff_low.fill_between(freqax, phase_diffs - phase_diff_uncers, phase_diffs + phase_diff_uncers, color = "gray", alpha = 0.5)

ax_phase_diff_low.set_xlim(min_freq_plot_full, max_freq_plot_low)
ax_phase_diff_low.set_xlabel("Frequency (Hz)", fontsize = 12)

format_phase_diff_ylabels(ax_phase_diff_low)

### Plot the reference velocities
depth1 = depth_dict[location1]
depth2 = depth_dict[location2]

depth_diff = depth1 - depth2

for i, vel in enumerate(vels_ref):
    freqs = array([freq0, freq0 + freq_inc])
    phase_diffs = phase_diff0 + (freqs - freq0) * 2 * pi * depth_diff / vel

    ax_phase_diff_low.plot(freqs, phase_diffs, color = "crimson", linewidth = line_width_ref)

    if i == 0:
        ax_phase_diff_low.text(freqs[1] + vel_label_gap, phase_diffs[1], f"{vel:.0f} m s$^{{-1}}$", 
                               fontsize = 12, color = "crimson",
                              ha = "left", va = "top")
    else:
        ax_phase_diff_low.text(freqs[1] + vel_label_gap, phase_diffs[1], f"{vel:.0f} m s$^{{-1}}$", 
                               fontsize = 12, color = "crimson",
                              ha = "left", va = "top")
        
### Add the zoom effect
add_zoom_effect(ax_phase_diff_full, ax_phase_diff_low,
                min_freq_plot_full, max_freq_plot_low,
                {"color": "crimson", "linewidth": line_width_box},
                {"edgecolor": "crimson", "facecolor": "none", "linewidth": line_width_box})

### Add the supertitle
time_str = starttime.strftime("%Y-%m-%d %H:%M:%S.%f")
fig.suptitle(f"{station}, {location1}-{location2}, {time_str}", fontsize = 14, fontweight = "bold", y = 0.9)

## Save the figure
time_str = starttime.strftime("%Y%m%d%H%M%S")
figname = f"low_freq_event_mt_cspec_{station}_{time_str}_{location1}-{location2}.png"
save_figure(fig, figname)
