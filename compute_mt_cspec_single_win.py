"""
Compute the multitaper cross-spectra  between two stations for a single time window
and plot the results
""" 

from os.path import join
from argparse import ArgumentParser
from numpy import nan, isnan, isrealobj, pi, setdiff1d, var
from pandas import DataFrame, Timedelta
from pandas import read_csv
from scipy.signal.windows import dpss
from matplotlib.pyplot import subplots
from time import time

from utils_basic import SPECTROGRAM_DIR as dirname_spec, MT_DIR as dirname_mt, GEO_COMPONENTS as components
from utils_basic import time2suffix, str2timestamp, power2db
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_mt import mt_cspec, get_avg_phase_diff, get_indep_phase_diffs
from utils_plot import component2label, save_figure, format_phase_diff_ylabels

###
# Inputs
###

# Command-line arguments
parser = ArgumentParser(description="Compute the multitaper cross-spectra between a channel of a geophone station pair for a single time window and plot the resullts")

parser.add_argument("--station1", type=str, help="Station 1")
parser.add_argument("--station2", type=str, help="Station 2")

parser.add_argument("--component", type=str, help="Component")

parser.add_argument("--starttime", type=str, help="Start time of the time window")
parser.add_argument("--window_length", type=float, help="Window length in seconds")

parser.add_argument("--freq_reson", type=float, help="Resonance frequency")

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--nw", type=float, help="Time-bandwidth product", default=3.0)
parser.add_argument("--min_cohe", type=float, help="Minimum coherence", default=0.85)
parser.add_argument("--decimate_factor", type=int, help="Decimation factor", default=10)
parser.add_argument("--bandwidth", type=float, help="Resonance bandwidth for computing the average frequency", default=0.02)

# Parse the command line inputs
args = parser.parse_args()

station1 = args.station1
station2 = args.station2

component = args.component

starttime = str2timestamp(args.starttime)
window_length = args.window_length

freq_reson = args.freq_reson

mode_name = args.mode_name
nw = args.nw
min_cohe = args.min_cohe
decimate_factor = args.decimate_factor
bandwidth = args.bandwidth

num_taper = int(2 * nw -1)

# Constants
figwidth = 10
figheight = 10
hspace = 0.2

linewidth_thick = 1.5
linewidth_thin = 1.0

min_db = -60
max_db = -30

axis_label_size = 12

suptitle_y = 0.93

# Get the day of the time window
day = starttime.strftime("%Y-%m-%d")

# Read the data
print(f"Reading data from {day}...")
endtime = starttime + Timedelta(window_length, "s")

stream1 = read_and_process_windowed_geo_waveforms(starttime, endtime = endtime, stations = station1, components = component, decimate = True, decimate_factor = decimate_factor)
stream2 = read_and_process_windowed_geo_waveforms(starttime, endtime = endtime, stations = station2, components = component, decimate = True, decimate_factor = decimate_factor)

sampling_rate = stream1[0].stats.sampling_rate

# Get the data
signal1 = stream1[0].data
signal2 = stream2[0].data

num_pts = len(signal1)

# Get the DPSS windows
print("Calculating the DPSS windows...")
dpss_mat, ratio_vec = dpss(num_pts, nw, num_taper, return_ratios=True)

# Perform multitaper cross-spectral analysis
print("Performing multitaper cross-spectral analysis...")
freqax, mt_aspec1, mt_aspec2, mt_trans, mt_cohe, mt_phase_diffs, mt_aspec1_lo, mt_aspec1_hi, mt_aspec2_lo, mt_aspec2_hi, mt_trans_uncer, mt_cohe_uncer, mt_phase_diff_uncers = mt_cspec(signal1, signal2, dpss_mat, ratio_vec, sampling_rate,
                                                                                                                                                                                            normalize = True)
mt_aspec1_db = power2db(mt_aspec1)
mt_aspec2_db = power2db(mt_aspec2)

# Get the average phase difference
print("Computing the average phase difference for the resonant frequency band...")
min_freq_reson = freq_reson - bandwidth / 2
max_freq_reson = freq_reson + bandwidth / 2
avg_phase_diff, avg_phase_diff_uncer, freq_inds_indep = get_avg_phase_diff((min_freq_reson, max_freq_reson), freqax, mt_phase_diffs, mt_phase_diff_uncers, mt_cohe, nw = nw, min_cohe = min_cohe, return_samples = True)

freqs_indep = freqax[freq_inds_indep]
phase_diffs_indep = mt_phase_diffs[freq_inds_indep]
phase_diff_uncers_indep = mt_phase_diff_uncers[freq_inds_indep]

# Plot the results
print("Plotting the results...")
fig, axes = subplots(3, 1, figsize=(figwidth, figheight), sharex=True)
fig.subplots_adjust(hspace=hspace)

print("Plotting the autospectra...")
axes[0].plot(freqax, mt_aspec1_db, label=f"{station1}", color = "tab:blue", linewidth = linewidth_thick)
axes[0].plot(freqax, mt_aspec2_db, label=f"{station2}", color = "tab:green", linewidth = linewidth_thick)
axes[0].axvline(min_freq_reson, color="crimson", linestyle="--", linewidth=linewidth_thin)
axes[0].axvline(max_freq_reson, color="crimson", linestyle="--", linewidth=linewidth_thin)
axes[0].axvline(freq_reson, color="crimson",  linewidth=linewidth_thin)
axes[0].set_ylabel("Normalized PSD (dB)", fontsize = axis_label_size)
axes[0].set_title("Auto-spectra", fontweight="bold")
axes[0].legend()

min_freq_plot = freq_reson - 5 * bandwidth / 2
max_freq_plot = freq_reson + 5 * bandwidth / 2

axes[0].set_xlim(min_freq_plot, max_freq_plot)
axes[0].set_ylim(min_db, max_db)

print("Plotting the coherence...")
axes[1].plot(freqax, mt_cohe, color = "black", linewidth = linewidth_thick)
axes[1].axvline(min_freq_reson, color="crimson", linestyle="--", linewidth=linewidth_thin)
axes[1].axvline(max_freq_reson, color="crimson", linestyle="--", linewidth=linewidth_thin)
axes[1].axvline(freq_reson, color="crimson", linewidth=linewidth_thin)
axes[1].axhline(min_cohe, color="crimson", linestyle="--", linewidth=linewidth_thin)
axes[1].set_ylabel("Coherence", fontsize = axis_label_size)
axes[1].set_title("Coherence", fontweight="bold")

axes[1].set_ylim(0, 1.0)

print("Plotting the phase differences...")
freqs_rest = setdiff1d(freqax, freqs_indep)
phase_diffs_rest = setdiff1d(mt_phase_diffs, phase_diffs_indep)
phase_diff_uncers_rest = setdiff1d(mt_phase_diff_uncers, phase_diff_uncers_indep)

axes[2].errorbar(freqax, mt_phase_diffs, yerr=mt_phase_diff_uncers, fmt="o", color = "gray", markersize=5, fillstyle="none", capsize=2, linewidth=linewidth_thin)
axes[2].errorbar(freqs_indep, phase_diffs_indep, yerr=phase_diff_uncers_indep, fmt="o", color = "crimson", markersize=5, fillstyle="none", capsize=2, linewidth=linewidth_thick)
axes[2].axvline(min_freq_reson, color="crimson", linestyle="--", linewidth=linewidth_thin)
axes[2].axvline(max_freq_reson, color="crimson", linestyle="--", linewidth=linewidth_thin)
axes[2].axvline(freq_reson, color="crimson", linewidth=linewidth_thin)
axes[2].set_xlabel("Frequency (Hz)", fontsize = axis_label_size)
axes[2].set_title("Phase difference", fontweight="bold")

axes[2].set_ylim(-pi, pi)
format_phase_diff_ylabels(axes[2])
print(f"The average phase difference between {station1} and {station2} is {avg_phase_diff:.2f} +/- {avg_phase_diff_uncer:.2f} rad.")

component_label = component2label(component)
fig.suptitle(f"{station1}-{station2}, {component_label.lower()} component, {starttime.strftime('%Y-%m-%d %H:%M:%S')}, {window_length:.0f} s", y = suptitle_y, fontweight = "bold")

# Save the plot
print("Saving the plot...")
figname = f"multitaper_cspec_{station1}_{station2}_{component.lower()}_{time2suffix(starttime)}_{window_length:.0f}s_min_cohe{min_cohe:.2f}.png"
save_figure(fig, figname)



