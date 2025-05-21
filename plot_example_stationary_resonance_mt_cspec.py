"""
Compute the multitaper cross-spectra of a station pair for a single time window and plot the results
""" 

from os.path import join
from argparse import ArgumentParser
from numpy import nan, isnan, isrealobj, pi, setdiff1d, var
from pandas import DataFrame, Timedelta
from pandas import read_csv
from scipy.signal.windows import dpss
from matplotlib.pyplot import subplots
from time import time

from utils_basic import SPECTROGRAM_DIR as dirpath_spec, MT_DIR as dirpath_mt, GEO_COMPONENTS as components
from utils_basic import time2suffix, str2timestamp, power2db
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_mt import mt_cspec, get_avg_phase_diff, get_indep_phase_diffs
from utils_plot import component2label, save_figure, format_phase_diff_ylabels

###
# Inputs
###

# Command-line arguments
parser = ArgumentParser(description="Compute the multitaper cross-spectra between a channel of a geophone station pair for a single time window and plot the results")

parser.add_argument("--station1", type=str, help="Station 1")
parser.add_argument("--station2", type=str, help="Station 2")

parser.add_argument("--component", type=str, help="Component")

parser.add_argument("--starttime", type=str, help="Start time of the time window")
parser.add_argument("--window_length_mt", type=float, help="Window length in seconds")
parser.add_argument("--window_length_stft", type=float, help="Window length in seconds")

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--nw", type=float, help="Time-bandwidth product", default=3.0)
parser.add_argument("--min_cohe", type=float, help="Minimum coherence", default=0.85)
parser.add_argument("--decimate_factor", type=int, help="Decimation factor", default=10)
parser.add_argument("--bandwidth", type=float, help="Resonance bandwidth for computing the average frequency", default=0.02)

parser.add_argument("--color1", type=str, help="Color 1", default="tab:purple")
parser.add_argument("--color2", type=str, help="Color 2", default="tab:yellow")

# Parse the command line inputs
args = parser.parse_args()


station1 = args.station1
station2 = args.station2

component = args.component
mode_name = args.mode_name
nw = args.nw
min_cohe = args.min_cohe
decimate_factor = args.decimate_factor
bandwidth = args.bandwidth

starttime = str2timestamp(args.starttime)
window_length_mt = args.window_length_mt
window_length_stft = args.window_length_stft

color1 = args.color1
color2 = args.color2

# Constants
num_taper = int(2 * nw -1)


figwidth = 10
figheight = 10
hspace = 0.2

linewidth_thick = 1.5
linewidth_thin = 1.0

min_db = -60
max_db = -30

axis_label_size = 12

suptitle_y = 0.93

###
# Read the data
###

# Load the MT time windows and get the resonance frequency of the time window
filename = f"stationary_resonance_mt_time_windows_{mode_name}_{station1}_{station2}_mt_win{window_length_mt:.0f}s_stft_win{window_length_stft:.0f}s.csv"
filepath = join(dirpath_mt, filename)
time_win_df = read_csv(filepath, parse_dates = ["start", "end"])

# Get the resonance frequency of the time window
freq_reson = time_win_df[ time_win_df["start"] == starttime ]["mean_freq"].values[0]
print(f"The resonance frequency of the time window is {freq_reson:.3f} Hz.")

# Get the day of the time window
day = starttime.strftime("%Y-%m-%d")

# Read the data
print(f"Reading data from {day}...")
endtime = starttime + Timedelta(window_length_mt, "s")

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
mt_cspec_params = mt_cspec(signal1, signal2, dpss_mat, ratio_vec, sampling_rate, normalize = True, get_uncer = True)
aspec1 = mt_cspec_params.aspec1
aspec2 = mt_cspec_params.aspec2
cohe = mt_cspec_params.cohe
phase_diff = mt_cspec_params.phase_diff
phase_diff_uncer = mt_cspec_params.phase_diff_uncer
freqax = mt_cspec_params.freqax

aspec1_db = power2db(aspec1)
aspec2_db = power2db(aspec2)

# Get the average phase difference
print("Computing the average phase difference for the resonant frequency band...")
min_freq_reson = freq_reson - bandwidth / 2
max_freq_reson = freq_reson + bandwidth / 2
avg_phase_diff, avg_phase_diff_uncer, freq_inds_indep, freq_inds_cohe = get_avg_phase_diff((min_freq_reson, max_freq_reson), freqax, phase_diff, phase_diff_uncer, cohe, nw = nw, min_cohe = min_cohe, return_samples = True)

print("--------------------------------")
print(f"The frequency indices meeting the coherence criterion are {freq_inds_cohe}.")
print(f"The frequency indices meeting the independence criterion are {freq_inds_indep}.")
print("--------------------------------")

freqs_indep = freqax[freq_inds_indep]
phase_diffs_indep = phase_diff[freq_inds_indep]
phase_diff_uncers_indep = phase_diff_uncer[freq_inds_indep]

###
# Plot the results
###

print("Plotting the results...")
fig, axes = subplots(3, 1, figsize=(figwidth, figheight), sharex=True)
fig.subplots_adjust(hspace=hspace)

print("Plotting the autospectra...")
ax_aspc = axes[0]
ax_aspc.plot(freqax, aspec1_db, label=f"{station1}", color = color1, linewidth = linewidth_thick)
ax_aspc.plot(freqax, aspec2_db, label=f"{station2}", color = color2, linewidth = linewidth_thick)
ax_aspc.axvline(min_freq_reson, color="crimson", linestyle="--", linewidth=linewidth_thin)
ax_aspc.axvline(max_freq_reson, color="crimson", linestyle="--", linewidth=linewidth_thin)
ax_aspc.axvline(freq_reson, color="crimson",  linewidth=linewidth_thin)
ax_aspc.set_ylabel("Normalized PSD (dB)", fontsize = axis_label_size)
ax_aspc.set_title("Auto-spectra", fontweight="bold")
ax_aspc.legend(loc = "upper right", fontsize = axis_label_size, frameon = True, fancybox = False, edgecolor = "black")

min_freq_plot = freq_reson - 5 * bandwidth / 2
max_freq_plot = freq_reson + 5 * bandwidth / 2

ax_aspc.set_xlim(min_freq_plot, max_freq_plot)
ax_aspc.set_ylim(min_db, max_db)

print("Plotting the coherence...")
ax_cohe = axes[1]
ax_cohe.plot(freqax, cohe, color = "black", linewidth = linewidth_thick)
ax_cohe.axvline(min_freq_reson, color="crimson", linestyle="--", linewidth=linewidth_thin)
ax_cohe.axvline(max_freq_reson, color="crimson", linestyle="--", linewidth=linewidth_thin)
ax_cohe.axvline(freq_reson, color="crimson", linewidth=linewidth_thin)
ax_cohe.axhline(min_cohe, color="crimson", linestyle="--", linewidth=linewidth_thin)
ax_cohe.set_ylabel("Coherence", fontsize = axis_label_size)
ax_cohe.set_title("Coherence", fontweight="bold")

ax_cohe.set_ylim(0, 1.0)

print("Plotting the phase differences...")
ax_phase_diff = axes[2]
ax_phase_diff.errorbar(freqax, phase_diff, yerr=phase_diff_uncer, fmt="o", color = "gray", markersize=5, fillstyle="none", capsize=2, linewidth=linewidth_thin)
ax_phase_diff.errorbar(freqs_indep, phase_diffs_indep, yerr=phase_diff_uncers_indep, fmt="o", color = "crimson", markersize=5, fillstyle="none", capsize=2, linewidth=linewidth_thick)
ax_phase_diff.axvline(min_freq_reson, color="crimson", linestyle="--", linewidth=linewidth_thin)
ax_phase_diff.axvline(min_freq_reson, color="crimson", linestyle="--", linewidth=linewidth_thin)
ax_phase_diff.axvline(max_freq_reson, color="crimson", linestyle="--", linewidth=linewidth_thin)
ax_phase_diff.axvline(freq_reson, color="crimson", linewidth=linewidth_thin)
ax_phase_diff.set_xlabel("Frequency (Hz)", fontsize = axis_label_size)
ax_phase_diff.set_title("Phase difference", fontweight="bold")

format_phase_diff_ylabels(ax_phase_diff)
print(f"The average phase difference between {station1} and {station2} is {avg_phase_diff:.2f} +/- {avg_phase_diff_uncer:.2f} rad.")

component_label = component2label(component)
fig.suptitle(f"{station1}-{station2}, {component_label.lower()} component, {starttime.strftime('%Y-%m-%d %H:%M:%S')}, {window_length_mt:.0f} s", y = suptitle_y, fontweight = "bold")

# Save the plot
print("Saving the plot...")
figname = f"stationary_resonance_mt_cspec_{mode_name}_{station1}_{station2}_{component.lower()}_{time2suffix(starttime)}_{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.png"
save_figure(fig, figname)



