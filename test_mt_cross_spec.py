# Test the Python implementation of multitaper cross-spectal analysis
from os.path import join
from numpy import isrealobj, var
from scipy.signal.windows import dpss
from matplotlib.pyplot import subplots
from obspy import read, UTCDateTime

from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_mt import mt_cspec, get_avg_phase_diff, get_indep_phase_diffs
from utils_plot import save_figure

# Define the parameters
nw = 3
num_taper = 2 * nw - 1

factor_ds = 10

station1 = "A01"
station2 = "A02"

component = "1"

starttime = "2020-01-13T15:30:00"
endtime = "2020-01-13T16:00:00"

min_freq_plot = 25.0
max_freq_plot = 26.0

min_freq_reson = 25.50
max_freq_reson = 25.52

min_cohe = 0.95

# Get the day of the time window
day = starttime.split("T")[0]

# Read the data
print(f"Reading data from {day}...")

stream1 = read_and_process_windowed_geo_waveforms(starttime, endtime = endtime, stations = station1, components = component, decimate = True, decimate_factor = factor_ds)
stream2 = read_and_process_windowed_geo_waveforms(starttime, endtime = endtime, stations = station2, components = component, decimate = True, decimate_factor = factor_ds)

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
freqax, mt_aspec1, mt_aspec2, mt_trans, mt_cohe, mt_phase_diff, mt_aspec1_lo, mt_aspec1_hi, mt_aspec2_lo, mt_aspec2_hi, mt_trans_uncer, mt_cohe_uncer, mt_phase_diff_uncer = mt_cspec(signal1, signal2, dpss_mat, ratio_vec, sampling_rate)

# Get the average phase difference
print("Computing the average phase difference for the resonant frequency band...")
avg_phase_diff, avg_phase_diff_uncer = get_avg_phase_diff((min_freq_reson, max_freq_reson), freqax, mt_phase_diff, mt_phase_diff_uncer, mt_cohe, nw = nw)

# Plot the results
print("Plotting the results...")
fig, axes = subplots(3, 1, figsize=(10, 8), sharex=True)
fig.subplots_adjust(hspace=0.2)

axes[0].plot(freqax, mt_aspec1, label=f"{station1}")
axes[0].plot(freqax, mt_aspec2, label=f"{station2}")
axes[0].axvline(min_freq_reson, color="crimson", linestyle="--", linewidth=0.5)
axes[0].axvline(max_freq_reson, color="crimson", linestyle="--", linewidth=0.5)
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Auto-spectra", fontweight="bold")
axes[0].legend()

axes[0].set_xlim(min_freq_plot, max_freq_plot)
axes[0].set_yscale("log")

axes[1].plot(freqax, mt_cohe)
axes[1].axvline(min_freq_reson, color="crimson", linestyle="--", linewidth=0.5)
axes[1].axvline(max_freq_reson, color="crimson", linestyle="--", linewidth=0.5)
axes[1].axhline(min_cohe, color="crimson", linestyle="--", linewidth=0.5)
axes[1].set_ylabel("Coherence")
axes[1].set_title("Coherence", fontweight="bold")
axes[1].set_xlim(min_freq_plot, max_freq_plot)

axes[2].errorbar(freqax, mt_phase_diff, yerr=mt_phase_diff_uncer, fmt="o",  markersize=5, fillstyle="none", capsize=2, linewidth=0.5)
axes[2].axvline(min_freq_reson, color="crimson", linestyle="--", linewidth=0.5)
axes[2].axvline(max_freq_reson, color="crimson", linestyle="--", linewidth=0.5)
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Phase difference (rad)")
axes[2].set_title("Phase difference", fontweight="bold")

print(f"The average phase difference between {station1} and {station2} is {avg_phase_diff:.2f} +/- {avg_phase_diff_uncer:.2f} rad.")

# Save the plot
print("Saving the plot...")
figname = f"mt_cspec_test_{station1}_{station2}_{day}.png"
save_figure(fig, figname)



