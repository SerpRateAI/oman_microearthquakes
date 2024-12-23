# Detect stationary resonances by finding peaks in the cumulative array spectral-peak counts

# Imports
from os.path import join
from pandas import read_csv, Timedelta
from numpy import log10
from scipy.signal import find_peaks
from matplotlib.pyplot import figure
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_stationary_resonance_freq_intervals, get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_harmonic_data_table
from utils_plot import plot_cum_freq_fractions, save_figure
from multiprocessing import Pool

# Inputs
# Spectrogram
window_length = 300.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Spectral-peak detection
prom_spec_threshold = 10.0
rbw_threshold = 3.0

min_freq_peak = None
max_freq_peak = 200

# Spectral-peak counting
count_threshold = 9

# Stationary-resonance detection
frac_threshold = -3.0 # In log10 units
prom_frac_threshold = 0.5 # In log10 units

height_factor = 0.3 # Height factor applied to the fraction prominence threshold while finding the peak widths

# Plotting
marker_size = 20.0
marker_offset = 1.1

linewidth_marker = 1.0

min_freq_plot = 0.0
max_freq_plot = 10.0

major_freq_spacing = 1.0
num_minor_freq_ticks = 5

# Print the inputs
print(f"### Detecting stationary resonances from the cumulative spectral-peak counts ###")
print(f"# Spectrogram computation #")
print(f"Window length: {window_length} s")
print(f"Overlap: {overlap}")
print(f"Downsample: {downsample}")
print(f"Downsample factor: {downsample_factor}")


print(f"# Spectral-peak detection #")
print(f"Prominence threshold: {prom_spec_threshold}")
print(f"Reverse-bandwidth threshold: {rbw_threshold} 1/Hz")
print(f"Frequency range: {min_freq_peak} - {max_freq_peak} Hz")

print(f"# Spectral-peak counting #")
print(f"Count threshold: {count_threshold}")

print(f"# Stationary-resonance detection #")
print(f"Fraction threshold: {frac_threshold}")
print(f"Fraction prominence threshold: {prom_frac_threshold}")
print(f"Height factor: {height_factor}")


# Read the spectral-peak cummulative counts
print("Reading the spectral peak counts...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
suffix_peak = get_spec_peak_file_suffix(prom_spec_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)

filename_in = f"geo_spectral_peak_time_cum_freq_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}_freq_sorted.csv"
inpath = join(indir, filename_in)
cum_count_df = read_csv(inpath, index_col = 0)
cum_count_df.sort_values('frequency', inplace = True)

# Detect peaks in the cumulative counts
print("Detecting the stationary resonances...")
fracs = cum_count_df['fraction'].values
log_fracs = log10(fracs)
i_freq_peaks, peak_dict = find_peaks(log_fracs, height = frac_threshold, prominence = prom_frac_threshold)
print(f"Detected {len(i_freq_peaks)} resonances.")

freqax = cum_count_df['frequency'].values
stationary_resonance_df = cum_count_df.iloc[i_freq_peaks].copy()
stationary_resonance_df["freq_index"] = i_freq_peaks
stationary_resonance_df["frac_prominence"] = peak_dict["prominences"]
stationary_resonance_df["freq_left_base"] = freqax[peak_dict["left_bases"]]
stationary_resonance_df["freq_right_base"] = freqax[peak_dict["right_bases"]]
stationary_resonance_df.reset_index(drop = True, inplace = True)

# Find the extraction frequency intervals
print("Finding the frequency bounds of the detected resonances...")
stationary_resonance_df = get_stationary_resonance_freq_intervals(stationary_resonance_df, peak_dict, cum_count_df, height_factor = height_factor)

# Save the detected resonances sorted by fraction and frequency
print("Saving the detected resonances sorted by frequency...")
filename_out = f"stationary_resonances_detected_{suffix_spec}_{suffix_peak}_count{count_threshold:d}_frac{frac_threshold:.1f}_prom{prom_frac_threshold:.1f}_hf{height_factor:.1f}_freq_sorted.csv"
outpath = join(indir, filename_out)
stationary_resonance_df.to_csv(outpath)
print(f"Saved the detected resonances sorted by frequency to {outpath}.")

# Save the detected resonances sorted by count
print("Saving the detected resonances sorted by fraction...")
filename_out = f"stationary_resonances_detected_{suffix_spec}_{suffix_peak}_count{count_threshold:d}_frac{frac_threshold:.1f}_prom{prom_frac_threshold:.1f}_hf{height_factor:.1f}_count_sorted.csv"
outpath = join(indir, filename_out)
stationary_resonance_df.sort_values('count', ascending = False, inplace = True, ignore_index = True)
stationary_resonance_df.to_csv(outpath)
print(f"Saved the detected resonances sorted by count to {outpath}.")

# Plot the fractions and the detected resonances
# Plot the fractions
print("Plotting the cumulative frequency fractions and the detected resonances...")
fig, ax = plot_cum_freq_fractions(cum_count_df,
                                  min_freq = min_freq_plot, max_freq = max_freq_plot,
                                  linewidth = 0.01, marker_size = 0.2,
                                  major_freq_spacing = major_freq_spacing, num_minor_freq_ticks = num_minor_freq_ticks)

# Plot triangles marking the detected resonances
for _, row in stationary_resonance_df.iterrows():
    freq = row['frequency']
    frac = row['fraction']
    ax.scatter(freq, frac * marker_offset, s = marker_size, marker = 'v', facecolors = 'none', edgecolors = 'crimson', linewidth = linewidth_marker, zorder = 10)

# Save the figure
figname = f"geo_spectral_peak_time_cum_freq_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}_resonance_detected_frac{frac_threshold:.1f}_prom{prom_frac_threshold:.1f}_plot{min_freq_plot:.0f}to{max_freq_plot:.0f}hz.png"
save_figure(fig, figname)