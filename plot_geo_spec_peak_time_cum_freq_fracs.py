# Plot the time cumulative spectral-peak fractions of each frequency of the geophone array

# Imports
from os.path import join
from pandas import read_csv

from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, get_stationary_resonance_file_suffix
from utils_plot import plot_cum_freq_fractions, save_figure

# Inputs
# Spectrogram computation
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Spectral-peak detection
prom_threshold = 10
rbw_threshold = 3.0

min_freq_peak = None
max_freq_peak = 200.0

# Spectral-peak counting
count_threshold = 9

# Stationary resonance detection
frac_threshold = -3.0
prom_frac_threshold = 0.5

# Plotting
marker_size_freq = 5.0
linewidth_freq = 0.2
min_freq_plot = 0.0
max_freq_plot = 5.0

marker_size_reson = 50.0
linewidth_reson = 1.5
marker_offset = 1.2

major_freq_spacing = 1.0
num_minor_freq_ticks = 10

# Read the time cumulative spectral-peak fractions
print("Reading the time cumulative spectral-peak fractions...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)
suffix_resonance = get_stationary_resonance_file_suffix(frac_threshold, prom_frac_threshold)

filename_in = f"geo_spectral_peak_time_cum_freq_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}_freq_sorted.csv"
inpath = join(indir, filename_in)
cum_frac_df = read_csv(inpath, index_col = 0)

# Read the detected stationary resonances
filename_in = f"stationary_resonances_detected_{suffix_spec}_{suffix_peak}_count{count_threshold}_{suffix_resonance}.csv"
inpath = join(indir, filename_in)
stationary_resonance_df = read_csv(inpath, index_col = 0)

# Plot the fractions and the detected resonances
# Plot the fractions
print("Plotting the cumulative frequency fractions and the detected resonances...")
fig, ax = plot_cum_freq_fractions(cum_frac_df,
                                  min_freq = min_freq_plot, max_freq = max_freq_plot,
                                  linewidth = linewidth_freq, marker_size = marker_size_freq,
                                  major_freq_spacing = major_freq_spacing, num_minor_freq_ticks = num_minor_freq_ticks)

# Plot triangles marking the detected resonances
for _, row in stationary_resonance_df.iterrows():
    freq = row['frequency']
    frac = row['fraction']
    ax.scatter(freq, frac * marker_offset, s = marker_size_reson, marker = 'v', facecolors = 'none', edgecolors = 'crimson', linewidth = linewidth_reson, zorder = 10)

# Save the figure
print("Saving the figure...")
figname = f"geo_spectral_peak_time_cum_freq_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}_resonance_detected_{suffix_resonance}_freq{min_freq_plot:.0f}to{max_freq_plot:.0f}hz.png"
save_figure(fig, figname)

