# Compute the cumulative counts of all geophone-spectral peak frequencies

# Imports
from os.path import join
from numpy import linspace
from time import time

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_spectral_peak_counts
from utils_plot import plot_cum_freq_counts, save_figure
from multiprocessing import Pool

# Inputs
# Spectrogram
window_length = 60.0
overlap = 0.0
downsample = True
downsample_factor = 60

# Spectral peaks
prom_threshold = 5.0
rbw_threshold = 0.2

min_freq_peak = None
max_freq_peak = None

file_ext_in = "h5"

# Grouping
count_threshold = 4

# Plotting
min_freq_plot = 0.0
max_freq_plot = 500.0

# Read the spectral peak counts
print("Reading the spectral peak counts...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)

filename_in = f"geo_spectral_peak_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.{file_ext_in}"
inpath = join(indir, filename_in)
count_df = read_spectral_peak_counts(inpath)
print(f"Read {count_df.shape[0]:d} spectral peaks.")
print("Done.")

# Compute the time-cumulative counts
print("Computing the cumulative frequency counts...")
cum_count_df = count_df.groupby('frequency').size().reset_index(name = 'count')
cum_count_df_sorted = cum_count_df.sort_values('count', ascending = False)
cum_count_df_sorted.reset_index(drop = True, inplace = True)
print("Done.")

# Save the results to file
print("Saving the cumulative frequency counts...")
filename_out = f"geo_spec_peak_cum_freq_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.csv"
outdir = indir
outpath = join(outdir, filename_out)
cum_count_df_sorted.to_csv(outpath)
print(f"Saved to {outpath}.")

# Plot the results
print("Plotting the cumulative frequency counts...")
fig, ax = plot_cum_freq_counts(cum_count_df, min_freq = min_freq_plot, max_freq = max_freq_plot)
print("Done.")

# Save the figure
print("Saving the figure...")
filename_fig = f"geo_spec_peak_cum_freq_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.png"
save_figure(fig, filename_fig)
