# Compute the cumulative time-window counts and fractions of the total time of all geophone-spectral peaks

# Imports
from os.path import join
from pandas import Timedelta
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_spectral_peak_counts
from utils_plot import plot_cum_freq_fractions, save_figure
from multiprocessing import Pool

# Inputs
# Spectrogram
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Spectral peaks
prom_threshold = 10.0
rbw_threshold = 3.0

min_freq_peak = None
max_freq_peak = 200

file_ext_in = "h5"

# Grouping
count_threshold = 9

# Plotting
min_freq_main_plot = 0.0
max_freq_main_plot = 200.0

linewidth_main = 0.1
marker_size_main = 1.0

major_freq_spacing_main = 20.0
num_minor_freq_ticks_main = 10

min_freq_inset1_plot = 22.0
max_freq_inset1_plot = 27.0

min_freq_inset2_plot = 35.0
max_freq_inset2_plot = 40.0

linewidth_inset = 1.0
marker_size_inset = 10.0

major_freq_spacing_inset = 1.0
num_minor_freq_ticks_inset = 10

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
print("Computing the spectral-peak counts...")
cum_count_df = count_df.groupby('frequency').size().reset_index(name = 'count')
cum_count_df_count_sorted = cum_count_df.sort_values('count', ascending = False)
cum_count_df_freq_sorted = cum_count_df.sort_values('frequency', ascending = True)
cum_count_df_count_sorted.reset_index(drop = True, inplace = True)
cum_count_df_freq_sorted.reset_index(drop = True, inplace = True)
print("Done.")

# Normalize the cumulative counts by the total number of time windows
print("Normalizing the cumulative spectral-peak counts...")
total_time = endtime - starttime
total_windows = int(total_time / Timedelta(seconds = window_length))
cum_count_df['fraction'] = cum_count_df['count'] / total_windows
cum_count_df_count_sorted['fraction'] = cum_count_df_count_sorted['count'] / total_windows
cum_count_df_freq_sorted['fraction'] = cum_count_df_freq_sorted['count'] / total_windows

# Save the results to file
print("Saving the cumulative spectral-peak counts and fractions...")
print("Saving the count-sorted dataframe...")
filename_out = f"geo_spec_peak_cum_freq_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}_count_sorted.csv"
outdir = indir
outpath = join(outdir, filename_out)
cum_count_df_count_sorted.to_csv(outpath)
print(f"Saved to {outpath}.")

print("Saving the frequency-sorted dataframe...")
filename_out = f"geo_spec_peak_cum_freq_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}_freq_sorted.csv"
outdir = indir
outpath = join(outdir, filename_out)
cum_count_df_freq_sorted.to_csv(outpath)
print(f"Saved to {outpath}.")

