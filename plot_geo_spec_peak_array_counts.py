# Plot the geophone array spectral peak counts
# Imports
from os.path import join
from time import time
from pandas import Series
from pandas import concat, read_hdf

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_basic import time2suffix
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_spec_peak_counts
from utils_plot import plot_array_spec_peak_counts, save_figure


# Inputs
# Data
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60
prom_threshold = 10
rbw_threshold = 3.0

min_freq_peak = None
max_freq_peak = 200.0

# Grouping
count_threshold = 9

# Plotting
starttime_plot = starttime
endtime_plot = endtime

min_freq_plot = 0.0
max_freq_plot = 5.0
major_freq_spacing = 1.0
num_minor_freq_ticks = 5

date_format = "%Y-%m-%d"
major_time_spacing = "1d"
num_minor_time_ticks = 4

example_counts = [10, 20, 30]

size_scale = 30

# Read the array spectral peak counts
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)
filename_in = f"geo_spectral_peak_array_counts_{suffix_spec}_{suffix_peak}_count{count_threshold}.h5"

print("Reading the array spectral peak counts...")
inpath = join(indir, filename_in)
count_df = read_spec_peak_counts(inpath, min_freq = min_freq_plot, max_freq = max_freq_plot)

# Plot the geophone array spectral peak counts
print("Plotting the geophone array spectral peak counts...")
fig, axes = plot_array_spec_peak_counts(count_df, starttime = starttime_plot, endtime = endtime_plot,
                                        min_freq = min_freq_plot, max_freq = max_freq_plot,
                                        date_format = date_format, major_time_spacing = major_time_spacing,
                                        major_freq_spacing = major_freq_spacing, num_minor_freq_ticks = num_minor_freq_ticks,
                                        num_minor_time_ticks = num_minor_time_ticks, example_counts = example_counts,
                                        size_scale = size_scale)

# Save the figure
print("Saving the figure...")
suffix_start = time2suffix(starttime_plot)
suffix_end = time2suffix(endtime_plot)
figname = f"geo_spectral_peak_counts_{suffix_spec}_{suffix_peak}_count{count_threshold}_{suffix_start}to{suffix_end}_freq{min_freq_plot:.2f}to{max_freq_plot:.2f}hz.png"

save_figure(fig, figname)