# Plot the total power spectrum, spectral-peak powers, array spectral-peak counts, and binarized array spectrogram of a specific time window
from os.path import join
from time import time
from numpy import bool_, linspace, zeros
from pandas import Timestamp
from pandas import date_range
from multiprocessing import Pool
import h5py

from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import get_geophone_days
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_binary_spectrogram, read_geo_spectrograms, read_geo_spec_peaks, read_spec_peak_array_counts
from utils_plot import plot_geo_total_psd_to_bin_array_spectrogram, save_figure

# Inputs
# Spectrograms
station_plot = "A01"
window_length = 60.0 #1.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Peak-finding
prom_threshold = 10
rbw_threshold = 3.0 #0.2
min_freq = None
max_freq = 200.0

# Peak-counting
count_threshold = 9

# Plotting
starttime_plot = "2020-01-13 20:00:00"
endtime_plot = "2020-01-13 21:00:00"

dbmin = -30
dbmax = 10

date_format = "%Y-%m-%d %H:%M:%S"
major_time_spacing = "15min"
num_minor_time_ticks = 3

size_scale = 30
marker_size = 2

# Read the spectrogram and compute the total power
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
filename_in = f"whole_deployment_daily_geo_spectrograms_{station_plot}_{suffix_spec}.h5"
inpath = join(indir, filename_in)

print(f"Reading spectrogram from {inpath} and computing the total power...")
stream_spec = read_geo_spectrograms(inpath, starttime =  starttime_plot, endtime = endtime_plot,  min_freq = min_freq, max_freq = max_freq)
trace_spec_total = stream_spec.get_total_power()
print("Done.")

# Read the spectral peaks
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq, max_freq = max_freq)
filename_in = f"geo_spectral_peaks_{station_plot}_{suffix_spec}_{suffix_peak}.h5"
inpath = join(indir, filename_in)

print(f"Reading spectral peaks from {inpath}...")
peak_df = read_geo_spec_peaks(inpath, starttime = starttime_plot, endtime = endtime_plot)
print("Done.")

# Read the spectral-peak count file
filename_in = f"geo_spectral_peak_array_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.h5"
inpath = join(indir, filename_in)

print(f"Reading spectral-peak array counts from {inpath}...")
count_df = read_spec_peak_array_counts(inpath, starttime = starttime_plot, endtime = endtime_plot)
print("Done.")

# Read the binarized array spectrogram
print(f"Reading the binarized array spectrogram from {inpath}...")
filename_in = f"geo_binary_array_spectrogram_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.h5"
inpath = join(indir, filename_in)

bin_spec_dict = read_binary_spectrogram(inpath, starttime = starttime_plot, endtime = endtime_plot, min_freq = min_freq, max_freq = max_freq)
print("Done.")

# Plotting
print("Plotting...")
fig, axes, power_cbar = plot_geo_total_psd_to_bin_array_spectrogram(trace_spec_total, peak_df, count_df, bin_spec_dict,
                                                                    min_freq = min_freq, max_freq = max_freq,
                                                                    dbmin = dbmin, dbmax = dbmax,
                                                                    size_scale = size_scale, marker_size = marker_size,
                                                                    major_time_spacing = major_time_spacing, num_minor_time_ticks = num_minor_time_ticks,
                                                                    date_format = date_format)
print("Done.")

# Save the plot
print("Saving the plot...")
figname = f"geo_binary_array_spec_extraction_example_{station_plot}.png"
save_figure(fig, figname)
print("Done.")