# Test the the speed of the geophone peak detection algorithm on a day of data

# Imports
from os.path import join
from pandas import concat
from time import time
from multiprocessing import Pool

from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import find_geo_station_spectral_peaks, get_spectrogram_file_suffix
from utils_spec import read_geo_spectrograms, string_to_time_label

# Inputs
# Data
station = "A01"
day = "2020-01-13"

window_length = 60.0
overlap = 0.0
downsample = False

# Finding peaks
num_process = 32
rbw_threshold = 3.0
prom_threshold = 10
min_freq = None
max_freq = 200.0

# Writing
to_csv = False
to_hdf = True

# Read the data
clock1 = time()
print("Reading the data...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)
filename_in = f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5"
inpath = join(indir, filename_in)

time_label = string_to_time_label(day)
stream_spec = read_geo_spectrograms(inpath, time_labels = [time_label])

# Find the peaks
print("Detecting the peaks...")
peak_df, _ = find_geo_station_spectral_peaks(stream_spec, num_process, 
                                            rbw_threshold = rbw_threshold, prom_threshold = prom_threshold, 
                                            min_freq = min_freq, max_freq = max_freq)
print(f"In total, {len(peak_df)} spectral peaks found.")

# Stop the clock and print the time
clock2 = time()
print(f"Time taken to detect the peaks: {clock2 - clock1:.1f} seconds.")