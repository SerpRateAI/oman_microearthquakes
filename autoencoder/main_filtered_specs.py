import os
from os.path import join
import sys
import urllib.request
from urllib.error import HTTPError
import h5py

import numpy as np
from numpy import bool_, linspace, zeros
from pandas import Timestamp
from pandas import date_range

from time import time
from datetime import datetime, timedelta

from scipy.ndimage import convolve





# Add the parent directory to the Python path
sys.path.append(os.path.abspath('..'))

# Import internal modules
from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import get_geophone_days
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_binary_spectrogram, read_geo_spectrograms, read_geo_spec_peaks, read_spec_peak_array_counts
from utils_plot import plot_geo_total_psd_to_bin_array_spectrogram, save_figure

# Initialize Variable Parameters
if len(sys.argv) == 4:
    station = str(sys.argv[1])
    window = int(sys.argv[2])
    threshold = int(sys.argv[3])
else:
    print("Using Default Values")
    station = 'A01'
    window = 64
    threshold = 500

# Initialize Constant Parameters
time_delta = timedelta(seconds=window)
window_length = 1.0 
overlap = 0.0
downsample = False
downsample_factor = 60
prom_threshold = 10
rbw_threshold = .2 
min_freq = None
max_freq = 200.0
count_threshold = 9
starttime_plot = "2020-01-10 20:00:00"
endtime_plot = "2020-01-30 21:00:00"
start_time = datetime.strptime(starttime_plot, "%Y-%m-%d %H:%M:%S")
end_time = datetime.strptime(endtime_plot, "%Y-%m-%d %H:%M:%S")
current_time = start_time
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq, max_freq = max_freq)

# Grab Binary Spectrograms
bin_slice_list = []
while current_time < end_time:
    next_time = current_time + time_delta
    if next_time > end_time:
        break # Break loop if end time is exceeded
    suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
    suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq, max_freq = max_freq)
    filename_in = f"geo_binary_array_spectrogram_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.h5"
    inpath = join(indir, filename_in)
    bin_spec_slice = read_binary_spectrogram(inpath, starttime = str(current_time), endtime = str(next_time), min_freq = min_freq, max_freq = max_freq)
    bin_slice_list.append(bin_spec_slice['data'])
    current_time = next_time


# Define Function to Filter Binary Spectrograms For Connectivity
def compute_mean_connectivity_idx(skeleton):
    struct_element = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])
    neighbors_maps = []
    for i in skeleton:
        neighbors_maps.append(convolve(np.array(i,dtype=int), struct_element, mode='constant', cval=0)*i)
    neighbors_maps = np.array(neighbors_maps)
    return neighbors_maps

# Filter Binary Spectrograms
neighbors_maps = compute_mean_connectivity_idx(bin_slice_list)
connect_list = []
for map in neighbors_maps:
    connect_list.append(np.sum(map))
connect_mask = np.array(connect_list) > threshold
connect_bin_slice_list = np.array(bin_slice_list)[connect_mask]
print(f'Printing Length of Filtered dataset {len(connect_bin_slice_list)}')

# Grab Full Power Spectrograms Corresponding to High Connectivity
connect_power_slice_list = []
current_time = start_time
for bool in connect_mask:
    if bool:
        next_time = current_time + time_delta
        if next_time > end_time:
            next_time = end_time
        filename_in = f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5"
        inpath = join(indir, filename_in)
        power_spec_slice = read_geo_spectrograms(inpath, starttime =  str(current_time), endtime = str(next_time),  min_freq = min_freq, max_freq = max_freq)
        power_spec_slice = power_spec_slice.get_total_power()
        current_time = next_time
        connect_power_slice_list.append(power_spec_slice)
connect_power_slice_list = np.array(connect_power_slice_list)

# Save Spectrograms 
np.savez(f"spectrograms/bin_spec_{window}_{threshold}.npz", spectrograms=connect_bin_slice_list)
np.savez(f"spectrograms/power_spec_{station}_{window}_{threshold}.npz", spectrograms = connect_power_slice_list)
print("Spectrograms saved")