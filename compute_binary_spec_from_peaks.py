# Compute binary array spectrograms from spectral peaks
from os.path import join
from time import time
from numpy import bool_, linspace, zeros
from pandas import Timestamp
from pandas import date_range
from multiprocessing import Pool

from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import get_geophone_days
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, get_spec_peak_time_freq_inds, read_spectral_peak_counts, save_binary_spectrogram

# Inputs
# Data
window_length = 1.0
overlap = 0.0
downsample = False
downsample_factor = 60
prom_threshold = 10
rbw_threshold = 0.2

min_freq = None
max_freq = 200.0

count_threshold = 4

file_ext_in = "h5"

# Binary spectrogram
num_process = 32

min_freq = 0.0
max_freq = 200.0

time_interval = "1s"
freq_interval = 1.0

outdir = indir

# Plotting
starttime_plot = "2020-01-13T20:00:00"
endtime_plot = "2020-01-13T21:00:00"

date_format = "%Y-%m-%d %H:%M:%S"
major_time_spacing = "15min"
minor_time_spacing = "5min"

size_scale = 30

# Read the spectral-peak count file
print("Reading spectral-peak count file...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq, max_freq = max_freq)

filename_in = f"geo_spectral_peak_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.{file_ext_in}"
inpath = join(indir, filename_in)

count_df = read_spectral_peak_counts(inpath)
num_count = len(count_df)
print(f"Read {num_count} spectral-peak counts.")
print("Done.")

# Construct the time and frequency axes
print("Constructing time and frequency axes...")
days = get_geophone_days()
starttime = Timestamp(days[0]).replace(hour = 0, minute = 0, second = 0, microsecond = 0)
endtime  = Timestamp(days[-1]).replace(hour = 23, minute = 59, second = 59, microsecond = 999999)
timeax = date_range(starttime, endtime, freq = time_interval)
num_time = len(timeax)

num_freq = int((max_freq - min_freq) / freq_interval) + 1
freqax = linspace(min_freq, max_freq, num_freq)
print("Done.")

# Divide the spectral-peak counts into chunks for parallel processing
print("Dividing spectral-peak counts into chunks for parallel processing...")
chunk_size = num_count // num_process
chunks = [count_df.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(num_process)]
args = [(chunk, timeax, freqax) for chunk in chunks]

print(f"Divided spectral-peak counts into {num_process} chunks.")

# Find the time and frequency indices for each chunk
print("Finding the time and frequency indices for each chunk...")
with Pool(num_process) as pool:
    results = pool.starmap(get_spec_peak_time_freq_inds, args)

time_inds = []
freq_inds = []
for result in results:
    time_inds.extend(result[0])
    freq_inds.extend(result[1])
print("Done.")

# Compute the binary spectrograms
print("Computing binary spectrograms...")
data = zeros((num_freq, num_time), dtype = bool_)

num_ones = len(time_inds)
for i in range(num_ones):
    data[freq_inds[i], time_inds[i]] = True
print("Done.")

# Save the binary spectrograms
print("Saving the binary spectrograms...")
filename_out = f"geo_binary_spectrogram_{suffix_spec}_{suffix_peak}_count{count_threshold}.h5"
outpath = join(outdir, filename_out)

save_binary_spectrogram(timeax, freqax, data, outpath)
print("Done.")

