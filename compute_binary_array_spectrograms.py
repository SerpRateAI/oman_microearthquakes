# Compute binary array spectrograms from spectral peak counts
from os.path import join
from time import time
from numpy import bool_, linspace, zeros
from pandas import Timestamp
from pandas import date_range
from multiprocessing import Pool

from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import get_geophone_days
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, get_spec_peak_time_freq_inds, read_spec_peak_array_counts, save_binary_spectrogram

# Inputs
# Data
# Spectrogram computation
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Spectral peak detection
prom_threshold = 15.0
rbw_threshold = 3.0

min_freq_peak = None
max_freq_peak = 200.0

count_threshold = 9

# Binary spectrogram
num_process = 32

min_freq_bin = 0.0
max_freq_bin = 200.0

time_interval = "1s"
freq_interval = 1.0

outdir = indir

# Print the inputs
print(f"### Computing binary array spectrograms from spectral peak counts in {num_process} processes ###")
print(f"Window length: {window_length} s")
print(f"Overlap: {overlap}")
print(f"Downsample: {downsample}")

if downsample:
    print(f"Downsample factor: {downsample_factor}")

print(f"Reverse-bandwidth threshold: {rbw_threshold} 1/Hz")
print(f"Prominence threshold: {prom_threshold} dB")
print(f"Frequency range: {min_freq_peak} - {max_freq_peak} Hz")
print("")

print(f"Count threshold: {count_threshold}")
print("")

print(f"Frequency range for binary spectrogram: {min_freq_bin} - {max_freq_bin} Hz")
print(f"Time interval: {time_interval}")
print(f"Frequency interval: {freq_interval} Hz")


# Read the spectral-peak count file
print("Reading spectral-peak count file...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)

filename_in = f"geo_spectral_peak_array_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.h5"
inpath = join(indir, filename_in)

count_df = read_spec_peak_array_counts(inpath)
num_count = len(count_df)
print(f"Read {num_count} spectral-peak counts.")
print("Done.")

# Construct the time and frequency axes
print("Constructing time and frequency axes...")
days = get_geophone_days()
starttime = days[0].replace(hour = 0, minute = 0, second = 0, microsecond = 0)
endtime  = days[-1].replace(hour = 23, minute = 59, second = 59, microsecond = 999999)
timeax = date_range(starttime, endtime, freq = time_interval)
num_time = len(timeax)

num_freq = int((max_freq_bin - min_freq_bin) / freq_interval) + 1
freqax = linspace(min_freq_bin, max_freq_bin, num_freq)
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
filename_out = f"geo_binary_array_spectrogram_{suffix_spec}_{suffix_peak}_count{count_threshold}.h5"
outpath = join(outdir, filename_out)

save_binary_spectrogram(timeax, freqax, data, outpath)
print("Done.")

