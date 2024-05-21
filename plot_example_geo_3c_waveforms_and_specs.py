# Plot the 3C geophone waveforms and spectrograms in a time window
from os.path import join
from time import time
from numpy import bool_, linspace, zeros
from pandas import Timestamp
from pandas import date_range
from multiprocessing import Pool

from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import get_geo_metadata, time2suffix
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_spec import concat_stream_spec, get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_binary_spectrogram, read_geo_spectrograms, read_spectral_peaks, read_spectral_peak_counts
from utils_plot import plot_geo_3c_waveforms_and_stfts, save_figure

# Inputs
# Stations
stations = ["A01", "A16", "A19", "B01", "B19", "B20"]

# Time window
starttime = "2020-01-13T20:00:00"
endtime = "2020-01-13T20:01:30"

# Waveforms
min_freq_filt = 40.0
max_freq_filt = 100.0

# Spectrograms
window_length = 1.0
overlap = 0.0
downsample = False
downsample_factor = 60

min_freq_spec = None
max_freq_spec = 200.0

# Plotting
dbmin = -30
dbmax = 10

# Read and process waveforms
print("Reading and processing waveforms...")
metadat = get_geo_metadata()
stream_wf = read_and_process_windowed_geo_waveforms(starttime, metadat, endtime = endtime, stations = stations, freqmin = min_freq_filt, freqmax = max_freq_filt)
print("Done.")

# Read the spectrograms
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
streams = []
for station in stations:
    filename_in = f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5"
    inpath = join(indir, filename_in)
    
    print(f"Reading spectrogram from {inpath}...")
    stream_spec = read_geo_spectrograms(inpath, starttime = starttime, endtime = endtime, min_freq = min_freq_spec, max_freq = max_freq_spec)
    streams.append(stream_spec)
    print("Done.")

print("Concatenating the spectrograms...")
stream_spec = concat_stream_spec(streams)
print("Done.")

# Plotting
print("Plotting...")
fig, axes, cbar = plot_geo_3c_waveforms_and_stfts(stream_wf, stream_spec, dbmin = dbmin, dbmax = dbmax, min_freq = min_freq_spec, max_freq = max_freq_spec)
print("Done.")

# Save the figure
print("Saving the figure...")
suffix_start = time2suffix(starttime)
suffix_end = time2suffix(endtime)
filename_out = f"example_geo_3c_waveforms_and_specs_{suffix_start}_{suffix_end}.png"

save_figure(fig, filename_out)

