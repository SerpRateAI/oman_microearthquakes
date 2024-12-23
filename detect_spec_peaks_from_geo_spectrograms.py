# Detect spectral peaks in the geophone spectrograms
# The results of each time label are saved in a separate group in the HDF file referenced by the time label

# Imports
from os.path import join
from pandas import Series
from pandas import concat
from time import time
from h5py import File


from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations
from utils_spec import find_trace_spectral_peaks, get_spectrogram_file_suffix, get_spec_peak_file_suffix 
from utils_spec import read_geo_spectrograms, read_spec_block_timings
from utils_plot import plot_geo_total_psd_and_peaks, save_figure

# Inputs
# Data
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Finding peaks
num_process = 32
prom_threshold = 15.0
rbw_threshold = 3.0

min_freq = None
max_freq = 200.0

# Loop over days and stations
print(f"### Detecting spectral peaks in the geophone power spectrograms in {num_process} processes ###")
print(f"Window length: {window_length} s")
print(f"Overlap: {overlap}")
print(f"Downsample: {downsample}")

if downsample:
    print(f"Downsample factor: {downsample_factor}")

print(f"Reverse-bandwidth threshold: {rbw_threshold} 1/Hz")
print(f"Prominence threshold: {prom_threshold} dB")
print(f"Frequency range: {min_freq} - {max_freq} Hz")
print("")

outdir = indir
for station in stations:
    print(f"### Working on {station}... ###")

    # Read the list of time labels
    suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
    filename_in = f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5"
    print(f"Proessing the file: {filename_in}")

    inpath = join(indir, filename_in)
    block_timing_df = read_spec_block_timings(inpath)

    suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq, max_freq = max_freq)
    filename_out = f"geo_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.h5"

    # Save the time labels
    print("Saving the time labels...")
    outpath = join(outdir, filename_out)
    block_timing_df.to_hdf(outpath, key = "block_timings", mode = "w")
    time_labels = block_timing_df["time_label"]
    
    # Process each time label
    for time_label in time_labels:
        clock1 = time()
        # Read the spectrograms
        print(f"Reading the spectrograms of {time_label}...")
        stream_spec = read_geo_spectrograms(inpath, time_labels = [time_label], min_freq = min_freq, max_freq = max_freq)

        # Compute the total power
        print("Computing the total power...")
        trace_total_power = stream_spec.get_total_power()

        # Find the peaks
        print("Detecting the peaks...")
        peak_df = find_trace_spectral_peaks(trace_total_power, num_process, 
                                            rbw_threshold = rbw_threshold, prom_threshold = prom_threshold,
                                            min_freq = min_freq, max_freq = max_freq)

        print(f"In total, {len(peak_df)} spectral peaks found.")
        peak_df["station"] = station
        peak_df.to_hdf(outpath, key = time_label, mode = "a")

        clock2 = time()
        elapse = clock2 - clock1
        print(f"Elapsed time: {elapse}")
        print("")