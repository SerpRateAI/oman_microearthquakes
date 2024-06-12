# Imports
from os.path import join
from pandas import concat
from time import time
from multiprocessing import Pool

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations
from utils_spec import find_geo_station_spectral_peaks, get_spectrogram_file_suffix, get_spec_peak_file_suffix 
from utils_spec import read_geo_spectrograms, read_geo_spec_headers, save_spectral_peaks
from utils_plot import plot_geo_total_psd_and_peaks, save_figure

# Inputs
# Data
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Finding peaks
num_process = 32
rbw_threshold = 3.0
prom_threshold = 10
min_freq = None
max_freq = 200.0

# Writing
to_csv = False
to_hdf = True

# Loop over days and stations
outdir = indir
for station in stations:
    print(f"### Working on {station}... ###")

    # The DataFrame for storing the peak detections for the stations
    peak_dfs = []

    # Read the list of time labels
    suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
    filename_in = f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5"
    print(f"Proessing the file: {filename_in}")

    inpath = join(indir, filename_in)

    header_dict = read_geo_spec_headers(inpath)
    time_labels = header_dict["time_labels"]

    # Process each time label
    for time_label in time_labels:
        clock1 = time()
        # Read the spectrograms
        print(f"Reading the spectrograms of {time_label}...")
        stream_spec = read_geo_spectrograms(inpath, time_labels = [time_label])

        # Find the peaks
        print("Detecting the peaks...")
        peak_df, _ = find_geo_station_spectral_peaks(stream_spec, num_process, 
                                                     rbw_threshold = rbw_threshold, prom_threshold = prom_threshold, 
                                                     min_freq = min_freq, max_freq = max_freq)
        print(f"In total, {len(peak_df)} spectral peaks found.")

        # Add the station to the dataframe
        peak_df["station"] = station

        # Append to the list
        peak_dfs.append(peak_df)

        clock2 = time()
        elapse = clock2 - clock1
        print(f"Elapsed time: {elapse}")

    # Save the results
    print("Saving the results...")
    peak_df = concat(peak_dfs)
    peak_df.drop_duplicates(subset = ["station", "time", "frequency"], inplace = True)
    peak_df.reset_index(drop = True, inplace = True)

    suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq, max_freq = max_freq)
    file_stem = f"geo_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}"

    if to_csv:
        print("Saving the CSV file...")
        clock1 = time()

        save_spectral_peaks(peak_df, file_stem, "csv")
    
        clock2 = time()
        elapse = clock2 - clock1
        print(f"Elapsed time: {elapse}")    
        print("")

    
    if to_hdf:
        print("Saving the HDF file...")
        clock1 = time()
    
        save_spectral_peaks(peak_df, file_stem, "hdf")
    
        clock2 = time()
        elapse = clock2 - clock1
        print(f"Elapsed time: {elapse}")    
        print("")
    
