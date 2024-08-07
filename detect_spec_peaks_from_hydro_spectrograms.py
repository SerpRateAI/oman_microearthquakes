# Detect spectral peaks in the hydrophone spectrograms
# The results of each time label are saved in a separate group in the HDF file referenced by the time label

# Imports
from os.path import join
from pandas import Series
from pandas import concat
from time import time
from h5py import File


from utils_basic import SPECTROGRAM_DIR as indir, HYDRO_LOCATIONS as location_dict
from utils_spec import find_trace_spectral_peaks, get_spectrogram_file_suffix, get_spec_peak_file_suffix 
from utils_spec import read_hydro_spectrograms, read_hydro_spec_time_labels, save_spectral_peaks
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


# Loop over days and stations
print(f"### Detecting spectral peaks in the hydrophone spectrograms in {num_process} processes ###")
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
for station, locations in location_dict.items():
    print(f"### Working on {station}... ###")

    # Read the list of time labels
    suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
    filename_in = f"whole_deployment_daily_hydro_spectrograms_{station}_{suffix_spec}.h5"
    print(f"Proessing the file: {filename_in}")

    inpath = join(indir, filename_in)
    time_labels = read_hydro_spec_time_labels(inpath)

    suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq, max_freq = max_freq)
    filename_out = f"hydro_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.h5"

    # Save the time labels
    print("Saving the time labels...")
    outpath = join(outdir, filename_out)
    time_label_sr = Series(time_labels, name = "time_label")
    time_label_sr.to_hdf(outpath, key = "time_label", mode = "w")

    # Save the locations
    print("Saving the locations...")
    location_sr = Series(locations, name = "location")
    location_sr.to_hdf(outpath, key = "location", mode = "a")
    
    # Process each time label
    for time_label in time_labels:
        clock1 = time()
        # Read the spectrograms
        print(f"Reading the spectrograms of {time_label}...")
        stream_spec = read_hydro_spectrograms(inpath, time_labels = [time_label], min_freq = min_freq, max_freq = max_freq)

        # Find the peaks
        print("Detecting the peaks...")
        peak_dfs = []
        for location in locations:
            print(f"Working on {location}...")
            trace_spec = stream_spec.select(location = location)[0]
            peak_df = find_trace_spectral_peaks(trace_spec, num_process, 
                                                rbw_threshold = rbw_threshold, prom_threshold = prom_threshold,
                                                min_freq = min_freq, max_freq = max_freq)

            print(f"In total, {len(peak_df)} spectral peaks found.")
            peak_df["station"] = station
            peak_df["location"] = location

            peak_dfs.append(peak_df)

        peak_df = concat(peak_dfs, ignore_index = True)

        # Save the results of the time label
        print("Saving the results...")
        peak_df.drop_duplicates(subset = ["station", "location", "time", "frequency"], inplace = True)
        peak_df.reset_index(drop = True, inplace = True)
        
        peak_df.to_hdf(outpath, key = time_label, mode = "a")

        clock2 = time()
        elapse = clock2 - clock1
        print(f"Elapsed time: {elapse}")
        print("")