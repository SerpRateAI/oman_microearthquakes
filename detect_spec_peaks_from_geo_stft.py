# Detect spectral peaks in the geophone STFTs containing power and phase information
# The results of each time label are saved in a separate group in the HDF file referenced by the time label

# Imports
from os.path import join
from argparse import ArgumentParser
from pandas import DataFrame
from pandas import concat, to_datetime
from time import time
from h5py import File

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations
from utils_preproc import get_envelope
from utils_spec import find_geo_station_spectral_peaks, get_spectrogram_file_suffix, get_spec_peak_file_suffix 
from utils_spec import read_geo_stft, read_spec_block_timings
from utils_plot import plot_geo_total_psd_and_peaks, save_figure

# Inputs
# Command line arguments
parser = ArgumentParser(description = "Detect spectral peaks in the geophone STFTs")
parser.add_argument("--min_prom", type = float, default = 15.0, help = "Minimum prominence in dB")
parser.add_argument("--min_rbw", type = float, default = 15.0, help = "Minimum reverse bandwidth in 1/Hz")
parser.add_argument("--min_freq", type = float, default = 0.0, help = "Minimum frequency in Hz")
parser.add_argument("--max_freq", type = float, default = 200.0, help = "Maximum frequency in Hz")
parser.add_argument("--max_mean_db", type = float, default = 10.0, help = "Maximum mean dB for excluding noisy windows")

parser.add_argument("--window_length", type = float, help = "Window length in seconds")
parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap in seconds")
parser.add_argument("--num_process", type = int, default = 32, help = "Number of processes for detecting the peaks")


# Parse the arguments
args = parser.parse_args()
min_prom = args.min_prom
min_rbw = args.min_rbw
min_freq = args.min_freq
max_freq = args.max_freq
max_mean_db = args.max_mean_db

window_length = args.window_length
overlap = args.overlap
num_process = args.num_process


# Print the parameters
print(f"### Detecting spectral peaks in the geophone STFTs in {num_process} processes ###")
print(f"Window length: {window_length} s")
print(f"Overlap: {overlap}")

print(f"Reverse-bandwidth threshold: {min_rbw} 1/Hz")
print(f"Prominence threshold: {min_prom} dB")
print(f"Frequency range: {min_freq} - {max_freq} Hz")
print(f"Maximum mean dB to exclude noisy windows: {max_mean_db} dB")
print("")

outdir = indir
for station in stations:
    print(f"### Working on {station}... ###")

    # Read the list of time labels
    suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
    filename_in = f"whole_deployment_daily_geo_stft_{station}_{suffix_spec}.h5"
    print(f"Proessing the file: {filename_in}")

    inpath = join(indir, filename_in)
    block_timing_in_df = read_spec_block_timings(inpath)

    suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)
    filename_out = f"geo_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.h5"

    # Extract the time labels
    print("Saving the block timings...")
    outpath = join(outdir, filename_out)
    
    # Process each time label
    time_labels_out = []
    for _, row in block_timing_in_df.iterrows():
        clock1 = time()
        time_label = row["time_label"]
        starttime = row["start_time"]
        endtime = row["end_time"]
        print(f"Processing the time label: {time_label} ({starttime} - {endtime})")

        # Read the spectrograms
        print(f"Reading the spectrograms...")
        stream_stft = read_geo_stft(inpath, time_labels = [time_label], min_freq = min_freq, max_freq = max_freq)

        # Find the peaks
        print("Detecting the peaks...")

        peak_df = find_geo_station_spectral_peaks(stream_stft, num_process, 
                                                  min_prom, min_rbw, max_mean_db,
                                                  min_freq = min_freq, max_freq = max_freq)

        if peak_df is None:
            print("No peaks found. Skipping...")
            continue

        print(f"In total, {len(peak_df)} spectral peaks found.")

        time_labels_out.append(time_label)

        # Save the results of the time label
        print("Saving the peak properties...")
        peak_df["station"] = station
        peak_df.to_hdf(outpath, key = time_label, mode = "a")

        clock2 = time()
        elapse = clock2 - clock1
        print(f"Elapsed time: {elapse}")
        print("")

    # Save the block timings
    print("Saving the block timings...")
    block_timing_out_df = block_timing_in_df.loc[block_timing_in_df["time_label"].isin(time_labels_out)]
    block_timing_out_df.to_hdf(outpath, key = "block_timing", mode = "a")
    print("")