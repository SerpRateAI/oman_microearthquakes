# Compute the time-cumulative counts of the spectral peaks for the hydrophone data

# Import the required libraries
from os.path import join
from time import time
from argparse import ArgumentParser
from pandas import Timedelta
from pandas import concat
from pandas import read_hdf

from utils_basic import HYDRO_LOCATIONS as location_dict, SPECTROGRAM_DIR as indir, STARTTIME_HYDRO as starttime, ENDTIME_HYDRO as endtime
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix

# Inputs
# Command-line arguments
parser = ArgumentParser(description="Compute the time-cumulative counts of the spectral peaks detected in the hydrophone data")
parser.add_argument("--window_length", type=float, default=300.0, help="Spectrogram window length in seconds")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap fraction between adjacent windows")
parser.add_argument("--min_prom", type=float, default=15.0, help="Minimum prominence threshold for peak detection")
parser.add_argument("--min_rbw", type=float, default=15.0, help="Minimum reverse bandwidth threshold for peak detection")
parser.add_argument("--min_freq", type=float, default=0.0, help="Minimum frequency in Hz for peak detection")
parser.add_argument("--max_freq", type=float, default=200.0, help="Maximum frequency in Hz for peak detection")
parser.add_argument("--max_mean_db", type=float, default=-10.0, help="Maximum mean dB for excluding noisy windows")

# Parse the command-line arguments
args = parser.parse_args()
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Print the command-line arguments
print("### Computing the time-cumulative counts of the hydrophone spectral-peak array detections ###")
print(f"Window length: {window_length:.0f} s")
print(f"Overlap: {overlap:.0%}")
print(f"Minimum prominence: {min_prom:.0f} dB")
print(f"Minimum reverse bandwidth: {min_rbw:.0f} Hz")
print("Maximum mean dB: ", max_mean_db)

# Read the array detections
# Assemble the file suffix
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

# Loop over the hydrophone stations
# Compute the total number of time windows in the deployment period
total_time = endtime - starttime
total_windows = int(total_time / Timedelta(seconds = window_length))

for station in location_dict.keys():
    print(f"Working on station {station}...")
    print("Reading the block timing data...")

    filename = f"hydro_spectral_peak_array_detections_{station}_{suffix_spec}_{suffix_peak}.h5"
    filepath = join(indir, filename)

    block_timing_df = read_hdf(filepath, key = "block_timing")
    print(f"Read {block_timing_df.shape[0]:d} blocks.")

    print("Processing the data of each block...")
    cum_freq_count_dfs = []
    for time_label in block_timing_df['time_label']:
        print(f"Reading the data of time label {time_label}...")
        array_detect_df = read_hdf(filepath, key = time_label)
        
        print("Grouping by frequency...")
        cum_freq_count_df = array_detect_df.groupby('frequency').size().reset_index(name = 'count')
        cum_freq_count_dfs.append(cum_freq_count_df)

    print("Concatenating the dataframes...")
    cum_freq_count_df = concat(cum_freq_count_dfs, axis = 0)

    print("Merging the dataframes...")
    cum_freq_count_df = cum_freq_count_df.groupby('frequency').sum().reset_index()
    cum_freq_count_df = cum_freq_count_df.sort_values("frequency", ascending = True)

    print("Normalizing the counts...")
    cum_freq_count_df['fraction'] = cum_freq_count_df['count'] / total_windows

    print("Saving the data to file...")
    filename_out = f"hydro_spectral_peak_time_cum_freq_counts_{station}_{suffix_spec}_{suffix_peak}.csv"
    outpath = join(indir, filename_out)
    cum_freq_count_df.to_csv(outpath, index = True)
    print(f"Saved to {outpath}.")



    
