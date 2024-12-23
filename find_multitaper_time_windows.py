### Find time windows for multitaper cross-spectral analysis between a pair of stations ###
# The script rank the multitaper windows by the number of STFT time windows in them in descending order

### Import libraries ###
from os.path import join
from argparse import ArgumentParser
from numpy import arange
from pandas import DataFrame, read_hdf, date_range

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
### Inputs ###
# Command-line arguments
parser = ArgumentParser(description = "Find time windows for multitaper analysis.")
parser.add_argument("--mode_name", type = str, help = "Mode name.")
parser.add_argument("--station1", type = str, help = "Name of the first station.")
parser.add_argument("--station2", type = str, help = "Name of the second station.")

parser.add_argument("--window_length_mt", type = float, default = 3600.0, help = "Window length in seconds for multitaper analysis.")
parser.add_argument("--window_length_stft", type = float, default = 300.0, help = "Window length in seconds for computing the STFT.")

parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap between consecutive windows for computing the STFT.")
parser.add_argument("--min_prom", type = float, default = 15.0, help = "Minimum prominence of a spectral peak.")
parser.add_argument("--min_rbw", type = float, default = 15.0, help = "Minimum reverse bandwidth of a spectral peak.")
parser.add_argument("--max_mean_db", type = float, default = 10.0, help = "Maximum mean dB value for excluding noise windows.")

# Parse the arguments
args = parser.parse_args()
mode_name = args.mode_name
station1 = args.station1
station2 = args.station2

window_length_mt = args.window_length_mt
window_length_stft = args.window_length_stft

overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Print the inputs
print(f"### Finding time windows for multitaper analysis ###")
print("")
print(f"Mode name: {mode_name}")
print(f"Window length for multitaper analysis: {window_length_mt} s")
print(f"Window length for computing the STFT: {window_length_stft} s")

### Read the stationary resonance properties ###
print("Reading the stationary resonance properties...")
suffix_spec = get_spectrogram_file_suffix(window_length_stft, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)
filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(indir, filename)


### Find the time windows when both stations recorded the resonance ###
resonance_df = read_hdf(filepath, key = "properties")
resonance_sta1_df = resonance_df[resonance_df["station"] == station1]
resonance_sta2_df = resonance_df[resonance_df["station"] == station2]
resonance_common_df = resonance_sta1_df.merge(resonance_sta2_df, on = ["time"], how = "inner", suffixes = ("_sta1", "_sta2"))

### Find the time windows for multitaper analysis ###
# Divide the time range between starttime and endtime into windows of length window_length_mt
# The windows are non-overlapping
# The windows are stored in a DataFrame with columns "start" and "end"
print("Finding the time windows for multitaper analysis...")
time_ranges = date_range(starttime, endtime, freq = f"{window_length_mt}s", tz = "UTC")

# Find the number of STFT time windows in each multitaper window
num_stft_windows = []
starttimes = []
endtimes = []
mean_freqs = []
std_freqs = []

for start, end in zip(time_ranges[:-1], time_ranges[1:]):
    resonance_win_df = resonance_common_df[(resonance_common_df["time"] >= start) & (resonance_common_df["time"] < end)]
    mean_freq = resonance_win_df["frequency_sta1"].mean()
    std_freq = resonance_win_df["frequency_sta1"].std()

    starttimes.append(start)
    endtimes.append(end)
    mean_freqs.append(mean_freq)
    std_freqs.append(std_freq)
    num_stft_windows.append(len(resonance_win_df))


# Create a DataFrame with the time windows and the number of STFT time windows in them
num_win_df = DataFrame({"start": starttimes, "end": endtimes, "mean_freq": mean_freqs, "std_freq": std_freqs, "num_stft_windows": num_stft_windows})

# Sort the time windows by the number of STFT time windows in them in descending order
num_win_df.sort_values("num_stft_windows", ascending = False, inplace = True)
num_win_df.reset_index(drop = True, inplace = True)

### Save the time windows for multitaper analysis ###
print("Saving the time windows for multitaper analysis...")
filename_out = f"multitaper_time_windows_{mode_name}_{station1}_{station2}_mt_win{window_length_mt:.0f}s_stft_win{window_length_stft:.0f}s.csv"
outpath = join(indir, filename_out)
num_win_df.to_csv(outpath, index = True)
print(f"Time windows saved to {outpath}.")