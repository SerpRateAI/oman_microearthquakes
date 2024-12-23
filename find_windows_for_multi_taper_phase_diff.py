# Find the windows for multi-taper phase difference measurements
#
### Import necessary modules ###
from os.path import join
from argparse import ArgumentParser
from json import loads
from pandas import DataFrame
from pandas import concat, read_csv, read_hdf

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix

### Inputs ###
# Command-line arguments
parser = ArgumentParser(description = "Find the windows for multi-taper phase difference measurements.")
parser.add_argument("--mode_name", type = str, help = "Mode name.")
parser.add_argument("--stations", type = str, help = "List of geophone stations.")

parser.add_argument("--window_length", type = float, default = 60.0, help = "Window length in seconds for computing the spectrogram.")
parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap between consecutive windows for computing the spectrogram.")
parser.add_argument("--min_prom", type = float, default = 15.0, help = "Minimum prominence of a spectral peak.")
parser.add_argument("--min_rbw", type = float, default = 3.0, help = "Minimum reverse bandwidth of a spectral peak.")
parser.add_argument("--max_mean_db", type = float, default = 10.0, help = "Maximum mean dB value for excluding noise windows.")

# Parse the arguments
args = parser.parse_args()
mode_name = args.mode_name
stations = loads(args.stations)

window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Print the inputs
print(f"### Finding the windows for multi-taper phase difference measurements ###")
print("")
print(f"Mode name: {mode_name}")
print(f"Stations: {stations}")
print("")

print(f"# Spectrogram computation #")
print(f"Window length: {window_length} s")
print(f"Overlap: {overlap}")
print("")

print(f"# Spectral-peak detection #")
print(f"Prominence threshold: {min_prom}")
print(f"Reverse bandwidth threshold: {min_rbw}")
print(f"Maximum mean dB value for excluding noise windows: {max_mean_db} dB")
print("")

### Read the stationary resonance properties ###
print("Reading the stationary resonance properties...")
# Read the spectral peak properties
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(indir, filename)
resonance_df = read_hdf(filepath, key = "properties")

resonance_df = resonance_df[resonance_df["station"].isin(stations)]

### Find the windows for multi-taper phase difference measurements ###
# Group the resonance properties by time
resonance_grouped = resonance_df.groupby("time")

# Find the windows when all stations register the resonance
windows = []
mean_total_powers = []
for time, resonance_time_df in resonance_grouped:
    if len(resonance_time_df) == len(stations):
        windows.append(time)
        mean_power = resonance_time_df["total_power"].mean()
        mean_total_powers.append(mean_power)

# Create a DataFrame for the windows
window_dict = {"time": windows, "mean_total_power": mean_total_powers}
window_df = DataFrame(window_dict)
print(f"Number of windows: {len(window_df)}")

### Save the windows ###
print("Saving the windows...")
filename = f"multi_taper_phase_diff_windows_{mode_name}_{suffix_spec}_{suffix_peak}.csv"
filepath = join(indir, filename)
window_df.to_csv(filepath, date_format = "%Y-%m-%d %H:%M:%S")
print(f"Windows saved to {filepath}.")

