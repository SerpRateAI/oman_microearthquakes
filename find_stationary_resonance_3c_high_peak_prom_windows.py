# Find the time windows with high peak prominence for all three components of a stationary resonance

### Import the required libraries ###
from os.path import join
from argparse import ArgumentParser
from pandas import DataFrame
from pandas import read_hdf

from utils_basic import SPECTROGRAM_DIR as indir, GEO_COMPONENTS as components

### Inputs ###
# Command line arguments
parser = ArgumentParser(description = "Find the time windows with high peak prominence for all three components of a stationary resonance")
parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--peak_prom_threshold", type = float, help = "Peak prominence threshold")
parser.add_argument("--min_num_stations", type = int, help = "Minimum number of stations satisfying the peak prominence threshold")

# Parse the command line arguments
args = parser.parse_args()
mode_name = args.mode_name
peak_prom_threshold = args.peak_prom_threshold
min_num_stations = args.min_num_stations

### Read the data ###
print(f"Reading the properties of {mode_name}...")
filename = f"stationary_resonance_phase_amplitude_diff_{mode_name}_geo.h5"
inpath = join(indir, filename)
properties_df = read_hdf(inpath, key = "properties")

### Find the time windows with high peak prominence for all three components ###
print("Finding the time windows with high peak prominence for all three components...")
# Get the unique time windows
time_windows = properties_df["time"].unique()

# For each time window, find the number of stations satisfying the peak prominence threshold for all three components
out_dicts = []
for time_window in time_windows:
    # Get the properties of the time window
    properties_window_df = properties_df[properties_df["time"] == time_window]

    # Find the number of stations satisfying the peak prominence threshold for all three components
    properties_station_df = properties_window_df[(properties_window_df["peak_prom_z"] >= peak_prom_threshold) &
                                                 (properties_window_df["peak_prom_1"] >= peak_prom_threshold) &
                                                 (properties_window_df["peak_prom_2"] >= peak_prom_threshold)]
    num_station = properties_station_df.shape[0]

    # If the number of stations is greater than or equal to the minimum number of stations, add the time window to the output dictionary
    if num_station >= min_num_stations:
        out_dict = {"time": time_window, "num_station": num_station}
        out_dicts.append(out_dict)

# Convert the output dictionary to a DataFrame
out_df = DataFrame(out_dicts)

# Sort the DataFrame by the number of stations in descending order
out_df = out_df.sort_values(by = ["num_station", "time"], ascending = [False, True])
out_df.reset_index(drop = True, inplace = True)

# Save the DataFrame to a CSV file
outpath = join(indir, f"stationary_resonance_time_windows_3c_high_peak_prom_{mode_name}_{peak_prom_threshold:.0f}db.csv")
out_df.to_csv(outpath, index = True)
print(f"Saved the time windows with high peak prominence for all three components to {outpath}")
