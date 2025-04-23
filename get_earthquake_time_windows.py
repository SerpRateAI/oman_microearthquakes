"""
Generate the time windows for multi-taper analysis of the earthquake signals
"""

# Import the necessary libraries
from os.path import join
import argparse
from utils_basic import LOC_DIR as dirpath_loc
from pandas import read_csv, date_range, DataFrame

# Input arguments
parser = argparse.ArgumentParser(description = "Generate the time windows for multi-taper analysis of the earthquake signals")
parser.add_argument("--earthquake_id", type = str, help = "The ID of the earthquake")
parser.add_argument("--window_length", type = float, default = 1.0, help = "The length of the time window in seconds")

args = parser.parse_args()
earthquake_id = args.earthquake_id
window_length = args.window_length


# Read the earthquake data
inpath = join(dirpath_loc, "earthquakes.csv")
earthquake_df = read_csv(inpath, dtype={"earthquake_id": str}, parse_dates=["start_time", "end_time"])

# Get the time windows
start_time = earthquake_df[ earthquake_df["earthquake_id"] == earthquake_id ]["start_time"].values[0]
end_time = earthquake_df[ earthquake_df["earthquake_id"] == earthquake_id ]["end_time"].values[0]

# Generate the time windows
time_windows = date_range(start = start_time, end = end_time, freq = f"{window_length}s")

# Assemble the time windows into a dataframe
start_times = time_windows[:-1]
end_times = time_windows[1:]
time_window_df = DataFrame({"start_time": start_times, "end_time": end_times})
time_window_df["window_id"] = time_window_df.index + 1

# Reorder columns to put window_id first
time_window_df = time_window_df[["window_id", "start_time", "end_time"]]

# Save the time windows
outpath = join(dirpath_loc, f"earthquake_time_windows_eq{earthquake_id}.csv")
time_window_df.to_csv(outpath, index = False)
print(f"The time windows for the earthquake {earthquake_id} have been saved to {outpath}")
