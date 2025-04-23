"""
Generate the time windows for multi-taper analysis of the vehicle signals   
"""

# Import the necessary libraries
from os.path import join
import argparse
from utils_basic import LOC_DIR as dirpath_loc
from pandas import read_csv, date_range, DataFrame

# Input arguments
parser = argparse.ArgumentParser(description = "Generate the time windows for multi-taper analysis of the vehicle signals")
parser.add_argument("--occurrence", type = str, help = "The occurrence of the vehicle")
parser.add_argument("--window_length", type = float, default = 1.0, help = "The length of the time window in seconds")

args = parser.parse_args()
occurrence = args.occurrence
window_length = args.window_length


# Read the vehicle data
inpath = join(dirpath_loc, "vehicles.csv")
vehicle_df = read_csv(inpath, dtype={"occurrence": str}, parse_dates=["start_time", "end_time"])

# Get the time windows
start_time = vehicle_df[ vehicle_df["occurrence"] == occurrence ]["start_time"].values[0]
end_time = vehicle_df[ vehicle_df["occurrence"] == occurrence ]["end_time"].values[0]

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
outpath = join(dirpath_loc, f"vehicle_time_windows_{occurrence}.csv")
time_window_df.to_csv(outpath, index = False)
print(f"The time windows for the vehicle {occurrence} have been saved to {outpath}")
