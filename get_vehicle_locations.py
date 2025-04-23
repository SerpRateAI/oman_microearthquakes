"""
Get the locations of the approaching vehicles for all time windows
The vehicle is assumed to be approaching from the south and stopped at the final location at the end of the time window
"""



###
# Import modules
###
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, DataFrame
from numpy import array
from utils_basic import LOC_DIR as dirpath

###
# Input arguments
###
parser = ArgumentParser()
parser.add_argument("--velocity", type = float, help = "The velocity of the vehicle in km/h", default = 40.0)
parser.add_argument("--final_x", type = float, help = "The final x-coordinate of the vehicle in m", default = 25.0)
parser.add_argument("--final_y", type = float, help = "The final y-coordinate of the vehicle in m", default = -90.0)
parser.add_argument("--occurrence", type = str, help = "The occurrence of the vehicle", default = "approaching")
args = parser.parse_args()

velocity = args.velocity
final_x = args.final_x
final_y = args.final_y
occurrence = args.occurrence
velocity = velocity * 1000.0 / 3600.0 # Convert to m/s

###
# Read the vehicle time windows
###
filename = f"vehicle_time_windows_{occurrence}.csv"
filepath = join(dirpath, filename)
time_window_df = read_csv(filepath, parse_dates = ["start_time", "end_time"])

###
# Get the locations of the vehicle for each time window
###
final_time = time_window_df["end_time"].max()
vehicle_xs = []
vehicle_ys = []
for _, row in time_window_df.iterrows():
    # Get the center time of the time window
    start_time = row["start_time"]
    end_time = row["end_time"]
    center_time = start_time + (end_time - start_time) / 2

    # Get the locations of the vehicle at the center time
    vehicle_y = final_y + velocity * (center_time - final_time).total_seconds()
    vehicle_x = final_x

    vehicle_xs.append(vehicle_x)
    vehicle_ys.append(vehicle_y)

time_window_df["vehicle_x"] = vehicle_xs
time_window_df["vehicle_y"] = vehicle_ys

###
# Write the vehicle locations to a file
filename = f"vehicle_locations_{occurrence}.csv"
filepath = join(dirpath, filename)
time_window_df.to_csv(filepath, index = False)
print(f"Vehicle locations written to {filepath}")






