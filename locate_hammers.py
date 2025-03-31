"""
Locate hammer shots using a grid search across Subarray A.
Only the picks on Array A are used.
The traveltime curve is computed using a 1D velocity model.
"""

###
# Import the necessary libraries
###

from os.path import join
from time import time
from argparse import ArgumentParser
from numpy import arange, meshgrid, sqrt, zeros, mean, nanargmin, unravel_index, isnan, nan
from pandas import read_csv, concat
from pandas import DataFrame, Timestamp
from numpy import interp
from utils_basic import VEL_MODEL_DIR as dirpath_vel, PICK_DIR as dirpath_pick, LOC_DIR as dirpath_loc, EASTMIN_A as min_east, EASTMAX_A as max_east, NORTHMIN_A as min_north, NORTHMAX_A as max_north
from utils_basic import get_geophone_coords
from utils_snuffler import read_normal_markers

###
# Input parameters
###

# Command line arguments
parser = ArgumentParser()
parser.add_argument("--model_name", type = str, default = "vp_1d", help = "Name of the velocity model")
parser.add_argument("--east_int", type = float, default = 1.0, help = "East interval (m)")
parser.add_argument("--north_int", type = float, default = 1.0, help = "North interval (m)")
parser.add_argument("--min_east", type = float, default = min_east, help = "Minimum east coordinate (m)")
parser.add_argument("--max_east", type = float, default = max_east, help = "Maximum east coordinate (m)")
parser.add_argument("--min_north", type = float, default = min_north, help = "Minimum north coordinate (m)")
parser.add_argument("--max_north", type = float, default = max_north, help = "Maximum north coordinate (m)")
parser.add_argument("--min_num_res", type = int, default = 5, help = "Minimum number of residuals")

# Parse the command line arguments
args = parser.parse_args()
model_name = args.model_name
east_int = args.east_int
north_int = args.north_int
min_east = args.min_east
max_east = args.max_east
min_north = args.min_north
max_north = args.max_north
min_num_res = args.min_num_res

###
# Load the input data
###

# Load the list of hammers
print(f"Loading the list of hammers...")
filename = f"hammer_ids.csv"
filepath = join(dirpath_loc, filename)
hammer_df = read_csv(filepath, dtype = {"hammer_id": str})

# Load the list of located hammers
print(f"Loading the list of located hammers...")
filename = f"hammer_locations.csv"
filepath = join(dirpath_loc, filename)
located_df = read_csv(filepath, dtype = {"hammer_id": str})

# Load the traveltime curve
print(f"Loading the traveltime curve for {model_name}...")
filename = f"hammer_traveltime_curve_{model_name}.csv"
filepath = join(dirpath_vel, filename)
traveltime_df = read_csv(filepath)

# Load the geophone coordinates
print(f"Loading the geophone coordinates...")
station_df = get_geophone_coords()

# Keep only the picks on Array A
station_df = station_df[station_df["subarray"] == "A"]

###
# Find the hammer shots that were not located
###

# Find the hammer shots that were not located
hammer_df = hammer_df[~hammer_df["hammer_id"].isin(located_df["hammer_id"])]

print(f"Number of hammers to locate: {len(hammer_df)}")


# ##
# Locate each hammer shot
###

# Define the grid
# Round the min/max coordinates to the nearest interval
min_east = east_int * round(min_east / east_int)
max_east = east_int * round(max_east / east_int)
min_north = north_int * round(min_north / north_int) 
max_north = north_int * round(max_north / north_int)

east_grid = arange(min_east, max_east, east_int)
north_grid = arange(min_north, max_north, north_int)

east_mesh, north_mesh = meshgrid(east_grid, north_grid)

# Loop through each hammer shot
output_dicts = []
for hammer_id in hammer_df["hammer_id"]:
    print("--------------------------------")
    print(f"Locating hammer {hammer_id}...")
    clock1 = time()
    
    # Load the pick data
    print(f"Loading the pick data for hammer {hammer_id}...")
    filename = f"hammer_{hammer_id}_p_geo.txt"
    filepath = join(dirpath_pick, filename)
    pick_df = read_normal_markers(filepath)

    # Convert pick times to nanoseconds for easier computation
    pick_df["time"] = pick_df["time"].astype("int64")

    # Keep only the picks with valid coordinates
    pick_df = pick_df[pick_df["station"].isin(station_df.index)]

    # Compute the RMS for each grid point
    num_east = len(east_grid)
    num_north = len(north_grid)
    rms_mat = zeros((num_north, num_east))
    otime_mat = zeros((num_north, num_east))

    print(f"Performing grid search...")
    for i in range(num_east):
        for j in range(num_north):
            east_shot = east_grid[i]
            north_shot = north_grid[j]
            
            otimes_sta = []
            traveltimes = []
            atimes = []
            for k, row in pick_df.iterrows():
                station = row["station"]
                atime = row["time"]

                # Get the station coordinates
                east_sta = station_df.loc[station, "east"]
                north_sta = station_df.loc[station, "north"]

                # Compute the distance
                dist_shot = sqrt((east_shot - east_sta) ** 2 + (north_shot - north_sta) ** 2)

                if dist_shot < traveltime_df["distance"].min() or dist_shot > traveltime_df["distance"].max():
                    continue
                
                # Interpolate the traveltime
                traveltime = interp(dist_shot, traveltime_df["distance"], traveltime_df["traveltime"])
                traveltime = traveltime * 1e9 # Convert to nanoseconds

                traveltimes.append(traveltime)
                atimes.append(atime)

                otime_sta = atime - traveltime
                otimes_sta.append(otime_sta)

            # Check if the number of residuals is greater than the minimum number of residuals
            if len(otimes_sta) < min_num_res:
                rms_mat[j, i] = nan
                otime_mat[j, i] = nan
                continue

            # Get the estimated origin time
            otime_shot = mean(otimes_sta)
            otime_mat[j, i] = otime_shot

            # Compute the RMS
            residuals = atimes - otime_shot - traveltimes
            rms = sqrt(mean(residuals ** 2))
            rms_mat[j, i] = rms

    # Find the grid point with the smallest RMS (ignoring NaNs)
    ind = nanargmin(rms_mat)

    # Convert the flattened index to 2D matrix indices
    j_min, i_min = unravel_index(ind, rms_mat.shape)

    # Get the minimum RMS and corresponding location
    min_rms = rms_mat[j_min, i_min] / 1e9 # Convert to seconds
    east_shot = east_grid[i_min] 
    north_shot = north_grid[j_min]
    otime_shot = otime_mat[j_min, i_min]

    # Convert the origin time to a datetime object
    otime_shot = Timestamp(otime_shot)

    print(f"Estimated origin time: {otime_shot}")
    print(f"Estimated shot coordinates: {east_shot}, {north_shot}")
    print(f"RMS: {min_rms} s")
    clock2 = time()
    print(f"Time taken: {clock2 - clock1} seconds")
    print("--------------------------------")

    output_dicts.append({"hammer_id": hammer_id, "origin_time": otime_shot, "east": east_shot, "north": north_shot, "rms": min_rms})

###
# Save the results
###

filename = f"hammer_locations.csv"
new_location_df = DataFrame(output_dicts)
if len(located_df) > 0:
    output_df = concat([located_df, new_location_df])
else:
    output_df = new_location_df

output_df = output_df.sort_values(by = "hammer_id")
output_df = output_df.reset_index(drop = True)

filepath = join(dirpath_loc, filename)
output_df.to_csv(filepath, index = False)
print(f"Results saved to {filepath}")
