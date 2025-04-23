"""
Get the vehicle apparent velocities of the station triads for each time window.
"""

###
# Import the necessary libraries
###

from argparse import ArgumentParser
from os.path import join
from pandas import read_csv, concat

from utils_basic import LOC_DIR as dirpath_loc, GEO_COMPONENTS as components
from utils_basic import get_geophone_triads

###
# Command-line arguments
###

parser = ArgumentParser(description = "Get the vehicle apparent velocities of the station triads for each time window.")

parser.add_argument("--occurrence", type=str, help="The occurrence of the vehicle signal", default="approaching")

# Parse the arguments
args = parser.parse_args()
occurrence = args.occurrence

###
# Load the data
###

# Load the time windows
filename = f"vehicle_time_windows_{occurrence}.csv"
filepath = join(dirpath_loc, filename)
window_df = read_csv(filepath, parse_dates = ["start_time", "end_time"])

# Load the station triads
triad_df = get_geophone_triads()

# Load the vehicle apparent velocities
vel_dfs = []
for _, row in triad_df.iterrows():
    station1 = row["station1"]
    station2 = row["station2"]
    station3 = row["station3"]

    filename = f"vehicle_station_triad_app_vel_vs_time_{occurrence}_{station1}_{station2}_{station3}.csv"
    filepath = join(dirpath_loc, filename)
    vel_df = read_csv(filepath)
    vel_df["station1"] = station1
    vel_df["station2"] = station2
    vel_df["station3"] = station3
    vel_dfs.append(vel_df)

vel_df = concat(vel_dfs)

###
# Process each time window
###

for _, row in window_df.iterrows():
    window_id = row["window_id"]
    start_time = row["start_time"]
    end_time = row["end_time"]
    print(f"Processing time window {window_id:d} from {start_time} to {end_time}...")

    # Keep only the rows within the time window
    vel_window_df = vel_df[vel_df["window_id"] == window_id]

    # Keep only the rows containing at least non-NaN value for the apparent velocities
    vel_window_df = vel_window_df[ (vel_window_df["app_vel_z"].notna()) | (vel_window_df["app_vel_1"].notna()) | (vel_window_df["app_vel_2"].notna()) ]

    print(f"Number of rows in the vehicle apparent velocities: {len(vel_window_df)}.")

    # Reorder the columns
    columns = ["station1", "station2", "station3"]
    for component in components:
        columns.extend([f"app_vel_{component.lower()}", f"app_vel_std_{component.lower()}", 
                        f"back_azi_{component.lower()}", f"back_azi_std_{component.lower()}", 
                        f"app_vel_east_{component.lower()}", f"app_vel_north_{component.lower()}", f"app_vel_cov_mat_{component.lower()}"])
    vel_window_df = vel_window_df[columns]

    # Save the vehicle apparent velocities
    filename = f"vehicle_station_triad_app_vels_{occurrence}_window{window_id:d}.csv"
    filepath = join(dirpath_loc, filename)
    vel_window_df.to_csv(filepath, index = False, na_rep = "nan")
    print(f"Saved the vehicle apparent velocities for time window {window_id:d} to {filepath}.")


