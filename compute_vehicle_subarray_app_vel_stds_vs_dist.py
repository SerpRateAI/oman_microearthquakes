"""
Compute the standard deviation of the apparent velocities of the two horizontal components of the vehicle signals on the two subarrays vs. the distance between the vehicle and the boreholes
The results of the two horizontal components are averaged
"""

###
# Import modules
###
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, DataFrame
from numpy import deg2rad, rad2deg, sqrt, nan

from utils_basic import LOC_DIR as dirpath
from utils_basic import INNER_STATIONS_A as inner_stations_a, INNER_STATIONS_B as inner_stations_b, MIDDLE_STATIONS_A as middle_stations_a, MIDDLE_STATIONS_B as middle_stations_b, OUTER_STATIONS_A as outer_stations_a, OUTER_STATIONS_B as outer_stations_b
from utils_basic import get_angles_std,get_borehole_coords

###
# Input arguments
###
parser = ArgumentParser()
parser.add_argument("--min_num_triad", type = int, help = "The minimum number of triads for a time window to be considered", default = 5)
parser.add_argument("--max_back_azi_std", type = float, help = "The maximum standard deviation of the back azimuth for a vector to be considered", default = 10.0)
parser.add_argument("--occurrence", type = str, help = "The occurrence of the vehicle", default = "approaching")
args = parser.parse_args()

occurrence = args.occurrence
min_num_triad = args.min_num_triad
max_back_azi_std = args.max_back_azi_std

###
# Read the inputs
###
# Vehicle locations
filename = f"vehicle_locations_{occurrence}.csv"
filepath = join(dirpath, filename)
vehicle_loc_df = read_csv(filepath, parse_dates = ["start_time", "end_time"])

# Borehole coordinates
borehole_loc_df = get_borehole_coords()

# Station triads to consider
stations_a = inner_stations_a + middle_stations_a + outer_stations_a
stations_b = inner_stations_b + middle_stations_b + outer_stations_b

###
# Compute the standard deviation for each time window
###
borehole_a_x = borehole_loc_df.loc["BA1A", "east"]
borehole_a_y = borehole_loc_df.loc["BA1A", "north"]
borehole_b_x = borehole_loc_df.loc["BA1B", "east"]
borehole_b_y = borehole_loc_df.loc["BA1B", "north"]


result_dicts = []
for _, row in vehicle_loc_df.iterrows():
    window_id = row["window_id"]
    start_time = row["start_time"]
    end_time = row["end_time"]
    vehicle_x = row["vehicle_x"]
    vehicle_y = row["vehicle_y"]

    print(f"Processing window {window_id} from {start_time} to {end_time}")

    # Compute the distance between the vehicle and the boreholes
    dist_a = sqrt((vehicle_x - borehole_a_x) ** 2 + (vehicle_y - borehole_a_y) ** 2)
    dist_b = sqrt((vehicle_x - borehole_b_x) ** 2 + (vehicle_y - borehole_b_y) ** 2)

    result_dict = {"window_id": window_id, "start_time": start_time, "end_time": end_time, "distance_a": dist_a, "distance_b": dist_b}

    # Read the apparent velocities for the time window
    filename = f"vehicle_station_triad_app_vels_{occurrence}_window{window_id}.csv"
    filepath = join(dirpath, filename)
    app_vel_df = read_csv(filepath)

    # Get the apparent velocities for Subarray A and Subarray B
    app_vel_a_df = app_vel_df[ (app_vel_df["station1"].isin(stations_a)) & (app_vel_df["station2"].isin(stations_a)) & (app_vel_df["station3"].isin(stations_a)) ]
    app_vel_b_df = app_vel_df[ (app_vel_df["station1"].isin(stations_b)) & (app_vel_df["station2"].isin(stations_b)) & (app_vel_df["station3"].isin(stations_b)) ]

    # Get the number of valid apparent velocities for Subarray A and Subarray B
    back_azis_a_1 = app_vel_a_df["back_azi_1"].dropna().values
    back_azi_stds_a_1 = app_vel_a_df["back_azi_std_1"].dropna().values
    back_azis_a_2 = app_vel_a_df["back_azi_2"].dropna().values
    back_azi_stds_a_2 = app_vel_a_df["back_azi_std_2"].dropna().values
    back_azis_b_1 = app_vel_b_df["back_azi_1"].dropna().values
    back_azi_stds_b_1 = app_vel_b_df["back_azi_std_1"].dropna().values
    back_azis_b_2 = app_vel_b_df["back_azi_2"].dropna().values
    back_azi_stds_b_2 = app_vel_b_df["back_azi_std_2"].dropna().values

    # Keep only the vectors with the standard deviation of the back azimuth less than the threshold
    back_azi_a_1_mask = back_azi_stds_a_1 < max_back_azi_std
    back_azi_a_2_mask = back_azi_stds_a_2 < max_back_azi_std
    back_azi_b_1_mask = back_azi_stds_b_1 < max_back_azi_std
    back_azi_b_2_mask = back_azi_stds_b_2 < max_back_azi_std

    back_azis_a_1 = back_azis_a_1[back_azi_a_1_mask]
    back_azis_a_2 = back_azis_a_2[back_azi_a_2_mask]
    back_azis_b_1 = back_azis_b_1[back_azi_b_1_mask]
    back_azis_b_2 = back_azis_b_2[back_azi_b_2_mask]

    if len(back_azis_a_1) >= min_num_triad and len(back_azis_a_2) >= min_num_triad:
        print(back_azis_a_1)
        print(back_azis_a_2)
        std_1 = rad2deg(get_angles_std(deg2rad(back_azis_a_1)))
        std_2 = rad2deg(get_angles_std(deg2rad(back_azis_a_2)))
        print(std_1)
        print(std_2)
        avg_std = (std_1 + std_2) / 2
        result_dict["back_azi_std_a"] = avg_std
        print(avg_std)

    if len(back_azis_b_1) >= min_num_triad and len(back_azis_b_2) >= min_num_triad:
        std_1 = rad2deg(get_angles_std(deg2rad(back_azis_b_1)))
        std_2 = rad2deg(get_angles_std(deg2rad(back_azis_b_2)))
        result_dict["back_azi_std_b"] = (std_1 + std_2) / 2

    result_dicts.append(result_dict)

result_df = DataFrame(result_dicts)

###
# Save the results
###
# Reorder the columns
result_df = result_df[["window_id", "start_time", "end_time", "distance_a", "back_azi_std_a", "distance_b", "back_azi_std_b"]]

# Save the results
outpath = join(dirpath, f"vehicle_subarray_back_azi_stds_vs_dist_{occurrence}_min_num_triad{min_num_triad:d}_max_back_azi_std{max_back_azi_std:.0f}.csv")
result_df.to_csv(outpath, index = False, na_rep = "nan")
print(f"The results are saved to {outpath}")





