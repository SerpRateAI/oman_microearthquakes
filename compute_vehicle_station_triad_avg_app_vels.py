"""
Compute the average apparent velocities of the vehicle signal for each station triad
"""

###
# Import the necessary libraries
###
from argparse import ArgumentParser
from os.path import join
from json import dumps
from numpy import mean, nan, array, cov, vstack, arctan2, sqrt, pi, rad2deg
from pandas import DataFrame
from pandas import read_csv

from utils_basic import MT_DIR as dirname_mt, LOC_DIR as dirname_loc,  GEO_COMPONENTS as components

###
# Define the input arguments
###
parser = ArgumentParser()
parser.add_argument("--min_num_obs", type=int, help="Minimum number of observations", default=3)
parser.add_argument("--max_back_azi_std", type=float, help="Maximum standard deviation of back azimuth", default=10.0)


parser.add_argument("--occurrence", type=str, help="The occurrence of the vehicle", default="approaching")

args = parser.parse_args()

min_num_obs = args.min_num_obs
occurrence = args.occurrence
max_back_azi_std = args.max_back_azi_std
###
# Load the data
###
# List of station triads
filename = "delaunay_station_triads.csv"
filepath = join(dirname_mt, filename)
triad_df = read_csv(filepath)

###
# Compute the average apparent velocities of the vehicle signal for each station triad
###
# Loop over each station triad
result_dicts = []
for _, row in triad_df.iterrows():
    # Get the station names
    station1, station2, station3 = row["station1"], row["station2"], row["station3"]
    result_dict = {"station1": station1, "station2": station2, "station3": station3}
    print(f"Processing {station1}-{station2}-{station3}...")

    # Read the apparent velocities
    filename = f"vehicle_station_triad_app_vel_vs_time_{occurrence}_{station1}_{station2}_{station3}.csv"
    filepath = join(dirname_loc, filename)
    app_vel_df = read_csv(filepath, parse_dates=["start_time", "end_time"])

    # Loop over each component
    for component in components:
        # Get the apparent velocities
        app_vels_east = app_vel_df[f"app_vel_east_{component.lower()}"].dropna().values
        app_vels_north = app_vel_df[f"app_vel_north_{component.lower()}"].dropna().values
        back_azi_std = app_vel_df[f"back_azi_std_{component.lower()}"].dropna().values

        # Keep only the apparent velocities with a standard deviation of back azimuth less than the maximum standard deviation of back azimuth
        app_vels_east = app_vels_east[back_azi_std < max_back_azi_std]
        app_vels_north = app_vels_north[back_azi_std < max_back_azi_std]

        # Check if the number of observations is greater than the minimum number of observations
        num_obs = len(app_vels_east)
        print(f"Number of observations for {component} in {station1}-{station2}-{station3}: {num_obs}")
        if num_obs < min_num_obs:
            print(f"Not enough observations for {component} in {station1}-{station2}-{station3}! Skipping...")
            result_dict[f"avg_app_vel_{component.lower()}"] = nan
            result_dict[f"avg_back_azi_{component.lower()}"] = nan
            result_dict[f"avg_app_vel_east_{component.lower()}"] = nan
            result_dict[f"avg_app_vel_north_{component.lower()}"] = nan
            result_dict[f"vel_app_cov_mat_{component.lower()}"] = array([])
            continue

        # Compute the average apparent velocity
        avg_app_vel_east = mean(app_vels_east)
        avg_app_vel_north = mean(app_vels_north)
        avg_app_vel = sqrt(avg_app_vel_east ** 2 + avg_app_vel_north ** 2)
        avg_back_azi = rad2deg(arctan2(avg_app_vel_east, avg_app_vel_north))

        # Compute the covariance matrix
        cov_mat = cov(vstack([app_vels_east, app_vels_north]))

        # Store the results
        result_dict[f"avg_app_vel_{component.lower()}"] = avg_app_vel
        result_dict[f"avg_back_azi_{component.lower()}"] = avg_back_azi
        result_dict[f"avg_app_vel_east_{component.lower()}"] = avg_app_vel_east
        result_dict[f"avg_app_vel_north_{component.lower()}"] = avg_app_vel_north
        result_dict[f"vel_app_cov_mat_{component.lower()}"] = cov_mat

    # Store the results
    result_dicts.append(result_dict)

# Convert the result_dicts to a DataFrame
result_df = DataFrame(result_dicts)

# Convert the covariance matrices to JSON strings
for component in components:
    result_df[f"vel_app_cov_mat_{component.lower()}"] = result_df[f"vel_app_cov_mat_{component.lower()}"].apply(lambda x: dumps(x.tolist()))

###
# Save the results
###
filename = f"vehicle_station_triad_avg_app_vels_{occurrence}_min_num_obs{min_num_obs:d}_max_back_azi_std{max_back_azi_std:.0f}.csv"
filepath = join(dirname_loc, filename)
result_df.to_csv(filepath, index=False, na_rep="nan")
print(f"Results saved to {filepath}")



