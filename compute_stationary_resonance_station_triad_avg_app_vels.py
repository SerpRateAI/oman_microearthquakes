"""
Compute the average apparent velocity of a stationary resonance mode at all station triads
"""

### Import packages ###
from argparse import ArgumentParser
from os.path import join
from json import dumps
from numpy import mean, nan, array, cov, vstack, arctan2, sqrt, pi, rad2deg
from pandas import DataFrame
from pandas import read_csv

from utils_basic import LOC_DIR as dirpath_loc, GEO_COMPONENTS as components
from utils_basic import get_geophone_triads
### Input arguments ###
parser = ArgumentParser()

parser.add_argument("--min_num_obs", type=int, help="Minimum number of observations", default=100)
parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--window_length_mt", type=float, help="Window length of the multitaper analysis", default=900.0)
parser.add_argument("--min_cohe", type=float, help="Minimum coherence of the multitaper analysis", default=0.85)

args = parser.parse_args()

mode_name = args.mode_name
window_length_mt = args.window_length_mt
min_cohe = args.min_cohe
min_num_obs = args.min_num_obs

### Load data ###
# List of station triads
triad_df = get_geophone_triads()

### Compute the average apparent velocity of all three components of each station triad ###
# Loop over each station triad
result_dicts = []
for _, row in triad_df.iterrows():
    # Get the station names
    station1, station2, station3 = row["station1"], row["station2"], row["station3"]
    result_dict = {"station1": station1, "station2": station2, "station3": station3}

    # Read the apparent velocities
    filename = f"stationary_resonance_station_triad_app_vels_{mode_name}_{station1}_{station2}_{station3}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
    filepath = join(dirpath_loc, filename)
    app_vel_df = read_csv(filepath, parse_dates=["time"])

    # Loop over each component
    for component in components:
        # Get the apparent velocities
        app_vel_east = app_vel_df[f"app_vel_east_{component.lower()}"].dropna().values
        app_vel_north = app_vel_df[f"app_vel_north_{component.lower()}"].dropna().values

        # Check if the number of observations is greater than the minimum number of observations
        if len(app_vel_east) < min_num_obs:
            result_dict[f"avg_app_vel_{component.lower()}"] = nan
            result_dict[f"avg_back_azi_{component.lower()}"] = nan
            result_dict[f"avg_app_vel_east_{component.lower()}"] = nan
            result_dict[f"avg_app_vel_north_{component.lower()}"] = nan
            result_dict[f"app_vel_cov_mat_{component.lower()}"] = array([])
            continue

        # Compute the average apparent velocity
        avg_app_vel_east = mean(app_vel_east)
        avg_app_vel_north = mean(app_vel_north)
        avg_app_vel = sqrt(avg_app_vel_east ** 2 + avg_app_vel_north ** 2)
        avg_back_azi = rad2deg(arctan2(avg_app_vel_east, avg_app_vel_north))

        # Compute the covariance matrix
        cov_mat = cov(vstack([app_vel_east, app_vel_north]))

        # Store the results
        result_dict[f"avg_app_vel_{component.lower()}"] = avg_app_vel
        result_dict[f"avg_back_azi_{component.lower()}"] = avg_back_azi
        result_dict[f"avg_app_vel_east_{component.lower()}"] = avg_app_vel_east
        result_dict[f"avg_app_vel_north_{component.lower()}"] = avg_app_vel_north
        result_dict[f"app_vel_cov_mat_{component.lower()}"] = cov_mat

    # Store the results
    result_dicts.append(result_dict)

# Convert the result_dicts to a DataFrame
result_df = DataFrame(result_dicts)

# Convert the covariance matrices to JSON strings
for component in components:
    result_df[f"app_vel_cov_mat_{component.lower()}"] = result_df[f"app_vel_cov_mat_{component.lower()}"].apply(lambda x: dumps(x.tolist()))

# Save the results
filename = f"stationary_resonance_station_triad_avg_app_vels_{mode_name}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}_min_num_obs{min_num_obs}.csv"
filepath = join(dirpath_loc, filename)
result_df.to_csv(filepath, na_rep="nan")
print(f"Saved the results to {filepath}")
