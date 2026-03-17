"""
Compute the average apparent velocity of a stationary resonance mode at all station triads
"""

### Import packages ###
from argparse import ArgumentParser
from os.path import join
from json import dumps, loads
from numpy import mean, nan, array, cov, vstack, arctan2, sqrt, pi, rad2deg
from pandas import DataFrame
from pandas import read_csv

from utils_basic import LOC_DIR as dirpath_loc, GEO_COMPONENTS as components
from utils_basic import get_geophone_triads

### Helper functions ###
def parse_cov(x):
    try:
        if isinstance(x, str):
            return array(loads(x))
        return array(x)
    except Exception:
        return array([])

def is_valid_cov(m):
    return hasattr(m, "shape") and m.shape == (2, 2)

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

    print(f"Working on Triad {station1}-{station2}-{station3}...")

    # Read the apparent velocities
    filename = f"stationary_resonance_station_triad_app_vels_{mode_name}_{station1}_{station2}_{station3}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
    filepath = join(dirpath_loc, filename)
    app_vel_df = read_csv(filepath, parse_dates=["time"])


    # Loop over each component
    for component in components:
        east_col = f"app_vel_east_{component.lower()}"
        north_col = f"app_vel_north_{component.lower()}"
        cov_col  = f"app_vel_cov_mat_{component.lower()}"

        app_vel_df[cov_col] = app_vel_df[cov_col].apply(parse_cov)

        # Get the apparent velocities
        app_vel_comp_df = app_vel_df[[east_col, north_col, cov_col]].dropna(subset=[east_col, north_col]).copy()
        num_obs = len(app_vel_comp_df)

        # Check if the number of observations is greater than the minimum number of observations
        if num_obs < min_num_obs:
            result_dict[f"avg_app_vel_{component.lower()}"] = nan
            result_dict[f"avg_back_azi_{component.lower()}"] = nan
            result_dict[f"avg_app_vel_east_{component.lower()}"] = nan
            result_dict[f"avg_app_vel_north_{component.lower()}"] = nan
            result_dict[f"app_vel_cov_mat_{component.lower()}"] = array([])
            result_dict[f"num_obs_{component.lower()}"] = num_obs
            continue

        # 4) Extract arrays
        app_vel_east  = app_vel_comp_df[east_col].to_numpy()
        app_vel_north = app_vel_comp_df[north_col].to_numpy()
        cov_mats      = app_vel_comp_df[cov_col].to_numpy()


        # Each covariance matrix is 2x2; get diagonal elements (variances)
        variances = array([m.diagonal() for m in cov_mats])

        # Inverse variances as weights (to give higher weight to lower uncertainty)
        weights_east = 1.0 / variances[:, 0]
        weights_north = 1.0 / variances[:, 1]

        # Weighted means of east and north apparent velocities
        avg_app_vel_east = (weights_east * app_vel_east).sum() / weights_east.sum()
        avg_app_vel_north = (weights_north * app_vel_north).sum() / weights_north.sum()

        # Weighted covariance of the ensemble
        cov_mat = cov(vstack((app_vel_east, app_vel_north)), aweights=(weights_east + weights_north) / 2)

        # Compute resultant average velocity and back azimuth
        avg_app_vel = sqrt(avg_app_vel_east ** 2 + avg_app_vel_north ** 2)
        avg_back_azi = rad2deg(arctan2(avg_app_vel_east, avg_app_vel_north))

        # Store the results
        result_dict[f"avg_app_vel_{component.lower()}"] = avg_app_vel
        result_dict[f"avg_back_azi_{component.lower()}"] = avg_back_azi
        result_dict[f"avg_app_vel_east_{component.lower()}"] = avg_app_vel_east
        result_dict[f"avg_app_vel_north_{component.lower()}"] = avg_app_vel_north
        result_dict[f"app_vel_cov_mat_{component.lower()}"] = cov_mat
        result_dict[f"num_obs_{component.lower()}"] = num_obs
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
