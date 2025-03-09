"""
Compute the average apparent velocity of all station triads
"""

### Import packages ###
from argparse import ArgumentParser
from os.path import join
from numpy import angle, exp, mean, nan, pi, std
from pandas import DataFrame
from pandas import read_csv


from utils_basic import MT_DIR as dirname_mt, GEO_COMPONENTS as components
from utils_basic import get_angle_std

### Input arguments ###
parser = ArgumentParser()

parser.add_argument("--min_num_obs", type=int, help="Minimum number of observations", default=10)
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
filename = "delaunay_station_triads.csv"
filepath = join(dirname_mt, filename)
triad_df = read_csv(filepath)

### Compute the average apparent velocity of all three components of each station triad ###
# Loop over each station triad
result_dicts = []
for _, row in triad_df.iterrows():
    # Get the station names
    station1, station2, station3 = row["station1"], row["station2"], row["station3"]
    result_dict = {"station1": station1, "station2": station2, "station3": station3}

    # Read the apparent velocities
    filename = f"station_triad_app_vels_{mode_name}_{station1}_{station2}_{station3}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
    filepath = join(dirname_mt, filename)
    app_vel_df = read_csv(filepath, parse_dates=["time"])

    # Loop over each component
    for component in components:
        # Get the apparent velocities
        app_vels = app_vel_df[f"vel_app_{component.lower()}"].dropna().values
        back_azis = app_vel_df[f"back_azi_{component.lower()}"].dropna().values
        
        # Check if the number of observations is greater than the minimum number of observations
        if len(app_vels) < min_num_obs:
            result_dict[f"avg_vel_app_{component.lower()}"] = nan
            result_dict[f"avg_back_azi_{component.lower()}"] = nan
            result_dict[f"std_back_azi_{component.lower()}"] = nan
            continue

        # Compute the average apparent velocity
        avg_app_vel_vec = mean(app_vels * exp(1j * back_azis * pi / 180))
        avg_app_vel = abs(avg_app_vel_vec)
        avg_back_azi = angle(avg_app_vel_vec) * 180 / pi

        # Compute the standard deviation of the back azimuths
        back_azi_std = get_angle_std(back_azis * pi / 180) * 180 / pi

        # Store the results
        result_dict[f"avg_vel_app_{component.lower()}"] = avg_app_vel
        result_dict[f"avg_back_azi_{component.lower()}"] = avg_back_azi
        result_dict[f"std_back_azi_{component.lower()}"] = back_azi_std

    # Store the results
    result_dicts.append(result_dict)

# Convert the result_dicts to a DataFrame
result_df = DataFrame(result_dicts)

# Save the results
filename = f"station_triad_avg_app_vels_{mode_name}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}_min_num_obs{min_num_obs}.csv"
filepath = join(dirname_mt, filename)
result_df.to_csv(filepath, na_rep="nan")
print(f"Saved the results to {filepath}")
