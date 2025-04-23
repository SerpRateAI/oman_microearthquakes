"""
Compute the back-azimuth standard deviations of the time-averaged apparent velocities of a stationary resonance over the two subarrays
Only the two horizontal components are considered
"""

from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, DataFrame
from numpy import rad2deg, deg2rad

from utils_basic import LOC_DIR as dirpath
from utils_basic import get_angles_std
from utils_basic import CORE_STATIONS_A as stations_a, CORE_STATIONS_B as stations_b

###
# Input arguments
###

parser = ArgumentParser()
parser.add_argument("--min_num_obs", type = int, required = True, help = "The minimum number of observations for time averaging the apparent velocities", default = 100)
parser.add_argument("--min_cohe", type = float, required = True, help = "The minimum coherence for MT analysis", default = 0.85)
parser.add_argument("--window_length", type = int, required = True, help = "The length of the window for MT analysis", default = 900)
parser.add_argument("--mode_name", type = str, required = True, help = "The name of the mode", default = "PR02549")

args = parser.parse_args()
min_num_obs = args.min_num_obs
min_cohe = args.min_cohe
window_length = args.window_length
mode_name = args.mode_name

###
# Load the data
###

filename = f"stationary_resonance_station_triad_avg_app_vels_{mode_name}_mt_win{window_length:d}s_min_cohe{min_cohe:.2f}_min_num_obs{min_num_obs:d}.csv"
filepath = join(dirpath, filename)

vel_df = read_csv(filepath)

###
# Compute the standard deviations of the time-averaged apparent velocities
###

# Get the data for the two subarrays
vel_a_df = vel_df[(vel_df["station1"].isin(stations_a)) & (vel_df["station2"].isin(stations_a)) & (vel_df["station3"].isin(stations_a))]
vel_b_df = vel_df[(vel_df["station1"].isin(stations_b)) & (vel_df["station2"].isin(stations_b)) & (vel_df["station3"].isin(stations_b))]

# Compute the back-azimuth standard deviations
back_azis_a_1 = vel_a_df["avg_back_azi_1"].dropna().values
back_azis_a_2 = vel_a_df["avg_back_azi_2"].dropna().values
back_azis_b_1 = vel_b_df["avg_back_azi_1"].dropna().values
back_azis_b_2 = vel_b_df["avg_back_azi_2"].dropna().values

# Compute the standard deviations of the back-azimuths
std_a_1 = rad2deg(get_angles_std(deg2rad(back_azis_a_1)))
std_a_2 = rad2deg(get_angles_std(deg2rad(back_azis_a_2)))
std_b_1 = rad2deg(get_angles_std(deg2rad(back_azis_b_1)))
std_b_2 = rad2deg(get_angles_std(deg2rad(back_azis_b_2)))

# Average the standard deviations of the back-azimuths
std_a = (std_a_1 + std_a_2) / 2
std_b = (std_b_1 + std_b_2) / 2

###
# Save the results
###
result_df = DataFrame({"subarray": ["A", "B"], "back_azi_std": [std_a, std_b]})
filename = f"stationary_resonance_subarray_back_azi_stds_{mode_name}_mt_win{window_length:d}s_min_cohe{min_cohe:.2f}_min_num_obs{min_num_obs:d}.csv"
result_df.to_csv(join(dirpath, filename), index = False)
print(f"Results saved to {join(dirpath, filename)}")


































