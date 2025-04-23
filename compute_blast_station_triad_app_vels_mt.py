"""
Compute the apparent velocities for a quarry blast on the station triads using the phase differences estimated using the multitaper method
"""


###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from json import loads
from numpy import zeros, mean, sqrt, rad2deg, nan, pi
from pandas import DataFrame
from pandas import read_csv, concat

from utils_basic import MT_DIR as dirpath_mt, GEO_COMPONENTS as components, LOC_DIR as dirpath_loc
from utils_basic import get_geophone_coords, get_angles_mean, get_angles_diff
from utils_loc import get_triad_app_vel, get_dist_mat_inv

###
# Input parameters
###

# Command line arguments
parser = ArgumentParser()
parser.add_argument("--blast_id", type = str, help = "The ID of the quarry blast")
parser.add_argument("--freq_target", type = float, default = 25.0, help = "The target frequency in Hz")
parser.add_argument("--min_cohe", type = float, default = 0.85, help = "The minimum coherence")
parser.add_argument("--window_length", type = float, default = 8.0, help = "The window length in seconds")
args = parser.parse_args()

blast_id = args.blast_id
freq_target = args.freq_target
min_cohe = args.min_cohe
window_length = args.window_length

###
# Read the input files
###

# Load the station information
print("Loading the station information...")
sta_df = get_geophone_coords()

# Load the station pair information
print("Loading the station pair information...")
inpath = join(dirpath_mt, "delaunay_station_pairs.csv")
pair_df = read_csv(inpath)

# Load the station triad information
print("Loading the station triad information...")
inpath = join(dirpath_mt, "delaunay_station_triads.csv")
triad_df = read_csv(inpath)

# Load the phase differences between each station pair
print("Loading the phase differences between each station pair...")
phase_diff_dfs = []
for _, row in pair_df.iterrows():
    station1 = row["station1"]
    station2 = row["station2"]

    # Load the phase differences between the two stations
    filename = f"blast_mt_inter_geo_sta_phase_diffs_{blast_id}_{station1}_{station2}_window{window_length:.0f}s.csv"

    inpath = join(dirpath_mt, filename)

    try:
        phase_diff_df = read_csv(inpath)
    except FileNotFoundError:
        print(f"The file {inpath} does not exist. Skipping the station pair {station1} and {station2}...")
        continue

    phase_diff_df["station1"] = station1
    phase_diff_df["station2"] = station2

    # Find the index of the frequency closest to the target frequency
    row_ind = (phase_diff_df["frequency"] - freq_target).abs().idxmin()

    phase_diff_freq_df = phase_diff_df.iloc[[row_ind]].copy()

    for component in components:
        # Convert the JSON strings to to arrays
        phase_diff_freq_df[f"phase_diff_jk_{component.lower()}"] = phase_diff_freq_df[f"phase_diff_jk_{component.lower()}"].apply(loads)

    phase_diff_dfs.append(phase_diff_freq_df)

phase_diff_df = concat(phase_diff_dfs)
phase_diff_df.reset_index(drop = True, inplace = True)

# print(phase_diff_df.columns)

# Compute the apparent velocities for each station triad
print("Computing the apparent velocities for each station triad...")
result_dicts = []
for _, row in triad_df.iterrows():
    station1 = row["station1"]
    station2 = row["station2"]
    station3 = row["station3"]

    # Check if the phase differences between the station pair exist
    if not (phase_diff_df["station1"] == station1).any() or not (phase_diff_df["station2"] == station2).any() or not (phase_diff_df["station1"] == station2).any() or not (phase_diff_df["station2"] == station3).any():
        print(f"The phase differences between the station pair {station1} and {station2} or {station2} and {station3} do not exist. Skipping the triad {station1}-{station2}-{station3}...")
        continue

    print(f"Computing the apparent velocities for the triad {station1}-{station2}-{station3}...")

    east_triad = row["east"]
    north_triad = row["north"]

    result_dict = {"station1": station1, "station2": station2, "station3": station3, "east": east_triad, "north": north_triad}

    # Get the coordinates of the stations
    east1, north1 = sta_df.loc[station1, ["east", "north"]]
    east2, north2 = sta_df.loc[station2, ["east", "north"]]
    east3, north3 = sta_df.loc[station3, ["east", "north"]]

    # Get the inverse of the distance matrix
    dist_mat_inv = get_dist_mat_inv(east1, north1, east2, north2, east3, north3)

    for component in components:
        # Determine if the coherence is above the threshold
        cohe_12 = phase_diff_df.loc[(phase_diff_df["station1"] == station1) & (phase_diff_df["station2"] == station2), f"cohe_{component.lower()}"].values[0]
        cohe_23 = phase_diff_df.loc[(phase_diff_df["station1"] == station2) & (phase_diff_df["station2"] == station3), f"cohe_{component.lower()}"].values[0]

        if cohe_12 < min_cohe or cohe_23 < min_cohe:
            result_dict[f"vel_app_{component.lower()}"] = nan
            result_dict[f"back_azi_{component.lower()}"] = nan
            result_dict[f"vel_app_east_{component.lower()}"] = nan
            result_dict[f"vel_app_north_{component.lower()}"] = nan

            result_dict[f"vel_app_std_{component.lower()}"] = nan
            result_dict[f"back_azi_std_{component.lower()}"] = nan
            result_dict[f"vel_app_east_var_{component.lower()}"] = nan
            result_dict[f"vel_app_north_var_{component.lower()}"] = nan
            result_dict[f"vel_app_cov_{component.lower()}"] = nan

            continue

        # Compute the apparent-velocity estimators
        phase_diff_12 = phase_diff_df.loc[(phase_diff_df["station1"] == station1) & (phase_diff_df["station2"] == station2), f"phase_diff_{component.lower()}"]
        phase_diff_23 = phase_diff_df.loc[(phase_diff_df["station1"] == station2) & (phase_diff_df["station2"] == station3), f"phase_diff_{component.lower()}"]

        time_diff_12 = phase_diff_12 / freq_target / 2 / pi
        time_diff_23 = phase_diff_23 / freq_target / 2 / pi

        vel_app, back_azi, vel_app_east, vel_app_north = get_triad_app_vel(time_diff_12, time_diff_23, dist_mat_inv)

        result_dict[f"vel_app_{component.lower()}"] = vel_app
        result_dict[f"back_azi_{component.lower()}"] = rad2deg(back_azi) # Convert to degrees
        result_dict[f"vel_app_east_{component.lower()}"] = vel_app_east
        result_dict[f"vel_app_north_{component.lower()}"] = vel_app_north

        # Compute the uncertainties of the apparent-velocity estimators using the jackknife method
        phase_diff_jk_12 = phase_diff_df.loc[(phase_diff_df["station1"] == station1) & (phase_diff_df["station2"] == station2), f"phase_diff_jk_{component.lower()}"].values[0]
        phase_diff_jk_23 = phase_diff_df.loc[(phase_diff_df["station1"] == station2) & (phase_diff_df["station2"] == station3), f"phase_diff_jk_{component.lower()}"].values[0]

        # print(phase_diff_jk_12)

        num_jk = len(phase_diff_jk_12)
        vel_app_jk = zeros(num_jk)
        back_azi_jk = zeros(num_jk)
        vel_app_east_jk = zeros(num_jk)
        vel_app_north_jk = zeros(num_jk)

        for i in range(num_jk):
            time_diff_12 = phase_diff_jk_12[i] / freq_target / 2 / pi
            time_diff_23 = phase_diff_jk_23[i] / freq_target / 2 / pi
            
            vel_app_jk[i], back_azi_jk[i], vel_app_east_jk[i], vel_app_north_jk[i] = get_triad_app_vel(time_diff_12, time_diff_23, dist_mat_inv)
        
        vel_app_jk_mean = mean(vel_app_jk)
        back_azi_jk_mean = get_angles_mean(back_azi_jk)

        vel_app_east_jk_mean = mean(vel_app_east_jk)
        vel_app_north_jk_mean = mean(vel_app_north_jk)

        vel_app_var = (num_jk - 1) / num_jk * sum((vel_app_jk - vel_app_jk_mean) ** 2)
        back_azi_var = (num_jk - 1) / num_jk * sum(get_angles_diff(back_azi_jk_mean, back_azi_jk) ** 2)

        vel_app_std = sqrt(vel_app_var)
        back_azi_std = sqrt(back_azi_var)

        vel_app_east_var = (num_jk - 1) / num_jk * sum((vel_app_east_jk - vel_app_east_jk_mean) ** 2)
        vel_app_north_var = (num_jk - 1) / num_jk * sum((vel_app_north_jk - vel_app_north_jk_mean) ** 2)
        vel_app_cov = (num_jk - 1) / num_jk * sum((vel_app_east_jk - vel_app_east_jk_mean) * (vel_app_north_jk - vel_app_north_jk_mean))
        
        result_dict[f"vel_app_std_{component.lower()}"] = vel_app_std
        result_dict[f"back_azi_std_{component.lower()}"] = rad2deg(back_azi_std) # Convert to degrees
        result_dict[f"vel_app_east_var_{component.lower()}"] = vel_app_east_var
        result_dict[f"vel_app_north_var_{component.lower()}"] = vel_app_north_var
        result_dict[f"vel_app_cov_{component.lower()}"] = vel_app_cov

    result_dicts.append(result_dict)

# Convert the result_dicts to a dataframe
result_df = DataFrame(result_dicts)

# Save the result dataframe
outpath = join(dirpath_loc, f"blast_station_triad_app_vels_mt_{blast_id}_{freq_target:.0f}hz_window{window_length:.0f}s_min_cohe{min_cohe:.2f}.csv")
result_df.to_csv(outpath, na_rep = "nan", index = False)
print(f"The result dataframe has been saved to {outpath}")