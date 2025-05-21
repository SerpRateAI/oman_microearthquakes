"""
Compute the vehicle-signal apparent velocity vs time for a specific station triad
"""

###
# Import modules
###
from os.path import join
from argparse import ArgumentParser
from numpy import array, nan, intersect1d, sqrt, mean, pi, rad2deg, zeros
from numpy.linalg import inv
from pandas import read_csv
from pandas import DataFrame
from json import loads, dumps

from utils_basic import MT_DIR as dirpath_mt, LOC_DIR as dirpath_loc, GEO_COMPONENTS as components
from utils_basic import get_geophone_coords
from utils_mt import get_avg_phase_diff, get_angles_mean, get_angles_diff
from utils_loc import get_triad_app_vel, get_dist_mat_inv

###
# Define functions
###


###
# Parse command-line arguments
###
parser = ArgumentParser()
parser.add_argument("--occurrence", type = str, default = "approaching", help = "Occurrence")
parser.add_argument("--station1", type = str, help = "Station 1 of the station triad")
parser.add_argument("--station2", type = str, help = "Station 2 of the station triad")
parser.add_argument("--station3", type = str, help = "Station 3 of the station triad")
parser.add_argument("--nw", type = float, default = 3.0, help = "Time-bandwidth product")
parser.add_argument("--min_cohe", type = float, default = 0.85, help = "Minimum coherence")
parser.add_argument("--freq_target", type = float, default = 25.5, help = "Target frequency")

args = parser.parse_args()
occurrence = args.occurrence
station1 = args.station1
station2 = args.station2
station3 = args.station3
nw = args.nw
min_cohe = args.min_cohe
freq_target = args.freq_target

###
# Read the input files
###
print(f"Computing the vehicle-signal apparent velocity vs time for the station triad {station1}-{station2}-{station3}...")

# Load the station information
print("Loading the station information...")
station_df = get_geophone_coords()

# Time window list
print("Loading the time window list...")
filename = f"vehicle_time_windows_{occurrence}.csv"
filepath = join(dirpath_loc, filename)
time_window_df = read_csv(filepath)

# Compute the inverse distance matrix
print("Computing the inverse distance matrix...")
east1 = station_df.loc[station1, "east"]
north1 = station_df.loc[station1, "north"]
east2 = station_df.loc[station2, "east"]
north2 = station_df.loc[station2, "north"]
east3 = station_df.loc[station3, "east"]
north3 = station_df.loc[station3, "north"]
dist_mat_inv = get_dist_mat_inv(east1, north1, east2, north2, east3, north3)

# Loop over the time windows
print("Looping over the time windows...")
result_dicts = []
for _, row in time_window_df.iterrows():
    window_id = row["window_id"]
    start_time = row["start_time"]
    end_time = row["end_time"]

    print(f"Processing window {window_id}...")

    # Read the cross-spectral results of the two station pairs
    filename = f"vehicle_mt_inter_geo_sta_phase_diffs_{occurrence}_window{window_id}_{station1}_{station2}.csv"
    filepath = join(dirpath_mt, filename)
    cspec_12_df = read_csv(filepath)

    filename = f"vehicle_mt_inter_geo_sta_phase_diffs_{occurrence}_window{window_id}_{station2}_{station3}.csv"
    filepath = join(dirpath_mt, filename)
    cspec_23_df = read_csv(filepath)

    # Process each component
    result_dict = {
        "window_id": window_id,
        "start_time": start_time,
        "end_time": end_time,
    }

    for component in components:
        print(f"Processing component {component}...")
        # Convert the JSON strings to to arrays
        cspec_12_df[f"phase_diff_jk_{component.lower()}"] = cspec_12_df[f"phase_diff_jk_{component.lower()}"].apply(loads)
        cspec_23_df[f"phase_diff_jk_{component.lower()}"] = cspec_23_df[f"phase_diff_jk_{component.lower()}"].apply(loads)
    
        # Get the data
        freqax = cspec_12_df["frequency"].values
        phase_diffs_12 = cspec_12_df[f"phase_diff_{component.lower()}"].values
        phase_diff_uncers_12 = cspec_12_df[f"phase_diff_uncer_{component.lower()}"].values
        phase_diff_jk_12 = cspec_12_df[f"phase_diff_jk_{component.lower()}"].values
        cohes_12 = cspec_12_df[f"cohe_{component.lower()}"].values

        phase_diffs_23 = cspec_23_df[f"phase_diff_{component.lower()}"].values
        phase_diff_uncers_23 = cspec_23_df[f"phase_diff_uncer_{component.lower()}"].values
        phase_diff_jk_23 = cspec_23_df[f"phase_diff_jk_{component.lower()}"].values
        cohes_23 = cspec_23_df[f"cohe_{component.lower()}"].values

        # Get the average phase difference in the bandwidth of the target frequency
        freq_interval = freqax[1] - freqax[0]
        min_freq = freq_target - nw * freq_interval
        max_freq = freq_target + nw * freq_interval

        print(f"Computing the average phase difference for {station1}-{station2}...")
        avg_phase_diff_12, avg_phase_diff_uncer_12, freq_inds_12, _ = get_avg_phase_diff((min_freq, max_freq), freqax, phase_diffs_12, phase_diff_uncers_12, cohes_12, 
                                                                                    min_cohe = min_cohe, nw = nw, return_samples = True)
        
        print(f"Computing the average phase difference for {station2}-{station3}...")
        avg_phase_diff_23, avg_phase_diff_uncer_23, freq_inds_23, _ = get_avg_phase_diff((min_freq, max_freq), freqax, phase_diffs_23, phase_diff_uncers_23, cohes_23, 
                                                                                    min_cohe = min_cohe, nw = nw, return_samples = True)
        
        if avg_phase_diff_12 is None or avg_phase_diff_23 is None:
            print(f"No phase difference meets the criteria for {component}! Skipping...")

            result_dict[f"app_vel_{component.lower()}"] = nan
            result_dict[f"app_vel_std_{component.lower()}"] = nan
            result_dict[f"back_azi_{component.lower()}"] = nan
            result_dict[f"back_azi_std_{component.lower()}"] = nan
            result_dict[f"app_vel_east_{component.lower()}"] = nan
            result_dict[f"app_vel_north_{component.lower()}"] = nan
            result_dict[f"app_vel_cov_mat_{component.lower()}"] = array([])
            continue

        # Get the common frequency indices
        freq_inds = intersect1d(freq_inds_12, freq_inds_23)
        num_freq_common = len(freq_inds)
        if num_freq_common == 0:
            print(f"No common frequency indices found for {component}! Skipping...")

            result_dict[f"app_vel_{component.lower()}"] = nan
            result_dict[f"app_vel_std_{component.lower()}"] = nan
            result_dict[f"back_azi_{component.lower()}"] = nan
            result_dict[f"back_azi_std_{component.lower()}"] = nan
            result_dict[f"app_vel_east_{component.lower()}"] = nan
            result_dict[f"app_vel_north_{component.lower()}"] = nan
            result_dict[f"app_vel_cov_mat_{component.lower()}"] = array([])
            continue

        print(f"Common frequencies: {freqax[freq_inds]}")

        print(f"Computing the apparent velocity and back azimuth for {component}...")

        # Compute the apparent velocity and back azimuth
        time_diff_12 = avg_phase_diff_12 / 2 / pi / freq_target
        time_diff_23 = avg_phase_diff_23 / 2 / pi / freq_target
        app_vel, back_azi, app_vel_east, app_vel_north = get_triad_app_vel(time_diff_12, time_diff_23, dist_mat_inv)
        
        result_dict[f"app_vel_{component.lower()}"] = app_vel
        result_dict[f"back_azi_{component.lower()}"] = back_azi

        # Estimate the uncertainty of the apparent velocity and back azimuth using the jackknife method
        print(f"Estimating the uncertainty of the apparent velocity for {component}...")

        phase_diff_jk_12 = phase_diff_jk_12[freq_inds]
        phase_diff_jk_23 = phase_diff_jk_23[freq_inds]

        result_dict[f"app_vel_east_{component.lower()}"] = app_vel_east
        result_dict[f"app_vel_north_{component.lower()}"] = app_vel_north

        # Compute the uncertainty of the apparent velocity
        vel_app_cov_mats = []
        vel_app_vars = []
        back_azi_vars = []

        for i in range(num_freq_common):
            phase_diffs_12 = phase_diff_jk_12[i]
            phase_diffs_23 = phase_diff_jk_23[i]

            num_jk = len(phase_diffs_12)
            freq = freqax[freq_inds[i]]

            vel_app_jk = zeros(num_jk)
            vel_app_east_jk = zeros(num_jk)
            vel_app_north_jk = zeros(num_jk)
            back_azi_jk = zeros(num_jk)
            for j in range(num_jk):
                phase_diff_12 = phase_diffs_12[j]
                phase_diff_23 = phase_diffs_23[j]

                time_diff_12 = phase_diff_12 / 2 / pi / freq
                time_diff_23 = phase_diff_23 / 2 / pi / freq
                vel_app, back_azi, vel_app_east, vel_app_north = get_triad_app_vel(time_diff_12, time_diff_23, dist_mat_inv)

                vel_app_jk[j] = vel_app
                vel_app_east_jk[j] = vel_app_east
                vel_app_north_jk[j] = vel_app_north
                back_azi_jk[j] = back_azi

            vel_app_var = (num_jk - 1) / num_jk * sum((vel_app_jk - mean(vel_app_jk)) ** 2)
            vel_app_east_var = (num_jk - 1) / num_jk * sum((vel_app_east_jk - mean(vel_app_east_jk)) ** 2)
            vel_app_north_var = (num_jk - 1) / num_jk * sum((vel_app_north_jk - mean(vel_app_north_jk)) ** 2)
            vel_app_cov = (num_jk - 1) / num_jk * sum((vel_app_east_jk - mean(vel_app_east_jk)) * (vel_app_north_jk - mean(vel_app_north_jk)))

            back_azi_mean = get_angles_mean(back_azi_jk)
            back_azi_var = (num_jk - 1) / num_jk * sum((get_angles_diff(back_azi_jk, back_azi_mean) ** 2))

            vel_app_vars.append(vel_app_var)
            vel_app_cov_mats.append(array([[vel_app_east_var, vel_app_cov], [vel_app_cov, vel_app_north_var]]))
            back_azi_vars.append(back_azi_var)

        # Combine the variances and covariances of the independent frequencies
        vel_app_vars = array(vel_app_vars)
        back_azi_vars = array(back_azi_vars)
        
        vel_app_var = 1 / sum(1 / vel_app_vars)
        back_azi_var = 1 / sum(1 / back_azi_vars)

        sum_of_inv = sum( inv(mat) for mat in vel_app_cov_mats )
        vel_app_cov_mat = inv(sum_of_inv)

        result_dict[f"app_vel_cov_mat_{component.lower()}"] = vel_app_cov_mat
        result_dict[f"app_vel_std_{component.lower()}"] = sqrt(vel_app_var)
        result_dict[f"back_azi_std_{component.lower()}"] = sqrt(back_azi_var)

        print("Done!")
    
    result_dicts.append(result_dict)
    print("")

result_df = DataFrame(result_dicts)

###
# Save the result
###
for component in components:
    # Convert the numpy array to a JSON string
    result_df[f"app_vel_cov_mat_{component.lower()}"] = result_df[f"app_vel_cov_mat_{component.lower()}"].apply(lambda x: dumps(x.tolist()))

    # Conver the back azimuth to degrees
    result_df[f"back_azi_{component.lower()}"] = result_df[f"back_azi_{component.lower()}"].apply(rad2deg)
    result_df[f"back_azi_std_{component.lower()}"] = result_df[f"back_azi_std_{component.lower()}"].apply(rad2deg)

# Reorder the columns
columns = ["window_id", "start_time", "end_time"]
for component in components:
    columns.extend([f"app_vel_{component.lower()}", f"app_vel_std_{component.lower()}", f"back_azi_{component.lower()}", f"back_azi_std_{component.lower()}",
                   f"app_vel_east_{component.lower()}", f"app_vel_north_{component.lower()}", f"app_vel_cov_mat_{component.lower()}"])
    
result_df = result_df[columns]

# Save the result
filename = f"vehicle_station_triad_app_vel_vs_time_{occurrence}_{station1}_{station2}_{station3}.csv"
filepath = join(dirpath_loc, filename)
result_df.to_csv(filepath, na_rep="nan", index=False)
print(f"Results saved to {filepath}")


















