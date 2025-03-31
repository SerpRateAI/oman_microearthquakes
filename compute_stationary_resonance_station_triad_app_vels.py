"""
Compute the apparent velocities of a stationary resonance on a delaunay station triad
"""

### Imports ###
from os.path import join
from argparse import ArgumentParser
from numpy import arctan2, array, nan, isnan, intersect1d, pi, setdiff1d, var, vstack, zeros, sum, sqrt, mean, rad2deg
from numpy.linalg import inv
from json import dumps, loads
from pandas import DataFrame, Timedelta
from pandas import merge, read_csv
from scipy.signal.windows import dpss
from matplotlib.pyplot import subplots
from time import time

from utils_basic import SPECTROGRAM_DIR as dirname_spec, MT_DIR as dirname_mt, GEO_COMPONENTS as components
from utils_basic import get_geophone_coords, get_angle_diff, get_angle_mean
from utils_mt import get_indep_freq_inds, get_triad_app_vel, get_dist_mat_inv

### Inputs ###

# Command-line arguments
parser = ArgumentParser(description="Compute the apparent velocities of a stationary resonance on a delaunay station triad")

parser.add_argument("--station1", type=str, help="Station 1")
parser.add_argument("--station2", type=str, help="Station 2")
parser.add_argument("--station3", type=str, help="Station 3")

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--window_length_mt", type=float, help="Window length for multitaper analysis", default=900.0)
parser.add_argument("--min_cohe", type=float, help="Minimum coherence", default=0.85)
parser.add_argument("--nw", type=int, help="Number of windows", default=3)

# Parse the command line inputs
args = parser.parse_args()

station1 = args.station1
station2 = args.station2
station3 = args.station3

mode_name = args.mode_name
window_length_mt = args.window_length_mt
min_cohe = args.min_cohe
nw = args.nw

### Read the station locations ###
print("Reading the station locations...")
coord_df = get_geophone_coords()

east1, north1 = coord_df.loc[station1, ["east", "north"]]
east2, north2 = coord_df.loc[station2, ["east", "north"]]
east3, north3 = coord_df.loc[station3, ["east", "north"]]

### Build the inverse matrix ###
print("Building the inverse matrix...")
dist_mat_inv = get_dist_mat_inv(east1, north1, east2, north2, east3, north3)

### Read the phase differences ###
print("Reading the phase differences...")

print(f"Reading the phase differences for {station1}-{station2}...")
filename = f"multitaper_inter_sta_phase_diffs_PR02549_{station1}_{station2}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
phase_diff_12_df = read_csv(join(dirname_mt, filename))

for col in phase_diff_12_df.columns:
    if col.startswith("freq_inds") or col.startswith("phase_diff_jks_"):
        phase_diff_12_df[col] = phase_diff_12_df[col].apply(lambda x: array(loads(x)))

print(f"Reading the phase differences for {station2}-{station3}...")
filename = f"multitaper_inter_sta_phase_diffs_PR02549_{station2}_{station3}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
phase_diff_23_df = read_csv(join(dirname_mt, filename))

for col in phase_diff_23_df.columns:
    if col.startswith("freq_inds") or col.startswith("phase_diff_jks_"):
        phase_diff_23_df[col] = phase_diff_23_df[col].apply(lambda x: array(loads(x)))

phase_diff_df = merge(phase_diff_12_df, phase_diff_23_df, on=["time"], suffixes=["_12", "_23"], how="inner")

### Compute the slowness ###
print("Computing the slownesses for each time window...")

result_dicts = []
for _, row in phase_diff_df.iterrows():
    centertime = row["time"]
    freq_reson = row["frequency_12"]

    print("########################################################")
    print(f"Computing the slowness vector for {centertime}...")
    print("########################################################")

    result_dict = {
        "time": centertime,
        "freq_reson": freq_reson,
    }

    for component in components:
        print(f"Computing the slowness vector for Component {component}...")
        # Determine if the two station pairs share frequency indices
        freq_inds_12 = row[f"freq_inds_{component.lower()}_12"]
        freq_inds_23 = row[f"freq_inds_{component.lower()}_23"]

        if len(intersect1d(freq_inds_12, freq_inds_23)) == 0:
            result_dict[f"vel_app_{component.lower()}"] = nan
            result_dict[f"vel_app_uncer_{component.lower()}"] = nan
            result_dict[f"back_azi_{component.lower()}"] = nan
            result_dict[f"back_azi_uncer_{component.lower()}"] = nan
            result_dict[f"vel_app_east_{component.lower()}"] = nan
            result_dict[f"vel_app_north_{component.lower()}"] = nan
            result_dict[f"vel_app_cov_mat_{component.lower()}"] = array([])
        
            print(f"No common frequency indices for Component {component}...Skipping.")
            continue

        # Compute the slowness vector estimate
        print(f"Computing the slowness vector estimate...")
        vel_app, back_azi, vel_app_east, vel_app_north = get_triad_app_vel(row[f"phase_diff_{component.lower()}_12"], row[f"phase_diff_{component.lower()}_23"], dist_mat_inv, freq_reson)

        result_dict[f"vel_app_{component.lower()}"] = vel_app
        result_dict[f"back_azi_{component.lower()}"] = back_azi
        result_dict[f"vel_app_east_{component.lower()}"] = vel_app_east
        result_dict[f"vel_app_north_{component.lower()}"] = vel_app_north

        # Estimate the uncertainty using the jackknife method
        print(f"Estimating the uncertainty using the jackknife method...")
        phase_diff_jks_12 = row[f"phase_diff_jks_{component.lower()}_12"]
        phase_diff_jks_23 = row[f"phase_diff_jks_{component.lower()}_23"]

        # Find the indices of the common frequency indices
        freq_inds_common, inds_12, inds_23 = intersect1d(freq_inds_12, freq_inds_23, return_indices=True)
        phase_diff_jks_12 = phase_diff_jks_12[:, inds_12]
        phase_diff_jks_23 = phase_diff_jks_23[:, inds_23]
        num_common = len(freq_inds_common)
        print(f"Number of common frequency indices : {num_common}")

        # Find the independent phase differences
        freq_inds_indep = get_indep_freq_inds(freq_inds_common, nw)
        _, _, inds_common_indep = intersect1d(freq_inds_common, freq_inds_indep, return_indices=True)
        
        phase_diff_jks_12 = phase_diff_jks_12[:, inds_common_indep]
        phase_diff_jks_23 = phase_diff_jks_23[:, inds_common_indep]
        num_indep = len(freq_inds_indep)
        print(f"Number of independent frequency indices: {num_indep}")

        # Use the jackknife estimate the covariance matrix for each pair of independent phase differences
        vel_app_cov_mats = []
        vel_app_vars = []
        back_azi_vars = []
        
        num_jk = phase_diff_jks_12.shape[0]
        for i in range(num_indep):
            vel_app_jk = zeros(num_jk)
            back_azi_jk = zeros(num_jk)
            vel_app_jk_east = zeros(num_jk)
            vel_app_jk_north = zeros(num_jk)
            for j in range(num_jk):
                phase_diff_12 = phase_diff_jks_12[j, i]
                phase_diff_23 = phase_diff_jks_23[j, i]

                vel_app, back_azi, vel_app_east, vel_app_north = get_triad_app_vel(phase_diff_12, phase_diff_23, dist_mat_inv, freq_reson)

                vel_app_jk[j] = vel_app
                back_azi_jk[j] = back_azi
                vel_app_jk_east[j] = vel_app_east
                vel_app_jk_north[j] = vel_app_north
                
            vel_app_var = (num_jk - 1) / num_jk * sum((vel_app_jk - mean(vel_app_jk)) ** 2)
            vel_app_east_var = (num_jk - 1) / num_jk * sum((vel_app_jk_east - mean(vel_app_jk_east)) ** 2)
            vel_app_north_var = (num_jk - 1) / num_jk * sum((vel_app_jk_north - mean(vel_app_jk_north)) ** 2)
            vel_app_cov = (num_jk - 1) / num_jk * sum((vel_app_jk_east - mean(vel_app_jk_east)) * (vel_app_jk_north - mean(vel_app_jk_north)))

            back_azi_mean = get_angle_mean(back_azi_jk)
            back_azi_var = (num_jk - 1) / num_jk * sum((get_angle_diff(back_azi_jk, back_azi_mean) ** 2))

            vel_app_cov_mats.append(array([[vel_app_east_var, vel_app_cov], [vel_app_cov, vel_app_north_var]]))
            vel_app_vars.append(vel_app_var)
            back_azi_vars.append(back_azi_var)

        # Combine the variances and covariances of the independent frequencies
        vel_app_vars = array(vel_app_vars)
        back_azi_vars = array(back_azi_vars)
        
        vel_app_var = 1 / sum(1 / vel_app_vars)
        back_azi_var = 1 / sum(1 / back_azi_vars)

        sum_of_inv = sum( inv(mat) for mat in vel_app_cov_mats )
        vel_app_cov_mat = inv(sum_of_inv)

        print(vel_app_cov_mat)

        result_dict[f"vel_app_cov_mat_{component.lower()}"] = vel_app_cov_mat
        result_dict[f"vel_app_uncer_{component.lower()}"] = sqrt(vel_app_var)
        result_dict[f"back_azi_uncer_{component.lower()}"] = sqrt(back_azi_var)

        print(f"Done with Component {component}.")
    
    print(f"Done with time {centertime}.")
    print("")
    result_dicts.append(result_dict)

result_df = DataFrame(result_dicts)

# Reorder the columns
print("Reordering the columns...")
result_df = result_df[["time", "freq_reson",
                       "vel_app_z", "vel_app_uncer_z", "back_azi_z", "back_azi_uncer_z",
                       "vel_app_east_z", "vel_app_north_z", "vel_app_cov_mat_z",
                       "vel_app_1", "vel_app_uncer_1", "back_azi_1", "back_azi_uncer_1",
                       "vel_app_east_1", "vel_app_north_1", "vel_app_cov_mat_1",
                       "vel_app_2", "vel_app_uncer_2", "back_azi_2", "back_azi_uncer_2",
                       "vel_app_east_2", "vel_app_north_2", "vel_app_cov_mat_2"]]

# Convert the covariance matrices to JSON strings
print("Converting the covariance matrices to JSON strings...")
result_df["vel_app_cov_mat_z"] = result_df["vel_app_cov_mat_z"].apply(lambda x: dumps(x.tolist()))
result_df["vel_app_cov_mat_1"] = result_df["vel_app_cov_mat_1"].apply(lambda x: dumps(x.tolist()))
result_df["vel_app_cov_mat_2"] = result_df["vel_app_cov_mat_2"].apply(lambda x: dumps(x.tolist()))

# Convert the back azimuths to degrees
print("Converting the back azimuths to degrees...")
result_df["back_azi_z"] = result_df["back_azi_z"].apply(lambda x: rad2deg(x))
result_df["back_azi_1"] = result_df["back_azi_1"].apply(lambda x: rad2deg(x))
result_df["back_azi_2"] = result_df["back_azi_2"].apply(lambda x: rad2deg(x))

# Convert the back azimuth uncertainties to degrees
print("Converting the back azimuth uncertainties to degrees...")
result_df["back_azi_uncer_z"] = result_df["back_azi_uncer_z"].apply(lambda x: rad2deg(x))
result_df["back_azi_uncer_1"] = result_df["back_azi_uncer_1"].apply(lambda x: rad2deg(x))
result_df["back_azi_uncer_2"] = result_df["back_azi_uncer_2"].apply(lambda x: rad2deg(x))

### Save the results ###
print("Saving the results...")
filename = f"stationary_resonance_station_triad_app_vels_{station1}_{station2}_{station3}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
filepath = join(dirname_mt, filename)
result_df.to_csv(filepath, na_rep="nan", index=False)
print(f"Results saved to {filepath}")









