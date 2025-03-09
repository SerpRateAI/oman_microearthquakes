"""
Compute the apparent velocities of one delaunay station triad
"""

### Imports ###
from os.path import join
from argparse import ArgumentParser
from numpy import arctan2, array, nan, isnan, isrealobj, pi, setdiff1d, var, vstack
from numpy.linalg import inv, norm
from pandas import DataFrame, Timedelta
from pandas import merge, read_csv
from scipy.signal.windows import dpss
from matplotlib.pyplot import subplots
from time import time

from utils_basic import SPECTROGRAM_DIR as dirname_spec, MT_DIR as dirname_mt, GEO_COMPONENTS as components
from utils_basic import get_geophone_coords
from utils_basic import time2suffix, str2timestamp, timestamp_to_utcdatetime
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_mt import mt_cspec, get_avg_phase_diff, get_indep_phase_diffs
from utils_plot import save_figure

### Inputs ###

# Command-line arguments
parser = ArgumentParser(description="Compute the slowness of the triangle formed by three stations")

parser.add_argument("--station1", type=str, help="Station 1")
parser.add_argument("--station2", type=str, help="Station 2")
parser.add_argument("--station3", type=str, help="Station 3")

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--window_length_mt", type=float, help="Window length for multitaper analysis", default=900.0)
parser.add_argument("--min_cohe", type=float, help="Minimum coherence", default=0.85)

parser.add_argument("--max_phase_uncer", type=float, help="Maximum phase uncertainty", default=0.2)

# Parse the command line inputs
args = parser.parse_args()

station1 = args.station1
station2 = args.station2
station3 = args.station3

mode_name = args.mode_name
window_length_mt = args.window_length_mt
min_cohe = args.min_cohe
max_phase_uncer = args.max_phase_uncer

### Read the station locations ###
print("Reading the station locations...")
coord_df = get_geophone_coords()

loc_vec_1 = array([coord_df.loc[station1, "east"], coord_df.loc[station1, "north"]])
loc_vec_2 = array([coord_df.loc[station2, "east"], coord_df.loc[station2, "north"]])
loc_vec_3 = array([coord_df.loc[station3, "east"], coord_df.loc[station3, "north"]])

loc_vec_12 = loc_vec_2 - loc_vec_1
loc_vec_23 = loc_vec_3 - loc_vec_2

### Build the inverse matrix ###
print("Building the inverse matrix...")
loc_mat = vstack([loc_vec_12, loc_vec_23])
loc_mat_inv = inv(loc_mat)

### Read the phase differences ###
print("Reading the phase differences...")

print(f"Reading the phase differences for {station1}-{station2}...")
filename = f"multitaper_inter_sta_phase_diffs_PR02549_{station1}_{station2}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
phase_diff_12_df = read_csv(join(dirname_mt, filename))

print(f"Reading the phase differences for {station2}-{station3}...")
filename = f"multitaper_inter_sta_phase_diffs_PR02549_{station2}_{station3}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
phase_diff_23_df = read_csv(join(dirname_mt, filename))

phase_diff_df = merge(phase_diff_12_df, phase_diff_23_df, on=["time"], suffixes=["_12", "_23"], how="inner")

### Compute the slowness ###
print("Computing the slownesses for each time window...")

result_dicts = []
for _, row in phase_diff_df.iterrows():
    centertime = row["time"]
    freq_reson = row["frequency_12"]

    result_dict = {
        "time": centertime,
        "freq_reson": freq_reson,
    }

    for component in components:
        phase_diff_12 = row[f"phase_diff_{component.lower()}_12"]
        phase_diff_23 = row[f"phase_diff_{component.lower()}_23"]

        phase_diff_uncer_12 = row[f"phase_diff_uncer_{component.lower()}_12"]
        phase_diff_uncer_23 = row[f"phase_diff_uncer_{component.lower()}_23"]

        if isnan(phase_diff_12) or isnan(phase_diff_23):
            result_dict[f"vel_app_{component.lower()}"] = nan
            result_dict[f"back_azi_{component.lower()}"] = nan
            continue

        if phase_diff_uncer_12 > max_phase_uncer or phase_diff_uncer_23 > max_phase_uncer:
            continue

        time_diff_12 = phase_diff_12 / freq_reson / 2 / pi
        time_diff_23 = phase_diff_23 / freq_reson / 2 / pi

        slow_vec = loc_mat_inv @ array([time_diff_12, time_diff_23])

        slow = norm(slow_vec)
        back_azi = arctan2(slow_vec[0], slow_vec[1]) * 180 / pi # The first element is the east component, and the second is the north component!

        vel_app = 1 / slow

        result_dict[f"vel_app_{component.lower()}"] = vel_app
        result_dict[f"back_azi_{component.lower()}"] = back_azi

    result_dicts.append(result_dict)

result_df = DataFrame(result_dicts)

### Save the results ###
print("Saving the results...")
filename = f"station_triad_app_vels_PR02549_{station1}_{station2}_{station3}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
filepath = join(dirname_mt, filename)
result_df.to_csv(filepath, na_rep="nan")
print(f"Results saved to {filepath}")









