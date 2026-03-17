"""
Remove hammer signals 
from STA/LTA detections
"""

# Import libraries
from os.path import join
from argparse import ArgumentParser

from utils_basic import (
    DETECTION_DIR as dirpath_detect,
    PICK_DIR as dirpath_pick,
    NETWORK as network,
    HAMMER_DAY as hammer_day,
    GEO_STATIONS as stations,
)
from utils_basic import get_freq_limits_string
from utils_sta_lta import get_sta_lta_suffix
from pandas import read_csv


# Command line arguments
parser = ArgumentParser()
parser.add_argument("--sta", type=float, default=5e-3)
parser.add_argument("--lta", type=float, default=5e-2)
parser.add_argument("--thr_on", type=float, default=4.0)
parser.add_argument("--thr_off", type=float, default=1.0)
parser.add_argument("--min_freq", type=float, default=20.0)
parser.add_argument("--max_freq", type=float, default=None)

args = parser.parse_args()

sta = args.sta
lta = args.lta
thr_on = args.thr_on
thr_off = args.thr_off
min_freq = args.min_freq
max_freq = args.max_freq

# Define the frequency limits string
freq_limits_string = get_freq_limits_string(min_freq, max_freq)
sta_lta_suffix = get_sta_lta_suffix(sta, lta, thr_on, thr_off)

# Process each station
for station in stations:
    filename = f"sta_lta_detections_{freq_limits_string}_{sta_lta_suffix}_{station}.csv"
    filepath = join(dirpath_detect, filename)
    det_df = read_csv(filepath, parse_dates=["starttime", "endtime"])
    print(f"Read the detections for {station}")
    print(f"Number of input detections: {len(det_df)}")

    # Process the detections
    det_df = det_df[det_df["starttime"].dt.date != hammer_day.date()]

    # Save the detections
    print(f"Saving the detections for {station}")
    det_df.to_csv(filepath, index=False)
    print("--------------------------------")
