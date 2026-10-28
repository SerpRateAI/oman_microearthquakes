"""
Compute the total number of repeating STA/LTA detections for each geophone station.
"""

from argparse import ArgumentParser
from pandas import read_csv, DataFrame
from pathlib import Path

from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix
from utils_basic import (
    DETECTION_DIR as dirpath,
    GEO_STATIONS as stations,
    get_freq_limits_string,
)

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--min_freq_filter", type=float, default=20.0, help="The minimum frequency for the filter")
    parser.add_argument("--max_freq_filter", type=float, default=None, help="The maximum frequency for the filter")
    parser.add_argument("--sta_window_sec", type=float, default=0.005, help="The STA window length in seconds")
    parser.add_argument("--lta_window_sec", type=float, default=0.05, help="The LTA window length in seconds")
    parser.add_argument("--on_threshold", type=float, default=4.0, help="The on threshold")
    parser.add_argument("--off_threshold", type=float, default=1.0, help="The off threshold")
    parser.add_argument("--min_cc", type=float, default=0.85, help="The minimum correlation coefficient")
    parser.add_argument("--min_num_similar", type=int, default=10, help="The minimum number of similar snippets")

    args = parser.parse_args()
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    on_threshold = args.on_threshold
    off_threshold = args.off_threshold
    min_cc = args.min_cc
    min_num_similar = args.min_num_similar

    print("Computing the total number of repeating STA/LTA detections for each station...")

    freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
    sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, on_threshold, off_threshold)
    repeating_snippet_suffix = get_repeating_snippet_suffix(min_cc, min_num_similar)

    det_numbers = []
    for station in stations:
        filename = (
            f"sta_lta_detections_repeating_{freq_str}_{sta_lta_suffix}_"
            f"{repeating_snippet_suffix}_{station}.csv"
        )
        filepath = Path(dirpath) / filename
        detection_df = read_csv(filepath, parse_dates=["starttime", "endtime"])
        num_det = len(detection_df)
        det_numbers.append(num_det)

    station_df = DataFrame({"station": stations, "num_of_detections": det_numbers})
    station_df.sort_values(by="num_of_detections", inplace=True, ascending=False)

    filename = (
        f"total_repeating_sta_lta_detection_numbers_{freq_str}_{sta_lta_suffix}_"
        f"{repeating_snippet_suffix}.csv"
    )
    filepath = Path(dirpath) / filename
    station_df.to_csv(filepath, index=False)
    print(f"Saved the station dataframe to {filepath}")
