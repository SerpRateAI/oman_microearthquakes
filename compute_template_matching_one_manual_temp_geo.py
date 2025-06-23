"""
Find the matches for a single manual template recorded on multiple stations
Input:
    - time windows in a Snuffler picker file
Output:
    - template matches
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from time import time
from pathlib import Path
from pandas import Timedelta, Timestamp, DataFrame
from obspy import UTCDateTime
from obspy.signal.cross_correlation import correlate_template
from numpy import float32, ndarray
from scipy.signal import find_peaks
from typing import Dict, List


from utils_basic import GEO_COMPONENTS as components, GEO_CHANNELS as channels, SAMPLING_RATE as sampling_rate, ROOTDIR_GEO as dirpath_geo, PICK_DIR as dirpath_pick, DETECTION_DIR as dirpath_det
from utils_basic import get_geophone_days
from utils_cc import TemplateMatches, Template, Match, plot_all_stations_template_waveforms
from utils_cont_waveform import DayLongWaveform, load_day_long_waveform_from_hdf, load_waveform_slice
from utils_snuffler import read_time_windows
from utils_plot import save_figure

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

# Load the template waveforms from an HDF5 file containing the preprocessed continuous waveforms
def load_template_waveforms(pick_df: DataFrame, template_id: str, hdf5_path: Path) -> Dict[str, Template]:
    template_dict = {}
    for _, row in pick_df.iterrows():
        station = row["station"]
        starttime = row["starttime"]
        endtime = row["endtime"]
        waveform_dict = load_waveform_slice(hdf5_path, station, starttime, endtime)

        # print(waveform_dict.keys())
        template_dict[station] = Template(id = template_id, station = station, starttime = starttime, num_pts = len(waveform_dict[channels[0]]), waveform = waveform_dict)

    return template_dict

# Get the template matches for a single day
def get_matches_for_day(day: str, template_dict: Dict[str, Template], match_dict: Dict[str, List[Match]], cc_threshold: float, hdf5_path: Path) -> Dict[str, List[Match]]:
    # Get the stations to read
    stations = list(template_dict.keys())

    # Read the day long waveforms
    print(f"Reading the day long waveforms for {day}...")
    clock1 = time()
    day_long_waveform_dict = {}
    for station in stations:
        try:
            day_long_waveform_dict[station] = load_day_long_waveform_from_hdf(hdf5_path, station, day)
        except KeyError:
            print(f"No data found for station {station} on {day}!")
            continue

    print(f"Time taken to read the day long waveforms: {time() - clock1} seconds")

    # Check if all stations have data
    if len(day_long_waveform_dict) != len(stations):
        print(f"Not all stations have data for {day}! Skipping...")

        return match_dict
    
    # Compute the cross-correlation for each station
    print(f"Computing cross-correlation for all stations on {day}...")
    clock2 = time()
    for station in stations:
        template = template_dict[station]
        day_long_waveform = day_long_waveform_dict[station]
        waveform_dict = day_long_waveform.waveform

        print(f"Computing cross-correlation for {station}...")
        clock3 = time()
        cc_sta = compute_cc(template, waveform_dict)
        print(f"Time taken to compute cross-correlation for {station}: {time() - clock3} seconds")
       
        clock4 = time()
        starttime_day = day_long_waveform.starttime
        matches = get_match_times_and_waveforms(cc_sta, cc_threshold, template, day_long_waveform)
        print(f"Time taken to get match times and waveforms for {station}: {time() - clock4} seconds")
        print(f"Found {len(matches)} matches for {station}.")


        match_dict[station].extend(matches)

    print(f"Time taken to compute cross-correlation for all stations on {day}: {time() - clock2} seconds")

    return match_dict

# Assemble the template matches for all stations
def assemble_template_matches(template_dict: Dict[str, Template], match_dict: Dict[str, List[Match]]) -> Dict[str, TemplateMatches]:
    tm_dict = {}
    for station in template_dict.keys():
        tm_dict[station] = TemplateMatches(template_dict[station], match_dict[station])
    return tm_dict

# Save the template matches for all stations
def save_template_matches(tm_dict: Dict[str, TemplateMatches], min_freq_filter: float):
    for station in tm_dict.keys():
        print(f"Saving template matches for {station}...")
        filepath_tm = Path(dirpath_det) / f"template_matches_manual_templates_freq{min_freq_filter:.0f}hz.h5"
        tm_dict[station].to_hdf(filepath_tm, overwrite=True)

# Compute the three-component cross-correlation between a template and a stream
def compute_cc(template: Template, waveform: Dict[str, ndarray]):
    for i, channel in enumerate(channels):
        template_waveform = template.waveform[channel]
        data_waveform = waveform[channel]
        cc = correlate_template(data_waveform, template_waveform)

        if i == 0:
            cc_sta = cc
        else:
            cc_sta += cc

    cc_sta /= 3.0

    return cc_sta

# Get the match times and waveforms
def get_match_times_and_waveforms(cc_sta: ndarray, cc_threshold: float, template: Template, day_long_waveform: DayLongWaveform) -> List[Match]:
    indices = find_peaks(cc_sta, height=cc_threshold)[0]
    num_pts = template.num_pts
    waveform_dict = day_long_waveform.waveform
    starttime_day = day_long_waveform.starttime

    matches = []
    for idx in indices:
        starttime_match = starttime_day + Timedelta(seconds = idx / sampling_rate)
        print(starttime_match)
        coeff = cc_sta[idx]

        waveform_match = {}
        for channel in channels:
            waveform_match[channel] = waveform_dict[channel][idx : idx + num_pts].copy()

        match = Match(starttime = starttime_match, coeff = coeff, waveform = waveform_match)
        matches.append(match)

    return matches

# -----------------------------------------------------------------------------
# Parse the command line arguments
# -----------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("--template_id", type=str, help="The ID of the template to match")

parser.add_argument("--min_freq_filter", type=float, help="The low corner frequency for filtering the data", default=20.0)
parser.add_argument("--cc_threshold", type=float, help="The cross-correlation threshold", default=0.95)
parser.add_argument("--test", action="store_true", help="Test mode: only process one day")
parser.add_argument("--day_test", type=str, help="The day to test", default="2020-01-14")

args = parser.parse_args()
template_id = args.template_id
min_freq_filter = args.min_freq_filter
cc_threshold = args.cc_threshold
test = args.test
day_test = args.day_test


# -----------------------------------------------------------------------------
# Main procedure
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Get the HDF5 file path
    hdf5_path = Path(dirpath_geo) / f"preprocessed_data_freq{min_freq_filter:.0f}hz.h5"

    # Load the template waveforms
    print(f"Loading template waveforms for {template_id}...")
    filepath_pick = Path(dirpath_pick) / f"template_windows_{template_id}.txt"
    pick_df = read_time_windows(filepath_pick)
    template_dict = load_template_waveforms(pick_df, template_id, hdf5_path)

    print(f"Stations to process for the template {template_id}: {list(template_dict.keys())}")

    # Plot the template waveforms
    fig, axs = plot_all_stations_template_waveforms(template_dict)
    save_figure(fig, f"template_waveforms_{template_id}.png")

    if test:
        days = [day_test]
        print(f"Running in test mode for {day_test} only...")
    else:
        days = get_geophone_days()
        print(f"Running in full mode for all days...")

    print(f"Initializing match dictionary...") 
    match_dict = {}
    for station in template_dict.keys():
        match_dict[station] = []

    for day in days:
        print(f"Computing template matches for {day}...")
        clock1 = time()
        match_dict = get_matches_for_day(day, template_dict, match_dict, cc_threshold, hdf5_path)

    # Print a summary of the matching results
    print(f"Summary:")
    for station in match_dict.keys():
        print(f"{station}: {len(match_dict[station])} total matches found")

    # Assemble the template matches
    print(f"Assembling template matches...")
    tm_dict = assemble_template_matches(template_dict, match_dict)

    # Save the template matches
    print(f"Saving template matches...")
    save_template_matches(tm_dict, min_freq_filter = min_freq_filter)