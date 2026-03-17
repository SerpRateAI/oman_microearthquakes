"""
Compute STA/LTA detections for geophone station using the home-grown code
"""

from os.path import join
from time import time
from argparse import ArgumentParser
from pandas import DataFrame, Timedelta, Timestamp
from typing import List, Tuple

from utils_sta_lta import (
    Snippets,
    compute_sta_lta,
    pick_triggers,
    merge_snippets,
)
from utils_basic import get_geophone_days, get_freq_limits_string, SAMPLING_RATE as sampling_rate
from utils_basic import ROOTDIR_GEO as dirpath_data, DETECTION_DIR as dirpath_detections, GEO_COMPONENTS as components
from utils_cont_waveform import load_day_long_waveform_from_hdf

# -----------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------
# Remove the hammer signals
# -----------------------------------------------------------------------------
def remove_hammer_signals(trigger_windows: List[Tuple[Timestamp, Timestamp]]) -> List[Tuple[Timestamp, Timestamp]]:
    """Remove the hammer signals from the trigger windows."""
    trigger_windows = [window for window in trigger_windows if (window[0] < starttime_hammer) or (window[0] > endtime_hammer)]
    return trigger_windows

# Function to extract the snippets
def extract_snippets(triggers, day_long_waveform, buffer_start, buffer_end):
    snippets = Snippets(day_long_waveform.station)

    for trigger in triggers:
        starttime = trigger[0]
        endtime = trigger[1]
        starttime = starttime - Timedelta(seconds=buffer_start)
        endtime = endtime + Timedelta(seconds=buffer_end)
        snippet = day_long_waveform.slice(starttime, endtime)
        snippets.append(snippet)

    return snippets

# Function to assemble the output
def assemble_output(triggers):
    output_dicts = []
    for trigger in triggers:
        output_dict = {"starttime": trigger[0], "endtime": trigger[1]}
        output_dicts.append(output_dict)

    output_df = DataFrame(output_dicts)
    return output_df

# Get parameter suffix
def get_param_suffix(window_length_sta, window_length_lta, on_thresh, off_thresh):
    sta_str = f"sta{window_length_sta * sampling_rate:.0f}" # in samples
    lta_str = f"lta{window_length_lta * sampling_rate:.0f}" # in samples
    on_str = f"on{on_thresh:.0f}"
    off_str = f"off{off_thresh:.0f}"
    suffix = f"{sta_str}_{lta_str}_{on_str}_{off_str}"

    return suffix

# -----------------------------------------------------------------------------
# Define the parameters
# -----------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--station", type=str, required=True)
parser.add_argument("--min_freq_filter", type=float, default=20.0)
parser.add_argument("--max_freq_filter", type=float, default=None)
parser.add_argument("--window_length_sta", type=float, default=5e-3)
parser.add_argument("--window_length_lta", type=float, default=5e-2)
parser.add_argument("--on_thresh", type=float, default=4.0)
parser.add_argument("--off_thresh", type=float, default=1.0)
parser.add_argument("--buffer_start", type=float, default=0.005)
parser.add_argument("--buffer_end", type=float, default=0.005)
parser.add_argument("--test", action="store_true")

args = parser.parse_args()
station = args.station
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
window_length_sta = args.window_length_sta
window_length_lta = args.window_length_lta
on_thresh = args.on_thresh
off_thresh = args.off_thresh
buffer_start = args.buffer_start
buffer_end = args.buffer_end
test = args.test

# -----------------------------------------------------------------------------
# Loop through the days
# -----------------------------------------------------------------------------
print(f"Computing STA/LTA detections for station {station}")
print(f"Min frequency filter: {min_freq_filter}")
print(f"Max frequency filter: {max_freq_filter}")
print(f"Window length STA: {window_length_sta}")
print(f"Window length LTA: {window_length_lta}")
print(f"On threshold: {on_thresh}")
print(f"Off threshold: {off_thresh}")
print(f"Buffer start: {buffer_start}")
print(f"Buffer end: {buffer_end}")
print(f"Test mode: {test}")

# Get the days
if test:
    print(f"Running in test mode. Only processing the second day.")
    days = get_geophone_days(timestamp=False)
    days = days[1:2]
else:
    days = get_geophone_days(timestamp=False)

# Loop through the days
freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
filename = f"preprocessed_data_{freq_str}.h5"
filepath = join(dirpath_data, filename)

print(f"Looping through the days")
trigger_windows_all = []
snippets_all = []
for day in days:
    day_long_waveform = load_day_long_waveform_from_hdf(filepath, station, day)
    if day_long_waveform is None:
        print(f"No data found for station {station} on {day}")
        continue
    
    print(f"Loaded the day-long waveform for {day}")
    starttime = day_long_waveform.starttime

    # Compute the three-component STA/LTA characteristic function
    print(f"Computing the STA/LTA characteristic function")
    for i, component in enumerate(components):
        print(f"Component {component}")
        clock3 = time()
        waveform = day_long_waveform.get_component(component)
        sampling_rate = day_long_waveform.sampling_rate
        cf = compute_sta_lta(waveform, sampling_rate, window_length_sta, window_length_lta)

        if i == 0:
            cf_stack = cf
        else:
            cf_stack = cf_stack + cf

        clock4 = time()
        print(f"Time taken: {clock4 - clock3} seconds")

    # Normalize the characteristic function
    cf_stack = cf_stack / len(components)

    # Find the trigger windows
    print(f"Picking the trigger windows")
    clock5 = time()
    trigger_dict = pick_triggers(cf_stack, on_thresh, off_thresh, sampling_rate = sampling_rate, starttime = starttime)
    trigger_windows = trigger_dict["event_windows"]
    if trigger_windows is None:
        print("Find no triggers. Skipping the day...")
        continue
    else:
        print(f"Detected {len(trigger_windows)} triggers.")

    clock6 = time()
    print(f"Time taken: {clock6 - clock5} seconds")

    # Remove the hammer signals
    print(f"Removing the hammer signals..")
    trigger_windows = remove_hammer_signals(trigger_windows)
    print(f"Number of trigger windows after removing the hammer signals: {len(trigger_windows)}")

    # Save the trigger windows
    print(f"Saving the trigger windows...")
    trigger_windows_all.extend(trigger_windows)

    # Extract the snippets and save
    snippets = extract_snippets(trigger_windows, day_long_waveform, buffer_start, buffer_end)
    snippets_all.append(snippets)

print("")
print("--------------------------------")
print(f"Total number of triggers: {len(trigger_windows_all)}")
print("--------------------------------")

# Merge the snippets
print(f"Merging the snippets...")
snippets_all = merge_snippets(snippets_all)

# Assemble the output
print(f"Assembling the output")
output_df = assemble_output(trigger_windows_all)

# Save the time windows to a CSV file
print(f"Saving the time windows to a CSV file")
suffix = get_param_suffix(window_length_sta, window_length_lta, on_thresh, off_thresh)
filepath = join(dirpath_detections, f"sta_lta_detections_{freq_str}_{suffix}_{station}.csv")
output_df.to_csv(filepath, index=False)
print(f"Saved the time windows to {filepath}")

# Save the snippets to an HDF5 file
print(f"Saving the snippets to an HDF5 file")
filepath = join(dirpath_detections, f"snippets_sta_lta_{freq_str}_{suffix}_{station}.h5")
snippets_all.to_hdf(filepath)
print(f"Saved the snippets to {filepath}")