"""
This script computes the waveform stacks for the hub event and its aligned events.
"""

from argparse import ArgumentParser
from os.path import join
from numpy import amax, ndarray
from pandas import Timestamp, Timedelta
from pandas import read_csv, to_datetime
from typing import Dict
from obspy import Stream, Trace

from utils_basic import DETECTION_DIR as dirpath_event, ROOTDIR_GEO as dirpath_waveform, SAMPLING_RATE as sampling_rate
from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix
from utils_basic import get_freq_limits_string
from utils_basic import INNER_STATIONS_A as stations_inner_a, INNER_STATIONS_B as stations_inner_b, MIDDLE_STATIONS_A as stations_middle_a, MIDDLE_STATIONS_B as stations_middle_b
from utils_cont_waveform import load_waveform_slice
from utils_plot import save_figure, get_geo_component_color, component2label
from utils_basic import GEO_COMPONENTS as components

#--------------------------------------------------------------------------------------------------
# Define the functions
#--------------------------------------------------------------------------------------------------

"""
Assemble the waveform stacks into a stream object.
"""
def assemble_stream(waveform_stack_dict: Dict[str, Dict[str, ndarray]], starttime: Timestamp, sampling_rate: float = sampling_rate) -> Stream:
    stream = Stream()
    for station, waveform_dict in waveform_stack_dict.items():
        for component, waveform in waveform_dict.items():
            stream.append(Trace(data=waveform, header={"network": "7F", "station": station, "channel": component, "starttime": starttime, "sampling_rate": sampling_rate}))

    return stream

# Parse the command line arguments
#--------------------------------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--group_label", type=int, required=True, help="The group label")
parser.add_argument("--min_cc", type=float, default=0.85)
parser.add_argument("--min_num_similar_snippet", type=int, default=10)
parser.add_argument("--min_num_similar_station", type=int, default=3)
parser.add_argument("--sta_window_sec", type=float, default=0.005)
parser.add_argument("--lta_window_sec", type=float, default=0.05)
parser.add_argument("--thr_on", type=float, default=4.0)
parser.add_argument("--thr_off", type=float, default=1.0)
parser.add_argument("--min_freq_filter", type=float, default=20.0)
parser.add_argument("--max_freq_filter", type=float, default=None)

parser.add_argument("--buffer_before_sec", type=float, default=0.05, help="The buffer before the first onset time in seconds")
parser.add_argument("--buffer_after_sec", type=float, default=0.2, help="The buffer after the first onset time in seconds")
parser.add_argument("--scale_factor", type=float, default=0.7, help="The scale factor for the waveforms")

args = parser.parse_args()
group_label = args.group_label
min_cc = args.min_cc
min_num_similar_snippet = args.min_num_similar_snippet
min_num_similar_station = args.min_num_similar_station
sta_window_sec = args.sta_window_sec
lta_window_sec = args.lta_window_sec
thr_on = args.thr_on
thr_off = args.thr_off
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
buffer_before_sec = args.buffer_before_sec
buffer_after_sec = args.buffer_after_sec
scale_factor = args.scale_factor

print("--------------------------------")
print("Computing the waveform stacks for the hub event and its aligned events...")
print("--------------------------------")
print(f"Group label: {group_label}")
print(f"Min CC: {min_cc}")
print(f"Min number of similar snippets: {min_num_similar_snippet}")
print(f"Min number of similar stations: {min_num_similar_station}")
print(f"STA window sec: {sta_window_sec}")
print(f"LTA window sec: {lta_window_sec}")
print(f"On threshold: {thr_on}")
print(f"Off threshold: {thr_off}")
print(f"Min frequency filter: {min_freq_filter}")
print(f"Max frequency filter: {max_freq_filter}")

# Build the suffix
suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
suffix_repeating_snippet = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar_snippet)
suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

# Load the event alignment information
print("Loading the event alignment information...")
filename = f"event_alignments_group{group_label:d}_{suffix_group}.csv"
filepath = join(dirpath_event, filename)
alignment_df = read_csv(filepath)
alignment_df["aligned_time"] = to_datetime(alignment_df["aligned_time"], format="ISO8601")
alignment_df["hub"] = alignment_df["hub"].astype(bool)
num_events = len(alignment_df)
print(f"Number of aligned events to stack: {num_events}")

# Get the hub event and stations to stack
id_hub = alignment_df.loc[alignment_df["hub"] == True, "id"].values[0]
station_first_onset = alignment_df.loc[alignment_df["hub"] == True, "aligned_station"].values[0]
print(f"Hub event ID: {id_hub}")
print(f"Hub event first onset station: {station_first_onset}")
if station_first_onset.startswith("A"):
    stations_to_stack = stations_inner_a + stations_middle_a
elif station_first_onset.startswith("B"):
    stations_to_stack = stations_inner_b + stations_middle_b
print(f"Stations to stack: {stations_to_stack}")

hub_aligned_time = alignment_df.loc[alignment_df["hub"], "aligned_time"].values[0]
stack_starttime = hub_aligned_time - Timedelta(seconds=buffer_before_sec)
print(f"Hub aligned time: {hub_aligned_time}")
print(f"Stack window starttime: {stack_starttime}")

# Get the waveform file path
filename = f"preprocessed_data_{suffix_freq}.h5"
filepath = join(dirpath_waveform, filename)

# Compute the waveform stacks for each station and component
print("Computing the waveform stacks for each station and component...")
waveform_stack_dict = {}
num_stacked_dict = {}
for station in stations_to_stack:
    print(f"Processing station {station}...")
    waveform_stack_dict[station] = {}
    num_stacked = 0
    for _, row in alignment_df.iterrows():
        event_id = row["id"]
        aligned_time = row["aligned_time"]
        aligned_starttime_plot = aligned_time - Timedelta(seconds=buffer_before_sec)
        aligned_endtime_plot = aligned_time + Timedelta(seconds=buffer_after_sec)
        try:
            waveform_dict, _ = load_waveform_slice(
                filepath, station, aligned_starttime_plot, endtime=aligned_endtime_plot, normalize=True
            )
        except Exception as e:
            print(f"Error loading waveform slice for event {event_id} at station {station}: {e}")
            continue

        num_stacked += 1
        for component in components:
            waveform = waveform_dict[component]
            waveform = waveform / amax(abs(waveform))
            if num_stacked == 1:
                waveform_stack_dict[station][component] = waveform
            else:
                waveform_stack_dict[station][component] += waveform

    num_stacked_dict[station] = num_stacked
    print(f"Stacked {num_stacked}/{num_events} events at station {station}")

# Normalize the waveform stacks
print("Normalizing the waveform stacks...")
stations_stacked = []
for station in stations_to_stack:
    num_stacked = num_stacked_dict[station]
    if num_stacked == 0:
        print(f"No successful waveform loads at station {station}; skipping.")
        del waveform_stack_dict[station]
        continue
    stations_stacked.append(station)
    for component in components:
        waveform_stack_dict[station][component] /= num_stacked

if not stations_stacked:
    raise RuntimeError("No waveform stacks could be computed for any station.")

# Assemble the waveform stacks into a stream object (starttime = first sample of loaded window)
stream = assemble_stream(waveform_stack_dict, stack_starttime)

# Save the waveform stacks
print("Saving the waveform stacks...")
filename = f"hub_event_waveform_stack_group{group_label:d}_{suffix_group}.mseed"
filepath = join(dirpath_event, filename)
stream.write(filepath, format="MSEED")
print(f"Saved the waveform stacks to {filepath}")