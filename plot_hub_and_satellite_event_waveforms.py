"""
Plot the waveforms of the hub event and all its similar events
"""

# Import libraries
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, read_hdf, read_json, Timedelta, to_datetime
from scipy.sparse import load_npz
from numpy import amax, arange
from matplotlib.pyplot import subplots

# Import modules
from utils_basic import DETECTION_DIR as dirpath_event, ROOTDIR_GEO as dirpath_waveform
from utils_basic import STARTTIME_GEO as starttime_bin, ENDTIME_GEO as endtime_bin
from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix
from utils_basic import get_freq_limits_string, get_geophone_coords
from utils_basic import (
    INNER_STATIONS_A as stations_inner_a, 
    INNER_STATIONS_B as stations_inner_b,
    MIDDLE_STATIONS_A as stations_middle_a,
    MIDDLE_STATIONS_B as stations_middle_b,
    GEO_COMPONENTS as components,
    SAMPLING_RATE as sample_rate,
)
from utils_cont_waveform import load_waveform_slice
from utils_plot import save_figure, get_geo_component_color, component2label


#--------------------------------------------------------------------------------------------------
# Define the functions
#--------------------------------------------------------------------------------------------------

"""
Plot the waveforms of an event
"""
def plot_waveforms(ax, stream, stations, color = "lightgray"):
    for i, station in enumerate(stations):
        trace = stream.select(station=station)[0]
        waveform = trace.data
        waveform = i + waveform / amax(abs(waveform))
        timeax = trace.times()
        ax.plot(timeax, waveform, color=color, linewidth=1.0)

"""
Convert the per-station times to datetime objects
"""
def convert_station_times(station_dict):
    return {
        station: to_datetime(times, utc=True, format="ISO8601")
        for station, times in station_dict.items()
    }
#--------------------------------------------------------------------------------------------------
# Parse the command line arguments
#--------------------------------------------------------------------------------------------------
# Parse the command line arguments
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
print("Plotting the waveforms of the hub event and its aligned events...")
print("--------------------------------")
print(f"Group label: {group_label}")
print(f"Min frequency filter: {min_freq_filter}")
print(f"Max frequency filter: {max_freq_filter}")
print(f"STA window sec: {sta_window_sec}")
print(f"LTA window sec: {lta_window_sec}")
print(f"On threshold: {thr_on}")
print(f"Off threshold: {thr_off}")

# Build the event suffix
suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
suffix_repeating_snippet = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar_snippet)
suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

# Load the event information
print("Loading the event information...")
filename = f"grouped_events_group{group_label:d}_{suffix_group}.jsonl"
filepath = join(dirpath_event, filename)
event_df = read_json(filepath, lines = True)
event_df["first_onset"] = to_datetime(event_df["first_onset"], errors="coerce")
event_df["per_station_times"] = event_df["per_station_times"].apply(convert_station_times)

# Get the event alignment information
print("Getting the event alignment information...")
filename = f"event_alignments_group{group_label:d}_{suffix_group}.csv"
filepath = join(dirpath_event, filename)
alignment_df = read_csv(filepath)
alignment_df["aligned_first_onset"] = to_datetime(alignment_df["aligned_first_onset"], format="ISO8601")
alignment_df["hub"] = alignment_df["hub"].astype(bool)
num_events = len(alignment_df)

# Get the stations to plot
id_hub = alignment_df.loc[alignment_df["hub"] == True, "id"].values[0]
station_first_onset = alignment_df.loc[alignment_df["hub"] == True, "first_onset_station"].values[0]
print(f"Hub event first onset station: {station_first_onset}")
if station_first_onset.startswith("A"):
    stations_to_plot = stations_inner_a + stations_middle_a
elif station_first_onset.startswith("B"):
    stations_to_plot = stations_inner_b + stations_middle_b
print(f"Stations to plot: {stations_to_plot}")

# Get the hub event per-station times
hub_dict = event_df.loc[event_df["event_id"] == id_hub].iloc[0].to_dict()
per_station_times_hub = hub_dict["per_station_times"]


# Load the station coordinates and sort by north coordinate
print("Loading the station coordinates and sorting by north coordinate...")
coord_df = get_geophone_coords()
coord_df = coord_df.sort_values(by="north")
coord_df = coord_df[ coord_df.index.isin(stations_to_plot) ]
stations_to_plot = coord_df.index.tolist()

# Get the waveform file path
filename = f"preprocessed_data_{suffix_freq}.h5"
filepath = join(dirpath_waveform, filename)

# Generate the figure and axes
print("Generating the figure and axes...")
fig, axes = subplots(1, 3, figsize=(15, 10), sharey=True)

# Plot the hub event and its aligned events waveforms
print("Plotting the waveforms of the hub event and its aligned events...")
for _, row in alignment_df.iterrows():
    for i_station, station in enumerate(stations_to_plot):
        id = row["id"]
        aligned_first_onset = row["aligned_first_onset"]
        hub = row["hub"]
        aligned_starttime_plot = aligned_first_onset - Timedelta(seconds=buffer_before_sec)
        aligned_endtime_plot = aligned_first_onset + Timedelta(seconds=buffer_after_sec)
        waveform_dict, _ = load_waveform_slice(filepath, station, aligned_starttime_plot, endtime = aligned_endtime_plot, normalize = True)

        for i_component, component in enumerate(components):
            if hub:
                color = get_geo_component_color(component)
                linewidth = 1.0
                zorder = 2
            else:
                color = "lightgray"
                linewidth = 0.1
                zorder = 0

            waveform = waveform_dict[component]
            timeax = arange(len(waveform)) / sample_rate
            waveform_plot = waveform / amax(abs(waveform)) * scale_factor + i_station
            axes[i_component].plot(timeax, waveform_plot, color=color, linewidth=linewidth, zorder=zorder)

            if hub and station in per_station_times_hub.keys():
                starttime = per_station_times_hub[station][0]
                endtime = per_station_times_hub[station][1]
                starttime_window = (starttime - aligned_starttime_plot).total_seconds()
                endtime_window = (endtime - aligned_starttime_plot).total_seconds()
                axes[i_component].vlines(x=starttime_window, ymin=i_station - 0.2, ymax=i_station + 0.2, color="crimson", linewidth= 2 *linewidth, zorder = 3)
                axes[i_component].vlines(x=endtime_window, ymin=i_station - 0.2, ymax=i_station + 0.2, color="crimson", linewidth= 2 *linewidth, zorder = 3)

# Plot the station label
for i_station, station in enumerate(stations_to_plot):
    axes[0].text(0.0, i_station, station, fontsize=12, fontweight="bold", ha="left", va="bottom")

# Set the axis limits
for i_component, component in enumerate(components):
    axes[i_component].set_xlim(0.0, buffer_after_sec + buffer_before_sec)
    axes[i_component].set_xlabel("Time (s)", fontsize=12)

    title = component2label(component)
    axes[i_component].set_title(title, fontsize=12, fontweight="bold")

# Turn off the y-axis labels and ticks
for i_component in range(3):
    axes[i_component].set_yticks([])
    axes[i_component].set_yticklabels([])

# Set the suptitle
fig.suptitle(f"Group {group_label}, Hub Event {id_hub}, {num_events} events", fontsize=14, fontweight="bold", y=0.95)

# Save the figure
print("Saving the figure...")
figname = f"hub_and_satellite_event_waveforms_group{group_label}_{suffix}.png"
save_figure(fig, figname)

