"""
Compute the waveform stacks of all matched events of a template
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List
from numpy import abs, arange, ndarray
from pandas import Timedelta, read_hdf, Timestamp
from obspy import Trace, Stream, UTCDateTime
from matplotlib.pyplot import subplots, close

from utils_basic import (
    GEO_COMPONENTS as components,
    SAMPLING_RATE as sampling_rate,
    ROOTDIR_GEO as dirpath_geo,
    DETECTION_DIR as dirpath_det,
    INNER_STATIONS_A as inner_stations_a,
    INNER_STATIONS_B as inner_stations_b,
    MIDDLE_STATIONS_A as middle_stations_a,
    MIDDLE_STATIONS_B as middle_stations_b,
    str2timestamp,
    geo_component2channel
)

from utils_cont_waveform import load_waveform_slice
from utils_plot import get_geo_component_color, save_figure

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

"""
Plot the template waveform for a given component.
"""
def plot_template_waveform(ax, template_waveform_dict, component, 
                           scale=0.7,
                           linewidth=1.0,
                           plot_station_label=True,
                           plot_title=True):

    # Plot each station
    for i, (station, waveform_dict) in enumerate(template_waveform_dict.items()):
        waveform = waveform_dict[component] * scale + i
        color = get_geo_component_color(component)
        timeax = arange(len(waveform)) / sampling_rate
        ax.plot(timeax, waveform, color=color, linewidth=linewidth)

        if plot_station_label:
            ax.text(0.00, i + 0.05, station, fontsize=12, fontweight="bold", va="bottom", ha="left")

    # Set the x-axis limits
    ax.set_xlim(timeax[0], timeax[-1])

    # Set the x-axis label
    ax.set_xlabel("Time (s)", fontsize=12)

    # Set the y-axis label
    ax.set_yticklabels([])

    # Set the title
    if plot_title:
        ax.set_title(f"Template", fontsize=14, fontweight="bold")

    return ax

"""
Plot the waveform stacks and the individual waveform contributing to the stack.
"""
def plot_waveform_stack(ax, stack_waveform_dict, all_waveform_dict, component,
                        scale=0.7,
                        linewidth_stack=1.0,
                        linewidth_indiv=0.5,
                        plot_station_label=True,
                        plot_title=True):

    # Plot the individual waveforms in light gray
    for i, (_, waveform_dicts) in enumerate(all_waveform_dict.items()):
        for waveform_dict in waveform_dicts:
            waveform = waveform_dict[component] * scale + i
            color = "lightgray"
            timeax = arange(len(waveform)) / sampling_rate
            ax.plot(timeax, waveform, color=color, linewidth=linewidth_indiv, zorder=0)

    # Plot the waveform stack for each station
    for i, (station, waveform_dict) in enumerate(stack_waveform_dict.items()):
        waveform = waveform_dict[component] * scale + i
        color = get_geo_component_color(component)
        timeax = arange(len(waveform)) / sampling_rate
        ax.plot(timeax, waveform, color=color, linewidth=linewidth_stack, zorder=1)

        if plot_station_label:
            ax.text(0.00, i + 0.05, station, fontsize=12, fontweight="bold", va="bottom", ha="left")

    # Set the x-axis limits
    ax.set_xlim(timeax[0], timeax[-1])

    # Set the x-axis label
    ax.set_xlabel("Time (s)", fontsize=12)

    # Set the y-axis label
    ax.set_yticklabels([])

    # Set the title
    if plot_title:
        ax.set_title(f"Stack", fontsize=14, fontweight="bold")

    return ax

"""
Assemble the waveform stacks into a stream object.
"""
def assemble_stream(stack_waveform_dict: Dict[str, Dict[str, ndarray]], starttime: Timestamp) -> Stream:
    stream = Stream()
    starttime = str2timestamp(starttime)

    for station, waveform_dict in stack_waveform_dict.items():
        for comp in components:
            channel = geo_component2channel(comp)
            trace = Trace(data=waveform_dict[comp], header={"station": station, "channel": channel, "starttime": starttime, "sampling_rate": sampling_rate})
            stream.append(trace)

    stream.sort()

    return stream
# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

# Input arguments
parser = ArgumentParser()
parser.add_argument("--template_id", type=str, help="Template ID")
parser.add_argument("--subarray_name", type=str, help="Subarray to process", default="A")
parser.add_argument("--min_freq_filter", type=float, help="The low corner frequency for filtering the data", default=20.0)
parser.add_argument("--max_freq_filter", type=float, help="The high corner frequency for filtering the data", default=None)
parser.add_argument("--cc_threshold", type=float, help="The cross-correlation threshold for matching events", default=0.85)
parser.add_argument("--max_num_unmatched_sta", type=int, default=0, help="Maximum number of unmatched stations to consider a matched event")

parser.add_argument("--buffer_time", type=float, help="The buffer time before the first match time", default=0.08)
parser.add_argument("--window_length", type=float, help="The window length", default=0.3)

parser.add_argument("--figwidth", type=float, help="The width of the figure", default=10.0)
parser.add_argument("--figheight", type=float, help="The height of the figure", default=5.0)

args = parser.parse_args()
template_id = args.template_id
subarray_name = args.subarray_name
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
cc_threshold = args.cc_threshold
max_num_unmatched_sta = args.max_num_unmatched_sta
buffer_time = args.buffer_time
window_length = args.window_length
figwidth = args.figwidth
figheight = args.figheight

if subarray_name == "A":
    stations_to_stack = inner_stations_a + middle_stations_a
elif subarray_name == "B":
    stations_to_stack = inner_stations_b + middle_stations_b
else:
    raise ValueError(f"Invalid subarray name: {subarray_name}")

# Load the matched events
if max_freq_filter is not None:
    filename = f"matched_events_manual_templates_min{min_freq_filter:.0f}hz_max{max_freq_filter:.0f}hz_cc{cc_threshold:.2f}_num_unmatch{max_num_unmatched_sta:d}.h5"
else:
    filename = f"matched_events_manual_templates_min{min_freq_filter:.0f}hz_cc{cc_threshold:.2f}_num_unmatch{max_num_unmatched_sta:d}.h5"
filepath = Path(dirpath_det) / filename
all_event_df = read_hdf(filepath, key=f"template_{template_id}")

# Set the hierarchical index
all_event_df = all_event_df.set_index(["first_match_time", "station"])
all_event_df.sort_index(inplace=True)

# Compute the waveform stack for each station
template_waveform_dict : Dict[str, Dict[str, ndarray]] = {}
all_waveform_dict : Dict[str, List[Dict[str, ndarray]]] = {}
stack_waveform_dict : Dict[str, Dict[str, ndarray]] = {}
if max_freq_filter is not None:
    data_path = Path(dirpath_geo) / f"preprocessed_data_min{min_freq_filter:.0f}hz_max{max_freq_filter:.0f}hz.h5"
else:
    data_path = Path(dirpath_geo) / f"preprocessed_data_min{min_freq_filter:.0f}hz.h5"
num_match = len(all_event_df.index.get_level_values("first_match_time").unique())
print(f"Number of matched events: {num_match}")

for i, (first_match_time, event_df) in enumerate(all_event_df.groupby(level=0)):
    print(f"Processing the {i + 1}-th event...")

    if event_df.loc[event_df.index[0], "self_match"] == True:
        self_match = True
        template_starttime = first_match_time
        print(f"The {i + 1}-th event is a self-match.")
    else:
        self_match = False

    starttime = first_match_time - Timedelta(seconds=buffer_time)
    endtime = first_match_time + Timedelta(seconds=window_length)

    for station in stations_to_stack:
        print(f"Processing Station {station}...")
        # Load the waveform slice
        waveform_dict = load_waveform_slice(data_path, station, starttime, endtime=endtime)

        # Normalize the waveform
        for comp in components:
            waveform_dict[comp] /= abs(waveform_dict[comp]).max()

        # Save the template waveform
        if self_match:
            template_waveform_dict[station] = waveform_dict

        # Update the waveform stack for the first event
        if i == 0:
            all_waveform_dict[station] = [waveform_dict]
            stack_waveform_dict[station] = waveform_dict
        # Update the waveform stack for the subsequent events
        else:
            for comp in components:
                all_waveform_dict[station].append(waveform_dict)
                stack_waveform_dict[station][comp] += waveform_dict[comp]

# Normalize the waveform stack
for station in stations_to_stack:
    for comp in components:
        stack_waveform_dict[station][comp] /= num_match

# Plot the three component template waveforms and waveform stacks
for component in components:
    fig, axs = subplots(1, 2, figsize=(figwidth, figheight))

    # Plot the template waveform
    ax = axs[0]
    ax = plot_template_waveform(ax, template_waveform_dict, component)

    # Plot the waveform stack
    ax = axs[1]
    ax = plot_waveform_stack(ax, stack_waveform_dict, all_waveform_dict, component)

    # Set the suptitle
    fig.suptitle(f"Template {template_id}, {num_match} matches", fontsize=14, fontweight="bold", y = 0.93)

    # Save the figure
    save_figure(fig, f"matched_waveform_stack_template{template_id}_{component.lower()}.png", dpi=300)
    close(fig)

# Assemble the waveform stacks into a stream object
stream = assemble_stream(stack_waveform_dict, template_starttime)

# Save the stream to an MSEED file
filepath = Path(dirpath_det) / f"matched_waveform_stack_template{template_id}.mseed"
stream.write(filepath, format="MSEED")
print(f"Saved the waveform stacks to {filepath}")