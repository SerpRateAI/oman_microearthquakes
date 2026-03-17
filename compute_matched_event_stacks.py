"""
Compute the waveform stacks of all matched events of a template
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List
from numpy import abs, arange, ndarray, mean, array, std
from pandas import Timedelta, read_hdf, Timestamp, DataFrame
from obspy import Trace, Stream
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

from utils_basic import get_freq_limits_string
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
def plot_waveform_stack(ax, waveform_stack_dict, waveform_plot_dict, component,
                        scale=0.7,
                        linewidth_stack=1.0,
                        linewidth_indiv=0.5,
                        plot_station_label=True,
                        plot_title=True):

    # Plot the individual waveforms in light gray
    for i, (station, waveform_plot_sta_dict) in enumerate(waveform_plot_dict.items()):
        waveforms = waveform_plot_sta_dict[component]
        for waveform in waveforms:
            waveform = waveform * scale + i
            color = "lightgray"
            timeax = arange(len(waveform)) / sampling_rate
            ax.plot(timeax, waveform, color=color, linewidth=linewidth_indiv, zorder=0)

    # Plot the waveform stack for each station
    for i, (station, waveform_stack) in enumerate(waveform_stack_dict.items()):
        waveform = waveform_stack[component] * scale + i
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
            trace = Trace(data=waveform_dict[comp], header={"network": "7F", "station": station, "channel": channel, "starttime": starttime, "sampling_rate": sampling_rate})
            stream.append(trace)

    stream.sort()

    return stream

"""
Compute the SNR of a waveform.
"""
def compute_snr(waveform: ndarray, noise_window_length: float) -> float:
    noise_window_length = int(noise_window_length * sampling_rate) + 1
    noise_window = waveform[:noise_window_length]
    signal_window = waveform[noise_window_length:]
    snr = mean(abs(signal_window)) / mean(abs(noise_window))

    return snr

"""
Get the amplitude threshold for screening the matched events.
"""
def get_amplitude_threshold(template_event_df: DataFrame, data_path: Path, buffer_time: float, window_length: float, amplitude_threshold_frac: float) -> float:
    max_amplitude = 0.0
    first_match_time = template_event_df.index[0][0]
    for station in template_event_df.index.get_level_values("station").unique():
        starttime = first_match_time - Timedelta(seconds=buffer_time)
        endtime = first_match_time + Timedelta(seconds=window_length)
        waveform_dict = load_waveform_slice(data_path, station, starttime, endtime=endtime)
        for comp in components:
            waveform = waveform_dict[comp]
            max_amplitude = max(max_amplitude, abs(waveform).max())

    amplitude_threshold = max_amplitude * amplitude_threshold_frac
    return amplitude_threshold

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
parser.add_argument("--snr_power", type=float, default=0.0, help="The power of the SNR weight")
parser.add_argument("--power_std_factor", type=float, default=5.0, help="The factor of the mean-power standard deviation threshold for removing waveforms with abnormally large amplitudes")

parser.add_argument("--buffer_time", type=float, help="The buffer time before the first match time", default=0.08)
parser.add_argument("--window_length", type=float, help="The window length", default=0.3)

parser.add_argument("--figwidth", type=float, help="The width of the figure", default=10.0)
parser.add_argument("--figheight", type=float, help="The height of the figure", default=2.0)

args = parser.parse_args()
template_id = args.template_id
cc_threshold = args.cc_threshold
subarray_name = args.subarray_name
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
max_num_unmatched_sta = args.max_num_unmatched_sta
buffer_time = args.buffer_time
window_length = args.window_length
figwidth = args.figwidth
figheight = args.figheight
snr_power = args.snr_power
power_std_factor = args.power_std_factor

if subarray_name == "A":
    stations_to_stack = inner_stations_a + middle_stations_a
elif subarray_name == "B":
    stations_to_stack = inner_stations_b + middle_stations_b
else:
    raise ValueError(f"Invalid subarray name: {subarray_name}")

# Load the matched events
freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
filename = f"matched_events_manual_templates_{freq_str}_cc{cc_threshold:.2f}_num_unmatch{max_num_unmatched_sta:d}.h5"
print(f"Loading the matched events from {filename}...")
filepath = Path(dirpath_det) / filename
all_event_df = read_hdf(filepath, key=f"template_{template_id}")

# Set the hierarchical index
all_event_df = all_event_df.set_index(["first_match_time", "station"])
all_event_df.sort_index(inplace=True)

# Get the data path
data_path = Path(dirpath_geo) / f"preprocessed_data_{freq_str}.h5"

print(f"Loading the data from {data_path}...")

num_match = len(all_event_df.index.get_level_values("first_match_time").unique())
print(f"Number of matched events: {num_match}")

# Initialize the average power dictionary
power_dict : Dict[str, Dict[str, List[float]]] = {}
for station in stations_to_stack:
    power_sta_dict = {}
    for comp in components:
        power_sta_dict[comp] = []

    power_dict[station] = power_sta_dict

## Read the waveform of all matched events and compute the mean powers
template_waveform_dict : Dict[str, Dict[str, ndarray]] = {station: {comp: None for comp in components} for station in stations_to_stack}
waveform_dict : Dict[str, Dict[str, List[ndarray]]] = {station: {comp: [] for comp in components} for station in stations_to_stack}

print(f"Processing the matched events...")
for i, (first_match_time, event_df) in enumerate(all_event_df.groupby(level=0)):
    #print(f"Processing the {i + 1}-th event...")

    if event_df.loc[event_df.index[0], "self_match"] == True:
        self_match = True
        template_starttime = first_match_time
        print(f"The {i + 1}-th event is a self-match.")
    else:
        self_match = False

    starttime = first_match_time - Timedelta(seconds=buffer_time)
    endtime = first_match_time + Timedelta(seconds=window_length)

    for station in stations_to_stack:
        #print(f"Processing Station {station}...")
        # Load the waveform slice
        waveform_sta_dict = load_waveform_slice(data_path, station, starttime, endtime=endtime)

        for comp in components:
            waveform = waveform_sta_dict[comp]


            ### Compute the mean power
            power = mean(waveform ** 2)

            power_dict[station][comp].append(power) # Save the mean power
            waveform_dict[station][comp].append(waveform) # Save the waveform

        # Save the template waveform
        if self_match:
            template_waveform_dict[station] = waveform_sta_dict

# Compute the standard deviation of the mean powers for each station and component
print(f"Computing the standard deviation of the mean powers...")
power_std_dict : Dict[str, Dict[str, float]] = {}
for station in stations_to_stack:
    power_std_sta_dict = {}
    for comp in components:
        power_std_sta_dict[comp] = std(array(power_dict[station][comp]))
        print(f"The standard deviation of the mean power of {station}.{comp} is {power_std_sta_dict[comp]}")
    power_std_dict[station] = power_std_sta_dict

# Compute the SNR-weighted waveform stack while ruling out waveforms with abnormally large amplitudes
print(f"Computing the SNR-weighted waveform stacks...")
waveform_weight_dict : Dict[str, Dict[str, List[ndarray]]] = {station: {comp: [] for comp in components} for station in stations_to_stack}
waveform_plot_dict : Dict[str, Dict[str, List[ndarray]]] = {station: {comp: [] for comp in components} for station in stations_to_stack}
snr_weight_dict: Dict[str, Dict[str, List[float]]] = {station: {comp: [] for comp in components} for station in stations_to_stack}
for station in stations_to_stack:
    for comp in components:
        waveforms = waveform_dict[station][comp]
        power_std = power_std_dict[station][comp]

        for i, waveform in enumerate(waveforms):
            power = mean(waveform ** 2)
            if power > power_std * power_std_factor:
                print(f"The power of {i + 1}-th waveform of {station}.{comp} is greater than {power_std_factor} times the standard deviation {power_std}, skipping...")
                continue

            ## Normalize the waveform
            waveform /= abs(waveform).max()

            ### Compute the SNR
            if snr_power > 0.0:
                snr_weight = compute_snr(waveform, buffer_time) ** snr_power # Use the buffer time as the noise window
            else:
                snr_weight = 1.0

            ## Compute the SNR-weighted waveform
            waveform_weight = waveform * snr_weight

            ## Save the waveform
            waveform_weight_dict[station][comp].append(waveform_weight)
            waveform_plot_dict[station][comp].append(waveform)
            snr_weight_dict[station][comp].append(snr_weight)

# Compute the waveform stack
print(f"Computing the waveform stack...")
waveform_stack_dict : Dict[str, Dict[str, ndarray]] = {station: {comp: None for comp in components} for station in stations_to_stack}
for station in stations_to_stack:
    for comp in components:
        # Initialize the waveform stack
        waveform_stack = waveform_weight_dict[station][comp][0]

        # Compute the waveform stack
        for waveform in waveform_weight_dict[station][comp][1:]:
            waveform_stack += waveform

        # Normalize the stack
        waveform_stack /= sum(array(snr_weight_dict[station][comp]))

        # Save the waveform stack
        waveform_stack_dict[station][comp] = waveform_stack


# Plot the three component template waveforms and waveform stacks
for component in components:
    fig, axs = subplots(1, 2, figsize=(figwidth, figheight))

    # Plot the template waveform
    ax = axs[0]
    ax = plot_template_waveform(ax, template_waveform_dict, component)

    # Plot the waveform stack
    ax = axs[1]
    ax = plot_waveform_stack(ax, waveform_stack_dict, waveform_plot_dict, component)

    # Set the suptitle
    fig.suptitle(f"Template {template_id}, {num_match} matches", fontsize=14, fontweight="bold", y = 0.93)

    # Save the figure
    filename = f"matched_waveform_stack_template{template_id}_{component.lower()}_{freq_str}_snr_power{snr_power:.0f}.png"

    save_figure(fig, filename, dpi=300)
    close(fig)

# Assemble the waveform stacks into a stream object
stream = assemble_stream(waveform_stack_dict, template_starttime)

# Save the stream to an MSEED file
filepath = Path(dirpath_det) / f"matched_waveform_stack_template{template_id}_{freq_str}_snr_power{snr_power:.0f}.mseed"

stream.write(filepath, format="MSEED")
print(f"Saved the waveform stacks to {filepath}")