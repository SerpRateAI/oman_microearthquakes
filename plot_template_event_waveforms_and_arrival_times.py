"""
Plot the waveforms and observed and predicted arrival times of a template event.
"""

#--------------------------------------------------------------------------------------------------
# Import the necessary libraries
#--------------------------------------------------------------------------------------------------

from argparse import ArgumentParser
from numpy import arange
from pandas import (
                    read_csv, 
                    read_hdf, 
                    Timedelta,
)
from pathlib import Path
from matplotlib.pyplot import (
    subplots,
)

from utils_basic import (
    DETECTION_DIR as dirpath_detection,
    ROOTDIR_GEO as dirpath_waveform,
    LOC_DIR as dirpath_loc,
    PICK_DIR as dirpath_pick,
)

from utils_basic import (
    SAMPLING_RATE as sample_rate,
)
from utils_snuffler import read_time_windows
from utils_loc import (
    load_location_info, 
    process_arrival_info,
)
from utils_cont_waveform import load_waveform_slice
from utils_plot import get_geo_component_color, save_figure

#--------------------------------------------------------------------------------------------------
# Define the functions
#--------------------------------------------------------------------------------------------------

"""
Plot the 3-C waveforms of a station.
"""
def plot_station_waveforms(ax, waveform_dict,
                            linewidth = 1.0,
                            max_abs_amplitude = 0.3):

    for component, waveform in waveform_dict.items():
        num_pts = len(waveform)
        timeax = arange(num_pts) / sample_rate
        color = get_geo_component_color(component)
        ax.plot(timeax, waveform, linewidth=linewidth, label=component, color=color, zorder=1)
    
    ax.set_ylim(-max_abs_amplitude, max_abs_amplitude)

    return ax

"""
Plot the observed arrival times of a station.
"""
def plot_station_pick(ax, arrival_df, station, starttime_waveform,
                     color = "crimson",
                     alpha = 0.3):
    
    starttime_pick = arrival_df[ arrival_df["station"] == station]["starttime"].values[0]
    endtime_pick = arrival_df[ arrival_df["station"] == station]["endtime"].values[0]

    begin = (starttime_pick - starttime_waveform).total_seconds()
    end = (endtime_pick - starttime_waveform).total_seconds()
    
    ax.axvspan(begin, end, facecolor=color, edgecolor="none", alpha=alpha, zorder=2)

    return ax
        
"""
Plot the predicted arrival times of a station.
"""
def plot_station_prediction(ax, prediction_time, starttime_waveform,
                              color = "black", linewidth = 2.0, linestyle = "-"):
    
    arrival_time_pred = (prediction_time - starttime_waveform).total_seconds()
    ax.axvline(arrival_time_pred, color=color, linewidth=linewidth, linestyle=linestyle, zorder=3)

    return ax

#--------------------------------------------------------------------------------------------------
# Define the parameters
#--------------------------------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("--template_id", type=str, required=True, help="The ID of the template event.")

parser.add_argument("--min_freq_filter", type=float, default=20.0, help="The minimum frequency of the filter.")
parser.add_argument("--cc_threshold", type=float, default=0.85, help="The threshold of the cross-correlation coefficient.")
parser.add_argument("--num_unmatch", type=int, default=0, help="The number of unmatched events.")
parser.add_argument("--buffer_time_start", type=float, default=0.02, help="The buffer time before the first match time.")
parser.add_argument("--time_window", type=float, default=0.1, help="The time window of the waveform slice.")

parser.add_argument("--figwidth", type=float, default=10.0, help="The width of the figure.")
parser.add_argument("--figheight", type=float, default=15.0, help="The height of the figure.")
parser.add_argument("--hspace", type=float, default=0.05, help="The height of the space between the subplots.")

args = parser.parse_args()
template_id = args.template_id
min_freq_filter = args.min_freq_filter
cc_threshold = args.cc_threshold
num_unmatch = args.num_unmatch
buffer_time_start = args.buffer_time_start
time_window = args.time_window
figwidth = args.figwidth
figheight = args.figheight
hspace = args.hspace

#--------------------------------------------------------------------------------------------------
# Load the data
#--------------------------------------------------------------------------------------------------

# Load the observed arrival times
filepath = Path(dirpath_pick) / f"template_arrivals_{template_id}.txt"
arrival_df = read_time_windows(filepath)
arrival_df = process_arrival_info(arrival_df, to_seconds = False)

starttime_waveform = arrival_df["starttime"].min() - Timedelta(seconds=buffer_time_start)
endtime_waveform = arrival_df["starttime"].min() + Timedelta(seconds=time_window)

# Load the location information
location_dict, arrival_time_pred_dict, easts_grid, norths_grid, depths_grid, rms_vol = load_location_info("template", template_id, "P")
origin_time = location_dict["origin_time"]

# Load the waveform slices
waveform_sta_dict = {}
filename = f"preprocessed_data_freq{min_freq_filter:.0f}hz.h5"
filepath = Path(dirpath_waveform) / filename

for _, row in arrival_df.iterrows():
    station = row["station"]
    waveform_dict = load_waveform_slice(filepath, station, starttime_waveform, endtime = endtime_waveform, normalize = True)
    waveform_sta_dict[station] = waveform_dict

#--------------------------------------------------------------------------------------------------
# Plot the waveforms and arrival times
#--------------------------------------------------------------------------------------------------

# Generate the axes
num_sta = len(waveform_sta_dict)
fig, axes = subplots(num_sta, 1, figsize=(figwidth, figheight), sharex=True)
fig.subplots_adjust(hspace=hspace)

# Plot the 3-C waveforms and arrival times for each station
for i_station, (station, waveform_dict) in enumerate(waveform_sta_dict.items()):
    # Plot the 3-C waveforms
    ax = axes[i_station]
    plot_station_waveforms(ax, waveform_dict)

    # Plot the observed arrival times
    plot_station_pick(ax, arrival_df, station, starttime_waveform)

    # Plot the origin time
    plot_station_prediction(ax, origin_time, starttime_waveform,
                            color = "gray", linestyle = "--")

    # Plot the predicted arrival times
    arrival_time_pred = arrival_time_pred_dict[station]
    plot_station_prediction(ax, arrival_time_pred, starttime_waveform,
                            color = "gray")
    
    # Plot the station label  
    ax.text(0.005, 0.98, station, fontsize=12, fontweight="bold",
            transform=ax.transAxes, ha="left", va="top", zorder=4)


# Set the x-axis label
axes[-1].set_xlabel("Time (s)")

# Set the y-axis label
axes[0].set_ylabel("Norm. amp.")

# Set the x-axis limit
axes[-1].set_xlim(0, time_window)

# Set the title
axes[0].set_title(f"Template {template_id}", fontsize=14, fontweight="bold")

# Save the figure
save_figure(fig, f"template_event_waveforms_and_arrival_times_{template_id}.png")