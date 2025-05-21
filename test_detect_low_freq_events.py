"""
Test using STA-LTA to detect low-frequency events.
"""

### Import the necessary libraries
from os.path import join
from argparse import ArgumentParser
from numpy import abs, amax, isnan, where
from pandas import Timestamp, Timedelta
from matplotlib.pyplot import subplots
from obspy import UTCDateTime

from utils_basic import HYDRO_LOCATIONS as loc_dict
from utils_preproc import read_and_process_day_long_hydro_waveforms, read_and_process_windowed_hydro_waveforms
from utils_plot import save_figure

### Parse the command line arguments
parser = ArgumentParser()
parser.add_argument("--day", type=str, default="2019-05-01", help="The day of the waveforms to process.")
parser.add_argument("--min_freq_filter", type=float, default=None, help="The minimum frequency of the filter.")
parser.add_argument("--max_freq_filter", type=float, default=10.0, help="The maximum frequency of the filter.")
parser.add_argument("--sta", type=float, default=0.2, help="The STA of the STA-LTA trigger.")
parser.add_argument("--lta", type=float, default=1.0, help="The LTA of the STA-LTA trigger.")

parser.add_argument("--station_to_plot", type=str, default="B00", help="The station to plot.")
parser.add_argument("--start_time_to_plot", type=Timestamp, default="2019-05-01T18:20:26", help="The start time of the waveforms to plot.")
parser.add_argument("--end_time_to_plot", type=Timestamp, default="2019-05-01T18:20:34", help="The end time of the waveforms to plot.")
parser.add_argument("--max_trigger_to_plot", type=float, default=5.0, help="The maximum trigger value to plot.")

args = parser.parse_args()

day = args.day
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
sta = args.sta
lta = args.lta
station_to_plot = args.station_to_plot
start_time_to_plot = args.start_time_to_plot
end_time_to_plot = args.end_time_to_plot
max_trigger_to_plot = args.max_trigger_to_plot

### Read the hydrophone waveforms and low-pass filter them
print(f"Reading and preprocessing the hydrophone waveforms for {day}...")
stream = read_and_process_windowed_hydro_waveforms(starttime = start_time_to_plot + Timedelta(seconds = -10), endtime = end_time_to_plot + Timedelta(seconds = 10), 
                                                   stations = station_to_plot, filter = True, filter_type = "butter", min_freq = min_freq_filter, max_freq = max_freq_filter)

waveform = stream[0].data
# print(where(~isnan(waveform))[0])

### Compute the trigger functions
print(f"Computing the trigger functions for {day}...")
stream_trigger = stream.copy() # Make a copy of the stream
stream_trigger.trigger("classicstalta", sta = sta, lta = lta)

triggers = stream_trigger[0].data
print(triggers)
# print(where(~isnan(triggers))[0])

### Plot the trigger functions
print(f"Plotting the trigger functions and waveforms in the time window {start_time_to_plot} to {end_time_to_plot} for {day}...")
locations_to_plot = loc_dict[station_to_plot]
stream.trim(starttime = UTCDateTime(start_time_to_plot), endtime = UTCDateTime(end_time_to_plot))
stream_trigger.trim(starttime = UTCDateTime(start_time_to_plot), endtime = UTCDateTime(end_time_to_plot))

print(stream_trigger[0].stats.starttime)
print(stream_trigger[0].stats.endtime)

# Create the figure
num_loc = len(locations_to_plot)
fig, axs = subplots(num_loc, 1, figsize = (10, 20), sharex = True)

for i, location in enumerate(locations_to_plot):
    ax_waveform = axs[i]

    # Plot the waveform
    trace = stream.select(location = location)[0]
    waveform = trace.data
    waveform = waveform / amax(abs(waveform))
    timeax = trace.times()
    ax_waveform.plot(timeax, waveform, color = "darkviolet")
    ax_waveform.set_ylim(-1, 1)

    # Set the x-axis limits
    ax_waveform.set_xlim(timeax[0], timeax[-1])

    # Set the y-axis to the color of the waveform
    ax_waveform.spines['left'].set_color('darkviolet')
    ax_waveform.tick_params(axis='y', colors='darkviolet')

    # Plot the trigger function
    ax_trigger = ax_waveform.twinx()
    trace_trigger = stream_trigger.select(location = location)[0]
    trigger_function = trace_trigger.data

    ax_trigger.plot(timeax, trigger_function, color = "violet")
    ax_trigger.set_ylim(0, max_trigger_to_plot)

    # Set the y-axis to the color of the trigger function
    ax_trigger.spines['right'].set_color('violet')
    ax_trigger.tick_params(axis='y', colors='violet')

    if i == num_loc - 1:
        ax_waveform.set_xlabel("Time (s)")

# Save the figure
start_time_to_plot_str = start_time_to_plot.strftime("%Y%m%dT%H%M%S")
end_time_to_plot_str = end_time_to_plot.strftime("%Y%m%d%H%M%S")
figname = f"test_detect_low_freq_events_{day}_{station_to_plot}_{start_time_to_plot_str}to{end_time_to_plot_str}.png"

save_figure(fig, figname)