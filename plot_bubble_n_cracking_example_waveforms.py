"""
This script plots the example waveforms for a bubble and a cracking events
"""

# Import the necessary libraries
from os.path import join
from argparse import ArgumentParser
from pandas import Timestamp, to_timedelta
from matplotlib.pyplot import subplots
from numpy import array, amin, amax
from utils_basic import HYDRO_LOCATIONS as loc_dict, HYDRO_DEPTHS as depth_dict, WATER_SOUND_SPEED as prop_speed
from utils_preproc import read_and_process_windowed_hydro_waveforms
from utils_plot import format_datetime_xlabels, save_figure


# Parse the command-line arguments
parser = ArgumentParser()
parser.add_argument("--station", type=str, help="Station to Plot", default="B00")
parser.add_argument("--starttime_bubble", type=Timestamp, help="Start time of the bubble events")
parser.add_argument("--starttime_crack", type=Timestamp, help="Start time of the cracking events")
parser.add_argument("--duration", type=float, help="Duration of the waveforms (in seconds)", default=0.5)
parser.add_argument("--time_reference", type=Timestamp, help="Reference time for plotting the line showing the propagation speed.")

parser.add_argument("--min_freq_filter", type=float, help="Minimum frequency of the filter", default=10.0)

parser.add_argument("--figwidth", type=float, help="Width of the figure (in inches)", default=15)
parser.add_argument("--figheight", type=float, help="Height of the figure (in inches)", default=15)

parser.add_argument("--scale_factor_crack", type=float, help="Scale factor for the crack waveforms", default=0.05)
parser.add_argument("--scale_factor_bubble", type=float, help="Scale factor for the bubble waveforms", default=0.01)

parser.add_argument("--linewidth_waveform", type=float, help="Line width", default=1.0)
parser.add_argument("--linewidth_reference", type=float, help="Line width", default=1.0)
parser.add_argument("--color_line", type=str, help="Color of the line", default="deepviolet")
parser.add_argument("--color_highlight", type=str, help="Color of the highlight", default="crimson")

parser.add_argument("--min_depth", type=float, help="Minimum depth to plot", default=0.0)
parser.add_argument("--max_depth", type=float, help="Maximum depth to plot", default=400.0)

parser.add_argument("--time_reference_crack_up1", type=float, help="Time reference for the line marking the up-going wave for the crack event", default=0.06)
parser.add_argument("--location_reference_crack_up1", type=str, help="Location reference for the line marking the up-going wave for the crack event", default="03")

parser.add_argument("--time_reference_crack_up2", type=float, help="Time reference for the line marking the up-going wave for the crack event", default=0.19)
parser.add_argument("--location_reference_crack_up2", type=str, help="Location reference for the line marking the up-going wave for the crack event", default="06")

parser.add_argument("--time_reference_crack_down1", type=float, help="Time reference for the line marking the down-going wave for the crack event", default=0.12)
parser.add_argument("--location_reference_crack_down1", type=str, help="Location reference for the line marking the down-going wave for the crack event", default="05")

parser.add_argument("--time_reference_bubble_down", type=float, help="Time reference for the line marking the bubble down", default=0.04)
parser.add_argument("--location_reference_bubble_down", type=str, help="Location reference for the line marking the bubble down", default="01")

parser.add_argument("--time_reference_bubble_up", type=float, help="Time reference for the line marking the bubble up", default=0.31)
parser.add_argument("--location_reference_bubble_up", type=str, help="Location reference for the line marking the bubble up", default="06")

args = parser.parse_args()

station = args.station
starttime_bubble = args.starttime_bubble
starttime_crack = args.starttime_crack
duration = args.duration
time_reference = args.time_reference

min_freq_filter = args.min_freq_filter

figwidth = args.figwidth
figheight = args.figheight

scale_factor_crack = args.scale_factor_crack
scale_factor_bubble = args.scale_factor_bubble

linewidth_waveform = args.linewidth_waveform
linewidth_reference = args.linewidth_reference
color_line = args.color_line
color_highlight = args.color_highlight

min_depth = args.min_depth
max_depth = args.max_depth

time_reference_crack_up1 = args.time_reference_crack_up1
location_reference_crack_up1 = args.location_reference_crack_up1

time_reference_crack_up2 = args.time_reference_crack_up2
location_reference_crack_up2 = args.location_reference_crack_up2

time_reference_crack_down1 = args.time_reference_crack_down1
location_reference_crack_down1 = args.location_reference_crack_down1

time_reference_bubble_down = args.time_reference_bubble_down
location_reference_bubble_down = args.location_reference_bubble_down

time_reference_bubble_up = args.time_reference_bubble_up
location_reference_bubble_up = args.location_reference_bubble_up


# Read the waveforms
## Read the waveforms for the bubble event
print(f"Reading the waveforms for the bubble event: {starttime_bubble}")
stream_bubble = read_and_process_windowed_hydro_waveforms(starttime_bubble,
                                                          dur = duration,
                                                          stations = station,
                                                          filter = True,
                                                          filter_type = "butter",
                                                          min_freq = min_freq_filter,
                                                          max_freq = None)

print(f"Reading the waveforms for the cracking event: {starttime_crack}")
stream_crack = read_and_process_windowed_hydro_waveforms(starttime_crack,
                                                          dur = duration,
                                                          stations = station,
                                                          filter = True,
                                                          filter_type = "butter",
                                                          min_freq = min_freq_filter,
                                                          max_freq = None)

# Plot the waveforms
## Create the figure and the axis
fig, ax = subplots(nrows = 2, ncols = 1, figsize = (figwidth, figheight))

## Plot the crack waveform
print("Plotting the crack waveforms...")
ax_crack = ax[0]

locations = loc_dict[station]
depths_loc = array([depth_dict[location] for location in locations])
min_depth_loc = amin(depths_loc)
max_depth_loc = amax(depths_loc)

for i, location in enumerate(locations):
    ## Get the trace and time axis
    trace = stream_crack.select(location = location)[0]
    timeax = trace.times()

    depth = depth_dict[location]

    waveform = trace.data
    waveform_to_plot = -waveform * scale_factor_crack + depth # Need to account for the reversed y axis

    ### Plot the waveform
    ax_crack.plot(timeax, waveform_to_plot, color = color_line, linewidth = linewidth_waveform)

    ### Plot the location label
    ax_crack.text(timeax[0], depth, location, fontsize = 12, fontweight = "bold", color = "black", va = "bottom", ha = "left")

    ### Plot the reference lines
    if location == location_reference_crack_up1:
        depths_reference = array([depth, min_depth_loc])
        times_reference = time_reference_crack_up1 - (depths_reference - depth) / prop_speed
        ax_crack.plot(times_reference, depths_reference, color = color_highlight, linewidth = linewidth_reference, linestyle = "--")

    if location == location_reference_crack_up2:
        depths_reference = array([depth, min_depth_loc])
        times_reference = time_reference_crack_up2 - (depths_reference - depth) / prop_speed
        ax_crack.plot(times_reference, depths_reference, color = color_highlight, linewidth = linewidth_reference, linestyle = "--")

    if location == location_reference_crack_down1:
        depths_reference = array([depth, max_depth_loc])
        times_reference = time_reference_crack_down1 + (depths_reference - depth) / prop_speed
        ax_crack.plot(times_reference, depths_reference, color = color_highlight, linewidth = linewidth_reference, linestyle = "--")

### Format the x-axis
ax_crack.set_xlim(timeax[0], timeax[-1])
ax_crack.set_ylim(min_depth, max_depth)

### Reverse the y-axis
ax_crack.invert_yaxis()

### Set the title
ax_crack.set_title(f"Cracking, {starttime_crack.strftime('%Y-%m-%d %H:%M:%S.%f')}", fontsize = 12, fontweight = "bold")

## Plot the bubble waveform
print("Plotting the bubble waveforms...")
ax_bubble = ax[1]

locations = loc_dict[station]
for i, location in enumerate(locations):
    ### Get the trace and time axis
    trace = stream_bubble.select(location = location)[0]
    timeax = trace.times()

    depth = depth_dict[location]

    waveform = trace.data
    waveform_to_plot = -waveform * scale_factor_bubble + depth

    ### Plot the waveform
    ax_bubble.plot(timeax, waveform_to_plot, color = color_line, linewidth = linewidth_waveform)

    ### Plot the location label
    ax_bubble.text(timeax[0], depth, location, fontsize = 12, fontweight = "bold", color = "black", va = "bottom", ha = "left")

    ### Plot the reference lines
    if location == location_reference_bubble_down:
        depths_reference = array([depth, max_depth_loc])
        times_reference = time_reference_bubble_down + (depths_reference - depth) / prop_speed
        ax_bubble.plot(times_reference, depths_reference, color = color_highlight, linewidth = linewidth_reference, linestyle = "--")

    if location == location_reference_bubble_up:
        depths_reference = array([depth, min_depth_loc])
        times_reference = time_reference_bubble_up - (depths_reference - depth) / prop_speed
        ax_bubble.plot(times_reference, depths_reference, color = color_highlight, linewidth = linewidth_reference, linestyle = "--")

### Format the x-axis
ax_bubble.set_xlim(timeax[0], timeax[-1])
ax_bubble.set_ylim(min_depth, max_depth)

### Reverse the y-axis
ax_bubble.invert_yaxis()

### Set the title
ax_bubble.set_title(f"Bubble, {starttime_bubble.strftime('%Y-%m-%d %H:%M:%S.%f')}", fontsize = 12, fontweight = "bold")

## Save the figure
print("Saving the figure...")
save_figure(fig, "bubble_n_cracking_example_waveforms.png")














