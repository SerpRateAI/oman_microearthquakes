"""
Plot the hour-long waveforms of slow migrating events on all hydrophones in a borehole
"""

# Import libraries 
from os.path import join
from argparse import ArgumentParser
from pandas import Timestamp, to_timedelta
from matplotlib.pyplot import subplots
from numpy import array
from utils_basic import HYDRO_LOCATIONS as loc_dict, HYDRO_DEPTHS as depth_dict
from utils_preproc import read_and_process_windowed_hydro_waveforms
from utils_plot import format_datetime_xlabels, save_figure

# Parser the command line arguments

parser = ArgumentParser()
parser.add_argument("--station", type=str, help="Station to Plot")
parser.add_argument("--starttime", type=Timestamp, help="Start time of the Borehole B")
parser.add_argument("--time_reference", type=Timestamp, help="Reference time for plotting the line showing the propagation speed.")

parser.add_argument("--duration", type=float, help="Duration of the waveforms (in seconds)", default=3600)
parser.add_argument("--min_freq_filter", type=float, help="Minimum frequency of the filter", default=10.0)

parser.add_argument("--figwidth", type=float, help="Width of the figure (in inches)", default=15)
parser.add_argument("--figheight", type=float, help="Height of the figure (in inches)", default=15)

parser.add_argument("--scale_factor", type=float, help="Scale factor for the original waveforms in Borehole A", default=1.0)
parser.add_argument("--prop_speed", type=float, help="Propagation speed of the swarms (in m/s)", default=0.1)

parser.add_argument("--linewidth_waveform", type=float, help="Line width", default=1.0)
parser.add_argument("--linewidth_reference", type=float, help="Line width", default=1.0)
parser.add_argument("--color_line", type=str, help="Color of the line", default="deepviolet")
parser.add_argument("--color_highlight", type=str, help="Color of the highlight", default="crimson")

parser.add_argument("--min_depth", type=float, help="Minimum depth to plot", default=0.0)
parser.add_argument("--max_depth", type=float, help="Maximum depth to plot", default=400.0)

parser.add_argument("--major_time_tick_spacing", type=str, default="30min")
parser.add_argument("--num_minor_time_ticks", type=int, default=6)

parser.add_argument("--ref_label_y", type=float, help="Y-coordinate of the reference label", default=25.0)

args = parser.parse_args()

station = args.station
starttime = args.starttime
duration = args.duration
min_freq_filter = args.min_freq_filter

figwidth = args.figwidth
figheight = args.figheight

scale_factor = args.scale_factor

linewidth_waveform = args.linewidth_waveform
linewidth_reference = args.linewidth_reference

color_line = args.color_line
color_highlight = args.color_highlight

min_depth = args.min_depth
max_depth = args.max_depth

major_time_tick_spacing = args.major_time_tick_spacing
num_minor_time_ticks = args.num_minor_time_ticks

prop_speed = args.prop_speed
time_reference = args.time_reference
ref_label_y = args.ref_label_y

# Read the waveforms
print(f"Reading the waveforms...")
stream = read_and_process_windowed_hydro_waveforms(starttime,
                                                     dur = duration,
                                                     stations = station,
                                                     filter = True,
                                                     filter_type = "butter",
                                                     min_freq = min_freq_filter,
                                                     max_freq = None)

# Plot the waveforms
print(f"Plotting the waveforms...")
fig, ax = subplots(1, 1, figsize = (figwidth, figheight))

for i, location in enumerate(loc_dict[station]):
    ## Get the trace and time axis
    trace = stream.select(location = location)[0]
    timeax = trace.times()

    timeax = starttime + to_timedelta(timeax, unit = "s")

    depth = depth_dict[location]

    waveform = trace.data
    waveform_to_plot = waveform * scale_factor + depth

    ## Plot the waveform
    ax.plot(timeax, waveform_to_plot, color = color_line, linewidth = linewidth_waveform)

    ## Plot the location label
    ax.text(timeax[0], depth, location, fontsize = 12, fontweight = "bold", color = "black", va = "bottom", ha = "left")

## Plot the reference line showing the propagation speed
depths = array(list(depth_dict.values()))
depth0 = depths[0]

time_shifts = (depths - depth0) / prop_speed 
times_reference = time_reference + to_timedelta(time_shifts, unit = "s")

ax.plot(times_reference, depths, color = color_highlight, linewidth = linewidth_reference, linestyle = "--")
ref_label_x = time_reference
ax.text(ref_label_x, ref_label_y, f"{prop_speed:.1f} m s$^{{-1}}$", fontsize = 12, fontweight = "bold", color = color_highlight, va = "bottom", ha = "center")

## Set the y-axis limits
ax.set_xlim(timeax[0], timeax[-1])
ax.set_ylim(min_depth, max_depth)

## Reverse the y-axis
ax.invert_yaxis()

# Set the x-axis and y-axis labels
format_datetime_xlabels(ax,
                        major_tick_spacing = major_time_tick_spacing,
                        num_minor_ticks = num_minor_time_ticks,
                        axis_label_size = 12)

ax.set_ylabel("Depth (m)", fontsize = 12)

## Set the title
ax.set_title(f"Downward propagating swarms recorded at {station}", fontsize = 14, fontweight = "bold")
## Save the figure
print(f"Saving the figure...")
time_str = starttime.strftime("%Y%m%d%H%M%S")
save_figure(fig, f"propagating_swarms_hour_long_waveforms_{station}_{time_str}_{duration:.0f}s.png")








