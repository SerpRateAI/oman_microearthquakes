"""
Plot example waveforms of the propagating swarms in two boreholes
"""



######
# Import the necessary modules
######

from os.path import join
from argparse import ArgumentParser
from pandas import Timestamp
from matplotlib.pyplot import subplots
from numpy import array
from utils_basic import HYDRO_LOCATIONS as loc_dict, HYDRO_DEPTHS as depth_dict
from utils_preproc import read_and_process_windowed_hydro_waveforms
from utils_plot import save_figure

######
# Parse the command line arguments
######

parser = ArgumentParser()
parser.add_argument("--starttime_a", type=Timestamp, required=True, help="Start time of the Borehole A")
parser.add_argument("--starttime_b", type=Timestamp, required=True, help="Start time of the Borehole B")
parser.add_argument("--duration", type=float, required=True, help="Duration of the waveforms (in seconds)", default=0.5)
parser.add_argument("--min_freq_filter", type=float, required=True, help="Minimum frequency of the filter", default=10.0)

parser.add_argument("--wavespeed_water", type=float, required=True, help="Wave speed of the water wave", default=1500.0)
parser.add_argument("--begin_water_down_a", type=float, required=True, help="Start time of the reference line for water wave in Borehole A", default=0.05)
parser.add_argument("--begin_water_down_b", type=float, required=True, help="Start time of the reference line for water wave in Borehole B", default=0.05)
parser.add_argument("--begin_water_up_a", type=float, required=True, help="Start time of the reference line for water wave in Borehole A", default=0.025)
parser.add_argument("--begin_water_up_b", type=float, required=True, help="Start time of the reference line for water wave in Borehole B", default=0.025)

parser.add_argument("--figwidth", type=float, required=True, help="Width of the figure (in inches)", default=10)
parser.add_argument("--figheight", type=float, required=True, help="Height of the figure (in inches)", default=5)

parser.add_argument("--scale_factor_norm", type=float, required=True, help="Scale factor for the normalized waveforms", default=50.0)
parser.add_argument("--scale_factor_orig_a", type=float, required=True, help="Scale factor for the original waveforms in Borehole A", default=1.0)
parser.add_argument("--scale_factor_orig_b", type=float, required=True, help="Scale factor for the original waveforms in Borehole B", default=1.0)

parser.add_argument("--linewidth_waveform", type=float, required=True, help="Line width", default=1.0)
parser.add_argument("--linewidth_reference", type=float, required=True, help="Line width", default=0.5)
parser.add_argument("--color_line", type=str, required=True, help="Color of the line", default="deepviolet")
parser.add_argument("--color_highlight", type=str, required=True, help="Color of the highlight", default="crimson")

parser.add_argument("--min_depth", type=float, required=True, help="Minimum depth to plot", default=150.0)
parser.add_argument("--max_depth", type=float, required=True, help="Maximum depth to plot", default=400.0)


args = parser.parse_args()

starttime_a = args.starttime_a
starttime_b = args.starttime_b
duration = args.duration
min_freq_filter = args.min_freq_filter

wavespeed_water = args.wavespeed_water
begin_water_down_a = args.begin_water_down_a
begin_water_down_b = args.begin_water_down_b
begin_water_up_a = args.begin_water_up_a
begin_water_up_b = args.begin_water_up_b

figwidth = args.figwidth
figheight = args.figheight

scale_factor_norm = args.scale_factor_norm
scale_factor_orig_a = args.scale_factor_orig_a
scale_factor_orig_b = args.scale_factor_orig_b

linewidth_waveform = args.linewidth_waveform
linewidth_reference = args.linewidth_reference
color_line = args.color_line
color_highlight = args.color_highlight

min_depth = args.min_depth
max_depth = args.max_depth


######
# Read the data
######

# Read the waveforms for Borehole A
station_a = "A00"
stream_a = read_and_process_windowed_hydro_waveforms(starttime_a,
                                                     dur = duration,
                                                     stations = station_a,
                                                     filter = True,
                                                     filter_type = "butter",
                                                     min_freq = min_freq_filter,
                                                     max_freq = None)

# Read the waveforms for Borehole B
station_b = "B00"
stream_b = read_and_process_windowed_hydro_waveforms(starttime_b,
                                                     dur = duration,
                                                     stations = station_b,
                                                     filter = True,
                                                     filter_type = "butter",
                                                     min_freq = min_freq_filter,
                                                     max_freq = None)

######
# Plot the waveforms
######
locations_to_plot = loc_dict[station_a]

# Plot the original waveforms
## Generate the figure 
fig, axs = subplots(1, 2, figsize=(figwidth, figheight))

## Plot the waveforms
depths = []
for i, location in enumerate(locations_to_plot):
    trace_a = stream_a.select(location=location)[0]
    trace_b = stream_b.select(location=location)[0]
    
    waveform_a = trace_a.data
    waveform_b = trace_b.data
    timeax = trace_a.times()
    depth = depth_dict[location]
    depths.append(depth)
    # Plot the original waveforms
    waveform_to_plot_a = waveform_a * scale_factor_orig_a + depth
    waveform_to_plot_b = waveform_b * scale_factor_orig_b + depth

    axs[0].plot(timeax, waveform_to_plot_a, color=color_line, linewidth=linewidth_waveform)
    axs[1].plot(timeax, waveform_to_plot_b, color=color_line, linewidth=linewidth_waveform)

    # Plot the location label
    axs[0].text(timeax[0], depth, location, fontweight="bold", color="black", va = "bottom", ha = "left")
    axs[1].text(timeax[0], depth, location, fontweight="bold", color="black", va = "bottom", ha = "left")

## Plot the reference lines for the water waves
depths = array(depths)
times_water_down_a = begin_water_down_a + (depths - depths[0]) / wavespeed_water
times_water_down_b = begin_water_down_b + (depths - depths[0]) / wavespeed_water
times_water_up_a = begin_water_up_a - (depths - depths[-1]) / wavespeed_water
times_water_up_b = begin_water_up_b - (depths - depths[-1]) / wavespeed_water

axs[0].plot(times_water_down_a, depths, color=color_highlight, linewidth=linewidth_reference, linestyle="--")
axs[0].plot(times_water_up_a, depths, color=color_highlight, linewidth=linewidth_reference, linestyle="--")
axs[1].plot(times_water_down_b, depths, color=color_highlight, linewidth=linewidth_reference, linestyle="--")
axs[1].plot(times_water_up_b, depths, color=color_highlight, linewidth=linewidth_reference, linestyle="--")

axs[0].text(begin_water_down_a, depths[0], "1500 m s$^{-1}$", color=color_highlight, va = "bottom", ha = "right", fontsize=10, rotation=-60)

## Set the axis limits
axs[0].set_xlim(timeax[0], timeax[-1])
axs[1].set_xlim(timeax[0], timeax[-1])

axs[0].set_ylim(min_depth, max_depth)
axs[1].set_ylim(min_depth, max_depth)

## Reverse the y axes
axs[0].invert_yaxis()
axs[1].invert_yaxis()

## Set the title and labels
axs[0].set_title("BA1A", fontweight="bold")
axs[1].set_title("BA1B", fontweight="bold")

axs[0].set_xlabel("Time (s)")
axs[1].set_xlabel("Time (s)")
axs[0].set_ylabel("Depth (m)")
axs[1].set_ylabel("Depth (m)")

# Save the figure
save_figure(fig,"propagating_swarm_example_waveforms_two_holes_original.png")

















