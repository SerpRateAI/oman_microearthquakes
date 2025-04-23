"""
Plot the waveforms of an earthquake
"""

###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from numpy import amax, abs
from pandas import read_csv, Timestamp, Timedelta
from matplotlib.pyplot import subplots

from utils_basic import LOC_DIR as dirpath_loc, GEO_COMPONENTS as geo_components
from utils_basic import get_geophone_coords, get_datetime_axis_from_trace
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_plot import component2label,save_figure, get_geo_component_color, format_datetime_xlabels

###
# Command line arguments
###

parser = ArgumentParser(description = "Plot the waveforms of an earthquake")
parser.add_argument("--earthquake_id", type = int, help = "Earthquake ID", default = 1)
parser.add_argument("--buffer_time", type = float, default = 10.0, help = "Buffer time in seconds")
parser.add_argument("--min_freq", type = float, default = 20.0, help = "Minimum frequency in Hz")
parser.add_argument("--max_freq", type = float, default = 30.0, help = "Maximum frequency in Hz")

parser.add_argument("--scale_factor", type = float, default = 0.8, help = "Scale factor")
parser.add_argument("--figwidth", type = float, default = 15.0, help = "Figure width in inches")
parser.add_argument("--figheight", type = float, default = 10.0, help = "Figure height in inches")
parser.add_argument("--linewidth_waveform", type = float, default = 0.5, help = "Line width")
parser.add_argument("--linewidth_marker", type = float, default = 1.0, help = "Line width")

parser.add_argument("--station_label_offset", type = float, default = 0.1, help = "Station label offset")
parser.add_argument("--station_label_size", type = float, default = 12, help = "Station label size")
parser.add_argument("--title_size", type = float, default = 14, help = "Title size")
parser.add_argument("--major_time_spacing", type = str, default = "10s", help = "Major tick spacing")

# Parse the command line arguments
args = parser.parse_args()
earthquake_id = args.earthquake_id
buffer_time = args.buffer_time
min_freq = args.min_freq
max_freq = args.max_freq

scale_factor = args.scale_factor
figwidth = args.figwidth
figheight = args.figheight
linewidth_waveform = args.linewidth_waveform
linewidth_marker = args.linewidth_marker
station_label_offset = args.station_label_offset
station_label_size = args.station_label_size
title_size = args.title_size
major_time_spacing = args.major_time_spacing
###
# Read the data
###

# Read the window list
inpath = join(dirpath_loc, "earthquakes.csv")
earthquake_df = read_csv(inpath, parse_dates = ["start_time", "end_time"])
start_time_eq = earthquake_df.loc[earthquake_df["earthquake_id"] == earthquake_id, "start_time"].values[0]
end_time_eq = earthquake_df.loc[earthquake_df["earthquake_id"] == earthquake_id, "end_time"].values[0]

# Read the station coordinates
station_df = get_geophone_coords()

# Read the waveforms
start_time_plot = Timestamp(start_time_eq) - Timedelta(seconds = buffer_time)
end_time_plot = Timestamp(end_time_eq) + Timedelta(seconds = buffer_time)
stream = read_and_process_windowed_geo_waveforms(start_time_plot, endtime = end_time_plot,
                                                 filter = True,
                                                 zerophase = False,
                                                 filter_type = "butter",
                                                 min_freq = min_freq,
                                                 max_freq = max_freq,
                                                 corners = 4)
###
# Plot the waveforms
###

# Set up the figure
fig, axs = subplots(nrows = 1, ncols = 3, figsize = (figwidth, figheight))

fig.subplots_adjust(wspace = 0.04)

# Sort the stations by their coordinates
station_df = station_df.sort_values(by = ["north"])

# Plot each component
for i, component in enumerate(geo_components):
    ax = axs[i]
    color = get_geo_component_color(component)

    # Plot each station
    for j, station in enumerate(station_df.index):
        # Get the traces
        trace = stream.select(component = component, station = station)[0]
        data = trace.data
        data = data / amax(abs(data)) * scale_factor + j

        # Get the datetime axis
        datetime_axis = get_datetime_axis_from_trace(trace)

        # Plot the trace
        ax.plot(datetime_axis, data, color = color, linewidth = linewidth_waveform)

        # Plot the station label
        if i == 0:
            ax.text(start_time_plot, j + station_label_offset, station, ha = "left", va = "bottom", fontsize = station_label_size, fontweight = "bold")

    # Mark the start and end times
    ax.axvline(x = start_time_eq, color = "crimson", linestyle = "--", linewidth = linewidth_marker)
    ax.axvline(x = end_time_eq, color = "crimson", linestyle = "--", linewidth = linewidth_marker)

    # Set the axis limits
    ax.set_xlim(start_time_plot, end_time_plot)

    # Format the axis
    format_datetime_xlabels(ax,
                            major_tick_spacing = major_time_spacing,
                            num_minor_ticks = 5,
                            va = "top", ha = "right",
                            rotation = 30)
    
    # Remove the y-axis label
    ax.set_yticklabels([])

    # Plot the title
    title = component2label(component)
    ax.set_title(title, fontsize = title_size, fontweight = "bold")

# Set the suptitle
fig.suptitle(f"Earthquake {earthquake_id}, {min_freq} - {max_freq} Hz", fontsize = title_size, fontweight = "bold", y = 0.95)

# Save the figure
figname = f"earthquake_waveforms_eq{earthquake_id}.png"
save_figure(fig, figname)










