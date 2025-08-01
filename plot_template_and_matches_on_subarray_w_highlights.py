"""
Plot the template waveform and the matches recorded on a subarray.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict
from numpy import asarray, arange
from pandas import Timedelta, Timestamp, date_range, read_hdf, to_timedelta
from matplotlib.pyplot import figure

from utils_basic import (
    INNER_STATIONS_A as inner_stations_a,
    INNER_STATIONS_B as inner_stations_b,
    MIDDLE_STATIONS_A as middle_stations_a,
    MIDDLE_STATIONS_B as middle_stations_b,
    ROOTDIR_GEO as dirpath_data,
    GEO_COMPONENTS as components,
    SAMPLING_RATE as sampling_rate,
    DETECTION_DIR as dirpath_det,
    EASTMIN_A_LOC as min_east_a,
    EASTMAX_A_LOC as max_east_a,
    NORTHMIN_A_LOC as min_north_a,
    NORTHMAX_A_LOC as max_north_a,
    EASTMIN_B_LOC as min_east_b,
    EASTMAX_B_LOC as max_east_b,
    NORTHMIN_B_LOC as min_north_b,
    NORTHMAX_B_LOC as max_north_b,
    get_geophone_coords,
)
from utils_cont_waveform import load_waveform_slice
from utils_satellite import load_maxar_image
from utils_plot import (format_datetime_xlabels, get_geo_component_color, save_figure, 
                        format_east_xlabels, format_north_ylabels, format_datetime_xlabels,
                        component2label)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def plot_subarray_waveforms(ax, waveform_sta_dict, coord_df, starttime, component, scale,
                            linewidth = 1.0,
                            rotation = 15,  
                            color = None,
                            marker_size = 20,
                            **kwargs):
    """
    Plot the waveforms with the true amplitude of a subarray in a time window
    """

    if color is None:
        color = get_geo_component_color(component)

    # Keep only the stations in the subarray
    coord_df = coord_df[coord_df.index.isin(waveform_sta_dict.keys())]

    # Sort the stations by their north coordinate
    coord_df = coord_df.sort_values(by="north")

    event_marker_df = kwargs.get("event_marker", None)
    if event_marker_df is not None:
        plot_marker = True
    else:
        plot_marker = False

    for i, station in enumerate(coord_df.index):

        # Get the waveform
        waveform = waveform_sta_dict[station][component]
        num_pts = waveform.shape[0]
        timeax = date_range(start=starttime, periods=num_pts, freq=f"{1 / sampling_rate}s")

        ax.plot(timeax, waveform * scale + i, color=color, linewidth=linewidth, zorder=1)
        ax.text(timeax[0], i, station, ha="left", va="bottom", fontsize=12, fontweight="bold")

        if plot_marker and station in event_marker_df["station"].values:
            event_marker_sta_df = event_marker_df[event_marker_df["station"] == station]

            for _, row in event_marker_sta_df.iterrows():
                marker_time = row["marker_time"]
                marker_color = row["color"]
                self_match = row["self_match"]
                if self_match:
                    ax.scatter(marker_time, i, marker="o", facecolor=marker_color, edgecolor="none", s=marker_size, zorder=3)
                else:
                    ax.scatter(marker_time, i, marker="o", facecolor=marker_color, edgecolor="none", s=marker_size, zorder=2)

    ax.set_xlim(timeax[0], timeax[-1])

    format_datetime_xlabels(ax,
                            major_tick_spacing = "10s",
                            num_minor_ticks = 10,
                            rotation = rotation, va = "top", ha = "right")
    
    return ax

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("--template_id", type=str, help="Template ID")
parser.add_argument("--starttime_plot", type=Timestamp, help="Start time of the plot")
parser.add_argument("--endtime_plot", type=Timestamp, help="End time of the plot")

parser.add_argument("--subarray", type=str, help="Subarray", default="A")
parser.add_argument("--min_freq_filter", type=float, help="Minimum frequency for filtering the data", default=20.0)
parser.add_argument("--cc_threshold", type=float, help="CC threshold", default=0.85)
parser.add_argument("--num_unmatch", type=int, help="Number of unmatch", default=0)

parser.add_argument("--figwidth", type = float, help="Figure width", default=15.0)
parser.add_argument("--figheight", type = float, help="Figure height", default=10.0)

parser.add_argument("--scale", type = float, help="Scale", default=0.01)
parser.add_argument("--linewidth", type = float, help="Line width", default=1.0)
parser.add_argument("--color_template", type = str, help="Color for the template", default="crimson")
parser.add_argument("--color_match", type = str, help="Color for the match", default="salmon")

parser.add_argument("--margin_x", type = float, help="Margin on the x-axis", default=0.02)
parser.add_argument("--margin_y", type = float, help="Margin on the y-axis", default=0.02)
parser.add_argument("--wspace", type = float, help="Width of the space between the two subplots", default=0.1)

parser.add_argument("--station_size", type = float, help="Size of the station markers", default=150.0)
parser.add_argument("--station_color", type = str, help="Color of the station markers", default="lightgray")

args = parser.parse_args()
template_id = args.template_id
subarray = args.subarray
starttime_plot = args.starttime_plot
endtime_plot = args.endtime_plot
min_freq_filter = args.min_freq_filter
scale = args.scale
linewidth = args.linewidth
color_template = args.color_template
color_match = args.color_match
cc_threshold = args.cc_threshold
num_unmatch = args.num_unmatch
figwidth = args.figwidth
figheight = args.figheight
margin_x = args.margin_x
margin_y = args.margin_y
wspace = args.wspace
station_size = args.station_size
station_color = args.station_color

# Get the stations to plot
if subarray == "A":
    stations_to_plot = inner_stations_a + middle_stations_a
elif subarray == "B":
    stations_to_plot = inner_stations_b + middle_stations_b
else:
    raise ValueError(f"Invalid subarray: {subarray}")

filename_data = f"preprocessed_data_freq{min_freq_filter:.0f}hz.h5"
filepath_data = Path(dirpath_data) / filename_data

# -----------------------------------------------------------------------------
# Load the data
# -----------------------------------------------------------------------------

# Load the template waveform
waveform_sta_dict = {}
for station in stations_to_plot:
    waveform_sta_dict[station] = load_waveform_slice(filepath_data, station, starttime_plot, endtime = endtime_plot)

# Load the matched events
filename = f"matched_events_manual_templates_freq{min_freq_filter:.0f}hz_cc{cc_threshold:.2f}_num_unmatch{num_unmatch}.h5"
filepath = Path(dirpath_det) / filename
matched_events_df = read_hdf(filepath, key=f"template_{template_id}")

# Get the coordinates of the stations
coord_df = get_geophone_coords()

# Load the satellite image
image, extent = load_maxar_image()

# -----------------------------------------------------------------------------
# Plot each component
# -----------------------------------------------------------------------------

# Assemble the event marker dataframe
event_marker_df = matched_events_df.copy().reset_index()
event_marker_df["color"] = event_marker_df["self_match"].map({True: color_template, False: color_match})
event_marker_df.rename(columns={"match_time": "marker_time"}, inplace=True)

# Compute the subplot dimensions
if subarray == "A":
    min_east = min_east_a
    max_east = max_east_a
    min_north = min_north_a
    max_north = max_north_a
    aspect_map = (max_north - min_north) / (max_east - min_east)
else:
    min_east = min_east_b
    max_east = max_east_b
    min_north = min_north_b
    max_north = max_north_b
    aspect_map = (max_north - min_north) / (max_east - min_east)

print(f"Aspect ratio of the map: {aspect_map}")
frac_map_width = (figheight * (1 - 2 * margin_y)) / aspect_map / figwidth
print(f"Fraction of the width of the map: {frac_map_width}")
frac_waveform_width = 1 - 2 * margin_x - frac_map_width

# Loop over the components
for component in components:
    # Create the figure
    fig = figure(figsize=(figwidth, figheight))

    # Generate the subplot for the waveform plot
    ax_waveform = fig.add_axes([margin_x, margin_y, frac_waveform_width, 1 - 2 * margin_y])

    # Plot the waveforms in the time window without the highlights
    plot_subarray_waveforms(ax_waveform, waveform_sta_dict, coord_df, starttime_plot, component, 
                            scale=scale, linewidth=linewidth, event_marker=event_marker_df)

    # Generate the subplot for the map plot
    ax_map = fig.add_axes([margin_x + frac_waveform_width + wspace, margin_y, frac_map_width, 1 - 2 * margin_y])

    # Plot the map
    ax_map.set_xlim(min_east, max_east)
    ax_map.set_ylim(min_north, max_north)

    # Plot the geophone locations
    for station, coords in coord_df.iterrows():
        if station in stations_to_plot:
            east = coords["east"]
            north = coords["north"]

            ax_map.scatter(east, north, marker="^", facecolor=station_color, edgecolor="black", s=station_size)
            ax_map.annotate(station, (east, north), xytext=(0, 10), textcoords="offset points", fontsize=12, fontweight="bold", ha="center", va="bottom")

    # Format the x-axis and y-axis
    format_east_xlabels(ax_map,
                        major_tick_spacing = 10.0,
                        num_minor_ticks = 5)
    
    format_north_ylabels(ax_map,
                         major_tick_spacing = 10.0,
                         num_minor_ticks = 5)
    
    # Set the title
    fig.suptitle(f"Template {template_id}, {component2label(component)}", fontsize=16, fontweight="bold", y=1.03)

    
    # Save the figure
    save_figure(fig, f"template_and_matches_on_subarray_{subarray}_{component.lower()}.png", dpi=300)
                            

