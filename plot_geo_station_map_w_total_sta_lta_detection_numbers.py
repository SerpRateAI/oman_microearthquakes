"""
Plot the map of the stations with the total number of STA/LTA detections shown in color.
"""

from argparse import ArgumentParser
from pathlib import Path
from pandas import read_csv
from matplotlib.pyplot import figure    
from matplotlib.colors import LogNorm
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable

from utils_sta_lta import get_sta_lta_suffix
from utils_basic import (
    EASTMIN_WHOLE as min_east,
    EASTMAX_WHOLE as max_east,
    NORTHMIN_WHOLE as min_north,
    NORTHMAX_WHOLE as max_north,
    DETECTION_DIR as dirpath,
    get_geophone_coords,
    get_freq_limits_string,
)
from utils_plot import save_figure, format_east_xlabels, format_north_ylabels

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--stations_to_highlight", type=str, nargs="+", default=["A04", "B04", "B15"], help="The stations to highlight")
    parser.add_argument("--min_freq_filter", type=float, default=20.0, help="The minimum frequency for the filter")
    parser.add_argument("--max_freq_filter", type=float, default=None, help="The maximum frequency for the filter")
    parser.add_argument("--sta_window_sec", type=float, default=0.005, help="The STA window length in seconds")
    parser.add_argument("--lta_window_sec", type=float, default=0.05, help="The LTA window length in seconds")
    parser.add_argument("--on_threshold", type=float, default=4.0, help="The on threshold")
    parser.add_argument("--off_threshold", type=float, default=1.0, help="The off threshold")
    parser.add_argument("--axis_label_size", type=int, default=12, help="The size of the axis labels")
    parser.add_argument("--station_label_size", type=int, default=12, help="The size of the station labels")
    parser.add_argument("--title_size", type=int, default=14, help="The size of the title")
    parser.add_argument("--min_count", type=int, default=5e3, help="The minimum count")
    parser.add_argument("--max_count", type=int, default=1e5, help="The maximum count")
    parser.add_argument("--figwidth", type=float, default=10, help="The width of the figure")
    parser.add_argument("--margin_x", type=float, default=0.02, help="The margin on the x-axis")
    parser.add_argument("--margin_y", type=float, default=0.02, help="The margin on the y-axis")

    args = parser.parse_args()
    stations_to_highlight = args.stations_to_highlight
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    on_threshold = args.on_threshold
    off_threshold = args.off_threshold
    axis_label_size = args.axis_label_size
    station_label_size = args.station_label_size
    title_size = args.title_size
    min_count = args.min_count
    max_count = args.max_count
    figwidth = args.figwidth
    margin_x = args.margin_x
    margin_y = args.margin_y

    print(f"Plotting the map of the stations with the total number of STA/LTA detections shown in color")

    # Load the station coordinates
    station_df = get_geophone_coords()
    station_df["station"] = station_df.index

    # Get the total number of STA/LTA detections for each station
    freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
    sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, on_threshold, off_threshold)
    filename = f"total_sta_lta_detection_numbers_{freq_str}_{sta_lta_suffix}.csv"
    filepath = Path(dirpath) / filename
    detection_df = read_csv(filepath)

    # Merge the station coordinates and the total number of STA/LTA detections
    station_df = station_df.merge(detection_df, on="station", how="left")

    # Create the figure and axes
    aspect_ratio_map = (max_north - min_north) / (max_east - min_east)
    figheight = figwidth * (1 - 2 * margin_x) * aspect_ratio_map / (1 - 2 * margin_y)

    fig = figure(figsize=(figwidth, figheight))
    ax = fig.add_axes([margin_x, margin_y, 1 - 2 * margin_x, 1 - 2 * margin_y])
    ax.set_facecolor("lightgray")

    # Plot the map
    cmap = colormaps["hot"] 
    norm = LogNorm(vmin=min_count, vmax=max_count)


    print(stations_to_highlight)
    for _, row in station_df.iterrows():
        station = row["station"]
        if station in stations_to_highlight:
            ax.scatter(row["east"], row["north"], c=row["num_of_detections"], cmap=cmap, norm=norm, s=100, marker="^", edgecolors="deepskyblue", linewidths=2)
            ax.annotate(station, (row["east"], row["north"]+3), fontsize=station_label_size, color="black", bbox=dict(facecolor="white", edgecolor="none", alpha=0.5), ha="center", va="bottom")
        else:
            ax.scatter(row["east"], row["north"], c=row["num_of_detections"], cmap=cmap, norm=norm, s=100, marker="^", edgecolors="black", linewidths=0.5)


    # Set the x and y limits
    format_east_xlabels(ax, major_tick_spacing=50, num_minor_ticks=5, axis_label_size=axis_label_size)
    format_north_ylabels(ax, major_tick_spacing=50, num_minor_ticks=5, axis_label_size=axis_label_size)
    ax.set_aspect("equal")

    # Set the title
    ax.set_title(f"Total number of STA/LTA detections", fontsize=title_size, fontweight="bold")

    # Add the colorbar
    cax = fig.add_axes([0.15, 0.05, 0.02, 0.2])
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(mappable, cax=cax, orientation="vertical")
    cbar.set_label("Number of detections", fontsize=axis_label_size)

    # Save the figure
    save_figure(fig, f"station_map_w_total_sta_lta_detection_numbers_{freq_str}_{sta_lta_suffix}.png")