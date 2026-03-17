"""
This script plots the geo station maps with the hourly detection counts shown in color and the hourly detection counts of three stations below the map.
"""

#--------------------------------
# Imports
#--------------------------------
from argparse import ArgumentParser
from pathlib import Path
from matplotlib.pyplot import close, figure
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from pandas import read_csv, concat
from tqdm import tqdm

from utils_plot import (
    add_day_night_shading,
    save_figure,
)

from utils_basic import (
    DETECTION_DIR as dirpath,
    EASTMIN_WHOLE as min_east,
    EASTMAX_WHOLE as max_east,
    NORTHMIN_WHOLE as min_north,
    NORTHMAX_WHOLE as max_north,
    STARTTIME_GEO as starttime_geo,
    ENDTIME_GEO as endtime_geo,
    get_geophone_coords,
    get_freq_limits_string,
)
from utils_sta_lta import get_sta_lta_suffix

#--------------------------------
# Functions
#--------------------------------

def get_subplots(margin_x = 0.05, margin_y = 0.05, fig_width = 10, hspace = 0.05, map_height_frac = 0.7):
    """
    Get the plot dimensions
    """
    # Get the figure dimensions
    map_ratio = (max_north - min_north) / (max_east - min_east)
    map_width = fig_width * (1 - 2 * margin_x)
    map_height = map_ratio * map_width
    fig_height = map_height / map_height_frac

    # Create the figure
    fig = figure(figsize=(fig_width, fig_height))
    
    # Add the count subplots
    count_height_frac = 1 - 2 * margin_y - hspace - map_height_frac
    map_width_frac = 1 - 2 * margin_x
    ax_count = fig.add_axes([margin_x, margin_y, map_width_frac, count_height_frac])

    # Add the map subplot
    ax_map = fig.add_axes([margin_x, margin_y + count_height_frac + hspace, map_width_frac, map_height_frac])

    return fig, ax_count, ax_map
    
#--------------------------------
# Main
#--------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--min_freq_filter", type=float, default=20)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--on_threshold", type=float, default=4.0)
    parser.add_argument("--off_threshold", type=float, default=1.0)
    parser.add_argument("--max_count_cmap", type=int, default=1000)
    parser.add_argument("--max_count_curve", type=int, default=3000)
    parser.add_argument("--stations_to_highlight", type=str, nargs="+", default=["A04", "B04", "B13"])
    parser.add_argument("--colormap_name", type=str, default="Accent")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    on_threshold = args.on_threshold
    off_threshold = args.off_threshold
    max_count_cmap = args.max_count_cmap
    max_count_curve = args.max_count_curve
    stations_to_highlight = args.stations_to_highlight
    colormap_name = args.colormap_name
    test = args.test

    # Get the frequency limits string
    freq_limits_string = get_freq_limits_string(min_freq_filter, max_freq_filter)
    sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, on_threshold, off_threshold)

    # Get the geophone coordinates
    coords_df = get_geophone_coords()

    # Read in the data
    print("Reading in the data...")
    count_dfs = []
    for station in coords_df.index:
        filename = f"sta_lta_detections_{freq_limits_string}_{sta_lta_suffix}_{station}.csv"
        filepath = Path(dirpath) / filename
        count_df = read_csv(filepath, parse_dates=["starttime", "endtime"])
        count_df["station"] = station
        count_dfs.append(count_df)

    # Concatenate the dataframes
    print("Concatenating the dataframes...")
    count_all_sta_df = concat(count_dfs, ignore_index=True)

    # Group the counts by hour and station
    print("Grouping the counts by hour and station...")
    count_grouped_df = count_all_sta_df.groupby("hour")

    # Get the color map for the stations
    print("Getting the color map...")
    cmap_stations = colormaps[colormap_name]

    # Get the color map for the counts
    print("Getting the color map...")
    cmap_count = colormaps["hot"]
    norm_count = LogNorm(vmin=1, vmax=max_count_cmap, clip=True)

    # Make the plots for each hour
    for i_hour, (hour, hour_count_df) in tqdm(enumerate(count_grouped_df), disable=test):
        # Get the subplots
        fig, ax_count, ax_map = get_subplots()
        ax_map.set_facecolor("lightgray")

        # Plot the counts
        for i_station, station in enumerate(stations_to_highlight):
            count_sta_df = count_all_sta_df.loc[count_all_sta_df["station"] == station]
            color_station = cmap_stations(i_station)
            ax_count.plot(count_sta_df["hour"], count_sta_df["count"], label=station, color=color_station, linewidth=2)
            add_day_night_shading(ax_count)

        ax_count.axvline(x=hour, color="black", linewidth=2)

        ax_count.set_yscale("log")
        ax_count.set_xlim(starttime_geo, endtime_geo)
        ax_count.set_ylim(1, max_count_curve)
        ax_count.set_xlabel("Time (UTC)")
        ax_count.set_ylabel("Count")
        ax_count.legend(loc="upper right")

        # Plot the map
        i_highlight = 0
        for station in hour_count_df["station"].unique():
            count = hour_count_df.loc[hour_count_df["station"] == station, "count"].iloc[0]
            east = coords_df.loc[station, "east"]
            north = coords_df.loc[station, "north"]

            if station in stations_to_highlight:
                color_station = cmap_stations(i_highlight)
                i_highlight += 1

                ax_map.scatter(east, north, c=count, cmap=cmap_count, norm=norm_count, s=100, marker="^", edgecolors=color_station, linewidths=3)
            else:
                ax_map.scatter(east, north, c=count, cmap=cmap_count, norm=norm_count, s=100, marker="^", edgecolors="black", linewidths=0.5)

        ax_map.set_xlim(min_east, max_east)
        ax_map.set_ylim(min_north, max_north)
        ax_map.set_aspect("equal")
        ax_map.set_title(f"{hour.strftime('%Y-%m-%d %H:%M:%S')}", fontweight="bold")
        ax_map.set_xlabel("East (m)")
        ax_map.set_ylabel("North (m)")

        # Add a color bar
        cax = ax_map.inset_axes([0.05, 0.05, 0.03, 0.3])
        fig.colorbar(mappable=ax_map.collections[0], cax=cax, label="Count")

        filename = f"geo_station_map_hourly_detection_counts_{freq_limits_string}_{hour.strftime('%Y%m%d%H%M%S')}.png"
        save_figure(fig, filename)
        close()

        if test and i_hour > 10:
            break

if __name__ == "__main__":
    main()