"""
Plot the hourly counts of the raw and refined snippets
"""

from argparse import ArgumentParser
from pathlib import Path
from numpy import load, full, finfo, maximum
from matplotlib.pyplot import subplots
from time import time

from utils_sta_lta import Snippets, get_sta_lta_suffix
from utils_basic import (
    get_freq_limits_string,
    DETECTION_DIR as dirpath,
    STARTTIME_GEO as starttime_bin,
    ENDTIME_GEO as endtime_bin,
)
from utils_plot import save_figure, add_day_night_shading, format_datetime_xlabels

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def plot_hourly_counts(bin_count_df, color,
                       linewidth = 2, label = None, linestyle = "solid", max_count = 5e3,
                       ax = None):
    """
    Plot the hourly counts of the snippets
    """
    # Create the figure and axes
    if ax is None:
        fig, ax = subplots(figsize=(12, 5))
    else:
        fig = ax.get_figure()

    # Plot the hourly counts
    ax.plot(bin_count_df["bin_center"], bin_count_df["bin_count"], label=label, color=color, linewidth=linewidth, linestyle=linestyle)

    # Set the y-axis to log scale
    ax.set_yscale("log")

    # Set the y-axis limits
    ax.set_ylim(1, max_count)

    # Set the x axis limit
    ax.set_xlim(bin_count_df["bin_center"].min(), bin_count_df["bin_center"].max())


    return fig, ax

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Define the input arguments
    parser = ArgumentParser()
    parser.add_argument("--station", type=str, required=True, help="The station to plot")
    parser.add_argument("--min_freq_filter", type=float, default=20.0, help="The minimum frequency for the filter")
    parser.add_argument("--max_freq_filter", type=float, default=None, help="The maximum frequency for the filter")
    parser.add_argument("--sta_window_sec", type=float, default=0.005, help="The STA window length in seconds")
    parser.add_argument("--lta_window_sec", type=float, default=0.05, help="The LTA window length in seconds")
    parser.add_argument("--on_threshold", type=float, default=4.0, help="The on threshold")
    parser.add_argument("--off_threshold", type=float, default=1.0, help="The off threshold")
    parser.add_argument("--cc_threshold", type=float, default=0.85, help="The threshold")
    parser.add_argument("--min_num_similar", type=int, default=10, help="The minimum number of similar snippets (including itself)")
    parser.add_argument("--axis_label_size", type=int, default=12, help="The size of the axis labels")
    parser.add_argument("--title_size", type=int, default=14, help="The size of the title")
    args = parser.parse_args()
    station = args.station
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    on_threshold = args.on_threshold
    off_threshold = args.off_threshold
    cc_threshold = args.cc_threshold
    min_num_similar = args.min_num_similar
    axis_label_size = args.axis_label_size
    title_size = args.title_size

    print(f"Plotting the hourly counts for station {station}...")

    # Load the raw snippets
    print(f"Loading the raw snippets...")
    freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
    sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, on_threshold, off_threshold)
    filename = f"snippets_sta_lta_{freq_str}_{sta_lta_suffix}_{station}.h5"
    filepath = Path(dirpath) / filename
    snippets = Snippets.from_hdf(filepath)

    # Load the refined snippets
    print(f"Loading the refined snippets...")
    filename = f"snippets_refined_{freq_str}_{sta_lta_suffix}_cc{cc_threshold:.2f}_min_num_similar{min_num_similar:d}.h5"
    filepath = Path(dirpath) / filename
    snippets_refined = Snippets.from_hdf(filepath)

    # Bin the snippets by hour
    print(f"Binning the snippets by hour...")
    snippets_bin_count_df = snippets.bin_by_hour(starttime_bin, endtime_bin)
    snippets_refined_bin_count_df = snippets_refined.bin_by_hour(starttime_bin, endtime_bin)

    # Create the figure and axes
    print(f"Creating the figure and axes...")
    fig, ax = subplots(figsize=(12, 5))

    # Plot the hourly counts
    print(f"Plotting the hourly counts...")
    fig, ax = plot_hourly_counts(snippets_bin_count_df, color="gray", label="Raw", ax=ax)
    fig, ax = plot_hourly_counts(snippets_refined_bin_count_df, color="darkorange", label="Refined", ax=ax)

    # Add the x axis label
    print(f"Adding the x axis label...")
    format_datetime_xlabels(ax, date_format="%Y-%m-%d", major_tick_spacing="5d", num_minor_ticks=5, axis_label_size=axis_label_size)

    # Add the y axis label
    print(f"Adding the y axis label...")
    ax.set_ylabel("Count", fontsize=axis_label_size)

    # Add the day and night shading
    print(f"Adding the day and night shading...")
    add_day_night_shading(ax)

    # Add the legend
    print(f"Adding the legend...")
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right", fontsize=axis_label_size)

    # Add the title
    print(f"Adding the title...")
    ax.set_title(f"{station}", fontsize=title_size, fontweight="bold")

    # Save the figure
    print(f"Saving the figure...")
    save_figure(fig, f"hourly_counts_raw_vs_refined_{freq_str}_{sta_lta_suffix}_cc{cc_threshold:.2f}_min_num_similar{min_num_similar:d}_{station}.png")