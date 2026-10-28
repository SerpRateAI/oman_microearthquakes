"""
Plot the geophone station map with total repeating STA/LTA detection counts in color (AGU Advances).
"""

from argparse import ArgumentParser
from pathlib import Path
from numpy import clip, log10, nan_to_num
from pandas import read_csv
from matplotlib.pyplot import figure
from matplotlib.colors import LogNorm
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable

from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix
from utils_basic import (
    EASTMIN_WHOLE as min_east,
    EASTMAX_WHOLE as max_east,
    NORTHMIN_WHOLE as min_north,
    NORTHMAX_WHOLE as max_north,
    DETECTION_DIR as dirpath,
    PLOTTING_DIR,
    get_geophone_coords,
    get_freq_limits_string,
)
from utils_plot import save_figure, format_east_xlabels, format_north_ylabels

DEFAULT_STATION_LABEL_COORDS_CSV = Path(PLOTTING_DIR) / "station_label_coords_repeating_detections.csv"

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def load_station_label_coords(filepath):
    """Return label coordinates indexed by station name."""
    df = read_csv(filepath)
    required = {"label", "east", "north", "va", "ha"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {filepath}; got {list(df.columns)}"
        )
    return df.set_index("label")


def plot_station_labels_with_leader_lines(
    ax,
    station_df,
    label_coords_df,
    *,
    station_label_size,
    linewidth_label,
):
    """Plot station labels at specified coordinates with leader lines to stations."""
    station_coords = station_df.set_index("station")
    for station, label_row in label_coords_df.iterrows():
        if station not in station_coords.index:
            continue
        station_east = float(station_coords.loc[station, "east"])
        station_north = float(station_coords.loc[station, "north"])
        label_east = float(label_row["east"])
        label_north = float(label_row["north"])
        ax.plot(
            [label_east, station_east],
            [label_north, station_north],
            color="black",
            linewidth=linewidth_label,
            zorder=3.5,
            clip_on=False,
        )
        ax.text(
            label_east,
            label_north,
            station,
            fontsize=station_label_size,
            fontweight="bold",
            color="black",
            ha=str(label_row["ha"]),
            va=str(label_row["va"]),
            zorder=5,
            clip_on=False,
        )


def log_scaled_marker_sizes(num_detections, size_min, size_max, log_vmin, log_vmax):
    """Marker areas linear in log10(detection count), using a shared log range."""
    counts = num_detections.astype(float)
    counts = nan_to_num(counts, nan=10.0 ** log_vmin)
    counts = clip(counts, 10.0 ** log_vmin, 10.0 ** log_vmax)
    log_vals = log10(counts)
    if log_vmax <= log_vmin:
        return size_min + 0.5 * (size_max - size_min)
    frac = clip((log_vals - log_vmin) / (log_vmax - log_vmin), 0.0, 1.0)
    return size_min + frac * (size_max - size_min)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--min_freq_filter", type=float, default=20.0, help="The minimum frequency for the filter")
    parser.add_argument("--max_freq_filter", type=float, default=None, help="The maximum frequency for the filter")
    parser.add_argument("--sta_window_sec", type=float, default=0.005, help="The STA window length in seconds")
    parser.add_argument("--lta_window_sec", type=float, default=0.05, help="The LTA window length in seconds")
    parser.add_argument("--on_threshold", type=float, default=4.0, help="The on threshold")
    parser.add_argument("--off_threshold", type=float, default=1.0, help="The off threshold")
    parser.add_argument("--min_cc", type=float, default=0.85, help="The minimum correlation coefficient")
    parser.add_argument("--min_num_similar", type=int, default=10, help="The minimum number of similar snippets")
    parser.add_argument("--axis_label_size", type=int, default=12, help="The size of the axis labels")
    parser.add_argument(
        "--station_label_coords_csv",
        type=str,
        default=str(DEFAULT_STATION_LABEL_COORDS_CSV),
        help="CSV with station label coordinates (columns: label, east, north, va, ha)",
    )
    parser.add_argument(
        "--station_label_size",
        type=float,
        default=10.0,
        help="Font size of geophone station labels",
    )
    parser.add_argument(
        "--linewidth_station_label",
        type=float,
        default=1.0,
        help="Line width for station label leader lines",
    )
    parser.add_argument("--title_size", type=int, default=14, help="The size of the title")
    parser.add_argument("--min_count", type=int, default=1e3, help="The minimum count")
    parser.add_argument("--max_count", type=int, default=1e5, help="The maximum count")
    parser.add_argument("--marker_size_min", type=float, default=30.0, help="Minimum marker area for log-scaled sizes")
    parser.add_argument("--marker_size_max", type=float, default=300.0, help="Maximum marker area for log-scaled sizes")
    parser.add_argument("--figwidth", type=float, default=10, help="The width of the figure")
    parser.add_argument("--margin_x", type=float, default=0.02, help="The margin on the x-axis")
    parser.add_argument("--margin_y", type=float, default=0.02, help="The margin on the y-axis")

    args = parser.parse_args()
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    on_threshold = args.on_threshold
    off_threshold = args.off_threshold
    min_cc = args.min_cc
    min_num_similar = args.min_num_similar
    axis_label_size = args.axis_label_size
    station_label_coords_csv = args.station_label_coords_csv
    station_label_size = args.station_label_size
    linewidth_station_label = args.linewidth_station_label
    title_size = args.title_size
    min_count = args.min_count
    max_count = args.max_count
    marker_size_min = args.marker_size_min
    marker_size_max = args.marker_size_max
    figwidth = args.figwidth
    margin_x = args.margin_x
    margin_y = args.margin_y

    print("Plotting the map of the stations with the total number of repeating STA/LTA detections shown in color")

    station_df = get_geophone_coords()
    station_df["station"] = station_df.index

    freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
    sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, on_threshold, off_threshold)
    repeating_snippet_suffix = get_repeating_snippet_suffix(min_cc, min_num_similar)
    filename = (
        f"total_repeating_sta_lta_detection_numbers_{freq_str}_{sta_lta_suffix}_"
        f"{repeating_snippet_suffix}.csv"
    )
    filepath = Path(dirpath) / filename
    detection_df = read_csv(filepath)

    station_df = station_df.merge(detection_df, on="station", how="left")
    missing = station_df["num_of_detections"].isna()
    if missing.any():
        missing_stations = station_df.loc[missing, "station"].tolist()
        print(f"Skipping {len(missing_stations)} stations with missing detection counts: {missing_stations}")
        station_df = station_df.loc[~missing].copy()

    aspect_ratio_map = (max_north - min_north) / (max_east - min_east)
    figheight = figwidth * (1 - 2 * margin_x) * aspect_ratio_map / (1 - 2 * margin_y)

    fig = figure(figsize=(figwidth, figheight))
    ax = fig.add_axes([margin_x, margin_y, 1 - 2 * margin_x, 1 - 2 * margin_y])
    ax.set_facecolor("lightgray")

    cmap = colormaps["hot"]
    norm = LogNorm(vmin=min_count, vmax=max_count)
    marker_sizes = log_scaled_marker_sizes(
        station_df["num_of_detections"],
        marker_size_min,
        marker_size_max,
        log10(min_count),
        log10(max_count),
    )

    ax.scatter(
        station_df["east"],
        station_df["north"],
        c=station_df["num_of_detections"],
        cmap=cmap,
        norm=norm,
        s=marker_sizes,
        marker="^",
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )

    label_coords_df = load_station_label_coords(station_label_coords_csv)
    plot_station_labels_with_leader_lines(
        ax,
        station_df,
        label_coords_df,
        station_label_size=station_label_size,
        linewidth_label=linewidth_station_label,
    )

    format_east_xlabels(ax, major_tick_spacing=50, num_minor_ticks=5, axis_label_size=axis_label_size)
    format_north_ylabels(ax, major_tick_spacing=50, num_minor_ticks=5, axis_label_size=axis_label_size)
    ax.set_aspect("equal")

    ax.set_title("Total number of repeating STA/LTA detections", fontsize=title_size, fontweight="bold")

    cax = fig.add_axes([0.15, 0.05, 0.02, 0.2])
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(mappable, cax=cax, orientation="vertical")
    cbar.set_label("Number of detections", fontsize=axis_label_size)

    save_figure(
        fig,
        f"aguadv_total_repeating_detection_number_station_map.png"
    )
