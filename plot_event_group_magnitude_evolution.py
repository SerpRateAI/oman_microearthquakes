"""
Plot log-magnitude versus time for two event groups in Subarrays A and B.

Left column: full time span. Right column: a 48-hour window starting 2020-01-16,
marked by crimson lines on the left. Each event is a dot at its first-onset time
and log magnitude, read from log_magnitudes_group{N}_{suffix}.csv.
"""

from argparse import ArgumentParser
from os.path import isfile, join

from matplotlib.pyplot import subplots
from pandas import Timedelta, Timestamp, read_csv, to_datetime

from utils_basic import DETECTION_DIR as dirpath_event, get_freq_limits_string
from utils_cc import get_repeating_snippet_suffix
from utils_plot import format_datetime_xlabels, save_figure
from utils_sta_lta import get_sta_lta_suffix

SUBARRAY_COLORS = {"A": "dodgerblue", "B": "darkorange"}
DEFAULT_GROUP_LABELS = (18, 20)
DEFAULT_ZOOM_START = "2020-01-16"
DEFAULT_ZOOM_HOURS = 48.0
PAD_FRACTION = 0.02
MARKER_SIZE = 8
MARKER_ALPHA = 0.45
WINDOW_LINE_COLOR = "crimson"
WINDOW_LINE_WIDTH = 1.5


def panel_label(index):
    """Return panel label string for zero-based index, e.g. '(a)'."""
    return f"({chr(ord('a') + index)})"


def load_subarray_by_label(group_info_csv):
    """Map event-group label to subarray from the selected-groups CSV."""
    if not isfile(group_info_csv):
        raise FileNotFoundError(f"Group info file not found: {group_info_csv}")

    group_df = read_csv(group_info_csv)
    required = {"label", "subarray"}
    missing = [col for col in required if col not in group_df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required)} in {group_info_csv}; missing {missing}."
        )

    return {
        int(row["label"]): str(row["subarray"])
        for _, row in group_df.iterrows()
    }


def load_log_magnitudes(group_label, suffix_group, dirpath):
    """Load first-onset times and log magnitudes for one event group."""
    infile = f"log_magnitudes_group{group_label:d}_{suffix_group}.csv"
    infile_path = join(dirpath, infile)
    if not isfile(infile_path):
        raise FileNotFoundError(f"Log magnitudes file not found: {infile_path}")

    df = read_csv(infile_path)
    required = {"first_onset", "log_magnitude"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required)} in {infile_path}; missing {missing}."
        )

    df["first_onset"] = to_datetime(df["first_onset"], utc=True, format="ISO8601")
    df = df.dropna(subset=["first_onset", "log_magnitude"])
    if df.empty:
        raise ValueError(f"No valid onset times and log magnitudes in {infile_path}.")

    return df


def padded_limits(values, pad_fraction):
    """Return min and max expanded by a fraction of the span."""
    vmin = min(series.min() for series in values)
    vmax = max(series.max() for series in values)
    pad = (vmax - vmin) * pad_fraction
    if pad == 0:
        pad = Timedelta(hours=1) if isinstance(vmin, Timestamp) else 0.1
    return vmin - pad, vmax + pad


def plot_magnitude_panel(
    ax,
    onset_times,
    log_mags,
    group_label,
    subarray,
    color,
    panel_index,
    plot_ylabel,
):
    """Scatter one group's events and label the panel."""
    ax.scatter(
        onset_times,
        log_mags,
        s=MARKER_SIZE,
        color=color,
        alpha=MARKER_ALPHA,
        linewidths=0,
        zorder=2,
    )
    if plot_ylabel:
        ax.set_ylabel("Log magnitude", fontsize=12)
    ax.set_title(
        f"Group {group_label}, Subarray {subarray}",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        -0.12,
        1.05,
        panel_label(panel_index),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )


def mark_zoom_window(ax, zoom_start, zoom_end):
    """Mark the zoom window with vertical lines."""
    ax.axvline(
        zoom_start,
        color=WINDOW_LINE_COLOR,
        linewidth=WINDOW_LINE_WIDTH,
        zorder=3,
    )
    ax.axvline(
        zoom_end,
        color=WINDOW_LINE_COLOR,
        linewidth=WINDOW_LINE_WIDTH,
        zorder=3,
    )


def main():
    parser = ArgumentParser(
        description=(
            "Plot log-magnitude versus time for two event groups in Subarrays A and B."
        ),
    )
    parser.add_argument(
        "--group_labels",
        type=int,
        nargs="+",
        default=list(DEFAULT_GROUP_LABELS),
        help="Two event group labels, top panel first (default: 18 20)",
    )
    parser.add_argument(
        "--group_info_csv",
        type=str,
        default=None,
        help="Selected-groups CSV with label and subarray columns",
    )
    parser.add_argument(
        "--figname",
        type=str,
        default="event_group_magnitude_evolution.png",
        help="Output figure filename under FIGURE_DIR",
    )
    parser.add_argument("--min_cc", type=float, default=0.85)
    parser.add_argument("--min_num_similar_snippet", type=int, default=10)
    parser.add_argument("--min_num_similar_station", type=int, default=3)
    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--thr_on", type=float, default=4.0)
    parser.add_argument("--thr_off", type=float, default=1.0)
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    parser.add_argument(
        "--zoom_start",
        type=str,
        default=DEFAULT_ZOOM_START,
        help="Start of the right-hand window (UTC)",
    )
    parser.add_argument(
        "--zoom_hours",
        type=float,
        default=DEFAULT_ZOOM_HOURS,
        help="Length of the right-hand window in hours",
    )
    args = parser.parse_args()

    group_labels = args.group_labels
    group_info_csv = args.group_info_csv
    figname = args.figname
    zoom_start = args.zoom_start
    zoom_hours = args.zoom_hours
    min_cc = args.min_cc
    min_num_similar_snippet = args.min_num_similar_snippet
    min_num_similar_station = args.min_num_similar_station
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    thr_on = args.thr_on
    thr_off = args.thr_off
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter

    if len(group_labels) != 2:
        raise ValueError(
            f"Expected two group labels for the two rows; got {group_labels}."
        )
    if zoom_hours <= 0:
        raise ValueError(f"zoom_hours must be positive; got {zoom_hours}.")

    zoom_start = to_datetime(zoom_start, utc=True)
    zoom_end = zoom_start + Timedelta(hours=zoom_hours)

    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=min_cc,
        min_num_similar=min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

    if group_info_csv is None:
        group_info_csv = join(dirpath_event, f"event_group_info_selected_{suffix_group}.csv")

    print("--------------------------------")
    print("Plotting event group magnitude evolution...")
    print("--------------------------------")
    print(f"Group labels: {group_labels}")
    print(f"Group info: {group_info_csv}")
    print(f"Suffix group: {suffix_group}")
    print(f"Zoom window: {zoom_start} to {zoom_end}")

    subarray_by_label = load_subarray_by_label(group_info_csv)
    group_frames = []
    for group_label in group_labels:
        if group_label not in subarray_by_label:
            raise ValueError(
                f"Group {group_label} not found in {group_info_csv}; "
                f"available groups: {sorted(subarray_by_label)}."
            )
        subarray = subarray_by_label[group_label]
        if subarray not in SUBARRAY_COLORS:
            raise ValueError(f"Unsupported subarray {subarray!r} for group {group_label}.")

        print(f"Loading group {group_label} (Subarray {subarray})...")
        mag_df = load_log_magnitudes(group_label, suffix_group, dirpath_event)
        print(f"  Number of events: {len(mag_df)}")
        group_frames.append((group_label, subarray, mag_df))

    time_min, time_max = padded_limits(
        [mag_df["first_onset"] for _, _, mag_df in group_frames],
        PAD_FRACTION,
    )
    mag_min, mag_max = padded_limits(
        [mag_df["log_magnitude"] for _, _, mag_df in group_frames],
        PAD_FRACTION,
    )

    fig, axes = subplots(2, 2, figsize=(16, 8), sharex="col", sharey=True)
    for i_row, (group_label, subarray, mag_df) in enumerate(group_frames):
        for i_col in (0, 1):
            ax = axes[i_row, i_col]
            plot_magnitude_panel(
                ax,
                mag_df["first_onset"],
                mag_df["log_magnitude"],
                group_label,
                subarray,
                SUBARRAY_COLORS[subarray],
                i_row * 2 + i_col,
                plot_ylabel=(i_col == 0),
            )
            if i_col == 0:
                mark_zoom_window(ax, zoom_start, zoom_end)
                format_datetime_xlabels(
                    ax,
                    plot_axis_label=(i_row == 1),
                    plot_tick_label=(i_row == 1),
                    major_tick_spacing="5d",
                    num_minor_ticks=5,
                    date_format="%Y-%m-%d",
                    axis_label_size=12,
                    tick_label_size=10,
                )
            else:
                format_datetime_xlabels(
                    ax,
                    plot_axis_label=(i_row == 1),
                    plot_tick_label=(i_row == 1),
                    major_tick_spacing="12h",
                    num_minor_ticks=4,
                    date_format="%m-%d %H:%M",
                    axis_label_size=12,
                    tick_label_size=10,
                )

    axes[0, 0].set_xlim(time_min, time_max)
    axes[0, 1].set_xlim(zoom_start, zoom_end)
    axes[0, 0].set_ylim(mag_min, mag_max)
    fig.subplots_adjust(hspace=0.25, wspace=0.18)
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
