"""
Plot magnitude evolution and magnitude-frequency relations for two event groups (AGU Advances).

One column per group (default: group 18 in Subarray A, group 20 in Subarray B). The top
row shows log magnitude versus first-onset time over the full span and the middle row a
48-hour window starting 2020-01-16, both with time increasing downward and nights shaded;
crimson lines in the top row mark the window. The bottom row shows the inverse cumulative
magnitude-frequency distribution with the maximum-likelihood b value, as in
plot_aguadv_event_group_magnitude_freq.py. All panels share the magnitude axis.
"""

from argparse import ArgumentParser
from os.path import join

from matplotlib.pyplot import subplots
from pandas import Timedelta, to_datetime

from plot_aguadv_event_group_magnitude_freq import (
    load_params_csv,
    plot_magnitude_freq_panel,
    select_group_params,
)
from plot_event_group_magnitude_evolution import (
    DEFAULT_ZOOM_HOURS,
    DEFAULT_ZOOM_START,
    MARKER_ALPHA,
    MARKER_SIZE,
    PAD_FRACTION,
    SUBARRAY_COLORS,
    WINDOW_LINE_COLOR,
    WINDOW_LINE_WIDTH,
    load_log_magnitudes,
    padded_limits,
    panel_label,
)
from utils_basic import (
    DETECTION_DIR as dirpath_event,
    PLOTTING_DIR,
    get_freq_limits_string,
    get_sunrise_sunset_times,
)
from utils_cc import get_repeating_snippet_suffix
from utils_plot import add_day_night_shading, format_datetime_ylabels, save_figure
from utils_sta_lta import get_sta_lta_suffix

DEFAULT_GROUP_LABELS = (18, 20)
DEFAULT_FIGNAME = "aguadv_magnitude_evolution_n_frequency_relation.png"
PANEL_LABEL_X_LEFT = -0.40
PANEL_LABEL_X_RIGHT = -0.12


def plot_rotated_magnitude_panel(ax, onset_times, log_mags, color):
    """Scatter one group's events with magnitude on x and time on y."""
    ax.scatter(
        log_mags,
        onset_times,
        s=MARKER_SIZE,
        color=color,
        alpha=MARKER_ALPHA,
        linewidths=0,
        zorder=2,
    )


def mark_zoom_window(ax, zoom_start, zoom_end):
    """Mark the zoom window with horizontal lines."""
    for window_time in (zoom_start, zoom_end):
        ax.axhline(
            window_time,
            color=WINDOW_LINE_COLOR,
            linewidth=WINDOW_LINE_WIDTH,
            zorder=3,
        )


def add_panel_label(ax, panel_index, x_label):
    """Write the panel label above the top-left corner of the axes."""
    ax.text(
        x_label,
        1.03,
        panel_label(panel_index),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )


def main():
    parser = ArgumentParser(
        description=(
            "Plot magnitude evolution and magnitude-frequency relations for two event groups."
        ),
    )
    parser.add_argument(
        "--group_labels",
        type=int,
        nargs="+",
        default=list(DEFAULT_GROUP_LABELS),
        help="Two event group labels, left column first (default: 18 20)",
    )
    parser.add_argument(
        "--params_csv",
        type=str,
        default=join(PLOTTING_DIR, "magnitude_number_plot.csv"),
        help="CSV with per-group subarray and magnitude of completeness (min_magnitude)",
    )
    parser.add_argument("--bin_size", type=float, default=0.05, help="Histogram bin width")
    parser.add_argument(
        "--zoom_start",
        type=str,
        default=DEFAULT_ZOOM_START,
        help="Start of the middle-row window (UTC)",
    )
    parser.add_argument(
        "--zoom_hours",
        type=float,
        default=DEFAULT_ZOOM_HOURS,
        help="Length of the middle-row window in hours",
    )
    parser.add_argument(
        "--min_mag_plot",
        type=float,
        default=2.0,
        help="Lower limit of the shared magnitude axis",
    )
    parser.add_argument(
        "--max_mag_plot",
        type=float,
        default=4.0,
        help="Upper limit of the shared magnitude axis",
    )
    parser.add_argument(
        "--figname",
        type=str,
        default=DEFAULT_FIGNAME,
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
    args = parser.parse_args()

    group_labels = args.group_labels
    params_csv = args.params_csv
    bin_size = args.bin_size
    zoom_start = args.zoom_start
    zoom_hours = args.zoom_hours
    min_mag_plot = args.min_mag_plot
    max_mag_plot = args.max_mag_plot
    figname = args.figname
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
            f"Expected two group labels for the two columns; got {group_labels}."
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

    print("--------------------------------")
    print("Plotting AGU Advances magnitude evolution and magnitude-frequency figure...")
    print("--------------------------------")
    print(f"Group labels: {group_labels}")
    print(f"Parameter file: {params_csv}")
    print(f"Bin size: {bin_size}")
    print(f"Zoom window: {zoom_start} to {zoom_end}")
    print(f"Suffix group: {suffix_group}")

    params_df = load_params_csv(params_csv)
    selected_rows = select_group_params(params_df, group_labels)

    group_frames = []
    for row in selected_rows:
        group_label = int(row["group_label"])
        subarray = str(row["subarray"])
        print(f"Loading group {group_label} (Subarray {subarray})...")
        mag_df = load_log_magnitudes(group_label, suffix_group, dirpath_event)
        print(f"  Number of events: {len(mag_df)}")
        group_frames.append((row, mag_df))

    time_min, time_max = padded_limits(
        [mag_df["first_onset"] for _, mag_df in group_frames],
        PAD_FRACTION,
    )
    if min_mag_plot >= max_mag_plot:
        raise ValueError(
            f"min_mag_plot ({min_mag_plot}) must be below max_mag_plot ({max_mag_plot})."
        )
    print(f"Magnitude axis: [{min_mag_plot:.2f}, {max_mag_plot:.2f}]")

    fig, axes = subplots(
        3,
        2,
        figsize=(9, 17),
        sharex=True,
        sharey="row",
        height_ratios=(3, 3, 2),
    )
    for i_col, (row, mag_df) in enumerate(group_frames):
        group_label = int(row["group_label"])
        subarray = str(row["subarray"])
        color = SUBARRAY_COLORS[subarray]
        plot_ylabel = i_col == 0

        ax_full = axes[0, i_col]
        plot_rotated_magnitude_panel(
            ax_full,
            mag_df["first_onset"],
            mag_df["log_magnitude"],
            color,
        )
        mark_zoom_window(ax_full, zoom_start, zoom_end)
        format_datetime_ylabels(
            ax_full,
            plot_axis_label=plot_ylabel,
            plot_tick_label=plot_ylabel,
            major_tick_spacing="5d",
            num_minor_ticks=5,
            date_format="%Y-%m-%d",
            axis_label_size=12,
            tick_label_size=10,
        )
        ax_full.set_title(
            f"Group {group_label}, Subarray {subarray}",
            fontsize=12,
            fontweight="bold",
        )

        ax_zoom = axes[1, i_col]
        plot_rotated_magnitude_panel(
            ax_zoom,
            mag_df["first_onset"],
            mag_df["log_magnitude"],
            color,
        )
        format_datetime_ylabels(
            ax_zoom,
            plot_axis_label=plot_ylabel,
            plot_tick_label=plot_ylabel,
            major_tick_spacing="12h",
            num_minor_ticks=4,
            date_format="%m-%d %H:%M",
            axis_label_size=12,
            tick_label_size=10,
        )

        ax_freq = axes[2, i_col]
        plot_magnitude_freq_panel(
            ax_freq,
            mag_df["log_magnitude"].to_numpy(),
            row,
            bin_size,
            color,
            plot_xlabel=True,
        )
        ax_freq.set_title("")
        if not plot_ylabel:
            ax_freq.set_ylabel("")

        x_label = PANEL_LABEL_X_LEFT if i_col == 0 else PANEL_LABEL_X_RIGHT
        for i_row in range(3):
            add_panel_label(axes[i_row, i_col], i_row * 2 + i_col, x_label)

    # plot_magnitude_freq_panel sets its own magnitude limits, so the shared limits go last.
    axes[0, 0].set_xlim(min_mag_plot, max_mag_plot)
    axes[0, 0].set_ylim(time_max, time_min)
    axes[1, 0].set_ylim(zoom_end, zoom_start)

    print("Adding the day and night shading...")
    sun_df = get_sunrise_sunset_times()
    for ax in axes[:2].flat:
        add_day_night_shading(ax, sun_df=sun_df, axis="y")

    fig.subplots_adjust(hspace=0.12, wspace=0.15)
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
