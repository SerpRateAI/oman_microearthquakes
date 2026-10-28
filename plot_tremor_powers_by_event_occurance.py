"""
Plot harmonic tremor power vs time for tremor windows with and without impulsive events.

Two vertically stacked panels show tremor power at one station, classified separately
by Subarray A and Subarray B event occurrence within each 300 s tremor window.
Windows without events are shown with solid markers; windows with events use open markers.
"""

from argparse import ArgumentParser
from pathlib import Path

from matplotlib.pyplot import subplots

from compute_mean_tremor_power_by_event_ocurrence import (
    DEFAULT_SUFFIX,
    classify_windows_with_events,
    load_mode_name,
    load_subarray_events,
    load_tremor_powers,
    merge_tremor_with_subarray_events,
)
from utils_basic import DETECTION_DIR, ENDTIME_GEO, SPECTROGRAM_DIR, STARTTIME_GEO
from utils_plot import (
    add_day_night_shading,
    format_datetime_xlabels,
    format_db_ylabels,
    save_figure,
)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot tremor power vs time for tremor windows with and without impulsive "
            "events, separately for subarrays A and B."
        ),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=DEFAULT_SUFFIX,
        help="Suffix for subarray event catalog JSONL files",
    )
    parser.add_argument(
        "--detection-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory containing grouped_events_subarray_*.jsonl files",
    )
    parser.add_argument(
        "--figname",
        type=str,
        default=None,
        help=(
            "Output figure filename "
            "(default: tremor_powers_by_event_occurance_<station>.png)"
        ),
    )
    parser.add_argument(
        "--color_mode_face",
        type=str,
        required=True,
        help="The color for solid markers (windows without events)",
    )
    parser.add_argument(
        "--color_mode_edge",
        type=str,
        required=True,
        help="The color for marker edges and open markers (windows with events)",
    )
    parser.add_argument(
        "--base_mode_name",
        type=str,
        default="PR02549",
        help="The base mode name",
    )
    parser.add_argument(
        "--base_mode_order",
        type=int,
        default=2,
        help="The base mode order",
    )
    parser.add_argument(
        "--mode_order",
        type=int,
        default=3,
        help="The mode order",
    )
    parser.add_argument(
        "--window_length",
        type=float,
        default=300.0,
        help="The window length in seconds",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="The overlap fraction",
    )
    parser.add_argument(
        "--min_prom",
        type=float,
        default=15.0,
        help="The minimum prominence",
    )
    parser.add_argument(
        "--min_rbw",
        type=float,
        default=15.0,
        help="The minimum reverse bandwidth",
    )
    parser.add_argument(
        "--max_mean_db",
        type=float,
        default=15.0,
        help="The maximum mean dB value",
    )
    parser.add_argument(
        "--station",
        type=str,
        required=True,
        help="Station whose harmonic tremor mode power is plotted",
    )
    return parser.parse_args()


def plot_subarray_tremor_panel(
    ax,
    merged_df,
    has_event_mask,
    args,
    subarray,
    mode_name,
    plot_xlabel,
):
    """Plot tremor power with solid (without event) and open (with event) markers."""
    without_df = merged_df.loc[~has_event_mask]
    with_df = merged_df.loc[has_event_mask]

    ax.scatter(
        without_df["time"],
        without_df["total_power"],
        marker="o",
        s=15,
        color=args.color_mode_face,
        edgecolors=args.color_mode_edge,
        label="Without events",
        zorder=2,
    )
    ax.scatter(
        with_df["time"],
        with_df["total_power"],
        marker="o",
        s=15,
        facecolors="none",
        edgecolors="darkgray",
        label="With events",
        zorder=3,
    )

    add_day_night_shading(ax)
    ax.set_xlim(STARTTIME_GEO, ENDTIME_GEO)
    format_datetime_xlabels(
        ax,
        plot_axis_label=plot_xlabel,
        plot_tick_label=plot_xlabel,
        major_tick_spacing="5d",
        num_minor_ticks=4,
        date_format="%Y-%m-%d",
        axis_label_size=12,
        tick_label_size=10,
    )
    format_db_ylabels(ax, axis_label_size=12, tick_label_size=10)
    ax.set_title(
        (
            f"Subarray {subarray}: Mode {args.mode_order} ({mode_name}) power on "
            f"{args.station} (events within {args.window_length:.0f} s window)"
        ),
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=1.0)


def main():
    args = parse_args()

    mode_name = load_mode_name(
        Path(SPECTROGRAM_DIR),
        args.base_mode_name,
        args.base_mode_order,
        args.mode_order,
    )
    print(f"Mode {args.mode_order}: {mode_name}")

    tremor_df = load_tremor_powers(
        Path(SPECTROGRAM_DIR),
        mode_name,
        args.window_length,
        args.overlap,
        args.min_prom,
        args.min_rbw,
        args.max_mean_db,
        args.station,
    )
    print(f"Loaded {len(tremor_df)} tremor windows at station {args.station}")

    fig, axes = subplots(2, 1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1)

    for ax, subarray, plot_xlabel in zip(axes, ("A", "B"), (False, True)):
        print(f"Plotting Subarray {subarray} tremor power...")
        catalog = load_subarray_events(args.detection_dir, subarray, args.suffix)
        merged_df = merge_tremor_with_subarray_events(
            tremor_df, catalog, args.window_length
        )
        has_event_mask = classify_windows_with_events(merged_df)
        plot_subarray_tremor_panel(
            ax,
            merged_df,
            has_event_mask,
            args,
            subarray,
            mode_name,
            plot_xlabel,
        )

    figname = args.figname or f"tremor_powers_by_event_occurance_{args.station}.png"
    save_figure(fig, figname)
    print(f"Saved {figname}")


if __name__ == "__main__":
    main()
