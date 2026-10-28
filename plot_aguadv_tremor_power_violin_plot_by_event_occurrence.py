"""
Plot violin distributions of harmonic tremor power for windows with and without events.

A single panel compares Subarray A (blue) and Subarray B (orange) tremor power
distributions for tremor windows without impulsive events vs windows with at least
one event within the 300 s analysis window.
"""

from argparse import ArgumentParser
from pathlib import Path

from numpy import amax, amin, median
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.pyplot import subplots

from compute_mean_tremor_power_by_event_ocurrence import (
    DEFAULT_SUFFIX,
    classify_windows_with_events,
    load_mode_name,
    load_subarray_events,
    load_tremor_powers,
    merge_tremor_with_subarray_events,
)
from utils_basic import DETECTION_DIR, SPECTROGRAM_DIR
from utils_plot import format_db_ylabels, save_figure

SUBARRAY_COLORS = {"A": "dodgerblue", "B": "darkorange"}
VIOLIN_ALPHA = 0.5
VIOLIN_WIDTH = 0.25
VIOLIN_EDGE_LINEWIDTH = 1.0
MEDIAN_LINEWIDTH = 2.5
MEAN_LINEWIDTH = 2.5
GROUP_CENTERS = {"without_event": 1.0, "with_event": 2.0}
SUBARRAY_OFFSETS = {"A": -0.15, "B": 0.15}


def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot violin distributions of tremor power for windows with and "
            "without impulsive events, comparing subarrays A and B."
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
            "(default: aguadv_tremor_power_violin_plot_by_event_occurrence_<station>.png)"
        ),
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


def collect_power_by_subarray_and_category(
    tremor_df, detection_dir, suffix, window_length
):
    """Return total_power arrays keyed by (subarray, has_event)."""
    groups = {}
    for subarray in ("A", "B"):
        catalog = load_subarray_events(detection_dir, subarray, suffix)
        merged_df = merge_tremor_with_subarray_events(
            tremor_df, catalog, window_length
        )
        has_event_mask = classify_windows_with_events(merged_df)
        groups[(subarray, False)] = merged_df.loc[
            ~has_event_mask, "total_power"
        ].values
        groups[(subarray, True)] = merged_df.loc[
            has_event_mask, "total_power"
        ].values
    return groups


def subarray_rgba(color):
    """Return a subarray color with the same alpha used for violin fills."""
    return to_rgba(color, VIOLIN_ALPHA)


def style_violin_parts(parts, color):
    """Apply subarray color and transparency to one violinplot result."""
    rgba = subarray_rgba(color)
    for body in parts["bodies"]:
        body.set_facecolor(rgba)
        body.set_edgecolor(rgba)
        body.set_linewidth(VIOLIN_EDGE_LINEWIDTH)


def draw_violin_extrema(ax, position, powers, color):
    """Draw the vertical center line and horizontal min/max caps."""
    vmin, vmax = amin(powers), amax(powers)
    half_width = VIOLIN_WIDTH / 2
    line_color = to_rgba(color, 1.0)
    ax.vlines(
        position,
        vmin,
        vmax,
        colors=[line_color],
        linewidth=VIOLIN_EDGE_LINEWIDTH,
        zorder=3,
    )
    ax.hlines(
        vmin,
        position - half_width,
        position + half_width,
        colors=[line_color],
        linewidth=VIOLIN_EDGE_LINEWIDTH,
        zorder=3,
    )
    ax.hlines(
        vmax,
        position - half_width,
        position + half_width,
        colors=[line_color],
        linewidth=VIOLIN_EDGE_LINEWIDTH,
        zorder=3,
    )


def plot_horizontal_marker(ax, y, position, color, linestyle, linewidth):
    """Draw a horizontal marker spanning the violin width."""
    half_width = VIOLIN_WIDTH / 2
    ax.hlines(
        y,
        position - half_width,
        position + half_width,
        colors=color,
        linewidth=linewidth,
        linestyles=linestyle,
        zorder=4,
    )


def plot_tremor_power_violins(ax, groups, mode_order, mode_name, station, window_length):
    """Draw grouped violins for subarrays A and B with/without events."""
    plot_specs = [
        ("without_event", False),
        ("with_event", True),
    ]
    annotations = []

    for category_key, has_event in plot_specs:
        center = GROUP_CENTERS[category_key]
        for subarray in ("A", "B"):
            position = center + SUBARRAY_OFFSETS[subarray]
            powers = groups[(subarray, has_event)]
            color = SUBARRAY_COLORS[subarray]
            if len(powers) == 0:
                annotations.append((position, 0, color))
                continue
            parts = ax.violinplot(
                [powers],
                positions=[position],
                widths=VIOLIN_WIDTH,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            style_violin_parts(parts, color)
            draw_violin_extrema(ax, position, powers, color)
            plot_horizontal_marker(
                ax,
                median(powers),
                position,
                color,
                linestyle="-",
                linewidth=MEDIAN_LINEWIDTH,
            )
            plot_horizontal_marker(
                ax,
                powers.mean(),
                position,
                color,
                linestyle="--",
                linewidth=MEAN_LINEWIDTH,
            )
            annotations.append((position, len(powers), color))

    ymin, ymax = ax.get_ylim()
    label_y = ymin + 0.01 * (ymax - ymin)
    for position, count, color in annotations:
        ax.text(
            position,
            label_y,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )

    ax.set_xticks([GROUP_CENTERS["without_event"], GROUP_CENTERS["with_event"]])
    ax.set_xticklabels(["Without", "With"], fontsize=10)
    ax.set_xlabel("Event occurrence", fontsize=12)
    format_db_ylabels(ax, axis_label_size=12, tick_label_size=10)
    ax.set_title(
        f"Mode {mode_order} power by event occurrence",
        fontweight="bold",
    )
    legend_handles = [
        Patch(facecolor=SUBARRAY_COLORS["A"], edgecolor=SUBARRAY_COLORS["A"], alpha=VIOLIN_ALPHA, label="Subarray A"),
        Patch(facecolor=SUBARRAY_COLORS["B"], edgecolor=SUBARRAY_COLORS["B"], alpha=VIOLIN_ALPHA, label="Subarray B"),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=MEDIAN_LINEWIDTH,
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=MEAN_LINEWIDTH,
            label="Mean",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=1.0)


def main():
    args = parse_args()
    suffix = args.suffix
    detection_dir = args.detection_dir
    figname = args.figname
    base_mode_name = args.base_mode_name
    base_mode_order = args.base_mode_order
    mode_order = args.mode_order
    window_length = args.window_length
    overlap = args.overlap
    min_prom = args.min_prom
    min_rbw = args.min_rbw
    max_mean_db = args.max_mean_db
    station = args.station

    mode_name = load_mode_name(
        Path(SPECTROGRAM_DIR),
        base_mode_name,
        base_mode_order,
        mode_order,
    )
    print(f"Mode {mode_order}: {mode_name}")

    tremor_df = load_tremor_powers(
        Path(SPECTROGRAM_DIR),
        mode_name,
        window_length,
        overlap,
        min_prom,
        min_rbw,
        max_mean_db,
        station,
    )
    print(f"Loaded {len(tremor_df)} tremor windows at station {station}")

    groups = collect_power_by_subarray_and_category(
        tremor_df,
        detection_dir,
        suffix,
        window_length,
    )
    for subarray in ("A", "B"):
        for has_event, label in ((False, "without_event"), (True, "with_event")):
            print(
                f"Subarray {subarray} {label}: n={len(groups[(subarray, has_event)])}"
            )

    fig, ax = subplots(1, 1, figsize=(8, 6))
    plot_tremor_power_violins(
        ax,
        groups,
        mode_order,
        mode_name,
        station,
        window_length,
    )

    figname = (
        figname
        or f"aguadv_tremor_power_violin_plot_by_event_occurrence_{station}.png"
    )
    save_figure(fig, figname)
    print(f"Saved {figname}")


if __name__ == "__main__":
    main()
