"""
Plot event-depth histograms for Subarrays A and B (AGU Advances).

Each selected group's member events are counted at the hub-event depth evaluated
at the preferred velocity scale factor for that subarray.
"""

from argparse import ArgumentParser
from pathlib import Path

from matplotlib.pyplot import subplots
from numpy import arange, histogram
from pandas import read_csv

from extract_hub_event_group_locations import (
    DEFAULT_SUFFIX,
    load_preferred_scale_factors,
)
from utils_basic import DETECTION_DIR, LOC_DIR, VEL_MODEL_DIR
from utils_loc import load_hub_event_location_from_hdf
from utils_plot import save_figure

SUBARRAY_COLORS = {"A": "dodgerblue", "B": "darkorange"}
SUBARRAYS = ("A", "B")
PANEL_LABELS = {"A": "(a)", "B": "(b)"}
MIN_DEPTH = 0.0
MAX_DEPTH = 15.0
BIN_WIDTH = 1.0
PANEL_LABEL_FONTSIZE = 14
DEFAULT_FIGNAME = "aguadv_event_depth_histograms.png"


def load_group_info(filepath):
    """Load selected event groups and require label, subarray, and num_events."""
    df = read_csv(filepath)
    required = {"label", "subarray", "num_events"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {filepath}; got {list(df.columns)}"
        )
    return df


def collect_event_depths(group_df, preferred_scale_by_subarray, loc_dir):
    """Return hub depths and event-count weights for each subarray."""
    depths = {subarray: [] for subarray in SUBARRAYS}
    weights = {subarray: [] for subarray in SUBARRAYS}

    for _, row in group_df.iterrows():
        label = int(row["label"])
        subarray = str(row["subarray"])
        num_events = int(row["num_events"])
        if subarray not in SUBARRAYS:
            raise ValueError(f"Unsupported subarray {subarray!r} for group {label:d}")
        if subarray not in preferred_scale_by_subarray:
            raise ValueError(
                f"No preferred scale factor for subarray {subarray!r} "
                f"(group {label:d}); available: {sorted(preferred_scale_by_subarray)}"
            )

        scale_factor_preferred = preferred_scale_by_subarray[subarray]
        filepath_h5 = loc_dir / f"hub_event_location_info_group{label:d}.h5"
        if not filepath_h5.is_file():
            raise FileNotFoundError(
                f"Missing localization HDF5 for group {label:d}: {filepath_h5}"
            )

        location_dict, scale_factor_used, _weight = load_hub_event_location_from_hdf(
            filepath_h5, scale_factor_preferred,
        )
        depth = float(location_dict["depth"])
        print(
            f"Group {label:d} (Subarray {subarray}): "
            f"scale {scale_factor_used:.4f}, depth {depth:.2f} m, "
            f"{num_events:d} events"
        )
        depths[subarray].append(depth)
        weights[subarray].append(num_events)

    return depths, weights


def add_panel_label(ax, label):
    """Add an AGU Advances-style panel label."""
    ax.text(
        -0.1,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
    )


def plot_depth_histogram(
    ax,
    depths,
    weights,
    color,
    subarray,
    bin_edges,
    plot_ylabel,
    *,
    show_title=True,
    show_panel_label=True,
    plot_xlabel=True,
    xlabel="Number of events",
    label_size=12,
    title_size=14,
    tick_label_size=None,
):
    """Plot one subarray's event-weighted depth histogram."""
    counts, edges = histogram(depths, bins=bin_edges, weights=weights)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    occupied = counts > 0
    ax.barh(
        bin_centers[occupied],
        counts[occupied],
        height=BIN_WIDTH,
        align="center",
        color=color,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_xscale("log")
    ax.set_ylim(MAX_DEPTH, MIN_DEPTH)
    if plot_xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size)
    if plot_ylabel:
        ax.set_ylabel("Depth (m)", fontsize=label_size)
    if tick_label_size is not None:
        ax.tick_params(axis="both", labelsize=tick_label_size)
    if show_title:
        ax.set_title(f"Subarray {subarray}", fontsize=title_size, fontweight="bold")
    if show_panel_label:
        add_panel_label(ax, PANEL_LABELS[subarray])
    if occupied.any():
        return float(counts[occupied].max())
    return 0.0


def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot event-depth histograms for Subarrays A and B using hub depths "
            "at preferred velocity scale factors"
        ),
    )
    parser.add_argument(
        "--group_info_csv",
        type=Path,
        default=Path(DETECTION_DIR) / f"event_group_info_selected_{DEFAULT_SUFFIX}.csv",
        help="CSV with selected event groups (columns include label, subarray, num_events)",
    )
    parser.add_argument(
        "--scale_factors_preferred_csv",
        type=Path,
        default=Path(VEL_MODEL_DIR) / "scale_factors_preferred.csv",
        help="Preferred velocity scale factor per subarray (columns subarray, scale_factor)",
    )
    parser.add_argument(
        "--loc_dir",
        type=Path,
        default=Path(LOC_DIR),
        help="Directory containing hub_event_location_info_group*.h5 files",
    )
    parser.add_argument(
        "--figname",
        type=str,
        default=DEFAULT_FIGNAME,
        help=f"Output figure filename under FIGURE_DIR (default: {DEFAULT_FIGNAME})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    group_info_csv = args.group_info_csv
    scale_factors_preferred_csv = args.scale_factors_preferred_csv
    loc_dir = args.loc_dir
    figname = args.figname

    print(f"Reading group info from {group_info_csv}...")
    group_df = load_group_info(group_info_csv)

    print(f"Reading preferred scale factors from {scale_factors_preferred_csv}...")
    preferred_scale_by_subarray = load_preferred_scale_factors(scale_factors_preferred_csv)
    for subarray, scale_factor in sorted(preferred_scale_by_subarray.items()):
        print(f"  Subarray {subarray}: scale factor {scale_factor:.4f}")

    depths, weights = collect_event_depths(group_df, preferred_scale_by_subarray, loc_dir)
    bin_edges = arange(MIN_DEPTH, MAX_DEPTH + BIN_WIDTH, BIN_WIDTH)

    fig, axes = subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True)
    max_count = 0.0
    for ax, subarray, plot_ylabel in zip(axes, SUBARRAYS, (True, False)):
        n_events = sum(weights[subarray])
        print(f"Subarray {subarray}: {n_events:d} events in {len(depths[subarray]):d} groups")
        panel_max = plot_depth_histogram(
            ax,
            depths[subarray],
            weights[subarray],
            SUBARRAY_COLORS[subarray],
            subarray,
            bin_edges,
            plot_ylabel,
        )
        max_count = max(max_count, panel_max)

    if max_count > 0:
        axes[0].set_xlim(1.0, max_count * 2.0)

    fig.subplots_adjust(wspace=0.15)
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
