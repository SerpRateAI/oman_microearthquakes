"""
Plot subarray hourly event counts and harmonic tremor mode power.

Two-panel figure taken from panels (b) and (c) of
plot_aguadv_event_hourly_counts_n_tremor_power.py with the
precipitation/water-table panel omitted: Subarray A and B hourly event
counts on top, harmonic mode power on the bottom.
"""

from argparse import ArgumentParser
from os.path import join
from pathlib import Path

from matplotlib import gridspec
from matplotlib.pyplot import figure
from pandas import read_csv, read_hdf

from utils_basic import (
    DETECTION_DIR,
    ENDTIME_GEO,
    SPECTROGRAM_DIR,
    STARTTIME_GEO,
)
from utils_plot import (
    add_day_night_shading,
    format_datetime_xlabels,
    format_db_ylabels,
    save_figure,
)
from utils_spec import get_spec_peak_file_suffix, get_spectrogram_file_suffix

DEFAULT_SUFFIX = (
    "freq20hz_sta5_lta50_on4_off1_cc0.85_num_sim_det10_num_sim_sta3"
)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot subarray hourly event counts and harmonic tremor mode power."
        ),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=DEFAULT_SUFFIX,
        help="Suffix for subarray hourly-count CSV files",
    )
    parser.add_argument(
        "--detection-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory containing hourly_counts_subarray_*.csv files",
    )
    parser.add_argument(
        "--figname",
        type=str,
        default=None,
        help=(
            "Output figure filename "
            "(default: fiber_optical_seis_event_hourly_count_n_tremor_power.png)"
        ),
    )
    parser.add_argument(
        "--color_mode_face",
        type=str,
        required=True,
        help="The color for the face of the harmonic mode",
    )
    parser.add_argument(
        "--color_mode_edge",
        type=str,
        required=True,
        help="The color for the edge of the harmonic mode",
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


def plot_subarray_hourly_counts(ax_event, detection_dir, suffix):
    """Plot hourly event counts for subarrays A and B."""
    print("Plotting the hourly event counts...")
    for subarray, color in (("a", "dodgerblue"), ("b", "darkorange")):
        filename = f"hourly_counts_subarray_{subarray}_{suffix}.csv"
        filepath = detection_dir / filename
        if not filepath.is_file():
            raise FileNotFoundError(f"Missing hourly counts file: {filepath}")

        count_df = read_csv(filepath, parse_dates=["hour"])
        label = f"Subarray {subarray.upper()}"
        ax_event.plot(
            count_df["hour"],
            count_df["count"],
            color=color,
            label=label,
        )

    print("Adding the day and night shading...")
    add_day_night_shading(ax_event)

    ax_event.set_xlim(STARTTIME_GEO, ENDTIME_GEO)
    ax_event.set_ylim(1, 500)
    format_datetime_xlabels(
        ax_event,
        plot_axis_label=False,
        plot_tick_label=False,
        major_tick_spacing="5d",
        num_minor_ticks=4,
        date_format="%Y-%m-%d",
        axis_label_size=12,
        tick_label_size=10,
    )
    ax_event.set_ylabel("Number of events per hour", fontsize=12)
    ax_event.set_yscale("log")
    ax_event.set_title("Hourly event counts", fontweight="bold")
    ax_event.legend(loc="upper right", fontsize=10, framealpha=1.0)


def plot_harmonic_mode_power(ax_spec, args):
    """Plot harmonic tremor mode power (same as reference script)."""
    print("Plotting the harmonic mode power...")

    filename = (
        f"stationary_harmonic_series_{args.base_mode_name}_base"
        f"{args.base_mode_order}.csv"
    )
    filepath = join(SPECTROGRAM_DIR, filename)
    harmonic_df = read_csv(filepath)
    mode_name = harmonic_df.loc[
        harmonic_df["mode_order"] == args.mode_order, "mode_name"
    ].values[0]

    suffix_spec = get_spectrogram_file_suffix(args.window_length, args.overlap)
    suffix_peak = get_spec_peak_file_suffix(
        args.min_prom, args.min_rbw, args.max_mean_db
    )
    filename_in = (
        f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
    )
    inpath = join(SPECTROGRAM_DIR, filename_in)
    resonance_df = read_hdf(inpath, key="properties")
    resonance_df = resonance_df[resonance_df["station"] == args.station]

    ax_spec.scatter(
        resonance_df["time"],
        resonance_df["total_power"],
        marker="o",
        s=15,
        color=args.color_mode_face,
        edgecolors=args.color_mode_edge,
        label=mode_name,
        zorder=2,
    )

    print("Adding the day and night shading...")
    add_day_night_shading(ax_spec)

    ax_spec.set_xlim(STARTTIME_GEO, ENDTIME_GEO)
    format_datetime_xlabels(
        ax_spec,
        plot_axis_label=True,
        plot_tick_label=True,
        major_tick_spacing="5d",
        num_minor_ticks=4,
        date_format="%Y-%m-%d",
        axis_label_size=12,
        tick_label_size=10,
    )
    format_db_ylabels(ax_spec, axis_label_size=12, tick_label_size=10)
    ax_spec.set_title(f"Mode {args.mode_order} power on {args.station}", fontweight="bold")


def main():
    args = parse_args()
    detection_dir = args.detection_dir
    suffix = args.suffix

    fig = figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.15)
    ax_event = fig.add_subplot(gs[0])
    ax_spec = fig.add_subplot(gs[1])
    panel_axes = (ax_event, ax_spec)
    panel_labels = ("(a)", "(b)")

    plot_subarray_hourly_counts(ax_event, detection_dir, suffix)
    plot_harmonic_mode_power(ax_spec, args)

    for ax, label in zip(panel_axes, panel_labels):
        ax.text(
            -0.1,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
        )

    figname = (
        args.figname
        or "fiber_optical_seis_event_hourly_count_n_tremor_power.png"
    )
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
