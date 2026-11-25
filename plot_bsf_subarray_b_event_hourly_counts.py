"""
Plot Subarray B hourly event counts.
"""

from argparse import ArgumentParser
from pathlib import Path

from matplotlib.pyplot import figure
from matplotlib.ticker import NullLocator
from pandas import read_csv

from utils_basic import DETECTION_DIR, ENDTIME_GEO, STARTTIME_GEO
from utils_plot import add_day_night_shading, format_datetime_xlabels, save_figure

DEFAULT_SUFFIX = (
    "freq20hz_sta5_lta50_on4_off1_cc0.85_num_sim_det10_num_sim_sta3"
)


def parse_args():
    parser = ArgumentParser(
        description="Plot Subarray B hourly event counts.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=DEFAULT_SUFFIX,
        help="Suffix for the Subarray B hourly-count CSV file",
    )
    parser.add_argument(
        "--detection-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory containing hourly_counts_subarray_b_*.csv",
    )
    parser.add_argument(
        "--figname",
        type=str,
        default=None,
        help=(
            "Output figure filename "
            "(default: bsf_subarray_b_event_hourly_counts.png)"
        ),
    )
    return parser.parse_args()


def plot_subarray_b_hourly_counts(ax_event, detection_dir, suffix):
    """Plot hourly event counts for Subarray B."""
    print("Plotting the Subarray B hourly event counts...")
    filename = f"hourly_counts_subarray_b_{suffix}.csv"
    filepath = detection_dir / filename
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing hourly counts file: {filepath}")

    count_df = read_csv(filepath, parse_dates=["hour"])
    ax_event.plot(
        count_df["hour"],
        count_df["count"],
        color="darkorange",
    )

    print("Adding the day and night shading...")
    add_day_night_shading(ax_event)

    ax_event.set_xlim(STARTTIME_GEO, ENDTIME_GEO)
    ax_event.set_ylim(1, 100)
    format_datetime_xlabels(
        ax_event,
        plot_axis_label=False,
        plot_tick_label=False,
        major_tick_spacing="1d",
        date_format="%Y-%m-%d",
        axis_label_size=12,
        tick_label_size=10,
    )
    ax_event.xaxis.set_minor_locator(NullLocator())
    ax_event.set_ylabel("Number of events per hour", fontsize=12)
    ax_event.set_yscale("log")


def main():
    args = parse_args()
    detection_dir = args.detection_dir
    suffix = args.suffix
    figname = args.figname

    fig = figure(figsize=(10, 4))
    ax_event = fig.add_subplot(111)
    plot_subarray_b_hourly_counts(ax_event, detection_dir, suffix)

    figname = figname or "bsf_subarray_b_event_hourly_counts.png"
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
