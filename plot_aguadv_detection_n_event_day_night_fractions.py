"""
Plot repeating STA/LTA detection and subarray event day fractions (AGU Advances).

Two-panel figure: (a) day fractions of repeating detections at two stations
(default A04 and B04) with the expected day fraction and binomial p-values;
(b) day fractions of events in Subarrays A and B with the same annotations.
"""

from argparse import ArgumentParser
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots

from binomial_test_station_repeating_detection_day_night import (
    DEFAULT_REPEATING_SUFFIX,
    DEFAULT_STATIONS,
    count_detections_by_station_and_period,
)
from binomial_test_subarray_event_day_night import (
    compute_expected_day_fraction,
    run_binomial_day_night_test,
)
from compute_mean_tremor_power_by_event_ocurrence import DEFAULT_SUFFIX as DEFAULT_EVENT_SUFFIX
from plot_subarray_event_count_day_fractions import (
    collect_day_fraction_results,
    format_p_value,
    plot_day_fractions,
)
from plot_subarray_event_day_night_counts import load_sunrise_sunset_times
from utils_basic import DETECTION_DIR, SUNRISE_SUNSET_PATH
from utils_plot import save_figure

STATION_COLORS = {"A04": "dodgerblue", "B04": "darkorange"}
BAR_WIDTH = 0.5
BAR_EDGE_LINEWIDTH = 1.5
REFERENCE_LINEWIDTH = 1.5
PANEL_LABEL_FONTSIZE = 14


def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot (a) repeating STA/LTA detection day fractions at two stations "
            "and (b) subarray event day fractions, each with binomial p-values."
        ),
    )
    parser.add_argument(
        "--stations",
        type=str,
        nargs=2,
        default=list(DEFAULT_STATIONS),
        metavar=("STA_A", "STA_B"),
        help="Two stations for panel (a) (default: A04 B04)",
    )
    parser.add_argument(
        "--repeating-suffix",
        type=str,
        default=DEFAULT_REPEATING_SUFFIX,
        help="Suffix for repeating STA/LTA detection CSV files",
    )
    parser.add_argument(
        "--event-suffix",
        type=str,
        default=DEFAULT_EVENT_SUFFIX,
        help="Suffix for subarray event catalog JSONL files",
    )
    parser.add_argument(
        "--detection-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory containing detection and event catalog files",
    )
    parser.add_argument(
        "--sunrise-sunset-csv",
        type=Path,
        default=Path(SUNRISE_SUNSET_PATH),
        help="CSV with day, sunrise, and sunset columns in local time",
    )
    parser.add_argument(
        "--alternative",
        type=str,
        default="greater",
        choices=("two-sided", "less", "greater"),
        help=(
            "Alternative hypothesis for the binomial test "
            "(default: greater, day rate > night rate)"
        ),
    )
    parser.add_argument(
        "--figname",
        type=str,
        default=None,
        help=(
            "Output figure filename "
            "(default: aguadv_detection_n_event_day_night_fractions.png)"
        ),
    )
    return parser.parse_args()


def station_color(station):
    """Return bar color for a station; fall back by subarray letter."""
    if station in STATION_COLORS:
        return STATION_COLORS[station]
    if station.startswith("A"):
        return "dodgerblue"
    if station.startswith("B"):
        return "darkorange"
    return "black"


def collect_station_day_fraction_results(
    detection_dir, repeating_suffix, stations, sun_df, p_null, alternative
):
    """Return binomial test results keyed by station."""
    counts = count_detections_by_station_and_period(
        detection_dir, repeating_suffix, stations, sun_df
    )
    results = {}
    for station in stations:
        n_day = counts[(station, "day")]
        n_night = counts[(station, "night")]
        n_total = n_day + n_night
        if n_total == 0:
            print(f"Warning: Station {station} has no classified detections.")
            continue
        results[station] = run_binomial_day_night_test(
            n_day,
            n_night,
            p_null=p_null,
            alternative=alternative,
        )
    return results


def plot_station_day_fractions(ax, results, stations, expected_day_fraction):
    """Draw station day-fraction bars, expected reference line, and p-value labels."""
    plot_stations = [station for station in stations if station in results]
    positions = {station: float(i) for i, station in enumerate(stations)}
    x_values = [positions[station] for station in plot_stations]
    fractions = [results[station]["day_fraction"] for station in plot_stations]
    colors = [station_color(station) for station in plot_stations]

    ax.bar(
        x_values,
        fractions,
        width=BAR_WIDTH,
        color=colors,
        edgecolor="black",
        linewidth=BAR_EDGE_LINEWIDTH,
    )
    ax.axhline(
        expected_day_fraction,
        color="crimson",
        linestyle="--",
        linewidth=REFERENCE_LINEWIDTH,
        zorder=0,
    )

    ymax = max(fractions + [expected_day_fraction]) if fractions else expected_day_fraction
    label_offset = 0.02 * ymax if ymax > 0 else 0.02
    for station, x in zip(plot_stations, x_values):
        fraction = results[station]["day_fraction"]
        p_text = format_p_value(results[station]["p_value"])
        ax.text(
            x,
            fraction + label_offset,
            p_text,
            ha="center",
            va="bottom",
            fontsize=10,
            color=station_color(station),
        )

    ax.set_xticks([positions[station] for station in stations])
    ax.set_xticklabels(list(stations), fontsize=10)
    ax.set_xlabel("Station", fontsize=12)
    ax.set_ylabel("Day fraction", fontsize=12)
    ax.set_title("Repeating detection day fractions", fontweight="bold")
    ax.set_ylim(0, ymax + 0.12 * ymax if ymax > 0 else 1.0)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="crimson",
            linestyle="--",
            linewidth=REFERENCE_LINEWIDTH,
            label="Expected day fraction",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=1.0)


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


def main():
    args = parse_args()
    stations = args.stations
    repeating_suffix = args.repeating_suffix
    event_suffix = args.event_suffix
    detection_dir = args.detection_dir
    sunrise_sunset_csv = args.sunrise_sunset_csv
    alternative = args.alternative
    figname = args.figname

    sun_df = load_sunrise_sunset_times(sunrise_sunset_csv)
    print(f"Loaded sunrise/sunset times from {sunrise_sunset_csv}")

    duration_info = compute_expected_day_fraction(sun_df)
    expected_day_fraction = duration_info["expected_day_fraction"]
    print(f"Expected day fraction: {expected_day_fraction:.4f}")

    station_results = collect_station_day_fraction_results(
        detection_dir,
        repeating_suffix,
        stations,
        sun_df,
        expected_day_fraction,
        alternative,
    )
    if not station_results:
        raise SystemExit("No station had classified detections; nothing to plot.")

    for station in stations:
        if station not in station_results:
            continue
        test_result = station_results[station]
        print(
            f"Station {station}: day_fraction={test_result['day_fraction']:.4f}, "
            f"p_value={test_result['p_value']:.6f}"
        )

    subarray_results = collect_day_fraction_results(
        detection_dir,
        event_suffix,
        sun_df,
        expected_day_fraction,
        alternative,
    )
    if not subarray_results:
        raise SystemExit("No subarray had classified events; nothing to plot.")

    for subarray in ("A", "B"):
        if subarray not in subarray_results:
            continue
        test_result = subarray_results[subarray]
        print(
            f"Subarray {subarray}: day_fraction={test_result['day_fraction']:.4f}, "
            f"p_value={test_result['p_value']:.6f}"
        )

    fig, (ax_a, ax_b) = subplots(1, 2, figsize=(12, 7.5))
    plot_station_day_fractions(
        ax_a, station_results, stations, expected_day_fraction
    )
    plot_day_fractions(ax_b, subarray_results, expected_day_fraction)
    legend_b = ax_b.get_legend()
    if legend_b is not None:
        legend_b.remove()
    ax_b.set_ylabel("")
    add_panel_label(ax_a, "(a)")
    add_panel_label(ax_b, "(b)")

    out_name = figname or "aguadv_detection_n_event_day_night_fractions.png"
    save_figure(fig, out_name)
    print(f"Saved {out_name}")


if __name__ == "__main__":
    main()
