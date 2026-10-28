"""
Plot subarray event day fractions with expected reference and p-values.

Shows the fraction of events occurring during the day for Subarrays A and B as
colored bars, overlays the duration-based expected day fraction, and labels each
bar with its one-sided binomial test p-value.
"""

from argparse import ArgumentParser
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots

from binomial_test_subarray_event_day_night import (
    compute_expected_day_fraction,
    run_binomial_day_night_test,
)
from compute_mean_tremor_power_by_event_ocurrence import DEFAULT_SUFFIX
from plot_subarray_event_day_night_counts import (
    count_events_by_subarray_and_period,
    load_sunrise_sunset_times,
)
from utils_basic import DETECTION_DIR, SUNRISE_SUNSET_PATH
from utils_plot import save_figure

SUBARRAY_COLORS = {"A": "dodgerblue", "B": "darkorange"}
BAR_WIDTH = 0.5
BAR_EDGE_LINEWIDTH = 1.5
REFERENCE_LINEWIDTH = 1.5


def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot subarray A and B event day fractions with the expected "
            "day fraction and one-sided binomial test p-values."
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
        "--sunrise-sunset-csv",
        type=Path,
        default=Path(SUNRISE_SUNSET_PATH),
        help="CSV with day, sunrise, and sunset columns in local time",
    )
    parser.add_argument(
        "--figname",
        type=str,
        default=None,
        help=(
            "Output figure filename "
            "(default: subarray_event_count_day_fractions_<suffix>.png)"
        ),
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
    return parser.parse_args()


def format_p_value(p_value):
    """Format p-value for bar annotation."""
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def collect_day_fraction_results(
    detection_dir, suffix, sun_df, p_null, alternative
):
    """Return binomial test results keyed by subarray."""
    counts = count_events_by_subarray_and_period(detection_dir, suffix, sun_df)
    results = {}
    for subarray in ("A", "B"):
        n_day = counts[(subarray, "day")]
        n_night = counts[(subarray, "night")]
        n_total = n_day + n_night
        if n_total == 0:
            print(f"Warning: Subarray {subarray} has no classified events.")
            continue
        results[subarray] = run_binomial_day_night_test(
            n_day,
            n_night,
            p_null=p_null,
            alternative=alternative,
        )
    return results


def plot_day_fractions(ax, results, expected_day_fraction):
    """Draw day-fraction bars, expected reference line, and p-value labels."""
    positions = {"A": 0.0, "B": 1.0}
    subarrays = [subarray for subarray in ("A", "B") if subarray in results]
    x_values = [positions[subarray] for subarray in subarrays]
    fractions = [results[subarray]["day_fraction"] for subarray in subarrays]
    colors = [SUBARRAY_COLORS[subarray] for subarray in subarrays]

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

    ymax = max(fractions + [expected_day_fraction])
    label_offset = 0.02 * ymax if ymax > 0 else 0.02
    for subarray, x in zip(subarrays, x_values):
        fraction = results[subarray]["day_fraction"]
        p_text = format_p_value(results[subarray]["p_value"])
        ax.text(
            x,
            fraction + label_offset,
            p_text,
            ha="center",
            va="bottom",
            fontsize=10,
            color=SUBARRAY_COLORS[subarray],
        )

    ax.set_xticks([positions[subarray] for subarray in ("A", "B")])
    ax.set_xticklabels(["A", "B"], fontsize=10)
    ax.set_xlabel("Subarray", fontsize=12)
    ax.set_ylabel("Day fraction", fontsize=12)
    ax.set_title("Subarray event day fractions", fontweight="bold")
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


def main():
    args = parse_args()

    sun_df = load_sunrise_sunset_times(args.sunrise_sunset_csv)
    print(f"Loaded sunrise/sunset times from {args.sunrise_sunset_csv}")

    duration_info = compute_expected_day_fraction(sun_df)
    expected_day_fraction = duration_info["expected_day_fraction"]
    print(f"Expected day fraction: {expected_day_fraction:.4f}")

    results = collect_day_fraction_results(
        args.detection_dir,
        args.suffix,
        sun_df,
        expected_day_fraction,
        args.alternative,
    )
    if not results:
        raise SystemExit("No subarray had classified events; nothing to plot.")

    for subarray in ("A", "B"):
        if subarray not in results:
            continue
        test_result = results[subarray]
        print(
            f"Subarray {subarray}: day_fraction={test_result['day_fraction']:.4f}, "
            f"p_value={test_result['p_value']:.6f}"
        )

    fig, ax = subplots(1, 1, figsize=(6, 5))
    plot_day_fractions(ax, results, expected_day_fraction)

    figname = args.figname or f"subarray_event_count_day_fractions_{args.suffix}.png"
    save_figure(fig, figname)
    print(f"Saved {figname}")


if __name__ == "__main__":
    main()
