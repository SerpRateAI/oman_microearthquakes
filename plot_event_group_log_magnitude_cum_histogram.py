"""
Plot the inverse cumulative distribution of log magnitudes for a classified event group.

Reads log_magnitudes_group{N}_{suffix}.csv produced by compute_event_group_log_magnitudes.py
and saves a complementary cumulative distribution figure to FIGURE_DIR.
"""

from argparse import ArgumentParser
from os.path import isfile, join

from matplotlib.pyplot import subplots
from numpy import arange, array, ceil, cumsum, floor, histogram, log10, polyfit
from pandas import read_csv

from utils_basic import DETECTION_DIR as dirpath_event, get_freq_limits_string
from utils_cc import get_repeating_snippet_suffix
from utils_plot import save_figure
from utils_sta_lta import get_sta_lta_suffix


def main():
    parser = ArgumentParser(
        description=(
            "Plot the inverse cumulative distribution of log magnitudes for an event group."
        ),
    )
    parser.add_argument("--group_label", type=int, default=18, help="Event group label")
    parser.add_argument("--bin_size", type=float, default=0.1, help="Histogram bin width")
    parser.add_argument(
        "--fit_mag_min",
        type=float,
        default=3.0,
        help="Minimum log magnitude for linear regression",
    )
    parser.add_argument(
        "--fit_mag_max",
        type=float,
        default=4.0,
        help="Maximum log magnitude for linear regression",
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

    group_label = args.group_label
    bin_size = args.bin_size
    fit_mag_min = args.fit_mag_min
    fit_mag_max = args.fit_mag_max
    min_cc = args.min_cc
    min_num_similar_snippet = args.min_num_similar_snippet
    min_num_similar_station = args.min_num_similar_station
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    thr_on = args.thr_on
    thr_off = args.thr_off
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter

    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=min_cc,
        min_num_similar=min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

    print("--------------------------------")
    print("Plotting event group log magnitude inverse cumulative distribution...")
    print("--------------------------------")
    print(f"Group label: {group_label}")
    print(f"Bin size: {bin_size}")
    print(f"Fit log magnitude range: [{fit_mag_min}, {fit_mag_max}]")
    print(f"Suffix group: {suffix_group}")

    infile = f"log_magnitudes_group{group_label:d}_{suffix_group}.csv"
    infile_path = join(dirpath_event, infile)
    if not isfile(infile_path):
        raise FileNotFoundError(f"Log magnitudes file not found: {infile_path}")

    df = read_csv(infile_path)
    if "log_magnitude" not in df.columns:
        raise ValueError(
            f"Expected 'log_magnitude' column in {infile_path}; got {list(df.columns)}"
        )

    log_mags = df["log_magnitude"].dropna().to_numpy()
    if log_mags.size == 0:
        raise ValueError(f"No valid log_magnitude values in {infile_path}.")

    print(f"Number of events plotted: {log_mags.size}")

    vmin = floor(log_mags.min() / bin_size) * bin_size
    vmax = ceil(log_mags.max() / bin_size) * bin_size
    bin_edges = arange(vmin, vmax + bin_size, bin_size)

    counts, _ = histogram(log_mags, bins=bin_edges)
    inv_cum_counts = cumsum(counts[::-1])[::-1]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid = inv_cum_counts > 0
    x_plot = bin_centers[valid]
    y_plot = log10(inv_cum_counts[valid])

    if fit_mag_min >= fit_mag_max:
        raise ValueError(
            f"fit_mag_min ({fit_mag_min}) must be less than fit_mag_max ({fit_mag_max})."
        )

    fit_mask = (x_plot >= fit_mag_min) & (x_plot <= fit_mag_max)
    x_fit = x_plot[fit_mask]
    y_fit = y_plot[fit_mask]
    if x_fit.size < 2:
        raise ValueError(
            f"Need at least two bins in fit range [{fit_mag_min}, {fit_mag_max}] for regression."
        )

    slope, intercept = polyfit(x_fit, y_fit, 1)
    b_value = -slope
    print(f"Regression b value (negative slope): {b_value:.4f}")

    fig, ax = subplots(figsize=(10, 5))
    ax.plot(
        x_plot,
        y_plot,
        "o",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.5,
        linestyle="none",
    )

    x_line = array([fit_mag_min, fit_mag_max])
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, color="crimson", linewidth=1.5)
    x_text = (fit_mag_min + fit_mag_max) / 2
    y_text = intercept + slope * x_text
    ax.text(
        x_text + 0.05,
        y_text,
        rf"$b={b_value:.2f}$",
        color="crimson",
        fontsize=12,
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Log magnitude", fontsize=12)
    ax.set_ylabel("Log number of events", fontsize=12)
    ax.set_xlim(2.5, 4.5)
    ax.set_title(
        f"Group {group_label}",
        fontsize=14,
        fontweight="bold",
    )

    figname = f"log_magnitude_cum_histogram_group{group_label:d}_{suffix_group}.png"
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
