"""
Plot inverse cumulative log-magnitude distributions for AGU Advances groups 18–20.

Three vertically stacked panels labeled (a)–(c), with per-group magnitudes of completeness
(min_magnitude) from magnitude_number_plot.csv and subarray-specific marker colors. The b
value is the maximum-likelihood estimate b = log10(e) / (mean(M) - M_min) over events with
M >= M_min, where M_min is min_magnitude, and the line with slope -b passes through the
number of those events at M_min and extends to magnitude 3.5. The b-value
uncertainty follows Shi and Bolt (1982).
"""

from argparse import ArgumentParser
from os.path import isfile, join

from matplotlib.pyplot import subplots
from numpy import arange, array, ceil, cumsum, e, floor, histogram, log10, mean, sqrt, sum
from pandas import read_csv

from utils_basic import (
    DETECTION_DIR as dirpath_event,
    PLOTTING_DIR,
    get_freq_limits_string,
)
from utils_cc import get_repeating_snippet_suffix
from utils_plot import save_figure
from utils_sta_lta import get_sta_lta_suffix

SUBARRAY_COLORS = {"A": "tab:blue", "B": "tab:orange"}
DEFAULT_GROUP_LABELS = (18, 19, 20)
PARAM_COLUMNS = ("group_label", "subarray", "min_magnitude")


def load_params_csv(params_csv):
    """Load and validate magnitude-number plot parameters."""
    if not isfile(params_csv):
        raise FileNotFoundError(f"Parameter file not found: {params_csv}")

    params_df = read_csv(params_csv)
    missing = [col for col in PARAM_COLUMNS if col not in params_df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {PARAM_COLUMNS} in {params_csv}; missing {missing}."
        )

    if len(params_df) != len(DEFAULT_GROUP_LABELS):
        raise ValueError(
            f"Expected {len(DEFAULT_GROUP_LABELS)} rows in {params_csv}, got {len(params_df)}."
        )

    group_labels = params_df["group_label"].astype(int).tolist()
    if group_labels != list(DEFAULT_GROUP_LABELS):
        raise ValueError(
            f"Expected group labels {DEFAULT_GROUP_LABELS} in {params_csv}, got {group_labels}."
        )

    for subarray in params_df["subarray"]:
        if str(subarray) not in SUBARRAY_COLORS:
            raise ValueError(f"Unsupported subarray {subarray!r} in {params_csv}.")

    return params_df


def select_group_params(params_df, group_labels):
    """Return parameter rows for the requested groups, in the requested order."""
    selected_rows = []
    for group_label in group_labels:
        match = params_df[params_df["group_label"].astype(int) == group_label]
        if match.empty:
            raise ValueError(
                f"Group {group_label} not found in parameter file; "
                f"available groups: {params_df['group_label'].astype(int).tolist()}."
            )
        selected_rows.append(match.iloc[0])
    return selected_rows


def panel_label(index):
    """Return panel label string for zero-based index, e.g. '(a)'."""
    return f"({chr(ord('a') + index)})"


def load_log_magnitudes(group_label, suffix_group, dirpath_event):
    """Load log_magnitude values for one event group."""
    infile = f"log_magnitudes_group{group_label:d}_{suffix_group}.csv"
    infile_path = join(dirpath_event, infile)
    if not isfile(infile_path):
        raise FileNotFoundError(f"Log magnitudes file not found: {infile_path}")

    df = read_csv(infile_path)
    if "log_magnitude" not in df.columns:
        raise ValueError(
            f"Expected 'log_magnitude' column in {infile_path}; got {list(df.columns)}."
        )

    log_mags = df["log_magnitude"].dropna().to_numpy()
    if log_mags.size == 0:
        raise ValueError(f"No valid log_magnitude values in {infile_path}.")

    return log_mags


def compute_inv_cum_distribution(log_mags, bin_size):
    """Return bin centers and log10 inverse cumulative event counts."""
    vmin = floor(log_mags.min() / bin_size) * bin_size
    vmax = ceil(log_mags.max() / bin_size) * bin_size
    bin_edges = arange(vmin, vmax + bin_size, bin_size)

    counts, _ = histogram(log_mags, bins=bin_edges)
    inv_cum_counts = cumsum(counts[::-1])[::-1]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid = inv_cum_counts > 0
    return bin_centers[valid], log10(inv_cum_counts[valid])


def plot_magnitude_freq_panel(ax, log_mags, row, bin_size, color, plot_xlabel=False):
    """Plot one inverse cumulative magnitude-frequency panel."""
    group_label = int(row["group_label"])
    fit_mag_min = float(row["min_magnitude"])
    fit_mag_max = 3.5

    x_plot, y_plot = compute_inv_cum_distribution(log_mags, bin_size)

    log_mags_fit = log_mags[log_mags >= fit_mag_min]
    num_fit = log_mags_fit.size
    if num_fit < 2:
        raise ValueError(
            f"Group {group_label}: need at least two events with magnitude >= {fit_mag_min}; "
            f"got {num_fit}."
        )

    b_value = log10(e) / (mean(log_mags_fit) - fit_mag_min)
    b_std = 2.30 * b_value**2 * sqrt(
        sum((log_mags_fit - mean(log_mags_fit)) ** 2) / (num_fit * (num_fit - 1))
    )
    print(
        f"Group {group_label}: maximum-likelihood b = {b_value:.1f} +/- {b_std:.1f} "
        f"({num_fit} events >= {fit_mag_min})"
    )

    log_num_fit = log10(num_fit)
    slope = -b_value
    intercept = log_num_fit + b_value * fit_mag_min

    ax.plot(
        x_plot,
        y_plot,
        "o",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.5,
        linestyle="none",
    )

    x_line = array([fit_mag_min, fit_mag_max])
    ax.plot(x_line, intercept + slope * x_line, color="crimson", linewidth=2.0)

    ax.axvline(fit_mag_min, color="crimson", linestyle="--", linewidth=1.5)

    ax.text(
        fit_mag_min + 0.15,
        log_num_fit,
        rf"$b={b_value:.1f}\pm{b_std:.1f}$",
        color="crimson",
        fontsize=12,
        ha="left",
        va="bottom",
    )

    ax.set_xlim(2.5, 4.0)
    ax.set_ylim(0.0, 4.0)
    ax.set_ylabel("Log number of events", fontsize=12)
    ax.set_title(f"Group {group_label}", fontsize=12, fontweight="bold")

    if plot_xlabel:
        ax.set_xlabel("Log magnitude", fontsize=12)


def main():
    parser = ArgumentParser(
        description=(
            "Plot AGU Advances magnitude-frequency distributions for groups 18, 19, and 20."
        ),
    )
    parser.add_argument(
        "--group_labels",
        type=int,
        nargs="+",
        default=list(DEFAULT_GROUP_LABELS),
        help="Event group labels to plot (default: all groups in parameter file)",
    )
    parser.add_argument(
        "--params_csv",
        type=str,
        default=join(PLOTTING_DIR, "magnitude_number_plot.csv"),
        help="CSV with per-group subarray and magnitude of completeness (min_magnitude)",
    )
    parser.add_argument("--bin_size", type=float, default=0.05, help="Histogram bin width")
    parser.add_argument(
        "--figname",
        type=str,
        default="aguadv_event_group_magnitude_freq.png",
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

    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=min_cc,
        min_num_similar=min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

    print("--------------------------------")
    print("Plotting AGU Advances event group magnitude-frequency figure...")
    print("--------------------------------")
    print(f"Group labels: {group_labels}")
    print(f"Parameter file: {params_csv}")
    print(f"Bin size: {bin_size}")
    print(f"Suffix group: {suffix_group}")

    params_df = load_params_csv(params_csv)
    selected_rows = select_group_params(params_df, group_labels)
    n_panels = len(selected_rows)

    fig_height = 4 * n_panels
    fig, axes = subplots(n_panels, 1, figsize=(fig_height / 2, fig_height), sharex=False)
    if n_panels == 1:
        axes = [axes]

    for i_row, row in enumerate(selected_rows):
        group_label = int(row["group_label"])
        subarray = str(row["subarray"])
        color = SUBARRAY_COLORS[subarray]
        ax = axes[i_row]

        print(f"Processing group {group_label} (Subarray {subarray})...")
        log_mags = load_log_magnitudes(group_label, suffix_group, dirpath_event)
        print(f"  Number of events: {log_mags.size}")

        plot_magnitude_freq_panel(
            ax,
            log_mags,
            row,
            bin_size,
            color,
            plot_xlabel=(i_row == n_panels - 1),
        )

        ax.text(
            -0.25,
            1.05,
            panel_label(i_row),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
        )

    fig.subplots_adjust(hspace=0.2)
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
