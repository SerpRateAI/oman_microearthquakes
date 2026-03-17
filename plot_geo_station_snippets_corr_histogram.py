"""
Plot the histogram of the correlation values of the raw STA/LTA detections of a geophone station
"""

from argparse import ArgumentParser
from numpy import linspace, zeros, int64, ones_like, fill_diagonal, diff, load, histogram
from matplotlib.pyplot import figure, close
from pathlib import Path
from time import time

from utils_basic import (
    DETECTION_DIR as dirpath,
    get_freq_limits_string,
)
from utils_sta_lta import get_sta_lta_suffix
from utils_plot import save_figure

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def plot_corr_histogram_streaming(corr_mat,
                                  bins=100,
                                  cc_range=(0.0, 1.0),
                                  fraction_range=(0.0, 0.05),
                                  chunk_size=1000,
                                  figsize=(8, 5)):
    """
    Build a histogram of off-diagonal correlation values in streaming fashion.
    Suitable for very large matrices (avoids flattening).
    """
    num_snips = corr_mat.shape[0]
    hist_counts = zeros(bins, dtype=int64)
    bin_edges = linspace(cc_range[0], cc_range[1], bins + 1)

    for start in range(0, num_snips, chunk_size):
        end = min(start + chunk_size, num_snips)
        block = corr_mat[start:end, :]
        # Mask diagonal inside this block
        if end - start == num_snips:
            # Full block includes diagonal
            mask = ones_like(block, dtype=bool)
            fill_diagonal(mask, 0)
            values = block[mask]
        else:
            values = block.ravel()
        # Update histogram
        counts, _ = histogram(values, bins=bin_edges)
        hist_counts += counts

    # Normalize the histogram
    hist_fractions = hist_counts / hist_counts.sum()

    fig = figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.bar(bin_edges[:-1], hist_fractions, width=diff(bin_edges),
           align="edge", color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xlim(cc_range[0], cc_range[1])
    ax.set_ylim(fraction_range[0], fraction_range[1])

    ax.set_xlabel("Correlation coefficient", fontsize=12)
    ax.set_ylabel("Fraction", fontsize=12)

    return fig, ax, bin_edges, hist_fractions

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("--station", type=str, required=True, help="The station name")
parser.add_argument("--min_freq_filter", type=float, default=20.0, help="The minimum frequency for the filter")
parser.add_argument("--max_freq_filter", type=float, default=None, help="The maximum frequency for the filter")
parser.add_argument("--sta_window_sec", type=float, default=0.005, help="The STA window length in seconds")
parser.add_argument("--lta_window_sec", type=float, default=0.05, help="The LTA window length in seconds")
parser.add_argument("--on_threshold", type=float, default=4.0, help="The threshold for the on-trigger")
parser.add_argument("--off_threshold", type=float, default=1.0, help="The threshold for the off-trigger")
parser.add_argument("--max_shift", type=int, default=20, help="The maximum shift for the correlation")
parser.add_argument("--chunk_size", type=int, default=3000, help="The chunk size for the correlation")

args = parser.parse_args()

station = args.station
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
sta_window_sec = args.sta_window_sec
lta_window_sec = args.lta_window_sec
on_threshold = args.on_threshold
off_threshold = args.off_threshold
max_shift = args.max_shift
chunk_size = args.chunk_size

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

print(f"Plotting the histogram of the correlation values for station {station}...")
t0 = time()

# Read the correlation matrix
freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, on_threshold, off_threshold)
filename = f"correlation_matrix_full_{freq_str}_{sta_lta_suffix}_{station}.npy"
filepath = Path(dirpath) / filename
corr_mat = load(filepath)
print(f"Loaded the correlation matrix from {filepath}")
t1 = time()
print(f"Time taken to load the correlation matrix: {t1 - t0:.2f} seconds")

# Plot the histogram
fig, ax, bin_edges, hist_fractions = plot_corr_histogram_streaming(corr_mat)
ax.set_title(f"{station}", fontsize=14, fontweight="bold")
figname = f"correlation_histogram_{freq_str}_{sta_lta_suffix}_{station}.png"
save_figure(fig, figname)
close()
t2 = time()
print(f"Time taken to plot the histogram of the correlation values: {t2 - t1:.2f} seconds")
