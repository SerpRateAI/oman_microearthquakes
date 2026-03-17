"""
Plot the correlation matrix of the raw STA/LTA detections of a geophone station
"""

from argparse import ArgumentParser
from pathlib import Path
from numpy import load, ceil
from matplotlib.pyplot import figure, close
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def plot_downsampled_matrix(matrix,
                        block_size=None,
                        target_tiles=10000,
                        min_block_size=1,
                        max_block_size=2048,
                        cmap='plasma',
                        figsize=(8, 8)):
    """
    Plot the downsampled correlation matrix.
    """

    num_snips = matrix.shape[0]

    # Auto block size if not provided
    if block_size is None:
        safe_target = max(target_tiles, 1)
        auto_bs = int(ceil(num_snips / safe_target))
        block_size = max(min_block_size, min(max_block_size, max(1, auto_bs)))

        if num_snips < block_size:
            downsampled = matrix
        else:
            down_num = num_snips // block_size
            if down_num == 0:
                downsampled = matrix
            else:
                downsampled = matrix[:down_num*block_size, :down_num*block_size] \
                    .reshape(down_num, block_size, down_num, block_size) \
                    .mean(axis=(1, 3))

        fig = figure(figsize=figsize)
        ax = fig.add_subplot(111)
        im = ax.imshow(downsampled, cmap=cmap, aspect='auto', vmin=0.0, vmax=1.0)

        # Create a new axis of equal height for the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Correlation")

        ax.set_xlabel("Index (downsampled)")
        ax.set_ylabel("Index (downsampled)")

        return fig, ax, block_size

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("--station", type=str, required=True, help="The station name")
parser.add_argument("--min_freq_filter", type=float, default=20.0)
parser.add_argument("--max_freq_filter", type=float, default=None)
parser.add_argument("--sta_window_sec", type=float, default=0.005)
parser.add_argument("--lta_window_sec", type=float, default=0.05)
parser.add_argument("--on_threshold", type=float, default=4.0)
parser.add_argument("--off_threshold", type=float, default=1.0)
parser.add_argument("--max_shift", type=int, default=20)
parser.add_argument("--chunk_size", type=int, default=3000)
parser.add_argument("--cc_threshold", type=float, default=0.85)

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
cc_threshold = args.cc_threshold

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

print(f"Loading the correlation matrix from {dirpath}")
t0 = time()
freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, on_threshold, off_threshold)
filename = f"correlation_matrix_full_{freq_str}_{sta_lta_suffix}_{station}.npy"
filepath = Path(dirpath) / filename
corr_mat = load(filepath)
print(f"Loaded the correlation matrix from {filepath}")
t1 = time()
print(f"Time taken to load the correlation matrix: {t1 - t0:.2f} seconds")

print("Plotting the downsampled correlation matrix...")
fig, ax, block_size = plot_downsampled_matrix(corr_mat)
ax.set_title(f"{station} (block={block_size})", fontsize=14, fontweight='bold')
figname = f"correlation_matrix_downsampled_{freq_str}_{sta_lta_suffix}_{station}.png"
save_figure(fig, figname)
close()
t2 = time()
print(f"Time taken to plot the downsampled correlation matrix: {t2 - t1:.2f} seconds")
