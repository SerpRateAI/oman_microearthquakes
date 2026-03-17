"""
Compute the correlation matrix of the raw STA/LTA detections of a geophone station
by averaging the three components.
"""
from pathlib import Path
from argparse import ArgumentParser
from time import time
from typing import Union
from numpy import (
    ndarray,
    zeros,
    float32,
    fill_diagonal,
    save,
    finfo,
    full,
    int16,
    linspace,
    diff,
    histogram,
    ones_like,
    int64,
    ceil,
)
from numpy.linalg import norm
from matplotlib.pyplot import figure, close
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from utils_basic import (
    DETECTION_DIR as dirpath,
    GEO_COMPONENTS as components,
    get_freq_limits_string,
)
from utils_sta_lta import Snippets, get_sta_lta_suffix
from utils_plot import save_figure, plot_downsampled_matrix, plot_corr_histogram_streaming


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def unitnorm(snippets):
    """
    Unit-normalize the 3-component snippets.
    Returns dict: comp -> (num_snips, num_points) float32 arrays.
    """
    num_snips = len(snippets)
    num_points = snippets[0].num_pts
    snip_dict = {comp: zeros((num_snips, num_points), dtype=float32) for comp in components}

    for idx, snip in enumerate(snippets):
        for comp in components:
            waveform = snip.waveform[comp].astype(float32, copy=False)
            waveform_norm = norm(waveform)
            if waveform_norm:
                waveform = waveform / waveform_norm
            snip_dict[comp][idx] = waveform
    return snip_dict


def symmetrize_with_lag(corr_mat, lag_mat, block=2048):
    """
    Make corr_mat symmetric by elementwise max, and sync lag_mat consistently.
    For (i,j) with i<j:
      if corr[i,j] >= corr[j,i]:
          set both corr to corr[i,j]
          lag[j,i] = -lag[i,j]
      else:
          set both corr to corr[j,i]
          lag[i,j] = -lag[j,i]
    """
    num_snips = corr_mat.shape[0]
    for i in range(0, num_snips, block):
        i2 = min(i + block, num_snips)
        # diagonal block
        corr_block = corr_mat[i:i2, i:i2]
        lag_block = lag_mat[i:i2, i:i2]
        # make diagonal lags 0
        for d in range(i2 - i):
            lag_block[d, d] = 0
        # off-diagonal inside block
        for r in range(i2 - i):
            for c in range(r + 1, i2 - i):
                a = corr_block[r, c]
                b = corr_block[c, r]
                if a >= b:
                    corr_block[c, r] = a
                    lag_block[c, r] = -lag_block[r, c]
                else:
                    corr_block[r, c] = b
                    lag_block[r, c] = -lag_block[c, r]
        # off-diagonal blocks
        for j in range(i2, num_snips, block):
            j2 = min(j + block, num_snips)
            corr_upper = corr_mat[i:i2, j:j2]
            corr_lower = corr_mat[j:j2, i:i2]
            lag_upper = lag_mat[i:i2, j:j2]
            lag_lower = lag_mat[j:j2, i:i2]
            for r in range(i2 - i):
                for c in range(j2 - j):
                    a = corr_upper[r, c]
                    b = corr_lower[c, r]
                    if a >= b:
                        corr_lower[c, r] = a
                        lag_lower[c, r] = -lag_upper[r, c]
                    else:
                        corr_upper[r, c] = b
                        lag_upper[r, c] = -lag_lower[c, r]


def build_corr_and_lag_avg_components(snip_dict, max_shift, chunk_size):
    """
    Build corr & lag by maximizing (over lag) the per-lag average
    correlation across components Z/1/2.
    """
    z = snip_dict['Z']
    c1 = snip_dict['1']
    c2 = snip_dict['2']
    num_snips, num_points = z.shape

    corr_mat = full((num_snips, num_snips), finfo(float32).min, dtype=float32)
    lag_mat = zeros((num_snips, num_snips), dtype=int16)

    for lag in tqdm(range(-max_shift, max_shift + 1), desc="lag"):
        if lag >= 0:
            sl, sr = slice(0, num_points - lag), slice(lag, num_points)
        else:
            sl, sr = slice(-lag, num_points), slice(0, num_points + lag)

        zr = z[:, sr]
        c1r = c1[:, sr]
        c2r = c2[:, sr]

        for row_start in range(0, num_snips, chunk_size):
            row_end = min(row_start + chunk_size, num_snips)

            zl = z[row_start:row_end, sl]
            c1l = c1[row_start:row_end, sl]
            c2l = c2[row_start:row_end, sl]

            zb = zl @ zr.T
            c1b = c1l @ c1r.T
            c2b = c2l @ c2r.T

            avg_block = (zb + c1b + c2b) / 3.0

            cur = corr_mat[row_start:row_end, :]
            improved = avg_block > cur
            cur[improved] = avg_block[improved]
            lag_mat[row_start:row_end, :][improved] = int16(lag)

    print(f"Symmetrizing the correlation and lag matrices...")
    symmetrize_with_lag(corr_mat, lag_mat)
    fill_diagonal(corr_mat, 1.0)
    fill_diagonal(lag_mat, 0)

    return corr_mat, lag_mat

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

print(f"Computing the correlation matrix for station {station}...")
print(f"Min frequency filter: {min_freq_filter} Hz")
print(f"Max frequency filter: {max_freq_filter} Hz")
print(f"STA window length: {sta_window_sec} s")
print(f"LTA window length: {lta_window_sec} s")
print(f"On threshold: {on_threshold} (STA/LTA ratio)")
print(f"Off threshold: {off_threshold} (STA/LTA ratio)")
print(f"Max shift: {max_shift} samples")
print(f"Chunk size: {chunk_size} snippets")

print(f"Loading detections from {dirpath}")
t0 = time()

freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, on_threshold, off_threshold)
h5_name = f"snippets_sta_lta_{freq_str}_{sta_lta_suffix}_{station}.h5"
h5_path = Path(dirpath) / h5_name
snippets = Snippets.from_hdf(h5_path)

t1 = time()
print(f"Loaded {len(snippets)} snippets")
print(f"Detections loaded in {t1 - t0:.2f} s.")

print("Unit-normalizing snippets...")
snip_dict = unitnorm(snippets)
t2 = time()
print(f"Snippets unit-normalized in {t2 - t1:.2f} s.")

print("Building averaged correlation and lag matrices...")
corr_mat, lag_mat = build_corr_and_lag_avg_components(
    snip_dict, max_shift=max_shift, chunk_size=chunk_size
)
t3 = time()
print(f"Averaged correlation and lag matrices built in {t3 - t2:.2f} s.")

print("Plotting the downsampled correlation matrix...")
fig, ax, block_size = plot_downsampled_matrix(corr_mat)
ax.set_title(f"{station} (block={block_size})", fontsize=14, fontweight='bold')
figname = f"correlation_matrix_downsampled_{freq_str}_{sta_lta_suffix}_{station}.png"
save_figure(fig, figname)
close()
t4 = time()
print(f"Time taken to plot the downsampled correlation matrix: {t4 - t3:.2f} seconds")

print("Plotting the histogram of the correlation values...")
fig, ax, bin_edges, hist_fractions = plot_corr_histogram_streaming(corr_mat)
ax.set_title(f"{station}", fontsize=14, fontweight="bold")
figname = f"correlation_histogram_{freq_str}_{sta_lta_suffix}_{station}.png"
save_figure(fig, figname)
close()
t5 = time()
print(f"Time taken to plot the histogram of the correlation values: {t5 - t4:.2f} seconds")

# Save dense correlation matrix
print("Saving the full averaged correlation matrix...")
npy_name = f"correlation_matrix_full_{freq_str}_{sta_lta_suffix}_{station}.npy"
npy_path = Path(dirpath) / npy_name
save(npy_path, corr_mat)
print(f"Averaged correlation matrix saved to {npy_path}")
t6 = time()
print(f"Averaged correlation matrix saved in {t6 - t5:.2f} s.")

# Save dense lag matrix 
print(f"Saving dense lag matrix ...")
npy_name = f"lag_matrix_full_{freq_str}_{sta_lta_suffix}_{station}.npy"
npy_path = Path(dirpath) / npy_name
save(npy_path, lag_mat)
print(f"Lag matrix saved to {npy_path}")
t7 = time()
print(f"Lag matrix saved in {t7 - t6:.2f} s.")

print(f"Total time taken: {t7 - t0:.2f} s.")