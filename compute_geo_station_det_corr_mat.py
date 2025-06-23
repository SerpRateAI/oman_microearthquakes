"""
Compute the correlation matrix of the raw STA/LTA detections of a geophone station and extract the popular ones
"""

from pathlib import Path
from argparse import ArgumentParser
import time
from numpy import histogram, log10

from numpy import (
    zeros,
    asarray,
    arange,
    full,
    empty,
    int8,
    int32,
    float32,
    where,
    nonzero,
)
from numpy.linalg import norm
from scipy.sparse import (
    isspmatrix,
    isspmatrix_csr,
    coo_matrix,
    save_npz
)
from tqdm import tqdm
from matplotlib.pyplot import subplots, imshow, axis, title, savefig, close, show

from utils_basic import DETECTION_DIR as dirpath, GEO_COMPONENTS as components
from utils_sta_lta import Snippets
from utils_cluster import save_csr_to_hdf5, plot_similarity_matrix
from utils_plot import save_figure


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

"""Unit-normalize the 3-component snippets"""
def unitnorm(snippets):
    num_snips = len(snippets)
    num_pts = snippets[0].num_pts

    # Initialize the normalized snippets
    snippet_dict = {comp: zeros((num_snips, num_pts), dtype=float32) for comp in components}

    # Normalize the snippets
    for idx, snip in enumerate(snippets):
        for comp in components:
            waveform = snip.waveform[comp]
            snip_norm = norm(waveform)
            if snip_norm:
                waveform /= snip_norm 

            snippet_dict[comp][idx, :] = waveform

    return snippet_dict


"""Build the similarity matrix"""
def build_similarity_matrix(snippets, cc_threshold, max_shift, chunk_size):
    

    num_snips, max_len = snippets.shape
    pair_max = full((num_snips, num_snips), -1.0, dtype=float32)
    hit_rows, hit_cols = [], []

    for lag in tqdm(range(-max_shift, max_shift + 1), desc="lag"):
        if lag >= 0:
            slice_left, slice_right = slice(0, max_len - lag), slice(lag, max_len)
        else:
            slice_left, slice_right = slice(-lag, max_len), slice(0, max_len + lag)

        view_left = snippets[:, slice_left]
        view_right = snippets[:, slice_right]

        for row_start in range(0, num_snips, chunk_size):
            row_end = min(row_start + chunk_size, num_snips)
            # explicit column slice for readability (all columns)
            block_left = view_left[row_start:row_end, :]
            corr_block = block_left @ view_right.T

            mask = corr_block > cc_threshold
            rows_rel, cols_rel = nonzero(mask)
            rows_rel += row_start
            keep = rows_rel < cols_rel
            rows_rel, cols_rel = rows_rel[keep], cols_rel[keep]

            improved = corr_block[rows_rel - row_start, cols_rel] > pair_max[
                rows_rel, cols_rel
            ]
            rows_rel, cols_rel = rows_rel[improved], cols_rel[improved]
            pair_max[rows_rel, cols_rel] = corr_block[rows_rel - row_start, cols_rel]

            hit_rows.extend(rows_rel)
            hit_cols.extend(cols_rel)

    if hit_rows:
        hit_rows = asarray(hit_rows, dtype=int32)
        hit_cols = asarray(hit_cols, dtype=int32)
        hit_data = full(hit_rows.shape, True, dtype=bool)
    else:
        hit_rows = hit_cols = hit_data = empty(0, dtype=int32)

    sim = coo_matrix((hit_data, (hit_rows, hit_cols)), shape=(num_snips, num_snips), dtype=bool)
    sim = sim + sim.T
    sim.setdiag(True)

    return sim.tocsr()

def plot_row_degrees(csr_matrix,
                     title = None,
                     max_degree = 300,
                     bin_width = 10,
                     figsize = (10, 5),
                     linewidth = 1,
                     color = 'tab:cyan',
                     ):
    row_degrees = csr_matrix.sum(axis=1).A1
    bin_edges = arange(0, max_degree, bin_width)
    hist, bin_edges = histogram(row_degrees, bins=bin_edges)
    bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
    fig, ax = subplots(figsize=figsize)
    ax.bar(bin_centers, hist, width=bin_width, linewidth=linewidth, color=color, edgecolor='black')
    ax.set_xlabel('Similarity degree', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    return fig, ax

# -----------------------------------------------------------------------------
#  Parse the command line arguments
# -----------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("--station", type=str, required=True, help="The station name")

parser.add_argument("--on_threshold", type=float, required=True, help="The threshold for the on-trigger")
parser.add_argument("--cc_threshold", type=float, default=0.85, help="The threshold for the cross-correlation")
parser.add_argument("--max_shift", type=int, default=20, help="The maximum shift for the sliding-lag search")
parser.add_argument("--chunk_size", type=int, default=3000, help="The chunk size for the sliding-lag search")

args = parser.parse_args()

station = args.station
on_threshold = args.on_threshold
cc_threshold = args.cc_threshold
max_shift = args.max_shift
chunk_size = args.chunk_size

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

# Load the detections
print(f"Loading detections from {dirpath}")
clock1 = time.time()
filename = f"raw_sta_lta_detections_{station}_on{on_threshold:.1f}.h5"
filepath = Path(dirpath) / filename
detections = Snippets.from_hdf(filepath)
clock2 = time.time()

# for detection in detections:
#     print(detection.num_pts)

print(f"Loaded {len(detections)} detections")
print(f"Detections loaded in {clock2 - clock1:.2f} seconds")

# Unit-normalize and pad the snippets
clock3 = time.time()
snippet_dict = unitnorm(detections)
clock4 = time.time()
print(f"Snippets unit-normalized in {clock4 - clock3:.2f} seconds")

# Build the similarity matrix
clock5 = time.time()
sim_matrix_dict = {}
for comp in components:
    print(f"Building the similarity matrix for Component {comp}...")
    sim_matrix_dict[comp] = build_similarity_matrix(snippet_dict[comp], cc_threshold, max_shift, chunk_size)

sim_matrix = sim_matrix_dict['Z'] * sim_matrix_dict['1'] * sim_matrix_dict['2']

# Plot the row degrees histogram
clock6 = time.time()
fig, ax = plot_row_degrees(sim_matrix, title=f'{station}', figsize=(10, 5))
filename = f"row_degrees_histogram_{station}_on{on_threshold:.1f}_cc{cc_threshold:.2f}.png"
filepath = Path(dirpath) / filename
save_figure(fig, filepath)
close()
clock7 = time.time()
print(f"Row degrees of the similarity matrix computed and plotted in {clock7 - clock6:.2f} seconds")

# Save the similarity matrix
clock7 = time.time()
filename = f"similarity_matrix_{station}_on{on_threshold:.1f}_cc{cc_threshold:.2f}.npz"
filepath = Path(dirpath) / filename
save_npz(filepath, sim_matrix)
clock8 = time.time()
close()
print(f"Similarity matrix saved to {filepath}")
print(f"Similarity matrix saved in {clock8 - clock7:.2f} seconds")
