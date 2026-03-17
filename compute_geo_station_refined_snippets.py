"""
Refine the STA/LTA detected snippets by removing the snippets that are not similar to enough other snippets
"""

from argparse import ArgumentParser
from pathlib import Path
from numpy import load, full, finfo, maximum, save
from scipy.sparse import csr_matrix
from matplotlib.pyplot import subplots
from time import time

from utils_basic import (
    DETECTION_DIR as dirpath, 
    SAMPLING_RATE as sampling_rate, 
    GEO_COMPONENTS as components,
    get_freq_limits_string,
)
from utils_sta_lta import Snippets, get_sta_lta_suffix
from utils_plot import save_figure

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def get_refined_snippet_indices(corr_matrix, cc_threshold, min_num_similar):
    """
    Get the indices of the refined snippets
    """
    """
    Get the indices of snippets that are similar to enough other snippets.

    Parameters
    ----------
    corr_matrix : ndarray
        The correlation matrix
    cc_threshold : float
        The threshold for the correlation coefficient
    min_num_similar : int
        The minimum number of similar snippets required

    Returns
    -------
    ndarray
        The indices of the refined snippets
    """
    # Convert to sparse matrix for efficiency since most elements are below threshold
    sparse_matrix = csr_matrix(corr_matrix > cc_threshold)
    
    # Count number of elements above threshold in each row
    num_similar = sparse_matrix.sum(axis=1).A1
    
    # Get indices where count exceeds minimum
    refined_indices = num_similar >= min_num_similar

    # Get the refined correlation matrix
    print(f"Getting the refined correlation matrix...")
    corr_matrix = corr_matrix[refined_indices, :][:, refined_indices]
    
    return refined_indices, corr_matrix

# -----------------------------------------------------------------------------
#  Parse the command line arguments
# -----------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("--station", type=str, required=True, help="The station name")

parser.add_argument("--min_freq_filter", type=float, default=20.0, help="The minimum frequency for the filter")
parser.add_argument("--max_freq_filter", type=float, default=None, help="The maximum frequency for the filter")
parser.add_argument("--sta_window_sec", type=float, default=0.005, help="The STA window length in seconds")
parser.add_argument("--lta_window_sec", type=float, default=0.05, help="The LTA window length in seconds")
parser.add_argument("--on_threshold", type=float, default=4.0, help="The threshold for the on-trigger")
parser.add_argument("--off_threshold", type=float, default=1.0, help="The threshold for the off-trigger")
parser.add_argument("--cc_threshold", type=float, default=0.85, help="The threshold for the correlation coefficient")
parser.add_argument("--min_num_similar", type=int, default=2, help="The minimum number of similar snippets (including itself)")


args = parser.parse_args()

station = args.station
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
sta_window_sec = args.sta_window_sec
lta_window_sec = args.lta_window_sec
on_threshold = args.on_threshold
off_threshold = args.off_threshold
cc_threshold = args.cc_threshold
min_num_similar = args.min_num_similar

print(f"Station: {station}")
print(f"Min frequency filter: {min_freq_filter}")
print(f"Max frequency filter: {max_freq_filter}")
print(f"STA window length: {sta_window_sec}")
print(f"LTA window length: {lta_window_sec}")
print(f"On threshold: {on_threshold}")
print(f"Off threshold: {off_threshold}")
print(f"CC threshold: {cc_threshold}")
print(f"Min. number of similar snippets (including itself): {min_num_similar}")


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

# Load the snippets

clock1 = time()
freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, on_threshold, off_threshold)
filename = f"snippets_sta_lta_{freq_str}_{sta_lta_suffix}_{station}.h5"
filepath = Path(dirpath) / filename
print(f"Loading snippets from {filepath}")
snippets = Snippets.from_hdf(filepath)

print(f"Loaded {len(snippets)} snippets")
clock2 = time()
print(f"Time taken to load the snippets: {clock2 - clock1:.2f} seconds")

# Load the correlation matrix
filename = f"correlation_matrix_full_{freq_str}_{sta_lta_suffix}_{station}.npy"
filepath = Path(dirpath) / filename
print(f"Loading correlation matrix from {filepath}")
corr_matrix = load(filepath)
clock3 = time()
print(f"Time taken to load the correlation matrix: {clock3 - clock2:.2f} seconds")

print(f"Loaded the correlation matrix with shape {corr_matrix.shape}")

# Get the indices of the refined snippets
print(f"Getting the indices of the refined snippets...")
refined_indices, corr_matrix_refined = get_refined_snippet_indices(corr_matrix, cc_threshold, min_num_similar)
print(f"Number of refined snippets: {sum(refined_indices)}")
clock4 = time()
print(f"Time taken to get the indices of the refined snippets: {clock4 - clock3:.2f} seconds")

# Get the refined snippets
snippets_refined = snippets[refined_indices]
clock5 = time()
print(f"Time taken to get the refined snippets: {clock5 - clock4:.2f} seconds")

# Save the refined snippets
filename = f"snippets_refined_{freq_str}_{sta_lta_suffix}_cc{cc_threshold:.2f}_min_num_similar{min_num_similar:d}.h5"
filepath = Path(dirpath) / filename
print(f"Saving the refined snippets to {filepath}")
snippets_refined.to_hdf(filepath)
print(f"Saved the refined snippets to {filepath}")

# Save the refined correlation matrix
filename = f"correlation_matrix_refined_{freq_str}_{sta_lta_suffix}_cc{cc_threshold:.2f}_min_num_similar{min_num_similar:d}.npy"
filepath = Path(dirpath) / filename
print(f"Saving the refined correlation matrix to {filepath}")
save(filepath, corr_matrix_refined)
print(f"Saved the refined correlation matrix to {filepath}")