"""
Extract the the information of the repeating signals, which are defined as the signals similar to over a certain number of other signals.
"""

from argparse import ArgumentParser
from numpy import load, where, save
from pathlib import Path
from time import time
from pandas import read_csv

from utils_basic import (
    DETECTION_DIR as dirpath,
    GEO_STATIONS as stations,
    get_freq_limits_string,
)
from utils_cc import get_repeating_snippet_suffix
from scipy.sparse import csr_matrix
from utils_sta_lta import get_sta_lta_suffix, Snippets


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_repeating_snippet_indices(corr_matrix, min_cc, min_num_similar):
    """
    Get the indices of the repeating snippets.
    """
    # Convert to sparse matrix for efficiency since most elements are below threshold
    sparse_matrix = csr_matrix(corr_matrix > min_cc)
    
    # Count number of elements above threshold in each row
    sim_numbers = sparse_matrix.sum(axis=1).A1
    
    # Get indices where count exceeds minimum
    repeating_indices = where(sim_numbers >= min_num_similar)[0]

    return repeating_indices

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)

    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--on_threshold", type=float, default=4.0)
    parser.add_argument("--off_threshold", type=float, default=1.0)
    parser.add_argument("--min_cc", type=float, default=0.85)
    parser.add_argument("--min_num_similar", type=int, default=10)

    args = parser.parse_args()
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    on_threshold = args.on_threshold
    off_threshold = args.off_threshold
    min_cc = args.min_cc
    min_num_similar = args.min_num_similar

    # Get the station list
    for station in stations:
        print(f"Processing station {station}...")

        # Load the full correlation matrix
        freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
        sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, on_threshold, off_threshold)
        filename = f"correlation_matrix_full_{freq_str}_{sta_lta_suffix}_{station}.npy"
        filepath = Path(dirpath) / filename
        corr_mat = load(filepath)
        num_snips = corr_mat.shape[0]
        print(f"Loaded the correlation matrix from {filepath}")
        print(f"Number of snippets: {num_snips}")

        # Get the suffix of the repeating snippet file
        repeating_snippet_suffix = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar)

        # Get the repeating signal indices
        repeating_indices = get_repeating_snippet_indices(corr_mat, min_cc = min_cc, min_num_similar = min_num_similar)
        print(f"Number of repeating snippets: {len(repeating_indices)}")
        print(f"Percentage of repeating snippets: {len(repeating_indices) / num_snips * 100:.2f}%")

        # Extract the correlation matrix of the repeating snippets
        corr_mat_repeating = corr_mat[repeating_indices, :][:, repeating_indices]
        print(f"Shape of the correlation matrix of the repeating snippets: {corr_mat_repeating.shape}")

        # Save the correlation matrix of the repeating snippets
        filename = f"correlation_matrix_repeating_{freq_str}_{sta_lta_suffix}_{repeating_snippet_suffix}_{station}.npy"
        filepath = Path(dirpath) / filename
        save(filepath, corr_mat_repeating)
        print(f"Saved the correlation matrix of the repeating snippets to {filepath}")

        # Free the memory
        del corr_mat
        del corr_mat_repeating

        # Load the full lag matrix
        filename = f"lag_matrix_full_{freq_str}_{sta_lta_suffix}_{station}.npy"
        filepath = Path(dirpath) / filename
        lag_mat = load(filepath)
        print(f"Loaded the lag matrix from {filepath}")
        print(f"Shape of the lag matrix: {lag_mat.shape}")

        # Extract the lag matrix of the repeating snippets
        lag_mat_repeating = lag_mat[repeating_indices, :][:, repeating_indices]
        print(f"Shape of the lag matrix of the repeating snippets: {lag_mat_repeating.shape}")

        # Save the lag matrix of the repeating snippets
        filename = f"lag_matrix_repeating_{freq_str}_{sta_lta_suffix}_{repeating_snippet_suffix}_{station}.npy"
        filepath = Path(dirpath) / filename
        save(filepath, lag_mat_repeating)
        print(f"Saved the lag matrix of the repeating snippets to {filepath}")

        # Free the memory
        del lag_mat
        del lag_mat_repeating

        # Load the snippets
        filename = f"snippets_sta_lta_{freq_str}_{sta_lta_suffix}_{station}.h5"
        filepath = Path(dirpath) / filename
        snippets = Snippets.from_hdf(filepath)
        print(f"Loaded the snippets from {filepath}")
        print(f"Number of snippets: {len(snippets)}")

        # Extract the snippets of the repeating snippets
        snippets_repeating = snippets[repeating_indices]
        # Save the snippets of the repeating snippets
        filename = f"snippets_repeating_{freq_str}_{sta_lta_suffix}_{repeating_snippet_suffix}_{station}.h5"
        filepath = Path(dirpath) / filename
        snippets_repeating.to_hdf(filepath)
        print(f"Saved the snippets of the repeating snippets to {filepath}")

        # Free the memory
        del snippets
        del snippets_repeating

        # Read the detection list
        filename = f"sta_lta_detections_{freq_str}_{sta_lta_suffix}_{station}.csv"
        filepath = Path(dirpath) / filename
        det_df = read_csv(filepath, parse_dates=["starttime", "endtime"])
        print(f"Loaded the detection list from {filepath}")
        print(f"Number of detections: {len(det_df)}")

        # Extract the detections of the repeating snippets
        det_df_repeating = det_df.iloc[repeating_indices]

        # Save the detections of the repeating snippets
        filename = f"sta_lta_detections_repeating_{freq_str}_{sta_lta_suffix}_{repeating_snippet_suffix}_{station}.csv"
        filepath = Path(dirpath) / filename
        det_df_repeating.to_csv(filepath, index=False)
        print(f"Saved the detections of the repeating snippets to {filepath}")

        # Free the memory
        del det_df
        del det_df_repeating