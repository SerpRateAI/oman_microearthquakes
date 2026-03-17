"""
Extract the the information of the associated snippets
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

def get_associated_snippet_indices(corr_matrix, min_cc, min_num_similar):
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
    parser.add_argument("--repeating", action="store_true", help="If set, process the repeating snippets only", default=True)
    parser.add_argument("--min_cc", type=float, default=0.85)
    parser.add_argument("--min_num_similar", type=int, default=10)
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)

    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--on_threshold", type=float, default=4.0)
    parser.add_argument("--off_threshold", type=float, default=1.0)

    args = parser.parse_args()
    repeating = args.repeating
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    on_threshold = args.on_threshold
    off_threshold = args.off_threshold
    min_cc = args.min_cc
    min_num_similar = args.min_num_similar

    # Assemble the file suffix
    freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
    sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, on_threshold, off_threshold)
    
    suffix = f"{freq_str}_{sta_lta_suffix}"
    if repeating:
        suffix = f"repeating_{suffix}"
        repeating_snippet_suffix = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar)
        suffix += f"_{repeating_snippet_suffix}"
        print(f"Suffix: {suffix}")

    # Load the associated detections
    filename = f"associated_detections_{suffix}.csv"
    filepath = Path(dirpath) / filename
    det_df = read_csv(filepath, parse_dates=["starttime", "endtime"])
    print(f"Loaded the associated detections from {filepath}")
    print(f"Number of associated detections: {len(det_df)}")

    # Extract the associated snippets for each station
    for station in stations:
        print("--------------------------------")
        print(f"Processing station {station}...")
        print("--------------------------------")

        # Extract the associated snippets for the station
        det_station_df = det_df[det_df["station"] == station]
        num_det_station = len(det_station_df)
        print(f"Number of associated detections for station {station}: {num_det_station}")

        if num_det_station < 2: # Need at least two detections to extract the associated snippets
            print(f"Not enough associated detections for station {station}. Skipping...")
            continue

        # Save the associated detections for the station
        filename = f"sta_lta_detections_associated_{suffix}_{station}.csv"
        filepath = Path(dirpath) / filename
        det_out_df = det_station_df.copy()
        det_out_df = det_out_df[["snippet_id", "starttime", "endtime"]]
        det_out_df.columns = ["id", "starttime", "endtime"]
        det_out_df.to_csv(filepath, index=False)
        print(f"Saved the associated detections for the station to {filepath}")

        # Read the snippets
        filename = f"snippets_{suffix}_{station}.h5"
        filepath = Path(dirpath) / filename
        snippets = Snippets.from_hdf(filepath)
        print(f"Loaded the snippets from {filepath}")
        print(f"Number of snippets: {len(snippets)}")

        # Extract the associated snippets for the station
        snippets_associated, indices = snippets.select_by_id(det_station_df["snippet_id"].tolist(), return_index=True)
        print(f"Number of associated snippets for station {station}: {len(snippets_associated)}")

        # Save the associated snippets for the station
        filename = f"snippets_associated_{suffix}_{station}.h5"
        filepath = Path(dirpath) / filename
        snippets_associated.to_hdf(filepath)
        print(f"Saved the associated snippets for the station to {filepath}")
        
        # Read the correlation matrix
        filename = f"correlation_matrix_{suffix}_{station}.npy"
        filepath = Path(dirpath) / filename
        corr_matrix = load(filepath)
        print(f"Loaded the correlation matrix from {filepath}")
        print(f"Shape of the correlation matrix: {corr_matrix.shape}")
        
        # Extract the correlation matrix of the associated snippets
        corr_matrix_associated = corr_matrix[indices, :][:, indices]
        print(f"Shape of the correlation matrix of the associated snippets: {corr_matrix_associated.shape}")

        # Save the correlation matrix of the associated snippets
        filename = f"correlation_matrix_associated_{suffix}_{station}.npy"
        filepath = Path(dirpath) / filename
        save(filepath, corr_matrix_associated)
        print(f"Saved the correlation matrix of the associated snippets to {filepath}")

        # Read the lag matrix
        filename = f"lag_matrix_{suffix}_{station}.npy"
        filepath = Path(dirpath) / filename
        lag_matrix = load(filepath)
        print(f"Loaded the lag matrix from {filepath}")
        print(f"Shape of the lag matrix: {lag_matrix.shape}")

        # Extract the lag matrix of the associated snippets
        lag_matrix_associated = lag_matrix[indices, :][:, indices]
        print(f"Shape of the lag matrix of the associated snippets: {lag_matrix_associated.shape}")

        # Save the lag matrix of the associated snippets
        filename = f"lag_matrix_associated_{suffix}_{station}.npy"
        filepath = Path(dirpath) / filename
        save(filepath, lag_matrix_associated)
        print(f"Saved the lag matrix of the associated snippets to {filepath}")

        # Free the memory
        del corr_matrix
        del lag_matrix
        del corr_matrix_associated
        del lag_matrix_associated