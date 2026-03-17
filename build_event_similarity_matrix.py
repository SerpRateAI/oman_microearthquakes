"""
Build the event similarity matrix
"""

from argparse import ArgumentParser
from pathlib import Path
from os.path import join
from numpy import load, zeros, save
from pandas import read_json, read_csv
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz
from utils_basic import (
    DETECTION_DIR as dirpath,
    GEO_STATIONS as stations,
    get_freq_limits_string,
)

from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def build_similarity_matrix(event_df, station_det_dict, station_corr_dict, min_cc, min_num_similar_station):
    """
    Build the similarity matrix

    Parameters
    ----------
    event_df : DataFrame
        The event dataframe
    station_det_dict : dict
        The dictionary of the associated detections for each station
    station_corr_dict : dict
        The dictionary of the correlation matrix for each station
    min_cc : float
        The minimum correlation coefficient
    min_num_similar_station : int
        The minimum number of similar stations required

    Returns
    -------
    ndarray
        The similarity matrix
    """
    # Get the number of events
    num_event = len(event_df)

    # Initialize the similarity matrix
    sim_matrix = zeros((num_event, num_event))

    # Build the per-station detection ID dictionary
    id2index_dict = {}
    for station, detection_df in station_det_dict.items():
        id2index_dict[station] = {}
        for index, row in detection_df.iterrows():
            id2index_dict[station][row["id"]] = index

    # Iterate over the events
    for i in tqdm(range(num_event), desc="Building similarity matrix"):
        stations_i = event_df["stations"][i]
        per_station_ids_i = event_df["per_station_ids"][i]

        for j in range(i+1, num_event):
            stations_j = event_df["stations"][j]
            per_station_ids_j = event_df["per_station_ids"][j]

            stations_common= list(set(stations_i) & set(stations_j))
            if len(stations_common) < min_num_similar_station:
                continue

            num_sim = 0
            for station in stations_common:
                # Get the correlation matrix for the station
                corr_matrix = station_corr_dict[station]

                # Get the detection dataframe for the station
                detection_df = station_det_dict[station]

                # Get the detection ID for
                detection_id_i = per_station_ids_i[station]
                detection_id_j = per_station_ids_j[station]

                # Get the index of the detection in the correlation matrix
                index_i = id2index_dict[station][detection_id_i]
                index_j = id2index_dict[station][detection_id_j]

                # Get the correlation value
                corr_value = corr_matrix[index_i, index_j]

                if corr_value >= min_cc:
                    num_sim += 1

            if num_sim >= min_num_similar_station:
                sim_matrix[i, j] = 1
                sim_matrix[j, i] = 1

    return sim_matrix

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--min_cc", type=float, default=0.85)
    parser.add_argument("--min_num_similar_snippet", type=int, default=10)
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--on_threshold", type=float, default=4.0)
    parser.add_argument("--off_threshold", type=float, default=1.0)
    parser.add_argument("--min_num_similar_station", type=int, default=3)

    args = parser.parse_args()
    min_cc = args.min_cc
    min_num_similar_snippet = args.min_num_similar_snippet
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    on_threshold = args.on_threshold
    off_threshold = args.off_threshold
    min_num_similar_station = args.min_num_similar_station

    # Assemble the file suffix
    freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
    sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, on_threshold, off_threshold)
    repeating_snippet_suffix = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar_snippet)
    suffix = f"{freq_str}_{sta_lta_suffix}_{repeating_snippet_suffix}"


    # Load the associated events
    print("--------------------------------")
    print("Loading the associated events...")
    print("--------------------------------")

    filename = f"associated_events_repeating_{suffix}.jsonl"
    filepath = Path(dirpath) / filename

    print("filepath:", filepath)
    print("exists:", filepath.exists())
    print("size:", filepath.stat().st_size if filepath.exists() else None)

    # show first few raw lines (including blanks)
    with filepath.open("r") as f:
        for i in range(5):
            line = f.readline()
            print(f"raw line {i+1}:", repr(line[:200]))

    event_df = read_json(filepath, lines = True)
    print(f"Loaded the associated events from {filepath}")
    print(f"Number of associated events: {len(event_df)}")

    # Load the associated detections for each station
    station_det_dict = {}
    print("--------------------------------")
    print("Loading the associated detections for each station...")
    print("--------------------------------")
    for station in stations:
        print(f"Processing station {station}...")
        filename = f"sta_lta_detections_associated_repeating_{suffix}_{station}.csv"
        filepath = Path(dirpath) / filename
        if not filepath.exists():
            print(f"File {filepath} does not exist. Skipping...")
            continue
        det_df = read_csv(filepath, parse_dates=["starttime", "endtime"])
        print(f"Loaded the associated detections from {filepath}")
        print(f"Number of associated detections: {len(det_df)}")
        station_det_dict[station] = det_df

    # Load the correlation matrix for each station
    station_corr_dict = {}
    print("--------------------------------")
    print("Loading the correlation matrix for each station...")
    print("--------------------------------")
    for station in stations:
        print(f"Processing station {station}...")
        filename = f"correlation_matrix_associated_repeating_{suffix}_{station}.npy"
        filepath = Path(dirpath) / filename
        if not filepath.exists():
            print(f"File {filepath} does not exist. Skipping...")
            continue
        corr_matrix = load(filepath)
        print(f"Loaded the correlation matrix from {filepath}")
        print(f"Shape of the correlation matrix: {corr_matrix.shape}")
        station_corr_dict[station] = corr_matrix

    # Build the event similarity matrix
    print("--------------------------------")
    print("Building the event similarity matrix...")
    print("--------------------------------")
    event_similarity_matrix = build_similarity_matrix(event_df, station_det_dict, station_corr_dict, min_cc, min_num_similar_station)

    # Find the sparsity of the event similarity matrix
    print("--------------------------------")
    print("Finding the sparsity of the event similarity matrix...")
    print("--------------------------------")
    sparsity = 1 - event_similarity_matrix.sum() / event_similarity_matrix.size
    print(f"Sparsity of the event similarity matrix: {sparsity:.2f}")

    # Convert the event similarity matrix to a CSR matrix
    event_similarity_matrix = csr_matrix(event_similarity_matrix, dtype=bool)

    # Save the event similarity matrix
    print("--------------------------------")
    print("Saving the CSR matrix")
    print("--------------------------------")
    suffix += f"_num_sim_sta{min_num_similar_station:d}"
    filename = f"event_similarity_matrix_repeating_{suffix}.npz"
    filepath = Path(dirpath) / filename
    save_npz(filepath, event_similarity_matrix)
    print(f"Saved the CSR matrix to {filepath}")