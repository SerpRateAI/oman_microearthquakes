"""
Classify the shallow impulsive events based on the similarity matrix.

The similarity matrix is a CSR matrix of shape (n_events, n_events) where each entry is 1 if the two events are similar and 0 otherwise.

"""

from argparse import ArgumentParser
from pathlib import Path
from pandas import read_json, DataFrame
from numpy import unique, where
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components
from matplotlib.pyplot import figure, hist

from utils_basic import (
    DETECTION_DIR as dirpath,
    GEO_STATIONS as stations,
)

from utils_basic import get_freq_limits_string
from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix

from utils_plot import save_figure

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def classify(similarity_matrix):
    """
    Classify the events into different categories based on the similarity matrix.
    """
    # Get the number of connected components
    num_comp, labels = connected_components(similarity_matrix)
    print(f"Number of connected components: {num_comp}")
    
    return labels, num_comp

def get_group_info(labels_unique, labels_event, similarity_matrix, event_df):
    """
    Get the group information, including the number of events, the hub event, and the degree of the hub event.
    """
    output_dicts = []
    for label in labels_unique:
        indices = where(labels_event == label)[0]
        num_events = len(indices)
        degrees = similarity_matrix[indices, :][:, indices].sum(axis=1)
        degree_hub = degrees.max()
        index_hub = indices[degrees.argmax()]
        id_hub = event_df.iloc[index_hub]["event_id"]
        output_dicts.append({"label": label, "num_events": num_events, "id_hub": id_hub, "degree_hub": degree_hub})

    output_df = DataFrame(output_dicts)

    return output_df


def plot_label_histogram(labels, num_comp):
    """
    Plot a histogram of the labels.
    """
    fig = figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.hist(labels, bins=range(num_comp + 1))
    ax.set_yscale("log")

    ax.set_xlabel("Group label", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Number of events per group", fontsize=14, fontweight="bold")

    ax.set_xlim(0, num_comp)
    
    return fig, ax

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
    event_df = read_json(filepath, lines = True)

    # Load the similarity matrix
    print("--------------------------------")
    print("Loading the similarity matrix...")
    print("--------------------------------")
    suffix_sim = f"{suffix}_num_sim_sta{min_num_similar_station:d}"
    filename = f"event_similarity_matrix_repeating_{suffix_sim}.npz"
    filepath = Path(dirpath) / filename
    similarity_matrix = load_npz(filepath)

    # Classify the events
    print("--------------------------------")
    print("Classifying the events...")
    print("--------------------------------")
    labels_event, num_comp = classify(similarity_matrix)

    # Get the number of connected components with more than one event
    print("--------------------------------")
    print("Getting the number of connected components with more than one event...")
    print("--------------------------------")
    labels_unique, counts = unique(labels_event, return_counts=True)
    num_comp_gt_1 = (counts > 1).sum()
    labels_unique_gt_1 = labels_unique[counts > 1]
    print(f"Number of connected components with more than one event: {num_comp_gt_1}")

    # Get the group information
    print("--------------------------------")
    print("Getting the group information...")
    print("--------------------------------")
    group_info_df = get_group_info(labels_unique_gt_1, labels_event, similarity_matrix, event_df)

    # Save the labels
    print("--------------------------------")
    print("Saving the labels...")
    print("--------------------------------")
    ids = event_df["event_id"].tolist()
    output_df = DataFrame({"id": ids, "label": labels_event})
    outpath = Path(dirpath) / f"event_labels_{suffix_sim}.csv"
    output_df.to_csv(outpath, index = False)
    print(f"Event labels saved to {outpath}")

    # Save the number of events per label for connected components with more than one event
    print("--------------------------------")
    print("Saving the group information for connected components with more than one event...")
    print("--------------------------------")
    outpath = Path(dirpath) / f"event_group_info_{suffix_sim}.csv"
    group_info_df = group_info_df.sort_values(by = "num_events", ascending = False)
    group_info_df.to_csv(outpath, index = False)
    print(f"Group information saved to {outpath}")

    # For each group, save the event IDs
    print("--------------------------------")
    print("Saving the event information for each group...")
    print("--------------------------------")
    for label in group_info_df["label"]:
        indices = where(labels_event == label)[0]
        event_group_df = event_df.iloc[indices]
        outpath = Path(dirpath) / f"grouped_events_group{label}_{suffix_sim}.jsonl"
        event_group_df.to_json(outpath, lines = True, orient = "records")
        print(f"Event information for group {label} saved to {outpath}")
    
    # Plot a histogram of the labels
    print("--------------------------------")
    print("Plotting the histogram of the labels...")
    print("--------------------------------")
    fig, _ = plot_label_histogram(labels_event, num_comp)
    save_figure(fig, f"event_label_histogram_{suffix_sim}.png")