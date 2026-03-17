"""
Align the satellite events with the hub event
"""

# Import libraries
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, read_hdf, read_json, Timedelta, to_datetime, DataFrame
from numpy import load, std, unique
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components
from matplotlib.pyplot import figure, subplots

# Import modules
from utils_basic import DETECTION_DIR as dirpath_event, ROOTDIR_GEO as dirpath_waveform
from utils_basic import STARTTIME_GEO as starttime_bin, ENDTIME_GEO as endtime_bin
from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix
from utils_plot import save_figure
from utils_basic import get_freq_limits_string, get_geophone_coords
from utils_basic import (
    INNER_STATIONS_A as stations_inner_a, 
    INNER_STATIONS_B as stations_inner_b,
    MIDDLE_STATIONS_A as stations_middle_a,
    MIDDLE_STATIONS_B as stations_middle_b,
    GEO_COMPONENTS as components,
    SAMPLING_RATE as sample_rate,
)

# --------------------------------------------------------------------------------------------------
# Define the functions
# --------------------------------------------------------------------------------------------------

"""
Convert the per-station times to datetime objects
"""
def convert_station_times(station_dict):
    return {
        station: to_datetime(times, utc=True, format="ISO8601")
        for station, times in station_dict.items()
    }

# --------------------------------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="Align the satellite events with the hub event")
    parser.add_argument("--group_label", type=int, required=True, help="The group label")
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--thr_on", type=float, default=4.0)
    parser.add_argument("--thr_off", type=float, default=1.0)
    parser.add_argument("--min_cc", type=float, default=0.85)
    parser.add_argument("--min_num_similar_snippet", type=int, default=10)
    parser.add_argument("--min_num_similar_station", type=int, default=3)

    args = parser.parse_args()
    group_label = args.group_label
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    thr_on = args.thr_on
    thr_off = args.thr_off
    min_cc = args.min_cc
    min_num_similar_snippet = args.min_num_similar_snippet
    min_num_similar_station = args.min_num_similar_station

    print("--------------------------------")
    print("Aligning the satellite events with the hub event...")
    print("--------------------------------")
    print(f"Group label: {group_label}")
    print(f"Min frequency filter: {min_freq_filter}")
    print(f"Max frequency filter: {max_freq_filter}")
    print(f"STA window sec: {sta_window_sec}")
    print(f"LTA window sec: {lta_window_sec}")
    print(f"On threshold: {thr_on}")
    print(f"Off threshold: {thr_off}")

    # Get the suffix
    print("Getting the suffix...")
    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    suffix_repeating_snippet = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar_snippet)
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

    # Load the event similarity matrix
    print("Loading the event similarity matrix...")
    filename = f"event_similarity_matrix_repeating_{suffix_group}.npz"
    filepath = join(dirpath_event, filename)
    event_similarity_matrix = load_npz(filepath)

    # Load the group information    
    print("Loading the group information...")
    filename = f"event_group_info_{suffix_group}.csv"
    filepath = join(dirpath_event, filename)
    group_df = read_csv(filepath)
    id_hub = group_df.loc[group_df["label"] == group_label, "id_hub"].values[0]
    print(f"Hub event ID: {id_hub}")

    # Load the event information
    print("Loading the event information...")
    filename = f"associated_events_repeating_{suffix}.jsonl"
    filepath = join(dirpath_event, filename)
    event_df = read_json(filepath, lines = True)
    event_df["first_onset"] = to_datetime(event_df["first_onset"], errors="coerce")
    event_df["per_station_times"] = event_df["per_station_times"].apply(convert_station_times)
    
    index_hub = event_df.index[event_df["event_id"] == id_hub].values[0]
    hub_dict = event_df.loc[event_df["event_id"] == id_hub].iloc[0].to_dict()
    stations_hub = hub_dict["stations"]
    print(f"Stations of the hub event: {stations_hub}")

    # Find the ids of the satellite events
    print("Finding the ids of the satellite events...")
    indices_satellite = event_similarity_matrix[index_hub, :].nonzero()[1]
    ids_satellite = event_df.iloc[indices_satellite]["event_id"].tolist()
    event_satellite_df = event_df.loc[event_df["event_id"].isin(ids_satellite)]
    print(f"Number of satellite events: {len(event_satellite_df)}")

    # Reading the detections for the hub-event stations
    print("Reading the detections for the hub-event stations...")
    det_station_dict = {}
    for station in stations_hub:
        filename = f"sta_lta_detections_associated_repeating_{suffix}_{station}.csv"
        filepath = join(dirpath_event, filename)
        det_df = read_csv(filepath, parse_dates=["starttime", "endtime"])
        det_station_dict[station] = dict(zip(det_df["id"], det_df.index))

    # Reading the lag matrices for the stations
    print("Reading the lag matrices for the stations...")
    lag_matrix_dict = {}
    for station in stations_hub:
        filename = f"lag_matrix_associated_repeating_{suffix}_{station}.npy"
        filepath = join(dirpath_event, filename)
        lag_matrix = load(filepath)
        lag_matrix_dict[station] = lag_matrix
        print(f"Lag matrix for {station}: {lag_matrix.shape}")

    # Align each satellite event with the hub event
    print("Aligning each satellite event with the hub event...")
    point_difference_stds = []
    event_alignment_dicts = [{"id": id_hub, "first_onset_station": hub_dict["first_onset_station"], "aligned_first_onset": hub_dict["first_onset"], "hub": True}]
    for _, row in event_satellite_df.iterrows():
        event_id = row["event_id"]

        station_first_onset = row["first_onset_station"]
        if station_first_onset != hub_dict["first_onset_station"]:
            continue

        print("--------------------------------")
        print(f"Aligning event {event_id} with the hub event...")
        print("--------------------------------")
        stations_satellite = row["stations"]
        stations_to_align = list(set(stations_satellite) & set(stations_hub))
        print(f"Stations to align: {stations_to_align}")

        starttime_aligned_dict = {}
        time_differences = []
        print("Aligning each station...")
        for station in stations_to_align:
            det_id_satellite = row["per_station_ids"][station]
            det_id_hub = hub_dict["per_station_ids"][station]

            det_index_satellite = det_station_dict[station][det_id_satellite]
            det_index_hub = det_station_dict[station][det_id_hub]

            lag_matrix = lag_matrix_dict[station]
            lag = lag_matrix[det_index_satellite, det_index_hub]
            print(f"Lag for {station}: {lag}")

            ## Apply the correction to the starttime of the satellite event
            starttime_satellite = row["per_station_times"][station][0]
            starttime_hub = hub_dict["per_station_times"][station][0]
            starttime_aligned = starttime_satellite - Timedelta(seconds = lag / sample_rate)
            print(f"Starttime aligned for {station}: {starttime_aligned}")
            time_difference = (starttime_aligned - starttime_hub).total_seconds()
            starttime_aligned_dict[station] = starttime_aligned
            time_differences.append(time_difference)

        time_difference_std = std(time_differences)
        point_difference_std = int(time_difference_std * sample_rate)

        if point_difference_std == 0:
            first_onset_aligned = starttime_aligned_dict[station_first_onset]
            event_alignment_dicts.append({"id": event_id, "first_onset_station": station_first_onset, "aligned_first_onset": first_onset_aligned, "hub": False})
            print(f"Aligned first onset: {first_onset_aligned}")

        print(f"Point difference standard deviation: {point_difference_std}")
        point_difference_stds.append(point_difference_std)

    point_difference_std_std = std(point_difference_stds)
    print(f"Point difference standard deviation standard deviation: {point_difference_std_std}")

print(f"Number of event alignments: {len(event_alignment_dicts)}")

# Save the event alignment information
print("Saving the event alignment information...")
event_alignment_df = DataFrame(event_alignment_dicts)
filename = f"event_alignments_group{group_label:d}_{suffix_group}.csv"
filepath = join(dirpath_event, filename)
event_alignment_df.to_csv(filepath, index=False)
print(f"Saved the event alignment information to {filepath}.")

# Plot a histogram of the point differences
print("Plotting a histogram of the point differences...")
unique_point_differences, point_difference_counts = unique(point_difference_stds, return_counts=True)
point_difference_fractions = point_difference_counts / point_difference_counts.sum()
fig, ax = subplots(figsize=(10, 5))
ax.bar(unique_point_differences, point_difference_fractions, color="lightskyblue", edgecolor="black")
ax.set_xlabel("Number of samples", fontsize=12)
ax.set_ylabel("Fraction", fontsize=12)
ax.set_title("Standard deviation of time differences", fontweight="bold", fontsize=14)
save_figure(fig, f"time_difference_std_histogram_group{group_label:d}_{suffix_group}.png")