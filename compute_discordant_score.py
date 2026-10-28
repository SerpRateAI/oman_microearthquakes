"""
Compute discordant scores between the hub event and its aligned satellite events.

The discordant score on a station is 2 * (cc_short - cc_long), where cc_short and
cc_long are normalized 3-component dot products between lag-aligned hub and
satellite waveforms in the hub-defined short window and a long window twice as
long (same start time). Event-level scores are averaged over hub recording stations.
"""

from argparse import ArgumentParser
from os.path import join

from numpy import asarray, float32, mean
from numpy.linalg import norm
from pandas import DataFrame, Timedelta, read_csv, read_json, to_datetime
from numpy import load

from utils_basic import (
    DETECTION_DIR as dirpath_event,
    GEO_COMPONENTS as components,
    ROOTDIR_GEO as dirpath_waveform,
    SAMPLING_RATE as sample_rate,
    get_freq_limits_string,
)
from utils_cc import get_repeating_snippet_suffix
from utils_cont_waveform import load_waveform_slice
from utils_sta_lta import get_sta_lta_suffix


def convert_station_times(station_dict):
    return {
        station: to_datetime(times, utc=True, format="ISO8601")
        for station, times in station_dict.items()
    }


def load_group_data(group_label, suffix_group):
    """Load grouped events JSONL and event alignments CSV for a group."""
    filename = f"grouped_events_group{group_label:d}_{suffix_group}.jsonl"
    event_df = read_json(join(dirpath_event, filename), lines=True)
    event_df["first_onset"] = to_datetime(event_df["first_onset"], errors="coerce")
    event_df["per_station_times"] = event_df["per_station_times"].apply(convert_station_times)

    filename = f"event_alignments_group{group_label:d}_{suffix_group}.csv"
    alignment_df = read_csv(join(dirpath_event, filename))
    alignment_df["aligned_time"] = to_datetime(alignment_df["aligned_time"], format="ISO8601")
    alignment_df["hub"] = alignment_df["hub"].astype(bool)

    hub_rows = alignment_df.loc[alignment_df["hub"]]
    if hub_rows.empty:
        raise ValueError(f"No hub event found in alignments for group {group_label}.")
    id_hub = int(hub_rows.iloc[0]["id"])

    hub_event_rows = event_df.loc[event_df["event_id"] == id_hub]
    if hub_event_rows.empty:
        raise ValueError(f"Hub event {id_hub} not found in grouped events for group {group_label}.")
    hub_dict = hub_event_rows.iloc[0].to_dict()

    return event_df, alignment_df, hub_dict, id_hub


def load_detection_maps_and_lag_matrices(stations_hub, suffix):
    """Load per-station detection id->index maps and lag matrices."""
    det_station_dict = {}
    lag_matrix_dict = {}
    for station in stations_hub:
        filename = f"sta_lta_detections_associated_repeating_{suffix}_{station}.csv"
        det_df = read_csv(join(dirpath_event, filename), parse_dates=["starttime", "endtime"])
        det_station_dict[station] = dict(zip(det_df["id"], det_df.index))

        filename = f"lag_matrix_associated_repeating_{suffix}_{station}.npy"
        lag_matrix_dict[station] = load(join(dirpath_event, filename))

    return det_station_dict, lag_matrix_dict


def get_short_long_windows(t0_hub, t1_hub, t0_sat, t1_sat, lag_samples):
    """
    Build short and long time windows for hub and lag-aligned satellite.

    Short: hub per_station_times; satellite times corrected by lag.
    Long: same start as short, duration = 2 * short duration.
    """
    lag_td = Timedelta(seconds=lag_samples / sample_rate)
    start_sat = t0_sat - lag_td
    end_sat = t1_sat - lag_td

    dur_short = t1_hub - t0_hub
    dur_long = 2 * dur_short

    hub_short = (t0_hub, t1_hub)
    hub_long = (t0_hub, t0_hub + dur_long)
    sat_short = (start_sat, end_sat)
    sat_long = (start_sat, start_sat + dur_long)

    return hub_short, hub_long, sat_short, sat_long


def trim_waveform_dicts_to_common_length(waveform_dict_a, waveform_dict_b):
    """Trim both waveform dicts to the same number of samples per component."""
    trimmed_a = {}
    trimmed_b = {}
    for comp in components:
        n_pts = min(len(waveform_dict_a[comp]), len(waveform_dict_b[comp]))
        trimmed_a[comp] = asarray(waveform_dict_a[comp][:n_pts], dtype=float32)
        trimmed_b[comp] = asarray(waveform_dict_b[comp][:n_pts], dtype=float32)
    return trimmed_a, trimmed_b


def unitnorm_waveform_dict(waveform_dict):
    """Unit-normalize each component in a 3-component waveform dict."""
    normed = {}
    for comp in components:
        waveform = asarray(waveform_dict[comp], dtype=float32)
        waveform_norm = norm(waveform)
        if waveform_norm:
            waveform = waveform / waveform_norm
        normed[comp] = waveform
    return normed


def compute_normalized_dot_product(waveform_dict_a, waveform_dict_b):
    """
    Normalized 3-component dot product between two aligned waveforms.

    Returns the mean dot product over Z, 1, and 2 components after unit
    normalization (equivalent to zero-lag cross-correlation).
    """
    trimmed_a, trimmed_b = trim_waveform_dicts_to_common_length(waveform_dict_a, waveform_dict_b)
    norm_a = unitnorm_waveform_dict(trimmed_a)
    norm_b = unitnorm_waveform_dict(trimmed_b)

    num_points = min(len(norm_a[comp]) for comp in components)
    if num_points < 1:
        raise ValueError("Waveforms too short for dot product.")

    dot_sum = 0.0
    for comp in components:
        dot_sum += float(norm_a[comp] @ norm_b[comp])
    return dot_sum / len(components)


def load_event_waveform(waveform_filepath, station, starttime, endtime):
    """Load a 3-component waveform slice; returns None on failure."""
    try:
        waveform_dict, _ = load_waveform_slice(
            waveform_filepath,
            station,
            starttime,
            endtime=endtime,
            normalize=False,
        )
        return waveform_dict
    except Exception as exc:
        print(f"Warning: failed to load waveform at {station} [{starttime}, {endtime}]: {exc}")
        return None


def get_lag_samples(hub_dict, sat_dict, station, det_station_dict, lag_matrix_dict):
    """Return lag in samples between satellite and hub detections at a station."""
    det_id_satellite = sat_dict["per_station_ids"][station]
    det_id_hub = hub_dict["per_station_ids"][station]
    det_index_satellite = det_station_dict[station][det_id_satellite]
    det_index_hub = det_station_dict[station][det_id_hub]
    return int(lag_matrix_dict[station][det_index_satellite, det_index_hub])


def compute_station_discordant_score(
    hub_dict,
    sat_dict,
    station,
    det_station_dict,
    lag_matrix_dict,
    waveform_filepath,
):
    """Compute aligned dot products and discordant score for one hub-satellite station pair."""
    if station not in sat_dict.get("per_station_ids", {}):
        return None

    t0_hub, t1_hub = hub_dict["per_station_times"][station]
    t0_sat, t1_sat = sat_dict["per_station_times"][station]
    lag_samples = get_lag_samples(hub_dict, sat_dict, station, det_station_dict, lag_matrix_dict)

    hub_short, hub_long, sat_short, sat_long = get_short_long_windows(
        t0_hub, t1_hub, t0_sat, t1_sat, lag_samples,
    )

    hub_wf_short = load_event_waveform(waveform_filepath, station, hub_short[0], hub_short[1])
    sat_wf_short = load_event_waveform(waveform_filepath, station, sat_short[0], sat_short[1])
    if hub_wf_short is None or sat_wf_short is None:
        return None

    hub_wf_long = load_event_waveform(waveform_filepath, station, hub_long[0], hub_long[1])
    sat_wf_long = load_event_waveform(waveform_filepath, station, sat_long[0], sat_long[1])
    if hub_wf_long is None or sat_wf_long is None:
        return None

    cc_short = compute_normalized_dot_product(hub_wf_short, sat_wf_short)
    cc_long = compute_normalized_dot_product(hub_wf_long, sat_wf_long)
    discordant_score = 2.0 * (cc_short - cc_long)

    return {
        "station": station,
        "cc_short": cc_short,
        "cc_long": cc_long,
        "discordant_score": discordant_score,
    }


def main():
    parser = ArgumentParser(
        description=(
            "Compute discordant scores between the hub event and aligned satellite events "
            "for one event group"
        ),
    )
    parser.add_argument("--group_label", type=int, required=True, help="Event group label")
    parser.add_argument("--min_cc", type=float, default=0.85)
    parser.add_argument("--min_num_similar_snippet", type=int, default=10)
    parser.add_argument("--min_num_similar_station", type=int, default=3)
    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--thr_on", type=float, default=4.0)
    parser.add_argument("--thr_off", type=float, default=1.0)
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    args = parser.parse_args()

    suffix_freq = get_freq_limits_string(args.min_freq_filter, args.max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(
        args.sta_window_sec, args.lta_window_sec, args.thr_on, args.thr_off,
    )
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=args.min_cc, min_num_similar=args.min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{args.min_num_similar_station:d}"

    print("--------------------------------")
    print("Computing discordant scores...")
    print("--------------------------------")
    print(f"Group label: {args.group_label}")
    print(f"Suffix group: {suffix_group}")

    event_df, alignment_df, hub_dict, id_hub = load_group_data(args.group_label, suffix_group)

    stations_hub = hub_dict["stations"]
    print(f"Hub event ID: {id_hub}")
    print(f"Hub stations: {stations_hub}")

    det_station_dict, lag_matrix_dict = load_detection_maps_and_lag_matrices(stations_hub, suffix)

    waveform_filepath = join(dirpath_waveform, f"preprocessed_data_{suffix_freq}.h5")

    event_id_to_row = {
        int(row["event_id"]): row.to_dict()
        for _, row in event_df.iterrows()
    }

    station_rows = []
    summary_rows = []

    satellite_df = alignment_df.loc[~alignment_df["hub"]]
    print(f"Number of aligned satellite events: {len(satellite_df)}")

    for _, align_row in satellite_df.iterrows():
        id_satellite = int(align_row["id"])
        if id_satellite not in event_id_to_row:
            print(f"Warning: satellite {id_satellite} not in grouped events; skipping.")
            continue

        sat_dict = event_id_to_row[id_satellite]
        station_scores = []

        for station in stations_hub:
            result = compute_station_discordant_score(
                hub_dict,
                sat_dict,
                station,
                det_station_dict,
                lag_matrix_dict,
                waveform_filepath,
            )
            if result is None:
                continue

            station_scores.append(result["discordant_score"])
            station_rows.append({
                "id_hub": id_hub,
                "id_satellite": id_satellite,
                "station": result["station"],
                "cc_short": result["cc_short"],
                "cc_long": result["cc_long"],
                "discordant_score": result["discordant_score"],
            })

        if not station_scores:
            print(f"Warning: no valid stations for hub {id_hub} / satellite {id_satellite}; skipping pair.")
            continue

        summary_rows.append({
            "id_hub": id_hub,
            "id_satellite": id_satellite,
            "discordant_score": mean(station_scores),
            "num_stations": len(station_scores),
        })

    station_outfile = (
        f"discordant_scores_station_group{args.group_label:d}_{suffix_group}.csv"
    )
    summary_outfile = f"discordant_scores_group{args.group_label:d}_{suffix_group}.csv"

    station_df = DataFrame(station_rows)
    summary_df = DataFrame(summary_rows)
    summary_df = summary_df.sort_values("discordant_score", ascending=False)

    if not station_df.empty and not summary_df.empty:
        sat_order = {sid: i for i, sid in enumerate(summary_df["id_satellite"])}
        station_df = station_df.assign(
            _sort_order=station_df["id_satellite"].map(sat_order),
        ).sort_values(["_sort_order", "station"]).drop(columns="_sort_order")

    station_path = join(dirpath_event, station_outfile)
    summary_path = join(dirpath_event, summary_outfile)

    station_df.to_csv(station_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved per-station scores to {station_path} ({len(station_df)} rows).")
    print(f"Saved per-pair summary to {summary_path} ({len(summary_df)} rows).")


if __name__ == "__main__":
    main()
