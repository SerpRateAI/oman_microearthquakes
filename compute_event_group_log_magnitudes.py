"""
Compute log magnitudes for all events in a classified event group.

For each event, load the requested component at the event's first-onset station
over a buffered window around first_onset, then record log10(max(abs(amplitude))).
"""

from argparse import ArgumentParser
from os.path import join

from numpy import amax, log10
from pandas import DataFrame, Timedelta, isna, read_json, to_datetime

from utils_basic import (
    DETECTION_DIR as dirpath_event,
    GEO_COMPONENTS as geo_components,
    ROOTDIR_GEO as dirpath_waveform,
    get_freq_limits_string,
)
from utils_cc import get_repeating_snippet_suffix
from utils_cont_waveform import load_waveform_slice
from utils_sta_lta import get_sta_lta_suffix


def load_group_events(group_label, suffix_group):
    """Load grouped events JSONL for one event group."""
    filename = f"grouped_events_group{group_label:d}_{suffix_group}.jsonl"
    event_df = read_json(join(dirpath_event, filename), lines=True)
    event_df["first_onset"] = to_datetime(event_df["first_onset"], format="ISO8601", errors="coerce")
    return event_df


def compute_log_magnitude(waveform_dict, component):
    """Return log10(max(abs(component))) or nan if amplitude is non-positive."""
    max_abs_amplitude = float(amax(abs(waveform_dict[component])))
    if max_abs_amplitude <= 0:
        return max_abs_amplitude, float("nan")
    return max_abs_amplitude, float(log10(max_abs_amplitude))


def main():
    parser = ArgumentParser(
        description="Compute log magnitudes for all events in a classified event group.",
    )
    parser.add_argument("--group_label", type=int, default=18, help="Event group label")
    parser.add_argument("--component", type=str, default="Z", help="Data component")
    parser.add_argument(
        "--buffer_before_sec",
        type=float,
        default=0.05,
        help="Buffer before first_onset in seconds",
    )
    parser.add_argument(
        "--buffer_after_sec",
        type=float,
        default=0.2,
        help="Buffer after first_onset in seconds",
    )
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

    group_label = args.group_label
    component = args.component
    buffer_before_sec = args.buffer_before_sec
    buffer_after_sec = args.buffer_after_sec
    min_cc = args.min_cc
    min_num_similar_snippet = args.min_num_similar_snippet
    min_num_similar_station = args.min_num_similar_station
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    thr_on = args.thr_on
    thr_off = args.thr_off
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter

    if component not in geo_components:
        raise ValueError(
            f"Component {component!r} not in supported components {geo_components}."
        )

    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=min_cc,
        min_num_similar=min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

    print("--------------------------------")
    print("Computing event group log magnitudes...")
    print("--------------------------------")
    print(f"Group label: {group_label}")
    print(f"Component: {component}")
    print(f"Buffer before sec: {buffer_before_sec}")
    print(f"Buffer after sec: {buffer_after_sec}")
    print(f"Suffix group: {suffix_group}")

    event_df = load_group_events(group_label, suffix_group)
    print(f"Number of events in group: {len(event_df)}")

    waveform_filepath = join(dirpath_waveform, f"preprocessed_data_{suffix_freq}.h5")

    rows = []
    for _, row in event_df.iterrows():
        event_id = int(row["event_id"])
        station = row["first_onset_station"]
        first_onset = row["first_onset"]
        if isna(first_onset):
            print(f"Warning: event {event_id} has invalid first_onset; skipping.")
            continue

        window_start = first_onset - Timedelta(seconds=buffer_before_sec)
        window_end = first_onset + Timedelta(seconds=buffer_after_sec)

        try:
            waveform_dict, _ = load_waveform_slice(
                waveform_filepath,
                station,
                window_start,
                endtime=window_end,
                normalize=False,
            )
        except Exception as exc:
            print(
                f"Warning: failed to load waveform for event {event_id} "
                f"at {station} [{window_start}, {window_end}]: {exc}"
            )
            continue

        max_abs_amplitude, log_magnitude = compute_log_magnitude(waveform_dict, component)
        rows.append({
            "event_id": event_id,
            "group_label": group_label,
            "first_onset_station": station,
            "component": component,
            "first_onset": first_onset,
            "window_start": window_start,
            "window_end": window_end,
            "max_abs_amplitude": max_abs_amplitude,
            "log_magnitude": log_magnitude,
        })

    result_df = DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values("first_onset").reset_index(drop=True)

    outfile = f"log_magnitudes_group{group_label:d}_{suffix_group}.csv"
    outfile_path = join(dirpath_event, outfile)
    result_df.to_csv(outfile_path, index=False)
    print(f"Saved log magnitudes to {outfile_path} ({len(result_df)} rows).")


if __name__ == "__main__":
    main()
