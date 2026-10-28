"""
Count associated events in Subarrays A and B.

Reads associated_events_repeating_<suffix>.jsonl, assigns each event to
Subarray A or B from the first character of first_onset_station, and writes
a two-row CSV of total counts and percentages of all associated events.
"""

from argparse import ArgumentParser
from pathlib import Path

from pandas import DataFrame, read_json

from utils_basic import DETECTION_DIR

DEFAULT_SUFFIX = "freq20hz_sta5_lta50_on4_off1_cc0.85_num_sim_det10"


def infer_subarray_from_station(station):
    """Infer Subarray A or B from the first character of a station name."""
    letter = str(station)[0].upper()
    if letter not in ("A", "B"):
        raise ValueError(f"Cannot infer subarray from station name {station!r}")
    return letter


def count_events_by_subarray(event_df):
    """Count events in Subarrays A and B and their share of the total."""
    subarrays = event_df["first_onset_station"].map(infer_subarray_from_station)
    counts = subarrays.value_counts()
    rows = [
        {"subarray": subarray, "num_events": int(counts.get(subarray, 0))}
        for subarray in ("A", "B")
    ]
    count_df = DataFrame(rows)
    total = int(count_df["num_events"].sum())
    if total == 0:
        count_df["percent"] = 0.0
    else:
        count_df["percent"] = 100.0 * count_df["num_events"] / total
    return count_df


def parse_args():
    parser = ArgumentParser(
        description=(
            "Count associated events in Subarrays A and B from "
            "first_onset_station and write counts and percentages to a CSV."
        ),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=DEFAULT_SUFFIX,
        help="Suffix for associated_events_repeating_<suffix>.jsonl",
    )
    parser.add_argument(
        "--detection-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory containing associated_events_repeating_*.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory for the subarray total-count CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = args.suffix
    detection_dir = args.detection_dir
    output_dir = args.output_dir

    events_jsonl = detection_dir / f"associated_events_repeating_{suffix}.jsonl"
    if not events_jsonl.is_file():
        raise FileNotFoundError(f"Missing associated events file: {events_jsonl}")

    print(f"Reading associated events from {events_jsonl}...")
    event_df = read_json(events_jsonl, lines=True)
    required = {"event_id", "first_onset_station"}
    if not required.issubset(event_df.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {events_jsonl}; "
            f"got {list(event_df.columns)}"
        )

    count_df = count_events_by_subarray(event_df)
    for _, row in count_df.iterrows():
        print(
            f"Subarray {row['subarray']}: {row['num_events']} events "
            f"({row['percent']:.2f}%)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"subarray_total_event_numbers_{suffix}.csv"
    count_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")


if __name__ == "__main__":
    main()
