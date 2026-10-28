"""
Plot stacked event-group size histograms for Subarrays A and B (AGU Advances).

Reads event group labels from CSV (as written by classify_events.py).
"""

import re
from argparse import ArgumentParser
from pathlib import Path

from pandas import read_csv, read_json

from utils_cluster import plot_label_histogram
from utils_basic import DETECTION_DIR
from utils_plot import save_figure

SUBARRAY_A_COLOR = "dodgerblue"
SUBARRAY_B_COLOR = "darkorange"
DEFAULT_OUTPUT_PNG = "aguadv_event_group_size_histograms.png"


def default_associated_events_jsonl(labels_csv: Path) -> Path:
    stem = labels_csv.stem
    prefix = "event_labels_"
    if not stem.startswith(prefix):
        raise ValueError(
            f"Cannot infer associated-events path from {labels_csv.name}; "
            f"expected stem event_labels_<suffix>.csv"
        )
    suffix_sim = stem[len(prefix) :]
    suffix = re.sub(r"_num_sim_sta\d+$", "", suffix_sim)
    return Path(DETECTION_DIR) / f"associated_events_repeating_{suffix}.jsonl"


def default_selected_groups_csv(labels_csv: Path) -> Path:
    stem = labels_csv.stem
    prefix = "event_labels_"
    if not stem.startswith(prefix):
        raise ValueError(
            f"Cannot infer selected-groups path from {labels_csv.name}; "
            f"expected stem event_labels_<suffix>.csv"
        )
    suffix_sim = stem[len(prefix) :]
    return Path(DETECTION_DIR) / f"event_group_info_selected_{suffix_sim}.csv"


def load_highlight_labels(selected_groups_csv: Path) -> list:
    df = read_csv(selected_groups_csv)
    if "label" not in df.columns:
        raise ValueError(
            f"Expected a 'label' column in {selected_groups_csv}; got {list(df.columns)}"
        )
    return df["label"].tolist()


def infer_subarray_from_station(station: str) -> str:
    letter = str(station)[0].upper()
    if letter not in ("A", "B"):
        raise ValueError(f"Cannot infer subarray from station name {station!r}")
    return letter


def load_event_subarrays(events_jsonl: Path, event_ids) -> dict:
    event_df = read_json(events_jsonl, lines=True)
    if "event_id" not in event_df.columns:
        raise ValueError(f"Expected 'event_id' in {events_jsonl}")
    if "first_onset_station" not in event_df.columns:
        raise ValueError(f"Expected 'first_onset_station' in {events_jsonl}")

    subarray_by_id = {
        int(row["event_id"]): infer_subarray_from_station(row["first_onset_station"])
        for _, row in event_df.iterrows()
    }
    missing = [eid for eid in event_ids if int(eid) not in subarray_by_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} event id(s) from labels CSV missing in {events_jsonl} "
            f"(e.g. {missing[:5]})"
        )
    return subarray_by_id


def parse_args():
    parser = ArgumentParser(
        description="Plot histogram of event group labels from classify_events CSV output."
    )
    parser.add_argument(
        "labels_csv",
        type=Path,
        help="Path to event_labels_*.csv (columns: id, label).",
    )
    parser.add_argument(
        "--events-jsonl",
        type=Path,
        default=None,
        help="Associated events JSONL with event_id and first_onset_station. "
        "Default: associated_events_repeating_<suffix>.jsonl inferred from labels_csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PNG,
        help="Output figure filename under FIGURE_DIR (utils_plot).",
    )
    parser.add_argument(
        "--selected-groups-csv",
        type=Path,
        default=None,
        help="CSV with group labels to outline in black (column 'label'). "
        "Default: event_group_info_selected_<suffix>.csv in DETECTION_DIR, "
        "inferred from labels_csv.",
    )

    return parser.parse_args()


def main(args):
    labels_csv_arg = args.labels_csv
    output = args.output

    labels_csv = labels_csv_arg.resolve()
    if not labels_csv.is_file():
        raise SystemExit(f"Labels file not found: {labels_csv}")

    df = read_csv(labels_csv)
    if "label" not in df.columns:
        raise SystemExit(f"Expected a 'label' column in {labels_csv}; got {list(df.columns)}")
    if "id" not in df.columns:
        raise SystemExit(f"Expected an 'id' column in {labels_csv}; got {list(df.columns)}")

    events_jsonl = (args.events_jsonl or default_associated_events_jsonl(labels_csv)).resolve()
    if not events_jsonl.is_file():
        raise SystemExit(f"Associated events file not found: {events_jsonl}")

    labels = df["label"].to_numpy()
    subarray_by_id = load_event_subarrays(events_jsonl, df["id"].tolist())
    subarrays = [subarray_by_id[int(eid)] for eid in df["id"]]

    selected_groups_csv = (
        args.selected_groups_csv or default_selected_groups_csv(labels_csv)
    ).resolve()
    if not selected_groups_csv.is_file():
        raise SystemExit(f"Selected-groups file not found: {selected_groups_csv}")
    highlight_labels = load_highlight_labels(selected_groups_csv)

    fig, _ = plot_label_histogram(
        labels,
        subarrays=subarrays,
        highlight_labels=highlight_labels,
        color_a=SUBARRAY_A_COLOR,
        color_b=SUBARRAY_B_COLOR,
    )
    save_figure(fig, output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
