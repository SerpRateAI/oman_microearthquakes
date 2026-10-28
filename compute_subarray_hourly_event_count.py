"""
Compute hourly event counts and merged event catalogs for each subarray.

Reads group metadata from ``event_group_info_selected_*.csv``, loads per-group
events from ``grouped_events_group{label}_*.jsonl``, merges all events in
subarrays A and B (sorted by ``first_onset``), writes one JSONL catalog per
subarray with an added ``label`` column, and bins ``first_onset`` times hourly
between ``STARTTIME_GEO`` and ``ENDTIME_GEO``.
"""

from argparse import ArgumentParser
from pathlib import Path

from pandas import DataFrame, Timestamp, concat, date_range, read_csv, read_json, to_datetime

from utils_basic import DETECTION_DIR, ENDTIME_GEO, STARTTIME_GEO

DEFAULT_SUFFIX = (
    "freq20hz_sta5_lta50_on4_off1_cc0.85_num_sim_det10_num_sim_sta3"
)


def load_group_events(group_label, detection_dir, suffix):
    """Load grouped events for one label; return full records with group label."""
    filepath = detection_dir / f"grouped_events_group{group_label:d}_{suffix}.jsonl"
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing grouped events file: {filepath}")

    event_df = read_json(filepath, lines=True)
    event_df["label"] = group_label
    return event_df


def build_subarray_catalog(group_df, subarray, detection_dir, suffix):
    """Merge events from all selected groups in one subarray, sorted by first_onset."""
    subarray_groups = group_df.loc[group_df["subarray"] == subarray, "label"]
    if subarray_groups.empty:
        return DataFrame()

    catalogs = [
        load_group_events(int(group_label), detection_dir, suffix)
        for group_label in subarray_groups
    ]
    catalog = concat(catalogs, ignore_index=True)
    catalog["first_onset"] = to_datetime(
        catalog["first_onset"], utc=True, format="ISO8601"
    )
    return catalog.sort_values("first_onset").reset_index(drop=True)


def save_subarray_catalog(catalog, output_path):
    """Write subarray event catalog in grouped-events JSONL format."""
    catalog.to_json(output_path, lines=True, orient="records")


def compute_hourly_counts(catalog, starttime, endtime):
    """Bin first_onset times into hourly counts over the full geo window."""
    start = Timestamp(starttime)
    end = Timestamp(endtime)
    geo_tz = start.tz

    catalog_win = catalog.loc[
        (catalog["first_onset"] >= start) & (catalog["first_onset"] <= end)
    ].copy()

    hour_index = date_range(
        start=start.floor("h"),
        end=end.floor("h"),
        freq="h",
        tz=geo_tz,
    )

    hourly_counts = (
        catalog_win.assign(hour=catalog_win["first_onset"].dt.floor("h"))
        .groupby("hour")
        .size()
        .reindex(hour_index, fill_value=0)
        .rename_axis("hour")
        .reset_index(name="count")
    )
    return hourly_counts


def parse_args():
    parser = ArgumentParser(
        description=(
            "Compute hourly event counts for subarrays A and B from selected event groups."
        ),
    )
    parser.add_argument(
        "--group-info-csv",
        type=Path,
        default=Path(DETECTION_DIR) / f"event_group_info_selected_{DEFAULT_SUFFIX}.csv",
        help="CSV with selected event groups (columns include label, subarray, …)",
    )
    parser.add_argument(
        "--detection-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory containing grouped_events_group*.jsonl files",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=DEFAULT_SUFFIX,
        help="Filename suffix shared by group info and grouped-events JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory for hourly-count CSV and subarray catalog JSONL output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    group_info_csv = args.group_info_csv
    detection_dir = args.detection_dir
    suffix = args.suffix
    output_dir = args.output_dir

    print(f"Reading group info from {group_info_csv}...")
    group_df = read_csv(group_info_csv)
    required = {"label", "subarray"}
    if not required.issubset(group_df.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {group_info_csv}; "
            f"got {list(group_df.columns)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for subarray in ("A", "B"):
        group_labels = group_df.loc[group_df["subarray"] == subarray, "label"].tolist()
        print(
            f"Subarray {subarray}: {len(group_labels)} group(s), "
            f"labels {group_labels}"
        )

        catalog = build_subarray_catalog(group_df, subarray, detection_dir, suffix)
        print(f"  Loaded {len(catalog)} events")

        output_jsonl = (
            output_dir / f"grouped_events_subarray_{subarray.lower()}_{suffix}.jsonl"
        )
        save_subarray_catalog(catalog, output_jsonl)
        print(f"  Saved {output_jsonl}")

        hourly_counts = compute_hourly_counts(catalog, STARTTIME_GEO, ENDTIME_GEO)
        total_count = int(hourly_counts["count"].sum())
        print(f"  {total_count} events in geo time window; {len(hourly_counts)} hourly bins")

        output_csv = output_dir / f"hourly_counts_subarray_{subarray.lower()}_{suffix}.csv"
        hourly_counts.to_csv(output_csv, index=False)
        print(f"  Saved {output_csv}")


if __name__ == "__main__":
    main()
