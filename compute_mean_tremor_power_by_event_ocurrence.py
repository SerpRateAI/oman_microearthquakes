"""
Compute mean and std Mode 3 tremor power for tremor windows with vs without events.

Loads harmonic tremor power at a chosen station from stationary resonance HDF5 files,
counts impulsive events whose first_onset falls within each tremor window
[time - window_length/2, time + window_length/2), where ``time`` is the window
center, classifies windows with at least one such event,
pools tremor windows in each category, and compares mean/std ``total_power`` (dB)
separately for subarrays A and B.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from pandas import DataFrame, read_csv, read_hdf, read_json, to_datetime, Timedelta

from utils_basic import DETECTION_DIR, SPECTROGRAM_DIR
from utils_spec import get_spec_peak_file_suffix, get_spectrogram_file_suffix

DEFAULT_SUFFIX = (
    "freq20hz_sta5_lta50_on4_off1_cc0.85_num_sim_det10_num_sim_sta3"
)


def load_mode_name(spectrogram_dir, base_mode_name, base_mode_order, mode_order):
    """Return the mode name for a given harmonic mode order."""
    filepath = (
        spectrogram_dir
        / f"stationary_harmonic_series_{base_mode_name}_base{base_mode_order}.csv"
    )
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing harmonic series file: {filepath}")

    harmonic_df = read_csv(filepath)
    matches = harmonic_df.loc[harmonic_df["mode_order"] == mode_order, "mode_name"]
    if matches.empty:
        raise ValueError(f"No mode with order {mode_order} in {filepath}")
    return matches.values[0]


def load_tremor_powers(
    spectrogram_dir,
    mode_name,
    window_length,
    overlap,
    min_prom,
    min_rbw,
    max_mean_db,
    station,
):
    """Load tremor total_power (dB) at one station for all time windows."""
    suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
    suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)
    filepath = (
        spectrogram_dir
        / f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
    )
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing tremor properties file: {filepath}")

    resonance_df = read_hdf(filepath, key="properties")
    resonance_df = resonance_df.loc[resonance_df["station"] == station].copy()
    if resonance_df.empty:
        raise ValueError(f"No tremor data for station {station} in {filepath}")

    resonance_df["time"] = to_datetime(resonance_df["time"], utc=True, format="ISO8601")
    return resonance_df.sort_values("time").reset_index(drop=True)


def load_subarray_events(detection_dir, subarray, suffix):
    """Load merged subarray event catalog from JSONL."""
    filepath = detection_dir / f"grouped_events_subarray_{subarray.lower()}_{suffix}.jsonl"
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing subarray event catalog: {filepath}")

    catalog = read_json(filepath, lines=True)
    first_onset = catalog["first_onset"]
    if first_onset.dtype.kind in "iu":
        catalog["first_onset"] = to_datetime(first_onset, utc=True, unit="ms")
    else:
        catalog["first_onset"] = to_datetime(first_onset, utc=True, errors="coerce")
    return catalog.sort_values("first_onset").reset_index(drop=True)


def count_events_in_tremor_windows(event_times, window_centers, window_length):
    """Count events with first_onset in the tremor window centered on each time."""
    half_window = Timedelta(seconds=window_length / 2)
    window_starts = window_centers - half_window
    window_ends = window_centers + half_window
    event_values = event_times.to_numpy()
    start_values = window_starts.to_numpy()
    end_values = window_ends.to_numpy()
    left = np.searchsorted(event_values, start_values, side="left")
    right = np.searchsorted(event_values, end_values, side="left")
    return right - left


def merge_tremor_with_subarray_events(tremor_df, catalog, window_length):
    """Count subarray events occurring within each tremor window."""
    half_window = Timedelta(seconds=window_length / 2)
    tremor_df = tremor_df.copy()
    tremor_df["time_start"] = tremor_df["time"] - half_window
    tremor_df["time_end"] = tremor_df["time"] + half_window
    tremor_df["count"] = count_events_in_tremor_windows(
        catalog["first_onset"], tremor_df["time"], window_length
    )
    return tremor_df[["time", "time_start", "time_end", "total_power", "count"]]


def classify_windows_with_events(merged_df):
    """Return boolean mask: True if at least one event occurs in the tremor window."""
    return merged_df["count"] > 0


def save_window_catalog(merged_df, has_event_mask, category, output_path):
    """Write window center, bounds, power, and in-window event count for one category."""
    mask_value = category == "with_event"
    window_df = merged_df.loc[
        has_event_mask == mask_value,
        ["time", "time_start", "time_end", "total_power", "count"],
    ].copy()
    window_df.to_csv(output_path, index=False)
    return len(window_df)


def window_catalog_path(output_dir, category, subarray, station, suffix):
    """Build the output path for a per-category tremor window CSV."""
    return (
        output_dir
        / f"tremor_windows_{category}_subarray_{subarray.lower()}_{station}_{suffix}.csv"
    )


def compute_category_stats(powers, has_event_mask):
    """Compute mean, std, and count for with_event and without_event groups."""
    results = []
    for category, mask_value in (("with_event", True), ("without_event", False)):
        subset = powers[has_event_mask == mask_value]
        results.append(
            {
                "category": category,
                "mean_power_db": subset.mean(),
                "std_power_db": subset.std(ddof=1) if len(subset) > 1 else float("nan"),
                "n_windows": len(subset),
            }
        )
    return results


def parse_args():
    parser = ArgumentParser(
        description=(
            "Compute mean and std Mode 3 tremor power for tremor windows with and "
            "without impulsive events, separately for subarrays A and B."
        ),
    )
    parser.add_argument(
        "--station",
        type=str,
        default="A07",
        help="Geophone station for tremor power (default: A07)",
    )
    parser.add_argument(
        "--base-mode-name",
        type=str,
        default="PR02549",
        help="Base mode name of the harmonic series",
    )
    parser.add_argument(
        "--base-mode-order",
        type=int,
        default=2,
        help="Base mode order of the harmonic series",
    )
    parser.add_argument(
        "--mode-order",
        type=int,
        default=3,
        help="Harmonic mode order whose power is analyzed",
    )
    parser.add_argument(
        "--window-length",
        type=float,
        default=300.0,
        help="STFT window length in seconds",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="STFT overlap fraction",
    )
    parser.add_argument(
        "--min-prom",
        type=float,
        default=15.0,
        help="Minimum spectral peak prominence (dB)",
    )
    parser.add_argument(
        "--min-rbw",
        type=float,
        default=15.0,
        help="Minimum reverse bandwidth threshold",
    )
    parser.add_argument(
        "--max-mean-db",
        type=float,
        default=15.0,
        help="Maximum mean dB for peak-detection window exclusion",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=DEFAULT_SUFFIX,
        help="Filename suffix for subarray event catalog JSONL files",
    )
    parser.add_argument(
        "--detection-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory containing grouped_events_subarray_*.jsonl files",
    )
    parser.add_argument(
        "--spectrogram-dir",
        type=Path,
        default=Path(SPECTROGRAM_DIR),
        help="Directory containing harmonic series and resonance HDF5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory for summary and per-category CSV output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    mode_name = load_mode_name(
        args.spectrogram_dir,
        args.base_mode_name,
        args.base_mode_order,
        args.mode_order,
    )
    print(f"Mode {args.mode_order}: {mode_name}")

    tremor_df = load_tremor_powers(
        args.spectrogram_dir,
        mode_name,
        args.window_length,
        args.overlap,
        args.min_prom,
        args.min_rbw,
        args.max_mean_db,
        args.station,
    )
    print(f"Loaded {len(tremor_df)} tremor windows at station {args.station}")
    print(
        f"Classification: with_event if any event occurs within the "
        f"{args.window_length:.0f} s tremor window (time is window center)"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []
    for subarray in ("A", "B"):
        catalog = load_subarray_events(args.detection_dir, subarray, args.suffix)
        print(f"Subarray {subarray}: {len(catalog)} events in catalog")

        merged_df = merge_tremor_with_subarray_events(
            tremor_df, catalog, args.window_length
        )
        has_event_mask = classify_windows_with_events(merged_df)

        for stats in compute_category_stats(merged_df["total_power"], has_event_mask):
            row = {
                "subarray": subarray,
                "station": args.station,
                "mode_order": args.mode_order,
                "mode_name": mode_name,
                "window_length_s": args.window_length,
                **stats,
            }
            result_rows.append(row)
            print(
                f"  {stats['category']}: mean={stats['mean_power_db']:.3f} dB, "
                f"std={stats['std_power_db']:.3f} dB, n={stats['n_windows']}"
            )

            output_path = window_catalog_path(
                args.output_dir,
                stats["category"],
                subarray,
                args.station,
                args.suffix,
            )
            n_saved = save_window_catalog(
                merged_df, has_event_mask, stats["category"], output_path
            )
            print(f"  Saved {n_saved} windows to {output_path}")

    result_df = DataFrame(result_rows)
    output_csv = (
        args.output_dir
        / f"tremor_power_by_event_occurrence_{args.station}_{args.suffix}.csv"
    )
    result_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")


if __name__ == "__main__":
    main()
