"""
Permutation test for tremor power in windows with vs without impulsive events.

For each subarray, tests whether mean total_power (dB) in tremor windows containing
at least one event is lower than in windows without events, using a one-sided
label-permutation test on the mean difference.
"""

from argparse import ArgumentParser
from pathlib import Path

from numpy import asarray, concatenate, count_nonzero, empty
from numpy.random import default_rng
from pandas import DataFrame

from compute_mean_tremor_power_by_event_ocurrence import (
    DEFAULT_SUFFIX,
    classify_windows_with_events,
    load_mode_name,
    load_subarray_events,
    load_tremor_powers,
    merge_tremor_with_subarray_events,
)
from utils_basic import DETECTION_DIR, SPECTROGRAM_DIR


def permutation_test_mean_lower_in_with_group(
    powers_with, powers_without, n_permutations, seed
):
    """
    One-sided test: is mean(powers_with) < mean(powers_without)?

    Returns observed_diff (with - without), effect (without - with), and p-value.
    """
    powers_with = asarray(powers_with, dtype=float)
    powers_without = asarray(powers_without, dtype=float)
    if len(powers_with) == 0 or len(powers_without) == 0:
        raise ValueError("Both with-event and without-event groups must be non-empty.")

    observed_diff = powers_with.mean() - powers_without.mean()
    combined = concatenate([powers_with, powers_without])
    n_with = len(powers_with)
    rng = default_rng(seed)

    null_diffs = empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        null_diffs[i] = perm[:n_with].mean() - perm[n_with:].mean()

    p_value = (count_nonzero(null_diffs <= observed_diff) + 1) / (n_permutations + 1)
    effect_db = -observed_diff
    return observed_diff, effect_db, p_value


def parse_args():
    parser = ArgumentParser(
        description=(
            "Permutation test: is tremor power lower in windows with impulsive "
            "events than in windows without events, separately for subarrays A and B?"
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
        help="Directory for permutation-test summary CSV output",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of label permutations (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for permutations (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    station = args.station
    base_mode_name = args.base_mode_name
    base_mode_order = args.base_mode_order
    mode_order = args.mode_order
    window_length = args.window_length
    overlap = args.overlap
    min_prom = args.min_prom
    min_rbw = args.min_rbw
    max_mean_db = args.max_mean_db
    suffix = args.suffix
    detection_dir = args.detection_dir
    spectrogram_dir = args.spectrogram_dir
    output_dir = args.output_dir
    n_permutations = args.n_permutations
    seed = args.seed

    mode_name = load_mode_name(
        spectrogram_dir,
        base_mode_name,
        base_mode_order,
        mode_order,
    )
    print(f"Mode {mode_order}: {mode_name}")

    tremor_df = load_tremor_powers(
        spectrogram_dir,
        mode_name,
        window_length,
        overlap,
        min_prom,
        min_rbw,
        max_mean_db,
        station,
    )
    print(f"Loaded {len(tremor_df)} tremor windows at station {station}")
    print(
        f"Test: mean power lower in with-event windows "
        f"({n_permutations} permutations, seed={seed})"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []

    for subarray in ("A", "B"):
        catalog = load_subarray_events(detection_dir, subarray, suffix)
        merged_df = merge_tremor_with_subarray_events(
            tremor_df, catalog, window_length
        )
        has_event_mask = classify_windows_with_events(merged_df)

        powers_with = merged_df.loc[has_event_mask, "total_power"].values
        powers_without = merged_df.loc[~has_event_mask, "total_power"].values

        observed_diff, effect_db, p_value = permutation_test_mean_lower_in_with_group(
            powers_with,
            powers_without,
            n_permutations,
            seed,
        )

        row = {
            "subarray": subarray,
            "station": station,
            "mode_order": mode_order,
            "mode_name": mode_name,
            "window_length_s": window_length,
            "n_with": len(powers_with),
            "n_without": len(powers_without),
            "mean_with_db": powers_with.mean(),
            "mean_without_db": powers_without.mean(),
            "observed_diff_db": observed_diff,
            "effect_db": effect_db,
            "p_value_one_sided_lower": p_value,
            "n_permutations": n_permutations,
            "seed": seed,
        }
        result_rows.append(row)

        print(f"Subarray {subarray}:")
        print(
            f"  with_event:    n={row['n_with']}, mean={row['mean_with_db']:.3f} dB"
        )
        print(
            f"  without_event: n={row['n_without']}, "
            f"mean={row['mean_without_db']:.3f} dB"
        )
        print(
            f"  observed_diff (with-without)={observed_diff:.3f} dB, "
            f"effect (without-with)={effect_db:.3f} dB"
        )
        print(f"  one-sided p-value={p_value:.6f}")

    result_df = DataFrame(result_rows)
    output_csv = (
        output_dir
        / f"tremor_power_permutation_test_{station}_{suffix}.csv"
    )
    result_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")


if __name__ == "__main__":
    main()
