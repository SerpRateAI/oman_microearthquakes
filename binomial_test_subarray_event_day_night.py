"""
Binomial test for subarray event day/night frequencies.

For each subarray, tests one-sided whether the day event rate exceeds the night
event rate, using the expected day fraction implied by sunrise/sunset durations
over the geophone deployment window as the null probability.
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from pandas import DataFrame, Timedelta
from scipy.stats import binomtest

from compute_mean_tremor_power_by_event_ocurrence import DEFAULT_SUFFIX
from plot_subarray_event_day_night_counts import (
    count_events_by_subarray_and_period,
    load_sunrise_sunset_times,
)
from utils_basic import (
    DETECTION_DIR,
    ENDTIME_GEO,
    STARTTIME_GEO,
    SUNRISE_SUNSET_PATH,
)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Binomial test: is the day event rate higher than the night event "
            "rate, separately for subarrays A and B?"
        ),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=DEFAULT_SUFFIX,
        help="Suffix for subarray event catalog JSONL files",
    )
    parser.add_argument(
        "--detection-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory containing grouped_events_subarray_*.jsonl files",
    )
    parser.add_argument(
        "--sunrise-sunset-csv",
        type=Path,
        default=Path(SUNRISE_SUNSET_PATH),
        help="CSV with day, sunrise, and sunset columns in local time",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory for binomial-test summary CSV output",
    )
    parser.add_argument(
        "--p-null",
        type=float,
        default=None,
        help=(
            "Override null day probability; default is computed from "
            "day/night durations in the deployment window"
        ),
    )
    parser.add_argument(
        "--alternative",
        type=str,
        default="greater",
        choices=("two-sided", "less", "greater"),
        help=(
            "Alternative hypothesis for the binomial test "
            "(default: greater, day rate > night rate)"
        ),
    )
    return parser.parse_args()


def _add_interval(total_day, total_night, start, end, window_start, window_end, is_day):
    """Add clipped interval duration to day or night total."""
    seg_start = max(start, window_start)
    seg_end = min(end, window_end)
    if seg_end <= seg_start:
        return total_day, total_night
    duration = seg_end - seg_start
    if is_day:
        return total_day + duration, total_night
    return total_day, total_night + duration


def compute_day_night_durations(sun_df, starttime=STARTTIME_GEO, endtime=ENDTIME_GEO):
    """
    Sum day and night durations over a time window from sunrise/sunset times.

    Uses the same first/middle/last-day interval logic as add_day_night_shading().
    """
    sun_df = sun_df.copy()
    sun_df["sunrise_next"] = sun_df["sunrise"].shift(-1)

    mask_start = (sun_df["sunset"] > starttime) | (sun_df["sunrise"] > starttime)
    mask_end = (sun_df["sunset"] < endtime) | (sun_df["sunrise"] < endtime)
    block = sun_df.loc[mask_start & mask_end]
    if block.empty:
        raise ValueError(
            "No sunrise/sunset rows overlap the deployment window; "
            "cannot compute expected day fraction."
        )

    total_day = Timedelta(0)
    total_night = Timedelta(0)
    first_idx = block.index[0]
    last_idx = block.index[-1]

    for idx, row in block.iterrows():
        sunrise = row["sunrise"]
        sunset = row["sunset"]
        sunrise_next = row["sunrise_next"]

        if idx == first_idx:
            if sunrise > starttime:
                total_day, total_night = _add_interval(
                    total_day, total_night, starttime, sunrise, starttime, endtime, False
                )
                total_day, total_night = _add_interval(
                    total_day, total_night, sunrise, sunset, starttime, endtime, True
                )
                if pd.notna(sunrise_next):
                    total_day, total_night = _add_interval(
                        total_day,
                        total_night,
                        sunset,
                        sunrise_next,
                        starttime,
                        endtime,
                        False,
                    )
            else:
                total_day, total_night = _add_interval(
                    total_day, total_night, starttime, sunset, starttime, endtime, True
                )
                if pd.notna(sunrise_next):
                    total_day, total_night = _add_interval(
                        total_day,
                        total_night,
                        sunset,
                        sunrise_next,
                        starttime,
                        endtime,
                        False,
                    )
        elif idx == last_idx:
            total_day, total_night = _add_interval(
                total_day, total_night, sunrise, sunset, starttime, endtime, True
            )
            if sunset < endtime:
                total_day, total_night = _add_interval(
                    total_day, total_night, sunset, endtime, starttime, endtime, False
                )
            else:
                total_day, total_night = _add_interval(
                    total_day, total_night, sunrise, endtime, starttime, endtime, True
                )
        else:
            total_day, total_night = _add_interval(
                total_day, total_night, sunrise, sunset, starttime, endtime, True
            )
            if pd.notna(sunrise_next):
                total_day, total_night = _add_interval(
                    total_day,
                    total_night,
                    sunset,
                    sunrise_next,
                    starttime,
                    endtime,
                    False,
                )

    return total_day, total_night


def compute_expected_day_fraction(sun_df, starttime=STARTTIME_GEO, endtime=ENDTIME_GEO):
    """Return expected day fraction and day/night durations for the deployment window."""
    total_day, total_night = compute_day_night_durations(sun_df, starttime, endtime)
    total_time = total_day + total_night
    if total_time <= Timedelta(0):
        raise ValueError("Deployment window has zero day/night duration.")

    day_duration_s = total_day.total_seconds()
    night_duration_s = total_night.total_seconds()
    expected_day_fraction = day_duration_s / (day_duration_s + night_duration_s)
    return {
        "day_duration_s": day_duration_s,
        "night_duration_s": night_duration_s,
        "expected_day_fraction": expected_day_fraction,
    }


def run_binomial_day_night_test(n_day, n_night, p_null, alternative="greater"):
    """Run a binomial test on day vs night event counts."""
    n_total = n_day + n_night
    result = binomtest(n_day, n_total, p=p_null, alternative=alternative)
    return {
        "n_day": n_day,
        "n_night": n_night,
        "n_total": n_total,
        "day_fraction": result.proportion_estimate,
        "p_null": p_null,
        "alternative": alternative,
        "p_value": result.pvalue,
    }


def main():
    args = parse_args()

    sun_df = load_sunrise_sunset_times(args.sunrise_sunset_csv)
    print(f"Loaded sunrise/sunset times from {args.sunrise_sunset_csv}")

    duration_info = compute_expected_day_fraction(sun_df)
    p_null = (
        args.p_null
        if args.p_null is not None
        else duration_info["expected_day_fraction"]
    )
    print(
        f"Deployment window: {STARTTIME_GEO} to {ENDTIME_GEO}"
    )
    print(
        f"Day duration:   {duration_info['day_duration_s'] / 3600:.2f} h"
    )
    print(
        f"Night duration: {duration_info['night_duration_s'] / 3600:.2f} h"
    )
    print(f"Expected day fraction: {duration_info['expected_day_fraction']:.4f}")
    if args.p_null is not None:
        print(f"Using overridden p_null: {p_null:.4f}")
    if args.alternative == "greater":
        test_label = f"Test: day fraction > {p_null:.4f} (one-sided binomial test)"
    elif args.alternative == "less":
        test_label = f"Test: day fraction < {p_null:.4f} (one-sided binomial test)"
    else:
        test_label = (
            f"Test: day fraction != {p_null:.4f} (two-sided binomial test)"
        )
    print(test_label)

    counts = count_events_by_subarray_and_period(
        args.detection_dir, args.suffix, sun_df
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []

    for subarray in ("A", "B"):
        n_day = counts[(subarray, "day")]
        n_night = counts[(subarray, "night")]
        n_total = n_day + n_night

        print(f"Subarray {subarray}:")
        print(f"  day:   n={n_day}")
        print(f"  night: n={n_night}")

        if n_total == 0:
            print("  Warning: no classified events; skipping binomial test.")
            continue

        test_result = run_binomial_day_night_test(
            n_day,
            n_night,
            p_null=p_null,
            alternative=args.alternative,
        )
        row = {
            "subarray": subarray,
            "suffix": args.suffix,
            **duration_info,
            **test_result,
        }
        result_rows.append(row)

        print(f"  total: n={test_result['n_total']}")
        print(f"  day_fraction={test_result['day_fraction']:.4f}")
        print(f"  p_value={test_result['p_value']:.6f}")

    if not result_rows:
        raise SystemExit("No subarray had classified events; no results to save.")

    result_df = DataFrame(result_rows)
    output_csv = (
        args.output_dir
        / f"binomial_test_subarray_event_day_night_{args.suffix}.csv"
    )
    result_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")


if __name__ == "__main__":
    main()
