"""
Binomial test for station repeating STA/LTA detection day/night frequencies.

For each of two stations (default A04 and B04), tests one-sided whether the day
detection rate exceeds the night detection rate, using the expected day fraction
implied by sunrise/sunset durations over the geophone deployment window as the
null probability.
"""

from argparse import ArgumentParser
from pathlib import Path

from pandas import DataFrame, Timestamp, read_csv, to_datetime

from binomial_test_subarray_event_day_night import (
    compute_expected_day_fraction,
    run_binomial_day_night_test,
)
from plot_subarray_event_day_night_counts import load_sunrise_sunset_times
from utils_basic import (
    DETECTION_DIR,
    ENDTIME_GEO,
    STARTTIME_GEO,
    SUNRISE_SUNSET_PATH,
)

DEFAULT_REPEATING_SUFFIX = (
    "freq20hz_sta5_lta50_on4_off1_cc0.85_num_sim_det10"
)
DEFAULT_STATIONS = ("A04", "B04")


def parse_args():
    parser = ArgumentParser(
        description=(
            "Binomial test: is the day repeating STA/LTA detection rate higher "
            "than the night rate, separately for two stations?"
        ),
    )
    parser.add_argument(
        "--stations",
        type=str,
        nargs=2,
        default=list(DEFAULT_STATIONS),
        metavar=("STA_A", "STA_B"),
        help="Two stations to test (default: A04 B04)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=DEFAULT_REPEATING_SUFFIX,
        help="Suffix for repeating STA/LTA detection CSV files",
    )
    parser.add_argument(
        "--detection-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory containing sta_lta_detections_repeating_*.csv files",
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


def load_repeating_detections(detection_dir, suffix, station):
    """Load repeating STA/LTA detections for one station."""
    filename = f"sta_lta_detections_repeating_{suffix}_{station}.csv"
    filepath = detection_dir / filename
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing repeating detections file: {filepath}")

    detection_df = read_csv(filepath, parse_dates=["starttime", "endtime"])
    detection_df["starttime"] = to_datetime(detection_df["starttime"], utc=False)

    geo_tz = Timestamp(STARTTIME_GEO).tz
    if geo_tz is not None and detection_df["starttime"].dt.tz is None:
        detection_df["starttime"] = detection_df["starttime"].dt.tz_localize(geo_tz)

    return detection_df


def filter_geo_window(detection_df):
    """Keep detections whose starttime lies within the geophone deployment window."""
    mask = (detection_df["starttime"] >= STARTTIME_GEO) & (
        detection_df["starttime"] <= ENDTIME_GEO
    )
    return detection_df.loc[mask].copy()


def classify_detections_day_night(detection_df, sun_df):
    """Label each detection as day or night from sunrise/sunset on its calendar day."""
    df = detection_df.copy()
    df["day"] = df["starttime"].dt.floor("D")
    merged = df.merge(sun_df[["day", "sunrise", "sunset"]], on="day", how="left")

    missing_mask = merged["sunrise"].isna()
    if missing_mask.any():
        print(
            f"Warning: {missing_mask.sum()} detection(s) lack sunrise/sunset times "
            "and will be excluded from counts."
        )
        merged = merged.loc[~missing_mask].copy()

    is_day = (merged["starttime"] >= merged["sunrise"]) & (
        merged["starttime"] <= merged["sunset"]
    )
    merged["period"] = "night"
    merged.loc[is_day, "period"] = "day"
    return merged


def count_detections_by_station_and_period(detection_dir, suffix, stations, sun_df):
    """Return detection counts keyed by (station, period)."""
    counts = {}
    for station in stations:
        detection_df = load_repeating_detections(detection_dir, suffix, station)
        detection_df = filter_geo_window(detection_df)
        classified = classify_detections_day_night(detection_df, sun_df)
        for period in ("day", "night"):
            counts[(station, period)] = int((classified["period"] == period).sum())
    return counts


def main():
    args = parse_args()
    stations = args.stations
    suffix = args.suffix
    detection_dir = args.detection_dir
    sunrise_sunset_csv = args.sunrise_sunset_csv
    output_dir = args.output_dir
    p_null_override = args.p_null
    alternative = args.alternative

    sun_df = load_sunrise_sunset_times(sunrise_sunset_csv)
    print(f"Loaded sunrise/sunset times from {sunrise_sunset_csv}")

    duration_info = compute_expected_day_fraction(sun_df)
    p_null = (
        p_null_override
        if p_null_override is not None
        else duration_info["expected_day_fraction"]
    )
    print(f"Deployment window: {STARTTIME_GEO} to {ENDTIME_GEO}")
    print(f"Day duration:   {duration_info['day_duration_s'] / 3600:.2f} h")
    print(f"Night duration: {duration_info['night_duration_s'] / 3600:.2f} h")
    print(f"Expected day fraction: {duration_info['expected_day_fraction']:.4f}")
    if p_null_override is not None:
        print(f"Using overridden p_null: {p_null:.4f}")
    if alternative == "greater":
        test_label = f"Test: day fraction > {p_null:.4f} (one-sided binomial test)"
    elif alternative == "less":
        test_label = f"Test: day fraction < {p_null:.4f} (one-sided binomial test)"
    else:
        test_label = (
            f"Test: day fraction != {p_null:.4f} (two-sided binomial test)"
        )
    print(test_label)

    counts = count_detections_by_station_and_period(
        detection_dir, suffix, stations, sun_df
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []

    for station in stations:
        n_day = counts[(station, "day")]
        n_night = counts[(station, "night")]
        n_total = n_day + n_night

        print(f"Station {station}:")
        print(f"  day:   n={n_day}")
        print(f"  night: n={n_night}")

        if n_total == 0:
            print("  Warning: no classified detections; skipping binomial test.")
            continue

        test_result = run_binomial_day_night_test(
            n_day,
            n_night,
            p_null=p_null,
            alternative=alternative,
        )
        row = {
            "station": station,
            "suffix": suffix,
            **duration_info,
            **test_result,
        }
        result_rows.append(row)

        print(f"  total: n={test_result['n_total']}")
        print(f"  day_fraction={test_result['day_fraction']:.4f}")
        print(f"  p_value={test_result['p_value']:.6f}")

    if not result_rows:
        raise SystemExit("No station had classified detections; no results to save.")

    result_df = DataFrame(result_rows)
    sta_a, sta_b = stations
    output_csv = (
        output_dir
        / f"binomial_test_station_repeating_detection_day_night_{suffix}_{sta_a}_{sta_b}.csv"
    )
    result_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")


if __name__ == "__main__":
    main()
