"""
Plot bar chart of subarray event counts by day and night.

Counts impulsive events in Subarrays A and B whose first_onset falls within the
geophone deployment window, classifies each event as day or night using site
sunrise/sunset times, and saves a grouped bar chart with white/light-gray fills and
blue/orange outlines for subarrays A and B.
"""

from argparse import ArgumentParser
from pathlib import Path

from matplotlib.patches import Patch
from matplotlib.pyplot import subplots
from pandas import read_csv, to_datetime

from compute_mean_tremor_power_by_event_ocurrence import (
    DEFAULT_SUFFIX,
    load_subarray_events,
)
from utils_basic import (
    DETECTION_DIR,
    ENDTIME_GEO,
    STARTTIME_GEO,
    SUNRISE_SUNSET_PATH,
)
from utils_plot import save_figure

SUBARRAY_EDGE_COLORS = {"A": "dodgerblue", "B": "darkorange"}
PERIOD_FILLS = {"day": "white", "night": "lightgray"}
GROUP_CENTERS = {"A": 0.0, "B": 1.0}
PERIOD_OFFSETS = {"day": -0.18, "night": 0.18}
BAR_WIDTH = 0.32
EDGE_LINEWIDTH = 1.8


def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot subarray A and B event counts by day and night using "
            "site sunrise/sunset times."
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
        "--figname",
        type=str,
        default=None,
        help=(
            "Output figure filename "
            "(default: subarray_event_day_night_counts_<suffix>.png)"
        ),
    )
    return parser.parse_args()


def load_sunrise_sunset_times(csv_path):
    """Load sunrise/sunset table with UTC-aware day, sunrise, and sunset columns."""
    sun_df = read_csv(csv_path, parse_dates=["day", "sunrise", "sunset"])
    for column in ("day", "sunrise", "sunset"):
        sun_df[column] = to_datetime(sun_df[column], utc=True)
    return sun_df


def filter_geo_window(catalog):
    """Keep events whose first_onset lies within the geophone deployment window."""
    mask = (catalog["first_onset"] >= STARTTIME_GEO) & (
        catalog["first_onset"] <= ENDTIME_GEO
    )
    return catalog.loc[mask].copy()


def classify_events_day_night(catalog, sun_df):
    """Label each event as day or night from sunrise/sunset on its calendar day."""
    df = catalog.copy()
    df["day"] = df["first_onset"].dt.floor("D")
    merged = df.merge(sun_df[["day", "sunrise", "sunset"]], on="day", how="left")

    missing_mask = merged["sunrise"].isna()
    if missing_mask.any():
        print(
            f"Warning: {missing_mask.sum()} event(s) lack sunrise/sunset times "
            "and will be excluded from counts."
        )
        merged = merged.loc[~missing_mask].copy()

    is_day = (merged["first_onset"] >= merged["sunrise"]) & (
        merged["first_onset"] <= merged["sunset"]
    )
    merged["period"] = "night"
    merged.loc[is_day, "period"] = "day"
    return merged


def count_events_by_subarray_and_period(detection_dir, suffix, sun_df):
    """Return event counts keyed by (subarray, period)."""
    counts = {}
    for subarray in ("A", "B"):
        catalog = load_subarray_events(detection_dir, subarray, suffix)
        catalog = filter_geo_window(catalog)
        classified = classify_events_day_night(catalog, sun_df)
        for period in ("day", "night"):
            counts[(subarray, period)] = int((classified["period"] == period).sum())
    return counts


def plot_day_night_counts(ax, counts):
    """Draw grouped day/night bars for each subarray."""
    ymax = max(counts.values()) if counts else 0
    label_offset = 0.02 * ymax if ymax > 0 else 0.5

    for subarray in ("A", "B"):
        center = GROUP_CENTERS[subarray]
        edge_color = SUBARRAY_EDGE_COLORS[subarray]
        for period in ("day", "night"):
            position = center + PERIOD_OFFSETS[period]
            count = counts[(subarray, period)]
            fill_color = PERIOD_FILLS[period]
            ax.bar(
                position,
                count,
                width=BAR_WIDTH,
                facecolor=fill_color,
                edgecolor=edge_color,
                linewidth=EDGE_LINEWIDTH,
            )
            ax.text(
                position,
                count + label_offset,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=edge_color,
            )

    ax.set_xticks([GROUP_CENTERS["A"], GROUP_CENTERS["B"]])
    ax.set_xticklabels(["A", "B"], fontsize=10)
    ax.set_xlabel("Subarray", fontsize=12)
    ax.set_ylabel("Number of events", fontsize=12)
    ax.set_title("Subarray event counts by day and night", fontweight="bold")

    if ymax > 0:
        ax.set_ylim(0, ymax + 0.12 * ymax)

    legend_handles = [
        Patch(
            facecolor=PERIOD_FILLS["day"],
            edgecolor="black",
            label="Day",
        ),
        Patch(
            facecolor=PERIOD_FILLS["night"],
            edgecolor="black",
            label="Night",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=1.0)


def main():
    args = parse_args()

    sun_df = load_sunrise_sunset_times(args.sunrise_sunset_csv)
    print(f"Loaded sunrise/sunset times from {args.sunrise_sunset_csv}")

    counts = count_events_by_subarray_and_period(
        args.detection_dir, args.suffix, sun_df
    )
    for subarray in ("A", "B"):
        for period in ("day", "night"):
            print(f"Subarray {subarray} {period}: n={counts[(subarray, period)]}")

    fig, ax = subplots(1, 1, figsize=(7, 5))
    plot_day_night_counts(ax, counts)

    figname = args.figname or f"subarray_event_day_night_counts_{args.suffix}.png"
    save_figure(fig, figname)
    print(f"Saved {figname}")


if __name__ == "__main__":
    main()
