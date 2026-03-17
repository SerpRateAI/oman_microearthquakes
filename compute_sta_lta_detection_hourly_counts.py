"""
This script computes the hourly counts of sta-lta detections for a geophone station.
"""

#--------------------------------------------------------------------------------------------------
# Imports
#--------------------------------------------------------------------------------------------------

# Standard library
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv
from pandas import to_datetime, date_range, Timestamp

from utils_basic import (
    DETECTION_DIR as dirpath,
    STARTTIME_GEO as starttime_geo,
    ENDTIME_GEO as endtime_geo,
)
from utils_basic import get_freq_limits_string

#--------------------------------------------------------------------------------------------------
# Main
#--------------------------------------------------------------------------------------------------

def main():
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--station", type=str, required=True)
    parser.add_argument("--min_freq_filter", type=float, default=20)
    parser.add_argument("--max_freq_filter", type=float, default=None)

    args = parser.parse_args()
    station = args.station
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter

    # Read in the data
    print(f"Reading in data for station {station}...")
    freq_limits_string = get_freq_limits_string(min_freq_filter, max_freq_filter)
    filename = f"sta_lta_detections_{freq_limits_string}_{station}.csv"
    filepath = join(dirpath, filename)
    detection_df = read_csv(filepath, parse_dates=["starttime", "endtime"])


    print("Computing hourly counts...")

    # Parse detection times (tz-naive)
    detection_df["starttime"] = to_datetime(detection_df["starttime"], utc=False)

    # Use the timezone from the geo limits
    geo_tz = Timestamp(starttime_geo).tz  # tzinfo or None

    if geo_tz is not None and detection_df["starttime"].dt.tz is None:
        # If your naive detection times are actually in the same timezone as the geo limits
        # (often UTC), localize them to that timezone.
        detection_df["starttime"] = detection_df["starttime"].dt.tz_localize(geo_tz)

    # Normalize window bounds to pandas Timestamps (preserve timezone)
    start = Timestamp(starttime_geo)
    end = Timestamp(endtime_geo)

    # Filter detections into the window [start, end)
    df_win = detection_df.loc[
        (detection_df["starttime"] >= start) & (detection_df["starttime"] < end)
    ].copy()

    # Build a full hourly index so empty hours appear with count=0
    hour_index = date_range(
        start=start.floor("h"),
        end=end.floor("h"),
        freq="h",
        tz=geo_tz,
    )

    # Bin by hour and count
    hourly_counts = (
        df_win.assign(hour=df_win["starttime"].dt.floor("h"))
              .groupby("hour")
              .size()
              .reindex(hour_index, fill_value=0)
              .rename_axis("hour")
              .reset_index(name="count")
    )

    # Save the hourly counts
    print(f"Saving hourly counts for station {station}...")
    filename = f"hourly_sta_lta_detection_counts_{freq_limits_string}_{station}.csv"
    filepath = join(dirpath, filename)
    hourly_counts.to_csv(filepath, index=False)

if __name__ == "__main__":
    main()