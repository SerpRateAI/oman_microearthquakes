"""
Associate STA/LTA detections across stations.

- Groups detections whose onset times fall within a window into events.
- Counts each station at most once per event (earliest onset wins).
- Emits:
    1) per-event summary CSV,
    2) per-event normalized station-times CSV,
    3) JSONL with full AssociatedEvent objects,
    4) Snuffler marker file (.mrk) for per-detection windows with seed_id = "{station}..BH1".
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from datetime import timedelta
import json

from numpy import asarray
from pandas import DataFrame, Timestamp, read_csv, concat

# Optional imports (kept for compatibility)
from matplotlib.pyplot import Axes, figure  # noqa: F401
from matplotlib.gridspec import GridSpec    # noqa: F401

# Project-specific imports
from utils_basic import DETECTION_DIR as dirpath_detect, GEO_STATIONS as stations
from utils_basic import get_freq_limits_string
from utils_sta_lta import get_sta_lta_suffix


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AssociatedEvent:
    """Container for an associated event across stations."""
    event_id: int
    first_onset: Timestamp
    first_onset_station: str
    stations: List[str]
    per_station_times: Dict[str, Tuple[Timestamp, Timestamp]]

    def to_summary_row(self) -> dict:
        return {
            "event_id": self.event_id,
            "first_onset": self.first_onset,
            "first_onset_station": self.first_onset_station,
            "stations": ",".join(self.stations),
            "n_stations": len(self.stations),
            "event_start": min(t[0] for t in self.per_station_times.values()),
            "event_end":   max(t[1] for t in self.per_station_times.values()),
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["first_onset"] = self.first_onset.isoformat()
        d["per_station_times"] = {
            sta: (t0.isoformat(), t1.isoformat())
            for sta, (t0, t1) in self.per_station_times.items()
        }
        return d


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------
def build_associated_event(event_id: int, block_full: DataFrame) -> AssociatedEvent:
    """Build an AssociatedEvent from a detection window."""
    b = block_full.copy()
    b["starttime"] = b["starttime"].astype("datetime64[ns]")
    b["endtime"]   = b["endtime"].astype("datetime64[ns]")

    row_first = b.sort_values(["starttime", "station"]).iloc[0]
    first_onset = Timestamp(row_first["starttime"])
    first_onset_station = str(row_first["station"])

    stations_in_evt = sorted(b["station"].unique().tolist())

    per_station_times: Dict[str, Tuple[Timestamp, Timestamp]] = {}
    for sta, g in b.groupby("station"):
        sta_start = Timestamp(g["starttime"].min())
        sta_end   = Timestamp(g["endtime"].max())
        per_station_times[str(sta)] = (sta_start, sta_end)

    return AssociatedEvent(
        event_id=event_id,
        first_onset=first_onset,
        first_onset_station=first_onset_station,
        stations=stations_in_evt,
        per_station_times=per_station_times,
    )


# -----------------------------------------------------------------------------
# Association logic
# -----------------------------------------------------------------------------
def associate_detections(det_df: DataFrame, assoc_window_sec: float, min_stations: int) -> Dict[str, DataFrame]:
    """
    Associate detections into events based on onset proximity.
    Duplicate stations within a window count only once toward `min_stations`.
    """
    from pandas import DataFrame as _DF

    if det_df is None or len(det_df) == 0:
        empty = det_df.copy() if det_df is not None else _DF(columns=["starttime", "endtime", "station"])
        return {
            "detections": empty.assign(event_id=asarray([], dtype=int)),
            "events": _DF(columns=[
                "event_id","first_onset","first_onset_station","stations",
                "n_stations","event_start","event_end"
            ]),
            "events_objs": []
        }

    df = det_df.copy()
    df["starttime"] = df["starttime"].astype("datetime64[ns]")
    df["endtime"]   = df["endtime"].astype("datetime64[ns]")
    df = df.sort_values("starttime").reset_index(drop=True)

    onsets = df["starttime"].to_numpy()
    n = len(df)

    event_ids = [-1] * n
    events_objs: List[AssociatedEvent] = []
    events_rows: List[dict] = []

    i = 0
    next_event_id = 0
    delta = timedelta(seconds=assoc_window_sec)

    while i < n:
        window_start = onsets[i]
        j = i
        while j + 1 < n and (onsets[j + 1] - window_start) <= delta:
            j += 1

        block = df.iloc[i:j+1]

        # Deduplicate stations for the threshold check
        block_unique = (
            block.sort_values("starttime")
                 .groupby("station", as_index=False)
                 .first()
        )
        n_unique = len(block_unique["station"])

        if n_unique >= min_stations:
            eid = next_event_id
            event_ids[i:j+1] = [eid] * (j - i + 1)
            evt = build_associated_event(eid, block_full=block)
            events_objs.append(evt)
            events_rows.append(evt.to_summary_row())
            next_event_id += 1

        i = j + 1

    det_out = df.copy()
    det_out["event_id"] = asarray(event_ids, dtype=int)

    from pandas import DataFrame as _DF2
    events_out = _DF2(events_rows).sort_values("event_id").reset_index(drop=True)

    return {"detections": det_out, "events": events_out, "events_objs": events_objs}


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def events_to_detail_df(events_objs) -> DataFrame:
    """Flatten per-station envelopes into a table."""
    rows = []
    for e in events_objs:
        for sta, (t0, t1) in e.per_station_times.items():
            rows.append({
                "event_id": e.event_id,
                "station": sta,
                "station_start": t0,
                "station_end": t1,
            })
    return DataFrame(rows).sort_values(["event_id", "station"]).reset_index(drop=True)


def write_snuffler_markers(outpath: Path, starttimes, endtimes, seed_ids):
    """Write detections to a Snuffler marker file."""
    with open(outpath, "w") as f:
        f.write("# Snuffler Markers File Version 0.2\n")
        for starttime, endtime, seed_id in zip(starttimes, endtimes, seed_ids):
            f.write(
                f"{starttime:%Y-%m-%d} {starttime:%H:%M:%S.%f} "
                f"{endtime:%Y-%m-%d} {endtime:%H:%M:%S.%f} "
                f"{(endtime - starttime).total_seconds():.3f} 0 {seed_id}\n"
            )


def write_outputs(out_dir: Path, base: str, det_out: DataFrame, events_out: DataFrame, events_objs):
    """Write summary, details, JSONL, and Snuffler outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) per-event summary CSV
    ev_csv = out_dir / f"associated_events_{base}.csv"
    events_out.to_csv(ev_csv, index=False)

    # 2) per-event detail (normalized) CSV
    detail_df = events_to_detail_df(events_objs)
    detail_csv = out_dir / f"associated_event_station_times_{base}.csv"
    detail_df.to_csv(detail_csv, index=False)

    # 3) JSONL of the full object
    jsonl = out_dir / f"associated_events_{base}.jsonl"
    with jsonl.open("w") as f:
        for e in events_objs:
            f.write(json.dumps(e.to_dict()) + "\n")

    # 4) Snuffler marker file
    snuffler_path = out_dir / f"associated_detections_{base}.mrk"
    seed_ids = [f"{sta}..BH1" for sta in det_out["station"]]
    write_snuffler_markers(snuffler_path, det_out["starttime"], det_out["endtime"], seed_ids)

    print("Wrote:")
    print(f"- {ev_csv}")
    print(f"- {detail_csv}")
    print(f"- {jsonl}")
    print(f"- {snuffler_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--thr_on", type=float, default=4.0)
    parser.add_argument("--thr_off", type=float, default=1.0)
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    parser.add_argument("--assoc_window_sec", type=float, default=0.1,
                        help="Association window on onset times (seconds).")
    parser.add_argument("--min_stations", type=int, default=3,
                        help="Minimum unique stations required for an event.")
    args = parser.parse_args()

    # STA/LTA parameters
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    thr_on = args.thr_on
    thr_off = args.thr_off
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    assoc_window_sec = args.assoc_window_sec
    min_stations = args.min_stations

    # Define suffixes
    freq_limits_string = get_freq_limits_string(min_freq_filter, max_freq_filter)
    sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)

    # Read detections
    det_dfs = []
    for station in stations:
        filename = f"sta_lta_detections_{freq_limits_string}_{sta_lta_suffix}_{station}.csv"
        filepath = Path(dirpath_detect) / filename
        det_df = read_csv(filepath, parse_dates=["starttime", "endtime"])
        det_df["station"] = station
        print(f"Read detections for {station}: {len(det_df)} rows")
        det_dfs.append(det_df)

    det_df = concat(det_dfs, ignore_index=True)

    # Associate detections
    out = associate_detections(det_df, assoc_window_sec=assoc_window_sec, min_stations=min_stations)
    det_out = out["detections"]
    events_out = out["events"]
    events_objs = out["events_objs"]

    print(f"\nAssociation window: {assoc_window_sec:.3f} s; min_stations: {min_stations}")
    print(f"Total detections: {len(det_out)}")
    print(f"Associated detections: {(det_out['event_id'] >= 0).sum()}")
    print(f"Number of events: {len(events_out)}")

    # Write outputs
    out_dir = Path(dirpath_detect)
    base = f"{freq_limits_string}_{sta_lta_suffix}_window{assoc_window_sec:.3f}s_min_sta{min_stations}"
    write_outputs(out_dir, base, det_out, events_out, events_objs)


if __name__ == "__main__":
    main()
