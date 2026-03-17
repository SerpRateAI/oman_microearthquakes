"""
Associate STA/LTA detections across stations with a spherical-wavefront constraint (meters & m/s).

Outputs:
  - per-event CSV with best-fit fields: east (m), north (m), vel_app (m/s), rmse (s)
  - per-event station-times CSV
  - JSONL of full AssociatedEvent objects (same fields)
  - Snuffler .mrk file for visualization
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import date
from utils_cc import get_repeating_snippet_suffix

from numpy import (
    array, asarray, arange, stack, meshgrid, ones_like,
    sqrt, sum as np_sum, mean as np_mean, isfinite,
    timedelta64, where
)
from pandas import DataFrame, Timestamp, read_csv, concat, to_datetime

# Project-specific
from utils_basic import (
    DETECTION_DIR as dirpath_detect,
    GEO_STATIONS as stations,
    NETWORK as network,
    get_freq_limits_string,
    get_geophone_coords,
)

from utils_basic import HAMMER_STARTTIME as starttime_hammer, HAMMER_ENDTIME as endtime_hammer
from utils_sta_lta import get_sta_lta_suffix


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AssociatedEvent:
    """Container for an associated event and its best-fit spherical-wave source."""
    event_id: int
    first_onset: Timestamp
    first_onset_station: str
    stations: List[str]
    per_station_times: Dict[str, Tuple[Timestamp, Timestamp]]
    per_station_ids: Dict[str, int]
    east: Optional[float] = None
    north: Optional[float] = None
    vel_app: Optional[float] = None
    rmse: Optional[float] = None

    def to_summary_row(self) -> dict:
        """Convert to a single-row dict for CSV output."""
        return {
            "event_id": self.event_id,
            "first_onset": self.first_onset,
            "first_onset_station": self.first_onset_station,
            "stations": ",".join(self.stations),
            "n_stations": len(self.stations),
            "event_start": min(t[0] for t in self.per_station_times.values()),
            "event_end":   max(t[1] for t in self.per_station_times.values()),
            "east": self.east,
            "north": self.north,
            "vel_app": self.vel_app,
            "rmse": self.rmse,
        }

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
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
def build_associated_event(
    event_id: int,
    block_full: DataFrame,
    *,
    east: Optional[float] = None,
    north: Optional[float] = None,
    vel_app: Optional[float] = None,
    rmse: Optional[float] = None,
) -> AssociatedEvent:
    """Build an AssociatedEvent object."""
    block = block_full.copy()

    first_row = block.sort_values(["starttime", "station"]).iloc[0]
    first_onset = Timestamp(first_row["starttime"])
    first_station = str(first_row["station"])
    stations_in_event = sorted(block["station"].unique().tolist())

    per_station_times: Dict[str, Tuple[Timestamp, Timestamp]] = {}
    per_station_ids: Dict[str, int] = {}
    for station, group in block.groupby("station"):
        station_start = Timestamp(group["starttime"].min())
        station_end   = Timestamp(group["endtime"].max())
        per_station_times[str(station)] = (station_start, station_end)
        per_station_ids[str(station)] = int(group["id"].iloc[0]) # integer id

    return AssociatedEvent(
        event_id=event_id,
        first_onset=first_onset,
        first_onset_station=first_station,
        stations=stations_in_event,
        per_station_times=per_station_times,
        per_station_ids=per_station_ids,
        east=east, north=north, vel_app=vel_app, rmse=rmse,
    )


# -----------------------------------------------------------------------------
# Coordinate loader (meters)
# -----------------------------------------------------------------------------
def build_station_xy(station_names: List[str], csv_override: str | None = None) -> Dict[str, Tuple[float, float]]:
    """Return {station: (x_m, y_m)} using utils_basic.get_geophone_coords() or a CSV override."""
    if csv_override:
        coords_df = read_csv(csv_override)
        coords_df = coords_df.rename(columns={c: c.strip().lower() for c in coords_df.columns})
        index_col = "station" if "station" in coords_df.columns else "name" if "name" in coords_df.columns else None
        if index_col is None:
            raise ValueError("station_coords_csv must have a 'station' or 'name' column")
        coords_df = coords_df.set_index(index_col)
    else:
        coords_df = get_geophone_coords()
        coords_df.columns = [c.strip().lower() for c in coords_df.columns]

    cols = set(coords_df.columns)
    if {"east", "north"} <= cols:
        xcol, ycol = "east", "north"
    elif {"x_m", "y_m"} <= cols:
        xcol, ycol = "x_m", "y_m"
    elif {"x", "y"} <= cols:
        xcol, ycol = "x", "y"
    else:
        raise ValueError("Could not find coordinate columns: need {east,north} or {x_m,y_m} or {x,y}")

    station_xy_m: Dict[str, Tuple[float, float]] = {}
    missing_stations = []
    for station in station_names:
        key = station if station in coords_df.index else str(station)
        if key not in coords_df.index:
            missing_stations.append(station)
            continue
        station_xy_m[station] = (
            float(coords_df.loc[key, xcol]),
            float(coords_df.loc[key, ycol])
        )

    if missing_stations:
        print(f"WARNING: Missing coordinates for stations: {', '.join(missing_stations)}")
    return station_xy_m


# -----------------------------------------------------------------------------
# Near-field helper (rectangular, onset-centered; vectorized; RMSE residual)
# -----------------------------------------------------------------------------
def _check_station_offset(first_station: str, block_unique: DataFrame, max_station_offset: float, station_xy: dict) -> bool:
    x_first, y_first = station_xy[first_station]
    x_stations = [station_xy[s][0] for s in block_unique["station"]]
    y_stations = [station_xy[s][1] for s in block_unique["station"]]
    distances = sqrt((x_first - array(x_stations))**2 + (y_first - array(y_stations))**2)
    max_distance = distances.max()
    if max_distance > max_station_offset:
        return False
    else:
        return True


def _ensure_xy(station_xy_m: dict, stations_needed: List[str]):
    if station_xy_m is None:
        raise KeyError("station_xy_m is None")
    missing = [s for s in stations_needed if s not in station_xy_m]
    if missing:
        raise KeyError(f"Missing XY for stations: {missing}")
    return array([station_xy_m[s] for s in stations_needed], dtype=float)


def _nearfield_fit_ok_rect_centered(
    block_unique: DataFrame,
    station_xy: dict,
    center_station: str,
    half_width: float,
    half_height: float,
    step: float,
    vmin: float,
    vmax: float,
    max_tt_rmse: float,
):
    """Vectorized near-field spherical-wave fit; all units in meters & m/s; residual metric = RMSE (L2)."""
    if len(block_unique) < 3:
        return True, {"reason": "skipped: fewer than 3 stations"}

    station_list = block_unique["station"].tolist()
    try:
        station_coords = _ensure_xy(station_xy, station_list)
        center_x, center_y = station_xy[center_station]
    except Exception as err:
        return True, {"reason": f"skipped: {err}"}

    # Relative times (seconds)
    times_abs = block_unique["starttime"]
    time_ref = times_abs.min()
    times_sec = (times_abs - time_ref).dt.total_seconds().to_numpy(dtype=float)
    ones = ones_like(times_sec)
    s1 = float(np_sum(ones))
    st = float(np_sum(times_sec))

    # Rectangular grid around the first-onset station
    grid_x = arange(center_x - half_width,  center_x + half_width  + 1e-9, step)
    grid_y = arange(center_y - half_height, center_y + half_height + 1e-9, step)
    grid_points = stack(meshgrid(grid_x, grid_y, indexing="xy"), axis=-1).reshape(-1, 2)
    if len(grid_points) == 0:
        return True, {"reason": "skipped: empty grid"}

    # Distances (meters)
    diffs = station_coords[None, :, :] - grid_points[:, None, :]
    distances = sqrt(np_sum(diffs**2, axis=2))

    # Closed-form LS (t ≈ alpha + beta * r)
    sum_r  = np_sum(distances, axis=1)
    sum_r2 = np_sum(distances * distances, axis=1)
    sum_rt = np_sum(distances * times_sec[None, :], axis=1)
    determinant = (s1 * sum_r2 - sum_r * sum_r)
    determinant = where((determinant < 1e-12) & (determinant > -1e-12), 1e-12, determinant)

    alpha = (sum_r2 * st - sum_r * sum_rt) / determinant
    beta  = (s1 * sum_rt - sum_r * st) / determinant
    beta  = where(beta <= 1e-12, float("nan"), beta)
    velocity = 1.0 / beta
    valid_velocity = (velocity >= vmin) & (velocity <= vmax) & isfinite(velocity)

    best_fit = {"rmse": float("inf")}
    valid_indices = where(valid_velocity)[0]
    if valid_indices.size == 0:
        return False, {"reason": "no nodes within velocity bounds"}

    for idx in valid_indices:
        predicted_times = alpha[idx] + beta[idx] * distances[idx]
        residuals = times_sec - predicted_times
        rmse = float(sqrt(np_mean(residuals * residuals)))
        if rmse < best_fit["rmse"]:
            best_fit = {
                "east": float(grid_points[idx, 0]),
                "north": float(grid_points[idx, 1]),
                "vel_app": float(velocity[idx]),
                "rmse": rmse,
            }

    ok = best_fit["rmse"] <= max_tt_rmse
    return ok, best_fit


# -----------------------------------------------------------------------------
# Association logic
# -----------------------------------------------------------------------------
def associate_detections(
    det_df: DataFrame,
    assoc_window_sec: float,
    min_stations: int,
    *,
    station_xy: dict | None = None,
    nearfield_rect_half_width: float = 10.0,
    nearfield_rect_half_height: float =10.0,
    nearfield_step: float = 2.0,
    vmin: float = 200.0,
    vmax: float = 1000.0,
    max_tt_rmse: float = 0.15,
    max_station_offset: float = 25.0,
) -> Dict[str, DataFrame]:

    """Associate detections into events with a near-field spherical-wave constraint (RMSE)."""
    if det_df is None or len(det_df) == 0:
        empty = det_df.copy() if det_df is not None else DataFrame(columns=["starttime", "endtime", "station"])
        return {"detections": empty.assign(event_id=[]),
                "events": DataFrame(), "events_objs": []}

    df = det_df.copy()
    df = df.sort_values("starttime").reset_index(drop=True)

    onset_times = df["starttime"].to_numpy()
    total_detections = len(df)
    event_ids = [-1] * total_detections
    events_objs: List[AssociatedEvent] = []
    events_rows: List[dict] = []

    time_delta = timedelta64(int(round(assoc_window_sec * 1e9)), "ns")
    index = 0
    next_event_id = 0

    while index < total_detections:
        window_start = onset_times[index]
        j = index
        while j + 1 < total_detections and (onset_times[j + 1] - window_start) <= time_delta:
            j += 1

        block = df.iloc[index:j+1]
        block_unique = block.sort_values("starttime").groupby("station", as_index=False).first()
        unique_stations = len(block_unique["station"])

        travel_time_ok = False
        best_fit = None
        if station_xy is not None and unique_stations >= 3:
            first_station = block_unique.sort_values("starttime").iloc[0]["station"]
            # Check if the largest station offset is below the limit
            station_offset_ok = _check_station_offset(first_station, block_unique, max_station_offset, station_xy)

            if station_offset_ok:
                travel_time_ok, best_fit = _nearfield_fit_ok_rect_centered(
                    block_unique=block_unique,
                    station_xy=station_xy,
                    center_station=first_station,
                    half_width=nearfield_rect_half_width,
                    half_height=nearfield_rect_half_height,
                    step=nearfield_step,
                    vmin=vmin,
                    vmax=vmax,
                    max_tt_rmse=max_tt_rmse,
                )

        if unique_stations >= min_stations and travel_time_ok:
            event_id = next_event_id
            event_ids[index:j+1] = [event_id] * (j - index + 1)

            event_obj = build_associated_event(
                event_id,
                block_full=block,
                east=(best_fit.get("east") if best_fit else None),
                north=(best_fit.get("north") if best_fit else None),
                vel_app=(best_fit.get("vel_app") if best_fit else None),
                rmse=(best_fit.get("rmse") if best_fit else None),
            )
            events_objs.append(event_obj)
            event_row = event_obj.to_summary_row()
            events_rows.append(event_row)

            print("-----------------")
            print("Event detected:")
            print("-----------------")
            print(f"First onset station: {event_row["first_onset_station"]}")
            print(f"Fist onset time: {event_row["first_onset"]}")
            next_event_id += 1

        index = j + 1

    detections_out = df.copy()
    detections_out["event_id"] = asarray(event_ids, dtype=int)
    detections_out = detections_out[ detections_out["event_id"] >= 0 ] # keep only detections that are associated

    events_out = DataFrame(events_rows).sort_values("event_id").reset_index(drop=True)
    return {"detections": detections_out, "events": events_out, "events_objs": events_objs}


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def events_to_detail_df(events_objs) -> DataFrame:
    rows = []
    for e in events_objs:
        per_station_ids = e.per_station_ids
        per_station_times = e.per_station_times
        for sta, id in per_station_ids.items():
            starttime = per_station_times[sta][0]
            endtime = per_station_times[sta][1]
            rows.append({
                "event_id": e.event_id,
                "station": sta,
                "snippet_id": id,
                "starttime": starttime,
                "endtime": endtime,
            })
    return DataFrame(rows).sort_values(["event_id", "station"]).reset_index(drop=True)


def write_snuffler_markers(outpath: Path, starttimes, endtimes, seed_ids):
    with open(outpath, "w") as f:
        f.write("# Snuffler Markers File Version 0.2\n")
        for starttime, endtime, seed_id in zip(starttimes, endtimes, seed_ids):
            f.write(
                f"{starttime:%Y-%m-%d} {starttime:%H:%M:%S.%f} "
                f"{endtime:%Y-%m-%d} {endtime:%H:%M:%S.%f} "
                f"{(endtime - starttime).total_seconds():.3f} 0 {seed_id}\n"
            )


def write_outputs(out_dir: Path, suffix: str, det_out: DataFrame, events_out: DataFrame, events_objs):
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-event detail CSV
    detail_df = events_to_detail_df(events_objs)
    detail_csv = out_dir / f"associated_detections_{suffix}.csv"
    detail_df.to_csv(detail_csv, index=False)

    # JSONL (includes east, north, vel_app, rmse)
    jsonl = out_dir / f"associated_events_{suffix}.jsonl"
    with jsonl.open("w") as f:
        for e in events_objs:
            f.write(json.dumps(e.to_dict()) + "\n")

    # Snuffler marker file
    snuffler_path = out_dir / f"associated_detections_{suffix}.mrk"
    seed_ids = [f"{network}.{sta}..GH1" for sta in det_out["station"]]
    write_snuffler_markers(snuffler_path, det_out["starttime"], det_out["endtime"], seed_ids)

    print("Wrote:")
    print(f"- {detail_csv}")
    print(f"- {jsonl}")
    print(f"- {snuffler_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--repeating", action="store_true", help="If set, process the repeating snippets only.")
    parser.add_argument("--min_cc", type=float, default=0.85)
    parser.add_argument("--min_num_similar", type=int, default=10)
    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--thr_on", type=float, default=4.0)
    parser.add_argument("--thr_off", type=float, default=1.0)
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    parser.add_argument("--assoc_window_sec", type=float, default=0.1)
    parser.add_argument("--min_stations", type=int, default=3)

    # Near-field spherical search (meters & m/s)
    parser.add_argument("--nearfield_rect_half_width", type=float, default=10.0)
    parser.add_argument("--nearfield_rect_half_height", type=float, default=10.0)
    parser.add_argument("--nearfield_step", type=float, default=2.0)
    parser.add_argument("--vmin", type=float, default=200.0)
    parser.add_argument("--vmax", type=float, default=1000.0)
    parser.add_argument("--max_station_offset", type=float, default=25.0)
    parser.add_argument("--max_tt_rmse", type=float, default=0.02)

    # NEW: test mode flag
    parser.add_argument("--test", action="store_true",
                        help="If set, process only detections with starttime on 2020-01-14.")

    args = parser.parse_args()
    repeating = args.repeating
    min_cc = args.min_cc
    min_num_similar = args.min_num_similar
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    thr_on = args.thr_on
    thr_off = args.thr_off
    assoc_window_sec = args.assoc_window_sec
    min_stations = args.min_stations
    max_station_offset = args.max_station_offset
    test = args.test
    max_tt_rmse = args.max_tt_rmse
    nearfield_rect_half_width = args.nearfield_rect_half_width
    nearfield_rect_half_height = args.nearfield_rect_half_height
    nearfield_step = args.nearfield_step
    vmin = args.vmin
    vmax = args.vmax
    test = args.test

    print("Associating STA/LTA detections with a spherical-wavefront constraint (RMSE residual)...")
    if repeating:
        print("Processing repeating snippets only...")
    else:
        print("Processing all snippets...")

    if test:
        target_day = date(2020, 1, 14)

    # STA/LTA suffixes for file names
    freq_limits_string = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix = f"{freq_limits_string}"

    sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    suffix += f"_{sta_lta_suffix}"
    if repeating:
        suffix = f"repeating_{suffix}"
        repeating_snippet_suffix = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar)
        suffix += f"_{repeating_snippet_suffix}"

    # Load detections
    det_dfs = []
    for station in stations:
        filename = f"sta_lta_detections_{suffix}_{station}.csv"
        filepath = Path(dirpath_detect) / filename
        det_df = read_csv(filepath, parse_dates=["starttime", "endtime"])
        det_df["station"] = station
        det_dfs.append(det_df)
        print(f"Read detections for {station}: {len(det_df)} rows")
    det_df = concat(det_dfs, ignore_index=True)
    det_df["starttime"] = to_datetime(det_df["starttime"], utc=True)
    det_df["endtime"] = to_datetime(det_df["endtime"], utc=True)

    # Remove the hammer signals
    det_df = det_df[(det_df["starttime"] < starttime_hammer) | (det_df["starttime"] > endtime_hammer)]
    print(f"Number of detections after removing the hammer signals: {len(det_df)}")

    # If in test mode, keep only starttime on 2020-01-14
    if test:
        target_day = date(2020, 1, 14)
        before = len(det_df)
        det_df = det_df[det_df["starttime"].dt.date == target_day].reset_index(drop=True)
        after = len(det_df)
        print(f"[TEST MODE] Filtered detections to {target_day.isoformat()}: {after}/{before} rows retained")

    # Load station coordinates (meters): default from utils_basic, optional override CSV
    station_xy = build_station_xy(stations)

    # Associate with spherical-wave constraint (RMSE residual)
    out = associate_detections(
        det_df,
        assoc_window_sec=assoc_window_sec,
        min_stations=min_stations,
        station_xy=station_xy,
        nearfield_rect_half_width=nearfield_rect_half_width,
        nearfield_rect_half_height=nearfield_rect_half_height,
        nearfield_step=nearfield_step,
        vmin=vmin,
        vmax=vmax,
        max_tt_rmse=max_tt_rmse,
        max_station_offset=max_station_offset,
    )

    detections_out = out["detections"]
    events_out = out["events"]
    events_objs = out["events_objs"]

    print(f"\nAssociation window: {assoc_window_sec:.3f} s; min_stations: {min_stations}")
    print(f"Near-field rectangular half width: {nearfield_rect_half_width:.3f} m")
    print(f"Near-field rectangular half height: {nearfield_rect_half_height:.3f} m")
    print(f"Near-field step: {nearfield_step:.3f} m")
    print(f"Velocity minimum: {vmin:.3f} m/s")
    print(f"Velocity maximum: {vmax:.3f} m/s")
    print(f"Maximum travel time RMSE: {max_tt_rmse:.3f} s")
    print(f"Maximum station offset: {max_station_offset:.3f} m")
    print(f"Associated detections: {len(detections_out)}")
    print(f"Number of events: {len(events_out)}")

    # Write outputs
    out_dir = Path(dirpath_detect)
    if test:
        suffix += "_test_2020-01-14"

    write_outputs(out_dir, suffix, detections_out, events_out, events_objs)


if __name__ == "__main__":
    main()
