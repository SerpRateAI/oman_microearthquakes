"""
Compute the maximum amplitudes of an event group's hub event as a function of event-station
distance at the inner-ring stations.

The hub event is read from event_alignments_group{N}_{suffix}.csv (written by
align_satellite_events.py) and merged with the grouped-events JSONL for its detecting stations
and first-onset station. The hub is placed at its 3D-localized horizontal position from
hub_event_locations_preferred_scale_*.csv. For each inner-ring station of the group's subarray,
the three-component waveforms are loaded over a window around the aligned onset time, and the
maximum absolute amplitude of each component is normalized by the value at the first-onset
station. Results are saved to a CSV and plotted as amplitude ratio versus distance in three
vertically aligned panels (Z, 1, 2).
"""

from argparse import ArgumentParser
from os.path import isfile, join

from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots
from numpy import amax, isfinite, sqrt
from pandas import DataFrame, Timedelta, read_csv, read_json, to_datetime

from utils_basic import (
    DETECTION_DIR as dirpath_event,
    GEO_COMPONENTS as components,
    INNER_STATIONS_A,
    INNER_STATIONS_B,
    LOC_DIR as dirpath_loc,
    ROOTDIR_GEO as dirpath_waveform,
    get_freq_limits_string,
    get_geophone_coords,
)
from utils_cc import get_repeating_snippet_suffix
from utils_cont_waveform import load_waveform_slice
from utils_plot import save_figure
from utils_sta_lta import get_sta_lta_suffix

INNER_STATIONS_BY_SUBARRAY = {"A": INNER_STATIONS_A, "B": INNER_STATIONS_B}
MARKER_SIZE = 60


def panel_label(index):
    """Return panel label string for zero-based index, e.g. '(a)'."""
    return f"({chr(ord('a') + index)})"


def load_hub_event(group_label, suffix_group):
    """Load the hub event of a group with its aligned onset time and first-onset station."""
    filename = f"event_alignments_group{group_label:d}_{suffix_group}.csv"
    filepath = join(dirpath_event, filename)
    if not isfile(filepath):
        raise FileNotFoundError(f"Event alignment file not found: {filepath}")

    alignment_df = read_csv(filepath)
    hub_df = alignment_df[alignment_df["hub"]].copy()
    if len(hub_df) != 1:
        raise ValueError(f"Expected exactly one hub event in {filepath}; found {len(hub_df)}.")
    hub_df["aligned_time"] = to_datetime(hub_df["aligned_time"], utc=True, format="ISO8601")

    filename = f"grouped_events_group{group_label:d}_{suffix_group}.jsonl"
    group_event_df = read_json(join(dirpath_event, filename), lines=True)
    group_event_df = group_event_df[["event_id", "stations", "first_onset_station"]]

    hub_df = hub_df.merge(group_event_df, left_on="id", right_on="event_id", how="inner")
    if hub_df.empty:
        raise ValueError(f"Hub event in {filepath} is missing from the grouped-events file.")

    return hub_df.drop(columns="id").iloc[0]


def load_hub_location(group_label, hub_location_csv):
    """Return (id_hub, east, north) of a group's 3D-localized hub event."""
    if not isfile(hub_location_csv):
        raise FileNotFoundError(f"Hub location file not found: {hub_location_csv}")

    loc_df = read_csv(hub_location_csv)
    required = {"label", "id_hub", "east", "north"}
    missing = [col for col in required if col not in loc_df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required)} in {hub_location_csv}; missing {missing}."
        )

    loc_df = loc_df[loc_df["label"] == group_label]
    if loc_df.empty:
        raise ValueError(f"Group {group_label} not found in {hub_location_csv}.")

    row = loc_df.iloc[0]
    return int(row["id_hub"]), float(row["east"]), float(row["north"])


def load_group_subarray(group_label, group_info_csv):
    """Return the subarray of an event group from the selected-groups CSV."""
    if not isfile(group_info_csv):
        raise FileNotFoundError(f"Group info file not found: {group_info_csv}")

    group_df = read_csv(group_info_csv)
    required = {"label", "subarray"}
    missing = [col for col in required if col not in group_df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required)} in {group_info_csv}; missing {missing}."
        )

    group_df = group_df[group_df["label"] == group_label]
    if group_df.empty:
        raise ValueError(f"Group {group_label} not found in {group_info_csv}.")

    return str(group_df.iloc[0]["subarray"])


def normalize_by_first_onset_station(result_df):
    """Add amplitude_ratio: max amplitude relative to the first-onset station, per component."""
    ref_df = result_df.loc[
        result_df["station"] == result_df["first_onset_station"],
        ["component", "max_abs_amplitude"],
    ].rename(columns={"max_abs_amplitude": "max_abs_amplitude_ref"})

    missing_components = sorted(set(result_df["component"]) - set(ref_df["component"]))
    if missing_components:
        raise ValueError(
            f"No first-onset-station amplitude for components {missing_components}."
        )
    zero_components = ref_df.loc[ref_df["max_abs_amplitude_ref"] == 0, "component"].tolist()
    if zero_components:
        raise ValueError(
            f"First-onset-station amplitude is zero for components {zero_components}."
        )

    norm_df = result_df.merge(ref_df, on="component", how="left")
    norm_df["amplitude_ratio"] = norm_df["max_abs_amplitude"] / norm_df["max_abs_amplitude_ref"]
    return norm_df.drop(columns="max_abs_amplitude_ref")


def plot_component_panel(ax, comp_df, component, station_colors, panel_index):
    """Plot amplitude ratio versus distance for one component, one marker per station."""
    for station, color in station_colors.items():
        sta_df = comp_df[comp_df["station"] == station]
        if sta_df.empty:
            continue

        distance = sta_df["distance"].iloc[0]
        ratio = sta_df["amplitude_ratio"].iloc[0]
        ax.scatter(
            distance,
            ratio,
            s=MARKER_SIZE,
            color=color,
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
        )
        ax.annotate(
            station,
            (distance, ratio),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=10,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", linestyle=":", linewidth=0.5)
    ax.set_title(f"Component {component}", fontsize=12, fontweight="bold")
    ax.text(
        -0.08,
        1.05,
        panel_label(panel_index),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )


def main():
    parser = ArgumentParser(
        description=(
            "Compute and plot the normalized maximum amplitudes of an event group's hub event "
            "versus event-station distance at the inner-ring stations."
        ),
    )
    parser.add_argument("--group_label", type=int, default=18, help="Event group label")
    parser.add_argument(
        "--buffer_before_sec",
        type=float,
        default=0.05,
        help="Buffer before the aligned onset time in seconds",
    )
    parser.add_argument(
        "--buffer_after_sec",
        type=float,
        default=0.2,
        help="Buffer after the aligned onset time in seconds",
    )
    parser.add_argument(
        "--group_info_csv",
        type=str,
        default=None,
        help="Selected-groups CSV with label and subarray columns",
    )
    parser.add_argument(
        "--hub_location_csv",
        type=str,
        default=None,
        help="Hub-event location CSV with label, id_hub, east, and north columns",
    )
    parser.add_argument(
        "--figname",
        type=str,
        default=None,
        help="Output figure filename (default: auto from group label and suffix)",
    )
    parser.add_argument("--min_cc", type=float, default=0.85)
    parser.add_argument("--min_num_similar_snippet", type=int, default=10)
    parser.add_argument("--min_num_similar_station", type=int, default=3)
    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--thr_on", type=float, default=4.0)
    parser.add_argument("--thr_off", type=float, default=1.0)
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    args = parser.parse_args()

    group_label = args.group_label
    buffer_before_sec = args.buffer_before_sec
    buffer_after_sec = args.buffer_after_sec
    group_info_csv = args.group_info_csv
    hub_location_csv = args.hub_location_csv
    figname = args.figname
    min_cc = args.min_cc
    min_num_similar_snippet = args.min_num_similar_snippet
    min_num_similar_station = args.min_num_similar_station
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    thr_on = args.thr_on
    thr_off = args.thr_off
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter

    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=min_cc,
        min_num_similar=min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

    if group_info_csv is None:
        group_info_csv = join(dirpath_event, f"event_group_info_selected_{suffix_group}.csv")
    if hub_location_csv is None:
        hub_location_csv = join(
            dirpath_loc,
            f"hub_event_locations_preferred_scale_event_group_info_selected_{suffix_group}.csv",
        )

    print("--------------------------------")
    print("Computing hub event maximum amplitudes versus distance...")
    print("--------------------------------")
    print(f"Group label: {group_label}")
    print(f"Buffer before sec: {buffer_before_sec}")
    print(f"Buffer after sec: {buffer_after_sec}")
    print(f"Suffix group: {suffix_group}")

    subarray = load_group_subarray(group_label, group_info_csv)
    if subarray not in INNER_STATIONS_BY_SUBARRAY:
        raise ValueError(f"Unsupported subarray {subarray!r} for group {group_label}.")
    stations = INNER_STATIONS_BY_SUBARRAY[subarray]
    print(f"Subarray: {subarray}")
    print(f"Inner-ring stations: {stations}")

    station_df = get_geophone_coords()

    hub_row = load_hub_event(group_label, suffix_group)
    event_id = int(hub_row["event_id"])
    aligned_time = hub_row["aligned_time"]
    detected_stations = set(hub_row["stations"])
    first_onset_station = hub_row["first_onset_station"]

    id_hub, east_hub, north_hub = load_hub_location(group_label, hub_location_csv)
    if event_id != id_hub:
        raise ValueError(
            f"Hub event {event_id} in the alignment file does not match "
            f"id_hub {id_hub} in {hub_location_csv}."
        )
    print(f"Hub event {id_hub} location: E={east_hub:.2f} m, N={north_hub:.2f} m")
    print(f"Aligned onset time: {aligned_time}")
    print(f"First-onset station: {first_onset_station}")

    if first_onset_station not in stations:
        raise ValueError(
            f"First-onset station {first_onset_station} is not an inner-ring station {stations}."
        )

    distance_by_station = {
        station: float(sqrt(
            (east_hub - station_df.loc[station, "east"]) ** 2
            + (north_hub - station_df.loc[station, "north"]) ** 2
        ))
        for station in stations
    }

    waveform_filepath = join(dirpath_waveform, f"preprocessed_data_{suffix_freq}.h5")
    window_start = aligned_time - Timedelta(seconds=buffer_before_sec)
    window_end = aligned_time + Timedelta(seconds=buffer_after_sec)

    rows = []
    for station in stations:
        try:
            waveform_dict, _ = load_waveform_slice(
                waveform_filepath,
                station,
                window_start,
                endtime=window_end,
                normalize=False,
            )
        except Exception as exc:
            print(
                f"Warning: failed to load waveform for event {event_id} "
                f"at {station} [{window_start}, {window_end}]: {exc}"
            )
            continue

        distance = distance_by_station[station]
        for component in components:
            waveform = waveform_dict[component]
            if not isfinite(waveform).all():
                print(
                    f"Warning: event {event_id} at {station}.{component} has non-finite data; "
                    f"skipping."
                )
                continue

            max_abs_amplitude = float(amax(abs(waveform)))
            print(
                f"  {station}.{component}: distance {distance:.2f} m, "
                f"max amplitude {max_abs_amplitude:.4g}"
            )
            rows.append({
                "event_id": event_id,
                "station": station,
                "first_onset_station": first_onset_station,
                "detected": station in detected_stations,
                "component": component,
                "distance": distance,
                "max_abs_amplitude": max_abs_amplitude,
            })

    result_df = DataFrame(rows)
    if result_df.empty:
        raise ValueError(f"No amplitudes computed for hub event {event_id}.")

    result_df = normalize_by_first_onset_station(result_df)

    outfile = f"hub_event_max_amplitudes_vs_distance_group{group_label:d}_{suffix_group}.csv"
    outfile_path = join(dirpath_event, outfile)
    result_df.to_csv(outfile_path, index=False)
    print(f"Saved amplitudes to {outfile_path} ({len(result_df)} rows).")

    print("Plotting amplitude ratios versus distance...")
    cmap = colormaps["tab10"]
    station_colors = {station: cmap(i_sta) for i_sta, station in enumerate(stations)}

    fig, axes = subplots(len(components), 1, figsize=(8, 10), sharex=True, sharey=True)
    for i_comp, component in enumerate(components):
        ax = axes[i_comp]
        comp_df = result_df[result_df["component"] == component]
        plot_component_panel(ax, comp_df, component, station_colors, i_comp)
        ax.set_ylabel("Max amplitude / first-onset station", fontsize=12)

    axes[-1].set_xlabel("Distance (m)", fontsize=12)
    axes[0].set_xlim(0.0, result_df["distance"].max() * 1.15)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor="black",
            label=station,
        )
        for station, color in station_colors.items()
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=10,
        framealpha=1.0,
    )

    fig.suptitle(
        f"Group {group_label:d} hub event {id_hub:d}: max amplitude vs distance",
        fontsize=14,
        fontweight="bold",
        y=0.95,
    )
    fig.subplots_adjust(hspace=0.25)

    if figname is None:
        figname = f"event_group_hub_max_amplitude_vs_distance_group{group_label:d}_{suffix}.png"
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
