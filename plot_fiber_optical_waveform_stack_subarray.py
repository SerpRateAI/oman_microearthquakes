"""
Plot hub/satellite waveforms and the hub-event waveform stack for group 20
(Subarray B, vertical component only).

One row of the 3×2 figure in plot_aguadv_waveform_stacks_subarray.py: hub and
satellite waveforms on the left, waveform stack on the right.
"""

from argparse import ArgumentParser
from os.path import join
from pathlib import Path

from matplotlib.pyplot import subplots
from numpy import amax, arange
from obspy import read
from pandas import Timedelta, read_csv, read_json, to_datetime

from utils_basic import (
    DETECTION_DIR as dirpath_event,
    INNER_STATIONS_B as stations_inner_b,
    MIDDLE_STATIONS_B as stations_middle_b,
    ROOTDIR_GEO as dirpath_waveform,
    SAMPLING_RATE as sample_rate,
    get_freq_limits_string,
    get_geophone_coords,
)
from utils_cc import get_repeating_snippet_suffix
from utils_cont_waveform import load_waveform_slice
from utils_plot import get_geo_component_color, save_figure
from utils_sta_lta import get_sta_lta_suffix

GROUP_LABEL = 20
VERTICAL_COMPONENT = "Z"
SUBARRAY = "B"
STATIONS = stations_inner_b + stations_middle_b


def convert_station_times(station_dict):
    return {
        station: to_datetime(times, utc=True, format="ISO8601")
        for station, times in station_dict.items()
    }


def get_stations_sorted(station_list):
    coord_df = get_geophone_coords()
    coord_df = coord_df.sort_values(by="north")
    coord_df = coord_df[coord_df.index.isin(station_list)]
    return coord_df.index.tolist()


def load_group_data(group_label, suffix_group):
    filename = f"grouped_events_group{group_label:d}_{suffix_group}.jsonl"
    event_df = read_json(join(dirpath_event, filename), lines=True)
    event_df["first_onset"] = to_datetime(event_df["first_onset"], errors="coerce")
    event_df["per_station_times"] = event_df["per_station_times"].apply(convert_station_times)

    filename = f"event_alignments_group{group_label:d}_{suffix_group}.csv"
    alignment_df = read_csv(join(dirpath_event, filename))
    alignment_df["aligned_time"] = to_datetime(alignment_df["aligned_time"], format="ISO8601")
    alignment_df["hub"] = alignment_df["hub"].astype(bool)

    id_hub = alignment_df.loc[alignment_df["hub"], "id"].values[0]
    hub_dict = event_df.loc[event_df["event_id"] == id_hub].iloc[0].to_dict()
    per_station_times_hub = hub_dict["per_station_times"]
    num_events = len(alignment_df)

    return alignment_df, per_station_times_hub, id_hub, num_events


def plot_hub_and_satellite_waveforms_z(
    ax,
    alignment_df,
    per_station_times_hub,
    stations,
    waveform_filepath,
    buffer_before_sec,
    buffer_after_sec,
    scale_factor,
):
    component = VERTICAL_COMPONENT
    color_hub = get_geo_component_color(component)

    for _, row in alignment_df.iterrows():
        aligned_time = row["aligned_time"]
        hub = row["hub"]
        aligned_starttime_plot = aligned_time - Timedelta(seconds=buffer_before_sec)
        aligned_endtime_plot = aligned_time + Timedelta(seconds=buffer_after_sec)

        if hub:
            color = color_hub
            linewidth = 1.0
            zorder = 2
        else:
            color = "lightgray"
            linewidth = 0.2
            zorder = 0

        for i_station, station in enumerate(stations):
            try:
                waveform_dict, _ = load_waveform_slice(
                    waveform_filepath,
                    station,
                    aligned_starttime_plot,
                    endtime=aligned_endtime_plot,
                    normalize=True,
                )
            except Exception as e:
                print(f"Error loading waveform slice for station {station}: {e}")
                continue

            waveform = waveform_dict[component]
            timeax = arange(len(waveform)) / sample_rate
            waveform_plot = waveform / amax(abs(waveform)) * scale_factor + i_station
            ax.plot(timeax, waveform_plot, color=color, linewidth=linewidth, zorder=zorder)

            if hub and station in per_station_times_hub:
                starttime = per_station_times_hub[station][0]
                endtime = per_station_times_hub[station][1]
                starttime_window = (starttime - aligned_starttime_plot).total_seconds()
                endtime_window = (endtime - aligned_starttime_plot).total_seconds()
                ax.vlines(
                    x=starttime_window,
                    ymin=i_station - 0.2,
                    ymax=i_station + 0.2,
                    color="crimson",
                    linewidth=2 * linewidth,
                    zorder=3,
                )
                ax.vlines(
                    x=endtime_window,
                    ymin=i_station - 0.2,
                    ymax=i_station + 0.2,
                    color="crimson",
                    linewidth=2 * linewidth,
                    zorder=3,
                )

    for i_station, station in enumerate(stations):
        ax.text(0.0, i_station, station, fontsize=12, fontweight="bold", ha="left", va="bottom")

    ax.set_xlim(0.0, buffer_after_sec + buffer_before_sec)
    ax.set_yticks([])
    ax.set_yticklabels([])


def plot_waveform_stack_z(ax, stream_stack, stations, scale_factor):
    component = VERTICAL_COMPONENT
    color = get_geo_component_color(component)
    timeax = None

    for i_station, station in enumerate(stations):
        stream_sta = stream_stack.select(station=station, channel=component)
        if len(stream_sta) == 0:
            print(f"No stack trace for station {station}, component {component}")
            continue

        waveform = stream_sta[0].data
        timeax = stream_sta[0].times()
        waveform_plot = waveform / amax(abs(waveform)) * scale_factor + i_station
        ax.plot(timeax, waveform_plot, color=color, linewidth=2.0, zorder=1)
        ax.text(0.0, i_station, station, fontsize=12, fontweight="bold", ha="left", va="bottom")

    if timeax is not None:
        ax.set_xlim(timeax[0], timeax[-1])
    ax.set_yticks([])
    ax.set_yticklabels([])


def main():
    parser = ArgumentParser(
        description=(
            "Plot hub/satellite and stack waveforms for group 20, Subarray B"
        ),
    )
    parser.add_argument("--figwidth", type=float, default=14.0)
    parser.add_argument("--figheight", type=float, default=5.0)
    parser.add_argument("--min_cc", type=float, default=0.85)
    parser.add_argument("--min_num_similar_snippet", type=int, default=10)
    parser.add_argument("--min_num_similar_station", type=int, default=3)
    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--thr_on", type=float, default=4.0)
    parser.add_argument("--thr_off", type=float, default=1.0)
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    parser.add_argument("--buffer_before_sec", type=float, default=0.05)
    parser.add_argument("--buffer_after_sec", type=float, default=0.2)
    parser.add_argument("--scale_factor", type=float, default=0.7)
    args = parser.parse_args()

    figwidth = args.figwidth
    figheight = args.figheight
    min_cc = args.min_cc
    min_num_similar_snippet = args.min_num_similar_snippet
    min_num_similar_station = args.min_num_similar_station
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    thr_on = args.thr_on
    thr_off = args.thr_off
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    buffer_before_sec = args.buffer_before_sec
    buffer_after_sec = args.buffer_after_sec
    scale_factor = args.scale_factor

    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=min_cc, min_num_similar=min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

    waveform_filepath = join(dirpath_waveform, f"preprocessed_data_{suffix_freq}.h5")

    stations = get_stations_sorted(STATIONS)
    print(f"Processing group {GROUP_LABEL:d} (Subarray {SUBARRAY})...")
    alignment_df, per_station_times_hub, id_hub, num_events = load_group_data(
        GROUP_LABEL, suffix_group,
    )

    stack_filepath = Path(dirpath_event) / (
        f"hub_event_waveform_stack_group{GROUP_LABEL:d}_{suffix_group}.mseed"
    )
    stream_stack = read(stack_filepath)

    print("Generating the 1×2 figure...")
    fig, axes = subplots(
        1,
        2,
        figsize=(figwidth, figheight),
        gridspec_kw={"wspace": 0.1},
    )

    ax_events = axes[0]
    plot_hub_and_satellite_waveforms_z(
        ax_events,
        alignment_df,
        per_station_times_hub,
        stations,
        waveform_filepath,
        buffer_before_sec,
        buffer_after_sec,
        scale_factor,
    )
    ax_events.set_title(
        f"Group {GROUP_LABEL:d}, Subarray {SUBARRAY}, Hub Event {id_hub}, {num_events} events",
        fontsize=12,
        fontweight="bold",
    )
    ax_events.set_xlabel("Time (s)", fontsize=12)

    ax_stack = axes[1]
    plot_waveform_stack_z(ax_stack, stream_stack, stations, scale_factor)
    ax_stack.set_title("Waveform stack", fontsize=12, fontweight="bold")
    ax_stack.set_xlabel("Time (s)", fontsize=12)

    save_figure(fig, "fiber_optical_waveform_stack_subarray.png")


if __name__ == "__main__":
    main()
