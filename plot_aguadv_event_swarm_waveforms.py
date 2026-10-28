"""
Plot the full swarm time window for Group 235 (AGU Advances, Subarray A).

Vertical-component waveforms at inner and middle stations are shown without
normalization so amplitude variation across the array is visible. Only the
start time of each per-station pick is marked with solid darkorange circles
for hub events and 80%-transparent darkorange circles for satellite events.
"""

from argparse import ArgumentParser
from os.path import join

from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from pandas import Timedelta, Timestamp, read_csv, read_json, to_datetime

from utils_basic import (
    DETECTION_DIR as dirpath_event,
    INNER_STATIONS_A as stations_inner_a,
    MIDDLE_STATIONS_A as stations_middle_a,
    ROOTDIR_GEO as dirpath_waveform,
    get_freq_limits_string,
    get_geophone_coords,
)
from utils_cc import get_repeating_snippet_suffix
from utils_cont_waveform import load_waveform_slice
from utils_plot import (
    format_datetime_xlabels,
    get_geo_component_color,
    save_figure,
)
from utils_sta_lta import get_sta_lta_suffix

GROUP_LABEL = 235
VERTICAL_COMPONENT = "Z"
SUBARRAY_A_STATIONS = stations_inner_a + stations_middle_a
COLOR_EVENT = "darkorange"
SATELLITE_MARKER_ALPHA = 0.5
SECONDS_PER_DAY = 86400.0
SCALE_BAR_LENGTH_SEC = 5.0


def add_time_scalebar(ax, length_sec, fontsize=10, size_vertical=0.05):
    scale_bar = AnchoredSizeBar(
        ax.transData,
        length_sec / SECONDS_PER_DAY,
        f"{length_sec:.0f} s",
        loc="lower left",
        pad=0.4,
        frameon=True,
        size_vertical=size_vertical,
        fontproperties=FontProperties(size=fontsize),
    )
    ax.add_artist(scale_bar)


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
    num_events = len(alignment_df)

    return event_df, alignment_df, id_hub, num_events


def compute_swarm_time_window(event_df, buffer_before_sec, buffer_after_sec):
    all_times = []
    for _, row in event_df.iterrows():
        for times in row["per_station_times"].values():
            all_times.extend(times)

    starttime_plot = min(all_times) - Timedelta(seconds=buffer_before_sec)
    endtime_plot = max(all_times) + Timedelta(seconds=buffer_after_sec)
    return starttime_plot, endtime_plot


def plot_event_swarm_waveforms_z(
    ax,
    stations,
    waveform_filepath,
    starttime_plot,
    endtime_plot,
    event_df,
    alignment_df,
    scale,
    linewidth,
    marker_size,
):
    component = VERTICAL_COMPONENT
    color = get_geo_component_color(component)
    station_to_index = {station: i for i, station in enumerate(stations)}

    for i_station, station in enumerate(stations):
        try:
            waveform_dict, time_array = load_waveform_slice(
                waveform_filepath,
                station,
                starttime_plot,
                endtime=endtime_plot,
                normalize=False,
            )
        except Exception as e:
            print(f"Error loading waveform slice for station {station}: {e}")
            continue

        waveform = waveform_dict[component]
        ax.plot(
            time_array,
            waveform * scale + i_station,
            color=color,
            linewidth=linewidth,
            zorder=1,
        )
        ax.text(
            time_array[0],
            i_station,
            station,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    aligned_events = event_df.merge(
        alignment_df,
        left_on="event_id",
        right_on="id",
    )

    for _, row in aligned_events.iterrows():
        hub = row["hub"]
        zorder = 3 if hub else 2

        for station, times in row["per_station_times"].items():
            if station not in station_to_index:
                continue

            i_station = station_to_index[station]
            pick_time = times[0]
            if hub:
                ax.scatter(
                    pick_time,
                    i_station,
                    marker="o",
                    facecolor=COLOR_EVENT,
                    edgecolor=COLOR_EVENT,
                    s=marker_size,
                    zorder=zorder,
                )
            else:
                ax.scatter(
                    pick_time,
                    i_station,
                    marker="o",
                    facecolor=COLOR_EVENT,
                    edgecolor=COLOR_EVENT,
                    alpha=SATELLITE_MARKER_ALPHA,
                    s=marker_size,
                    zorder=zorder,
                )

    ax.set_xlim(starttime_plot, endtime_plot)
    ax.set_ylim(-1.0, len(stations))
    ax.set_yticks([])
    ax.set_yticklabels([])
    format_datetime_xlabels(
        ax,
        major_tick_spacing="20s",
        num_minor_ticks=4,
        rotation=0,
        va="top",
        ha="center",
    )
    add_time_scalebar(ax, SCALE_BAR_LENGTH_SEC)


def main():
    parser = ArgumentParser(
        description="Plot the full swarm waveforms for Group 235 (AGU Advances, Subarray A)",
    )
    parser.add_argument("--group_label", type=int, default=GROUP_LABEL)
    parser.add_argument("--figwidth", type=float, default=15.0)
    parser.add_argument("--figheight", type=float, default=10.0)
    parser.add_argument("--min_cc", type=float, default=0.85)
    parser.add_argument("--min_num_similar_snippet", type=int, default=10)
    parser.add_argument("--min_num_similar_station", type=int, default=3)
    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--thr_on", type=float, default=4.0)
    parser.add_argument("--thr_off", type=float, default=1.0)
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    parser.add_argument(
        "--starttime_plot", type=Timestamp, default=None,
        help="Start time of the plot window (UTC). If omitted, inferred from event picks.",
    )
    parser.add_argument(
        "--endtime_plot", type=Timestamp, default=None,
        help="End time of the plot window (UTC). If omitted, inferred from event picks.",
    )
    parser.add_argument("--buffer_before_sec", type=float, default=0.05)
    parser.add_argument("--buffer_after_sec", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--linewidth", type=float, default=1.0)
    parser.add_argument("--marker_size", type=float, default=20.0)
    args = parser.parse_args()

    group_label = args.group_label
    suffix_freq = get_freq_limits_string(args.min_freq_filter, args.max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(
        args.sta_window_sec, args.lta_window_sec, args.thr_on, args.thr_off,
    )
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=args.min_cc, min_num_similar=args.min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{args.min_num_similar_station:d}"

    print(f"Loading group {group_label:d} event data...")
    event_df, alignment_df, id_hub, num_events = load_group_data(group_label, suffix_group)

    if args.starttime_plot is None and args.endtime_plot is None:
        starttime_plot, endtime_plot = compute_swarm_time_window(
            event_df, args.buffer_before_sec, args.buffer_after_sec,
        )
    elif args.starttime_plot is not None and args.endtime_plot is not None:
        starttime_plot = args.starttime_plot
        endtime_plot = args.endtime_plot
    else:
        raise ValueError("Both --starttime_plot and --endtime_plot must be provided together.")

    stations = get_stations_sorted(SUBARRAY_A_STATIONS)
    waveform_filepath = join(dirpath_waveform, f"preprocessed_data_{suffix_freq}.h5")

    print(f"Plotting swarm window {starttime_plot} to {endtime_plot}...")
    fig, ax = subplots(figsize=(args.figwidth, args.figheight))
    plot_event_swarm_waveforms_z(
        ax,
        stations,
        waveform_filepath,
        starttime_plot,
        endtime_plot,
        event_df,
        alignment_df,
        args.scale,
        args.linewidth,
        args.marker_size,
    )

    ax.set_title(
        f"Group {group_label:d}, Subarray A, Hub Event {id_hub}, {num_events} events",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Time (UTC)", fontsize=12)

    legend_handles = [
        Line2D(
            [], [], linestyle="none", marker="o", markersize=6,
            markerfacecolor=COLOR_EVENT, markeredgecolor=COLOR_EVENT,
            label="Hub event",
        ),
        Line2D(
            [], [], linestyle="none", marker="o", markersize=6,
            markerfacecolor=COLOR_EVENT, markeredgecolor=COLOR_EVENT,
            alpha=SATELLITE_MARKER_ALPHA,
            label="Satellite event",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=1.0)

    figname = f"aguadv_event_swarm_waveforms_group{group_label:d}.png"
    save_figure(fig, figname)
    print(f"Saved figure to {figname}")


if __name__ == "__main__":
    main()
