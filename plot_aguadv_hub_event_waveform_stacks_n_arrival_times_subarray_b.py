"""
Plot hub-event waveform stack and arrival-time picks for group 20 (Subarray B) (AGU Advances).

Based on plot_aguadv_hub_event_waveform_stacks_n_arrival_times_subarray_a.py and
plot_hub_event_waveform_stack_and_arrival_time_picks.py.
"""

from argparse import ArgumentParser
from pathlib import Path

from matplotlib.pyplot import subplots
from obspy import read

from utils_basic import (
    DETECTION_DIR as dirpath_detection,
    INNER_STATIONS_B as stations_inner_b,
    LOC_DIR as dirpath_loc,
    MIDDLE_STATIONS_B as stations_middle_b,
    PICK_DIR as dirpath_pick,
    GEO_COMPONENTS as components,
    get_freq_limits_string,
    get_geophone_coords,
)
from utils_cc import get_repeating_snippet_suffix
from utils_loc import (
    load_hub_event_predicted_arrival_times_from_hdf,
    process_arrival_info,
)
from utils_plot import get_geo_component_color, save_figure
from utils_snuffler import read_time_windows
from utils_sta_lta import get_sta_lta_suffix

GROUP_LABEL = 20
SUBARRAY_B_STATIONS = stations_inner_b + stations_middle_b

linewidth_waveform = 2.0
linewidth_marker = 2.0
STATION_LABEL_SIZE = 12
STATION_LABEL_X = 0.005
STATION_LABEL_Y = 0.98


def get_stations_sorted(station_list):
    coord_df = get_geophone_coords()
    coord_df = coord_df.sort_values(by="north")
    coord_df = coord_df[coord_df.index.isin(station_list)]
    return coord_df.index.tolist()


def load_group_stack_and_arrivals(group_label, suffix_group):
    filename = f"hub_event_waveform_stack_group{group_label:d}_{suffix_group}.mseed"
    stream_stack = read(Path(dirpath_detection) / filename)

    filename = f"hub_event_picks_group{group_label:d}_{suffix_group}.mkr"
    arrival_df = read_time_windows(Path(dirpath_pick) / filename, phase_marker=False)
    arrival_df = process_arrival_info(arrival_df, "manual_stack")
    return stream_stack, arrival_df


def load_predicted_arrivals(group_label, scale_factor):
    filename = f"hub_event_location_info_group{group_label:d}.h5"
    return load_hub_event_predicted_arrival_times_from_hdf(
        Path(dirpath_loc) / filename, scale_factor,
    )


def plot_station_waveform_stack_and_arrivals(
    ax,
    station,
    stream_stack,
    arrival_df,
    arrival_time_dict,
    origin_time,
    max_amp,
):
    stream_sta = stream_stack.select(station=station)
    if len(stream_sta) == 0:
        ax.set_visible(False)
        return

    starttime_trace = stream_sta[0].stats.starttime.timestamp

    for component in components:
        stream_comp = stream_sta.select(component=component)
        if len(stream_comp) == 0:
            continue
        waveform_comp = stream_comp[0].data
        timeax = stream_comp[0].times()
        color = get_geo_component_color(component)
        ax.plot(timeax, waveform_comp, color=color, linewidth=linewidth_waveform)

    arrival_time_abs = arrival_df.loc[arrival_df["station"] == station, "arrival_time"].values[0]
    arrival_time_uncer = arrival_df.loc[arrival_df["station"] == station, "uncertainty"].values[0]
    arrival_time_rel = arrival_time_abs - starttime_trace
    arrival_time_rel_min = arrival_time_rel - arrival_time_uncer
    arrival_time_rel_max = arrival_time_rel + arrival_time_uncer
    ax.axvline(arrival_time_rel_min, color="crimson", linewidth=linewidth_marker, linestyle="--")
    ax.axvline(arrival_time_rel_max, color="crimson", linewidth=linewidth_marker, linestyle="--")
    ax.axvline(arrival_time_rel, color="crimson", linewidth=linewidth_marker)

    origin_time_rel = origin_time.timestamp() - starttime_trace
    ax.axvline(origin_time_rel, color="gray", linewidth=linewidth_marker)

    arrival_time_pred_rel = arrival_time_dict[station].timestamp() - starttime_trace
    ax.axvline(arrival_time_pred_rel, color="black", linewidth=linewidth_marker)

    ax.text(
        STATION_LABEL_X,
        STATION_LABEL_Y,
        station,
        fontsize=STATION_LABEL_SIZE,
        fontweight="bold",
        transform=ax.transAxes,
        ha="left",
        va="top",
        zorder=4,
    )

    ax.set_xlim(timeax[0], timeax[-1])
    ax.set_ylim(-max_amp, max_amp)


def main():
    parser = ArgumentParser(
        description=(
            "Plot hub-event waveform stack and arrival-time picks for group 20 "
            "(Subarray B) (AGU Advances)"
        ),
    )
    parser.add_argument("--figwidth", type=float, default=14.0)
    parser.add_argument("--figheight", type=float, default=15.0)
    parser.add_argument(
        "--scale_factor_prefer",
        type=float,
        default=1.1,
        help="Preferred velocity scale factor for origin and predicted arrivals",
    )
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--thr_on", type=float, default=4.0)
    parser.add_argument("--thr_off", type=float, default=1.0)
    parser.add_argument("--min_cc", type=float, default=0.85)
    parser.add_argument("--min_num_similar_snippet", type=int, default=10)
    parser.add_argument("--min_num_similar_station", type=int, default=3)
    parser.add_argument("--max_amp", type=float, default=0.2, help="Maximum amplitude")
    args = parser.parse_args()

    scale_factor_prefer = args.scale_factor_prefer
    suffix_freq = get_freq_limits_string(args.min_freq_filter, args.max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(
        args.sta_window_sec, args.lta_window_sec, args.thr_on, args.thr_off,
    )
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=args.min_cc, min_num_similar=args.min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{args.min_num_similar_station:d}"

    stations = get_stations_sorted(SUBARRAY_B_STATIONS)
    num_sta = len(stations)

    print("Loading waveform stack, arrival picks, and predicted times...")
    stream_stack, arrival_df = load_group_stack_and_arrivals(GROUP_LABEL, suffix_group)
    arrival_time_dict, origin_time = load_predicted_arrivals(GROUP_LABEL, scale_factor_prefer)

    print("Plotting waveform stack and arrival times...")
    fig, axes = subplots(
        num_sta,
        1,
        figsize=(args.figwidth, args.figheight),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.15},
    )
    if num_sta == 1:
        axes = [axes]

    axes[0].set_title(f"Group {GROUP_LABEL:d}", fontsize=14, fontweight="bold")

    for i_sta, station in enumerate(stations):
        if station not in arrival_df["station"].values:
            axes[i_sta].set_visible(False)
            continue
        if station not in arrival_time_dict:
            axes[i_sta].set_visible(False)
            continue

        print(f"Station {station}")
        plot_station_waveform_stack_and_arrivals(
            axes[i_sta],
            station,
            stream_stack,
            arrival_df,
            arrival_time_dict,
            origin_time,
            args.max_amp,
        )

        if i_sta == num_sta - 1:
            axes[i_sta].set_xlabel("Time (s)", fontsize=12)

    fig.suptitle(
        f"Hub event waveform stack, Subarray B",
        fontsize=16,
        fontweight="bold",
        y=0.93,
    )

    save_figure(fig, "aguadv_hub_event_waveform_stacks_n_arrival_times_subarray_b.png")


if __name__ == "__main__":
    main()
