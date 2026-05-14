"""
Plot the waveform stack and arrival time picks of a hub event
"""

from argparse import ArgumentParser
from pathlib import Path
from matplotlib.pyplot import subplots
from obspy import read

from utils_basic import LOC_DIR as dirpath_loc, DETECTION_DIR as dirpath_detection, PICK_DIR as dirpath_pick
from utils_basic import get_freq_limits_string
from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix
from utils_snuffler import read_time_windows
from utils_loc import process_arrival_info, load_hub_event_predicted_arrival_times_from_hdf
from utils_basic import GEO_COMPONENTS as components
from utils_plot import save_figure, get_geo_component_color

# Define the functions


#--------------------------------------------------------------------------------------------------
# Define the main function
#--------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="Plot the waveform stack and arrival time picks of a hub event")
    parser.add_argument("--group_label", type=int, required=True, help="Group label")

    parser.add_argument("--scale_factor", type=float, default=1.0, help="Scale factor")
    parser.add_argument("--weight", help="Weight the RMS by the uncertainties", action="store_true", default=True)
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
    group_label = args.group_label
    scale_factor = args.scale_factor
    weight = args.weight
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    thr_on = args.thr_on
    thr_off = args.thr_off
    min_cc = args.min_cc
    min_num_similar_snippet = args.min_num_similar_snippet
    min_num_similar_station = args.min_num_similar_station
    max_amp = args.max_amp

    # Constants
    linewidth_waveform = 2.0
    linewidth_marker = 2.0
    station_label_size = 12
    station_label_x = 0.005
    station_label_y = 0.98

    # Build the suffix
    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    suffix_repeating_snippet = get_repeating_snippet_suffix(min_cc, min_num_similar_snippet)
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

    # Load the event waveform stack
    print(f"Loading the event waveform stacks...")
    filename = f"hub_event_waveform_stack_group{group_label:d}_{suffix_group}.mseed"
    filepath = Path(dirpath_detection) / filename
    stream_stack = read(filepath)

    # Load the arrival time picks
    print(f"Loading the arrival time picks...")
    filename = f"hub_event_picks_group{group_label:d}_{suffix_group}.mkr"
    filepath = Path(dirpath_pick) / filename
    arrival_df = read_time_windows(filepath, phase_marker = False)
    arrival_df = process_arrival_info(arrival_df, "manual_stack")

    # Load the predicted arrival time picks
    filename =f"hub_event_location_info_group{group_label:d}.h5"
    filepath = Path(dirpath_loc) / filename
    arrival_time_dict, origin_time = load_hub_event_predicted_arrival_times_from_hdf(filepath, scale_factor)
    origin_time_abs = origin_time.timestamp()

    # Plot the waveform stack and arrival time picks
    print(f"Plotting the waveform stack and arrival time picks...")
    num_sta = len(arrival_df)
    fig, axes = subplots(num_sta, 1, figsize=(15, 15), sharex=True)
    for i_sta, station in enumerate(arrival_df["station"]):
        print(f"Plotting the waveform stack and arrival time picks for station {station}")
        stream_sta = stream_stack.select(station = station)
        starttime_trace = stream_sta[0].stats.starttime.timestamp

        for component in components:
            stream_comp = stream_sta.select(component = component)
            waveform_comp = stream_comp[0].data
            timeax = stream_comp[0].times()
            color = get_geo_component_color(component)
            axes[i_sta].plot(timeax, waveform_comp, color=color, linewidth=linewidth_waveform)

        arrival_time_abs = arrival_df.loc[arrival_df["station"] == station, "arrival_time"].values[0]
        print(f"Arrival time for station {station}: {arrival_time_abs}")
        arrival_time_uncer = arrival_df.loc[arrival_df["station"] == station, "uncertainty"].values[0]
        arrival_time_rel = arrival_time_abs - starttime_trace
        arrival_time_rel_min = arrival_time_rel - arrival_time_uncer
        arrival_time_rel_max = arrival_time_rel + arrival_time_uncer
        axes[i_sta].axvline(arrival_time_rel_min, color="crimson", linewidth=linewidth_marker, linestyle = "--")
        axes[i_sta].axvline(arrival_time_rel_max, color="crimson", linewidth=linewidth_marker, linestyle = "--")
        axes[i_sta].axvline(arrival_time_rel, color="crimson", linewidth=linewidth_marker)

        # Plot the origin time
        origin_time_rel = origin_time_abs - starttime_trace
        axes[i_sta].axvline(origin_time_rel, color="gray", linewidth=linewidth_marker)

        # Plot the predicted arrival time
        arrival_time_pred = arrival_time_dict[station].timestamp()
        arrival_time_pred_rel = arrival_time_pred - starttime_trace
        axes[i_sta].axvline(arrival_time_pred_rel, color="black", linewidth=linewidth_marker)

        # Label the station
        axes[i_sta].text(station_label_x, station_label_y, station, fontsize=station_label_size, fontweight="bold",
                         transform=axes[i_sta].transAxes, ha="left", va="top", zorder=4)

        # Set the axis limits
        axes[i_sta].set_xlim(timeax[0], timeax[-1])
        axes[i_sta].set_ylim(-max_amp, max_amp)

        # Set the axis labels
        if i_sta == num_sta - 1:
            axes[i_sta].set_xlabel("Time (s)", fontsize=12)

    # Set the suptitle
    title = f"Hub event waveform stack, Group {group_label}, Scale factor {scale_factor:.2f}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.9)

    # Save the figure
    save_figure(fig, f"hub_event_waveform_stack_and_arrival_time_picks_group{group_label:d}_scale{scale_factor:.2f}.png")