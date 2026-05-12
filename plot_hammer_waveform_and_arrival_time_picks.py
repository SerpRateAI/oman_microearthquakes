"""
Plot the waveforms and arrival time picks of a hammer shot
"""

from argparse import ArgumentParser
from pathlib import Path
from pandas import Timedelta, Timestamp
from numpy import amax
from matplotlib.pyplot import subplots

from utils_basic import LOC_DIR as dirpath_loc, PICK_DIR as dirpath_pick
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_snuffler import read_time_windows
from utils_loc import process_arrival_info, load_hammer_predicted_arrival_times_from_hdf
from utils_basic import GEO_COMPONENTS as components
from utils_plot import save_figure, get_geo_component_color

# Define the functions


#--------------------------------------------------------------------------------------------------
# Define the main function
#--------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="Plot the waveforms and arrival time picks of a hammer shot")

    parser.add_argument("--hammer_id", type=str, required=True, help="Hammer ID")
    parser.add_argument("--weight", help="Weight the RMS by the uncertainties", action="store_true", default=True)
    parser.add_argument("--buffer_begin", type=float, default=0.05, help="Buffer begin in seconds")
    parser.add_argument("--buffer_end", type=float, default=0.2, help="Buffer end in seconds")
    parser.add_argument("--max_amp", type=float, default=0.2, help="Maximum amplitude")

    args = parser.parse_args()
    hammer_id = args.hammer_id
    weight = args.weight
    buffer_begin = args.buffer_begin
    buffer_end = args.buffer_end
    max_amp = args.max_amp

    # Constants
    linewidth_waveform = 2.0
    linewidth_marker = 2.0
    station_label_size = 12
    station_label_x = 0.005
    station_label_y = 0.98

    # Load the arrival time picks
    print(f"Loading the arrival time picks...")
    filename = f"hammer_{hammer_id}.mkr"
    filepath = Path(dirpath_pick) / filename
    arrival_df = read_time_windows(filepath, phase_marker = True)
    arrival_df = process_arrival_info(arrival_df, "manual_stack")

    # Load the predicted arrival time picks
    print(f"Loading the predicted arrival time picks...")
    filename = f"hammer_location_info_{hammer_id}.h5"
    filepath = Path(dirpath_loc) / filename
    arrival_time_dict, origin_time = load_hammer_predicted_arrival_times_from_hdf(filepath)
    origin_time_abs = origin_time.timestamp()
    print(f"Origin time: {origin_time_abs}")

    # Plot the waveforms and arrival time picks
    print(f"Loading the hammer waveform...")
    first_arrival_time = arrival_df["arrival_time"].min()
    starttime_waveform = Timestamp(first_arrival_time, unit="s") - Timedelta(seconds=buffer_begin)
    endtime_waveform = Timestamp(first_arrival_time, unit="s") + Timedelta(seconds=buffer_end)
    stations = arrival_df["station"].values.tolist()
    stream = read_and_process_windowed_geo_waveforms(starttime_waveform, endtime = endtime_waveform, stations = stations)

    num_sta = len(arrival_df)
    fig, axes = subplots(num_sta, 1, figsize=(15, 15), sharex=True)
    for i_sta, station in enumerate(stations):
        stream_sta = stream.select(station = station)
        starttime_trace = stream_sta[0].stats.starttime.timestamp

        for component in components:
            stream_comp = stream_sta.select(component = component)
            waveform_comp = stream_comp[0].data
            waveform_comp = waveform_comp / amax(abs(waveform_comp)) 

            timeax = stream_comp[0].times()
            color = get_geo_component_color(component)
            axes[i_sta].plot(timeax, waveform_comp, color=color, linewidth=linewidth_waveform)

        # Plot the arrival time picks
        arrival_time_abs = arrival_df.loc[arrival_df["station"] == station, "arrival_time"].values[0]
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
title = f"Hammer shot {hammer_id}"
fig.suptitle(title, fontsize=16, fontweight="bold", y=0.9)

# Save the figure
save_figure(fig, f"hammer_waveform_and_arrival_time_picks_{hammer_id}.png")