"""
Plot Z waveforms and continuous wavelet transforms for a hammer shot and the
hub event of an event group at one station.
"""

from argparse import ArgumentParser
from os.path import join
from pathlib import Path

from matplotlib.pyplot import subplots
from numpy import amax, abs as npabs, arange, linspace
from pandas import Timedelta, read_csv, to_datetime

from utils_basic import (
    DETECTION_DIR as dirpath_event,
    GEO_COMPONENTS as components,
    PICK_DIR as dirpath_pick,
    SAMPLING_RATE as sample_rate,
    get_freq_limits_string,
    power2db,
)
from utils_cc import get_repeating_snippet_suffix
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_plot import (
    format_freq_ylabels,
    get_geo_component_color,
    save_figure,
)
from utils_snuffler import read_time_windows
from utils_sta_lta import get_sta_lta_suffix
from utils_wavelet import get_trace_cwt

VERTICAL_COMPONENT = "Z"
HAMMER_ID = "100129"
DEFAULT_STATION = "A04"

LINEWIDTH_WAVEFORM = 2.0


def hammer_pick_filepath(hammer_id):
    return Path(dirpath_pick) / f"hammer_{hammer_id}.mkr"


def load_hub_alignment(group_label, suffix_group):
    """Return hub event id and aligned_time for a group."""
    filename = f"event_alignments_group{group_label:d}_{suffix_group}.csv"
    alignment_df = read_csv(join(dirpath_event, filename))
    alignment_df["aligned_time"] = to_datetime(alignment_df["aligned_time"], format="ISO8601")
    alignment_df["hub"] = alignment_df["hub"].astype(bool)
    hub_row = alignment_df.loc[alignment_df["hub"]].iloc[0]
    return int(hub_row["id"]), hub_row["aligned_time"]


def load_hammer_arrival_time(pick_filepath, station):
    """Return the manual P-wave arrival time (pick window center) for the given station."""
    arrival_df = read_time_windows(pick_filepath, phase_marker=True)
    station_df = arrival_df[arrival_df["station"] == station]
    if station_df.empty:
        raise ValueError(f"No pick for station {station} in {pick_filepath}")
    row = station_df.iloc[0]
    return row["starttime"] + (row["endtime"] - row["starttime"]) / 2


def load_station_stream_at_arrival(arrival_time, station, buffer_before_sec, buffer_after_sec):
    """Load unfiltered 3-component waveforms around arrival_time; return the ObsPy stream."""
    starttime_plot = arrival_time - Timedelta(seconds=buffer_before_sec)
    endtime_plot = arrival_time + Timedelta(seconds=buffer_after_sec)
    stream = read_and_process_windowed_geo_waveforms(
        starttime_plot,
        endtime=endtime_plot,
        stations=[station],
        components=components,
        filter=False,
        normalize=True,
    )
    if stream is None or len(stream) == 0:
        raise ValueError(f"No waveform data for station {station}.")
    return stream


def select_z_trace(stream, station):
    """Return the Z-component trace for a station."""
    stream_z = stream.select(station=station, component=VERTICAL_COMPONENT)
    if len(stream_z) == 0:
        raise ValueError(f"No {VERTICAL_COMPONENT} trace for station {station}.")
    return stream_z[0]


def compute_cwt_power_db(trace, wavelet, bandwidth, center_freq, scales):
    """Compute CWT power in dB relative to the peak power."""
    spec = get_trace_cwt(
        trace,
        wavelet=wavelet,
        bandwidth=bandwidth,
        center_freq=center_freq,
        scales=scales,
    )
    power = spec.get_power(db=False)
    power_db = power2db(power, reference_type="max")
    timeax = arange(power_db.shape[1]) / sample_rate
    return timeax, spec.freqs, power_db


def plot_z_waveform(ax, waveform, window_sec):
    """Plot normalized Z component vs time (seconds from window start)."""
    timeax = arange(len(waveform)) / sample_rate
    if amax(npabs(waveform)) > 0:
        waveform_plot = waveform / amax(npabs(waveform))
    else:
        waveform_plot = waveform
    color = get_geo_component_color(VERTICAL_COMPONENT)
    ax.plot(timeax, waveform_plot, color=color, linewidth=LINEWIDTH_WAVEFORM)
    ax.set_xlim(0.0, window_sec)
    ax.set_ylim(-1.05, 1.05)


def plot_cwt_power(ax, timeax, freqs, power_db, window_sec, min_freq, max_freq, min_db, max_db):
    """Plot CWT power scalogram."""
    quadmesh = ax.pcolormesh(
        timeax,
        freqs,
        power_db,
        shading="auto",
        cmap="inferno",
        vmin=min_db,
        vmax=max_db,
    )
    ax.set_xlim(0.0, window_sec)
    ax.set_ylim(min_freq, max_freq)
    return quadmesh


def main():
    parser = ArgumentParser(
        description=(
            "Plot Z waveforms and CWTs for a hammer shot and the hub event "
            "of an event group at one station"
        ),
    )
    parser.add_argument("--group_label", type=int, required=True, help="Event group label")
    parser.add_argument("--station", type=str, default=DEFAULT_STATION, help="Station to plot")
    parser.add_argument("--hammer_id", type=str, default=HAMMER_ID, help="Hammer shot ID")
    parser.add_argument(
        "--hammer_pick_file",
        type=str,
        default=None,
        help="Snuffler pick file for the hammer shot (default: PICK_DIR/hammer_{hammer_id}.mkr)",
    )
    parser.add_argument("--figwidth", type=float, default=12.0)
    parser.add_argument("--figheight", type=float, default=8.0)
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
    parser.add_argument("--wavelet", type=str, default="cmor", help="PyWavelets wavelet family")
    parser.add_argument("--bandwidth", type=float, default=30.0, help="CWT bandwidth parameter")
    parser.add_argument("--center_freq", type=float, default=1.0, help="CWT center frequency")
    parser.add_argument("--min_scale", type=float, default=1.0)
    parser.add_argument("--max_scale", type=float, default=100.0)
    parser.add_argument("--num_scales", type=int, default=401)
    parser.add_argument("--min_freq_plot", type=float, default=10.0)
    parser.add_argument(
        "--max_freq_plot",
        type=float,
        default=200.0,
        help="Maximum frequency for CWT panels",
    )
    parser.add_argument("--min_db", type=float, default=-40.0, help="CWT color-scale minimum (dB)")
    parser.add_argument("--max_db", type=float, default=0.0, help="CWT color-scale maximum (dB)")
    args = parser.parse_args()

    hammer_pick_file = (
        Path(args.hammer_pick_file)
        if args.hammer_pick_file is not None
        else hammer_pick_filepath(args.hammer_id)
    )

    suffix_freq = get_freq_limits_string(args.min_freq_filter, args.max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(
        args.sta_window_sec, args.lta_window_sec, args.thr_on, args.thr_off,
    )
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=args.min_cc, min_num_similar=args.min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{args.min_num_similar_station:d}"

    window_sec = args.buffer_before_sec + args.buffer_after_sec
    scales = linspace(args.min_scale, args.max_scale, args.num_scales)

    print(f"Loading hammer shot {args.hammer_id} waveforms at {args.station}...")
    hammer_arrival_time = load_hammer_arrival_time(hammer_pick_file, args.station)
    hammer_stream = load_station_stream_at_arrival(
        hammer_arrival_time,
        args.station,
        args.buffer_before_sec,
        args.buffer_after_sec,
    )
    hammer_trace = select_z_trace(hammer_stream, args.station)

    print(f"Loading hub event for group {args.group_label:d} at {args.station}...")
    id_hub, hub_aligned_time = load_hub_alignment(args.group_label, suffix_group)
    print(f"Hub event ID: {id_hub}")
    hub_stream = load_station_stream_at_arrival(
        hub_aligned_time,
        args.station,
        args.buffer_before_sec,
        args.buffer_after_sec,
    )
    hub_trace = select_z_trace(hub_stream, args.station)

    print("Computing CWTs...")
    hammer_timeax, hammer_freqs, hammer_power_db = compute_cwt_power_db(
        hammer_trace, args.wavelet, args.bandwidth, args.center_freq, scales,
    )
    hub_timeax, hub_freqs, hub_power_db = compute_cwt_power_db(
        hub_trace, args.wavelet, args.bandwidth, args.center_freq, scales,
    )

    print("Generating the 2x2 figure...")
    fig, axes = subplots(
        2,
        2,
        figsize=(args.figwidth, args.figheight),
        sharex="col",
        gridspec_kw={"hspace": 0.25, "wspace": 0.2},
    )

    # Row 0: waveforms (hammer | hub)
    plot_z_waveform(axes[0, 0], hammer_trace.data, window_sec)
    axes[0, 0].set_title(
        f"Hammer {args.hammer_id}, {args.station}",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 0].set_ylabel("Normalized amplitude", fontsize=12)

    plot_z_waveform(axes[0, 1], hub_trace.data, window_sec)
    axes[0, 1].set_title(
        f"Group {args.group_label:d}, Hub {id_hub}, {args.station}",
        fontsize=12,
        fontweight="bold",
    )

    # Row 1: CWTs (hammer | hub)
    quadmesh = plot_cwt_power(
        axes[1, 0],
        hammer_timeax,
        hammer_freqs,
        hammer_power_db,
        window_sec,
        args.min_freq_plot,
        args.max_freq_plot,
        args.min_db,
        args.max_db,
    )
    axes[1, 0].set_xlabel("Time (s)", fontsize=12)
    format_freq_ylabels(axes[1, 0], major_tick_spacing=50.0)

    plot_cwt_power(
        axes[1, 1],
        hub_timeax,
        hub_freqs,
        hub_power_db,
        window_sec,
        args.min_freq_plot,
        args.max_freq_plot,
        args.min_db,
        args.max_db,
    )
    axes[1, 1].set_xlabel("Time (s)", fontsize=12)
    format_freq_ylabels(
        axes[1, 1],
        plot_axis_label=False,
        major_tick_spacing=50.0,
    )

    cax = fig.add_axes([0.55, 0.1, 0.3, 0.02])
    cbar = fig.colorbar(quadmesh, cax=cax, orientation="horizontal")
    cbar.set_label("Power (dB)", fontsize=11)

    fig.suptitle(
        f"Hammer vs hub-event waveforms and CWT ({args.station})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(bottom=0.12, top=0.90)

    figname = (
        f"hammer_n_event_cwt_hammer{args.hammer_id}_"
        f"group{args.group_label:d}_{args.station}.png"
    )
    print(f"Saving figure to {figname}...")
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
