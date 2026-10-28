"""
Plot Z waveforms and continuous wavelet transforms of Hammer 100129 and the
hub events of groups 18, 19, and 20 in a 4×2 figure (AGU Advances).

Rows 0–1: hammer (A04) and group 18 (A04). Rows 2–3: group 19 (A04) and group 20 (B04).
"""

from argparse import ArgumentParser
from pathlib import Path

from matplotlib.pyplot import subplots
from numpy import linspace

from utils_basic import get_freq_limits_string
from utils_cc import get_repeating_snippet_suffix
from utils_plot import (
    format_freq_ylabels,
    format_norm_amp_ylabels,
    save_figure,
)
from utils_sta_lta import get_sta_lta_suffix

from plot_hammer_n_event_cwt import (
    HAMMER_ID,
    compute_cwt_power_db,
    hammer_pick_filepath,
    load_hammer_arrival_time,
    load_hub_alignment,
    load_station_stream_at_arrival,
    plot_cwt_power,
    plot_z_waveform,
    select_z_trace,
)

HAMMER_STATION = "A04"
GROUP_PANELS = [
    {"label": 18, "station": "A04"},
    {"label": 19, "station": "A04"},
    {"label": 20, "station": "B04"},
]


def load_hub_z_trace(group_label, station, suffix_group, buffer_before_sec, buffer_after_sec):
    """Return (hub event id, Z trace) for a group at one station."""
    id_hub, hub_aligned_time = load_hub_alignment(group_label, suffix_group)
    stream = load_station_stream_at_arrival(
        hub_aligned_time,
        station,
        buffer_before_sec,
        buffer_after_sec,
    )
    return id_hub, select_z_trace(stream, station)


def add_panel_letter(ax, letter):
    ax.text(
        -0.1,
        1.07,
        f"({letter})",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )


def format_waveform_axes(ax_left, ax_right):
    format_norm_amp_ylabels(ax_left, major_tick_spacing=0.5)
    format_norm_amp_ylabels(ax_right, plot_axis_label=False, major_tick_spacing=0.5)
    ax_left.tick_params(axis="x", labelbottom=False)
    ax_right.tick_params(axis="x", labelbottom=False)


def format_cwt_axes(ax_left, ax_right):
    format_freq_ylabels(ax_left, major_tick_spacing=50.0)
    format_freq_ylabels(ax_right, plot_axis_label=False, major_tick_spacing=50.0)
    ax_left.set_xlabel("Time (s)", fontsize=12)
    ax_right.set_xlabel("Time (s)", fontsize=12)
    ax_left.tick_params(axis="x", labelbottom=True)
    ax_right.tick_params(axis="x", labelbottom=True)


def plot_lower_freq_limit(ax, freq_hz):
    """Mark a spectrum lower-frequency limit on a CWT panel."""
    ax.axhline(
        freq_hz,
        color="white",
        linestyle=":",
        linewidth=2.0,
        zorder=3,
    )


def main():
    parser = ArgumentParser(
        description=(
            "Plot Z waveforms and CWTs of Hammer 100129 and hub events of "
            "groups 18, 19, and 20 (AGU Advances)"
        ),
    )
    parser.add_argument("--hammer_id", type=str, default=HAMMER_ID, help="Hammer shot ID")
    parser.add_argument(
        "--hammer_pick_file",
        type=str,
        default=None,
        help="Snuffler pick file for the hammer shot (default: PICK_DIR/hammer_{hammer_id}.mkr)",
    )
    parser.add_argument("--figwidth", type=float, default=12.0)
    parser.add_argument("--figheight", type=float, default=14.0)
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
    hammer_id = args.hammer_id
    hammer_pick_file_arg = args.hammer_pick_file
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
    wavelet = args.wavelet
    bandwidth = args.bandwidth
    center_freq = args.center_freq
    min_scale = args.min_scale
    max_scale = args.max_scale
    num_scales = args.num_scales
    min_freq_plot = args.min_freq_plot
    max_freq_plot = args.max_freq_plot
    min_db = args.min_db
    max_db = args.max_db

    hammer_pick_file = (
        Path(hammer_pick_file_arg)
        if hammer_pick_file_arg is not None
        else hammer_pick_filepath(hammer_id)
    )

    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(
        sta_window_sec, lta_window_sec, thr_on, thr_off,
    )
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=min_cc, min_num_similar=min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

    window_sec = buffer_before_sec + buffer_after_sec
    scales = linspace(min_scale, max_scale, num_scales)

    print(f"Loading hammer shot {hammer_id} waveforms at {HAMMER_STATION}...")
    hammer_arrival_time = load_hammer_arrival_time(hammer_pick_file, HAMMER_STATION)
    hammer_stream = load_station_stream_at_arrival(
        hammer_arrival_time,
        HAMMER_STATION,
        buffer_before_sec,
        buffer_after_sec,
    )
    hammer_trace = select_z_trace(hammer_stream, HAMMER_STATION)

    hub_traces = {}
    for group_panel in GROUP_PANELS:
        group_label = group_panel["label"]
        station = group_panel["station"]
        print(f"Loading hub event for group {group_label:d} at {station}...")
        id_hub, hub_trace = load_hub_z_trace(
            group_label,
            station,
            suffix_group,
            buffer_before_sec,
            buffer_after_sec,
        )
        print(f"Hub event ID: {id_hub}")
        hub_traces[group_label] = hub_trace

    print("Computing CWTs...")
    hammer_timeax, hammer_freqs, hammer_power_db = compute_cwt_power_db(
        hammer_trace, wavelet, bandwidth, center_freq, scales,
    )
    hub_cwts = {}
    for group_label, hub_trace in hub_traces.items():
        hub_cwts[group_label] = compute_cwt_power_db(
            hub_trace, wavelet, bandwidth, center_freq, scales,
        )

    print("Generating the 4x2 figure...")
    fig, axes = subplots(
        4,
        2,
        figsize=(figwidth, figheight),
        sharex="col",
        gridspec_kw={"hspace": 0.30, "wspace": 0.2},
    )

    plot_z_waveform(axes[0, 0], hammer_trace.data, window_sec)
    axes[0, 0].set_title(
        f"Hammer {hammer_id}, {HAMMER_STATION}",
        fontsize=12,
        fontweight="bold",
    )
    plot_z_waveform(axes[0, 1], hub_traces[18].data, window_sec)
    axes[0, 1].set_title(
        f"Group 18, {GROUP_PANELS[0]['station']}",
        fontsize=12,
        fontweight="bold",
    )
    add_panel_letter(axes[0, 0], "a")
    add_panel_letter(axes[0, 1], "b")
    format_waveform_axes(axes[0, 0], axes[0, 1])

    quadmesh = plot_cwt_power(
        axes[1, 0],
        hammer_timeax,
        hammer_freqs,
        hammer_power_db,
        window_sec,
        min_freq_plot,
        max_freq_plot,
        min_db,
        max_db,
    )
    timeax_18, freqs_18, power_db_18 = hub_cwts[18]
    plot_cwt_power(
        axes[1, 1],
        timeax_18,
        freqs_18,
        power_db_18,
        window_sec,
        min_freq_plot,
        max_freq_plot,
        min_db,
        max_db,
    )
    plot_lower_freq_limit(axes[1, 1], 40.0)
    format_cwt_axes(axes[1, 0], axes[1, 1])

    plot_z_waveform(axes[2, 0], hub_traces[19].data, window_sec)
    axes[2, 0].set_title(
        f"Group 19, {GROUP_PANELS[1]['station']}",
        fontsize=12,
        fontweight="bold",
    )
    plot_z_waveform(axes[2, 1], hub_traces[20].data, window_sec)
    axes[2, 1].set_title(
        f"Group 20, {GROUP_PANELS[2]['station']}",
        fontsize=12,
        fontweight="bold",
    )
    add_panel_letter(axes[2, 0], "c")
    add_panel_letter(axes[2, 1], "d")
    format_waveform_axes(axes[2, 0], axes[2, 1])

    timeax_19, freqs_19, power_db_19 = hub_cwts[19]
    plot_cwt_power(
        axes[3, 0],
        timeax_19,
        freqs_19,
        power_db_19,
        window_sec,
        min_freq_plot,
        max_freq_plot,
        min_db,
        max_db,
    )
    plot_lower_freq_limit(axes[3, 0], 40.0)
    timeax_20, freqs_20, power_db_20 = hub_cwts[20]
    plot_cwt_power(
        axes[3, 1],
        timeax_20,
        freqs_20,
        power_db_20,
        window_sec,
        min_freq_plot,
        max_freq_plot,
        min_db,
        max_db,
    )
    plot_lower_freq_limit(axes[3, 1], 60.0)
    format_cwt_axes(axes[3, 0], axes[3, 1])

    cax = fig.add_axes([0.35, 0.03, 0.30, 0.015])
    cbar = fig.colorbar(quadmesh, cax=cax, orientation="horizontal")
    cbar.set_label("Power (dB)", fontsize=11)

    fig.suptitle(
        "Hammer vs hub-event waveforms and CWT",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(bottom=0.10, top=0.93)

    print("Saving figure to aguadv_event_n_hammer_cwt.png...")
    save_figure(fig, "aguadv_event_n_hammer_cwt.png")


if __name__ == "__main__":
    main()
