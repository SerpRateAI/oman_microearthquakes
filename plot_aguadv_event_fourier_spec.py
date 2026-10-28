"""
Plot hub-event Z waveforms and Nuttall-window Fourier power spectra for groups 18 and 19
at A04 (Subarray A) and group 20 at B04 (Subarray B) in a 3×2 figure (AGU Advances).

Hammer-shot Z spectrum (A04) is overlaid on spectrum panels for groups 18 and 19 only.
"""

from argparse import ArgumentParser
from os.path import join
from pathlib import Path

from scipy.signal.windows import nuttall

from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots
from numpy import amax, arange
from pandas import Timedelta, read_csv, to_datetime

from utils_basic import (
    DETECTION_DIR as dirpath_event,
    GEO_COMPONENTS as components,
    PICK_DIR as dirpath_pick,
    SAMPLING_RATE as sample_rate,
    geo_channel2component,
    get_freq_limits_string,
    power2db,
)
from utils_cc import get_repeating_snippet_suffix
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_plot import (
    component2label,
    format_freq_xlabels,
    format_norm_psd_ylabels,
    get_geo_component_color,
    save_figure,
)
from utils_snuffler import read_time_windows
from utils_spec import get_fft_psd
from utils_sta_lta import get_sta_lta_suffix

VERTICAL_COMPONENT = "Z"
HAMMER_ID = "100129"
HAMMER_STATION = "A04"

GROUP_ROWS = [
    {"label": 18, "station": "A04", "plot_hammer": True},
    {"label": 19, "station": "A04", "plot_hammer": True},
    {"label": 20, "station": "B04", "plot_hammer": False},
]

LINEWIDTH_WAVEFORM = 2.0
LINEWIDTH_SPECTRUM = 2.0
LINEWIDTH_HAMMER_SPECTRUM = 3.0
HAMMER_SPECTRUM_COLOR = "black"


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


def stream_station_to_waveform_dict(stream, station):
    """Extract 3-component arrays from a processed geophone stream for one station."""
    stream_sta = stream.select(station=station)
    if len(stream_sta) != len(components):
        raise ValueError(
            f"Expected {len(components)} components at {station}, got {len(stream_sta)} traces."
        )
    waveform_dict = {}
    for trace in stream_sta:
        component = geo_channel2component(trace.stats.channel)
        waveform_dict[component] = trace.data
    return waveform_dict


def load_station_waveforms_at_arrival(arrival_time, station, buffer_before_sec, buffer_after_sec):
    """Load unfiltered 3-component waveforms for the 0.25-s window centered on arrival_time."""
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
    return stream_station_to_waveform_dict(stream, station), buffer_before_sec + buffer_after_sec


def load_hammer_arrival_time(pick_filepath, station):
    """Return the manual P-wave arrival time (pick window center) for the given station."""
    arrival_df = read_time_windows(pick_filepath, phase_marker=True)
    station_df = arrival_df[arrival_df["station"] == station]
    if station_df.empty:
        raise ValueError(f"No pick for station {station} in {pick_filepath}")
    row = station_df.iloc[0]
    return row["starttime"] + (row["endtime"] - row["starttime"]) / 2


def compute_normalized_psd_db(waveform):
    """Nuttall-window PSD normalized to peak power, returned in dB."""
    freqax, psd = get_fft_psd(waveform, sample_rate, window=nuttall)
    psd_peak = amax(psd)
    if psd_peak > 0:
        psd = psd / psd_peak
    return freqax, power2db(psd)


def plot_z_waveform(ax, waveform, window_sec, normalize_display=True):
    """Plot Z component vs time (seconds from window start)."""
    timeax = arange(len(waveform)) / sample_rate
    if normalize_display and amax(abs(waveform)) > 0:
        waveform_plot = waveform / amax(abs(waveform))
    else:
        waveform_plot = waveform
    color = get_geo_component_color(VERTICAL_COMPONENT)
    ax.plot(timeax, waveform_plot, color=color, linewidth=LINEWIDTH_WAVEFORM)
    ax.set_xlim(0.0, window_sec)


def plot_component_spectra(ax, waveform_dict, min_freq_plot, max_freq_plot):
    """Plot Nuttall-window PSD in dB for Z, 1, and 2 (each normalized to its peak)."""
    for component in components:
        freqax, psd_db = compute_normalized_psd_db(waveform_dict[component])
        color = get_geo_component_color(component)
        ax.plot(
            freqax,
            psd_db,
            color=color,
            linewidth=LINEWIDTH_SPECTRUM,
            label=component2label(component),
        )
    ax.set_xlim(min_freq_plot, max_freq_plot)
    ax.grid(True, linestyle=":", which="both", axis="x")


def plot_hammer_z_spectrum(ax, waveform_dict):
    """Overlay hammer-shot Z-component PSD as a dotted black line."""
    freqax, psd_db = compute_normalized_psd_db(waveform_dict[VERTICAL_COMPONENT])
    ax.plot(
        freqax,
        psd_db,
        color=HAMMER_SPECTRUM_COLOR,
        linewidth=LINEWIDTH_HAMMER_SPECTRUM,
        linestyle=":",
        zorder=3,
    )


def main():
    parser = ArgumentParser(
        description=(
            "Plot hub-event Z waveforms and Nuttall-window Fourier spectra for "
            "groups 18 and 19 (A04) and group 20 (B04) (AGU Advances)"
        ),
    )
    parser.add_argument("--figwidth", type=float, default=14.0)
    parser.add_argument("--figheight", type=float, default=12.0)
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
    parser.add_argument(
        "--max_freq_plot",
        type=float,
        default=None,
        help="Maximum frequency for spectrum panels (default: Nyquist)",
    )
    parser.add_argument("--min_db", type=float, default=-45.0, help="Spectrum y-axis minimum (dB)")
    parser.add_argument("--max_db", type=float, default=5.0, help="Spectrum y-axis maximum (dB)")
    parser.add_argument("--hammer_id", type=str, default=HAMMER_ID, help="Hammer shot ID")
    parser.add_argument(
        "--hammer_pick_file",
        type=str,
        default=None,
        help="Snuffler pick file for the hammer shot (default: PICK_DIR/hammer_{hammer_id}.mkr)",
    )
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

    min_freq_plot = args.min_freq_filter
    max_freq_plot = args.max_freq_plot if args.max_freq_plot is not None else sample_rate / 2.0
    window_sec = args.buffer_before_sec + args.buffer_after_sec
    print(f"Loading hammer shot {args.hammer_id} waveforms at {HAMMER_STATION}...")
    hammer_arrival_time = load_hammer_arrival_time(hammer_pick_file, HAMMER_STATION)
    hammer_waveform_dict, _ = load_station_waveforms_at_arrival(
        hammer_arrival_time,
        HAMMER_STATION,
        args.buffer_before_sec,
        args.buffer_after_sec,
    )

    print("Generating the 3×2 figure...")
    fig, axes = subplots(
        len(GROUP_ROWS),
        2,
        figsize=(args.figwidth, args.figheight),
        sharex="col",
        gridspec_kw={"hspace": 0.2, "wspace": 0.15},
    )

    legend_handles = [
        Line2D(
            [0], [0],
            color=get_geo_component_color(comp),
            linewidth=LINEWIDTH_SPECTRUM,
            linestyle="-",
            label=component2label(comp),
        )
        for comp in components
    ]
    legend_handles.append(
        Line2D(
            [0], [0],
            color=HAMMER_SPECTRUM_COLOR,
            linewidth=LINEWIDTH_HAMMER_SPECTRUM,
            linestyle=":",
            label="Hammer",
        ),
    )

    for i_row, group_row in enumerate(GROUP_ROWS):
        group_label = group_row["label"]
        station = group_row["station"]
        plot_hammer = group_row["plot_hammer"]

        print(f"Processing group {group_label:d} at {station}...")
        ax_wf = axes[i_row, 0]
        ax_spec = axes[i_row, 1]

        try:
            id_hub, aligned_time = load_hub_alignment(group_label, suffix_group)
            waveform_dict, _ = load_station_waveforms_at_arrival(
                aligned_time,
                station,
                args.buffer_before_sec,
                args.buffer_after_sec,
            )
        except Exception as exc:
            print(f"Warning: skipping group {group_label:d}: {exc}")
            ax_wf.set_visible(False)
            ax_spec.set_visible(False)
            continue

        plot_z_waveform(ax_wf, waveform_dict[VERTICAL_COMPONENT], window_sec)
        plot_component_spectra(ax_spec, waveform_dict, min_freq_plot, max_freq_plot)
        if plot_hammer:
            plot_hammer_z_spectrum(ax_spec, hammer_waveform_dict)

        ax_spec.set_ylim(args.min_db, args.max_db)

        ax_wf.set_title(
            f"Group {group_label:d}, Hub event {id_hub}, {station}",
            fontsize=12,
            fontweight="bold",
        )

        ax_wf.text(
            -0.08,
            1.05,
            f"({chr(97 + i_row)})",
            fontsize=14,
            fontweight="bold",
            transform=ax_wf.transAxes,
            ha="right",
            va="top",
        )

        if i_row == len(GROUP_ROWS) - 1:
            ax_wf.set_xlabel("Time (s)", fontsize=12)
            format_freq_xlabels(
                ax_spec,
                plot_axis_label=True,
                plot_tick_label=True,
                major_tick_spacing=100.0,
            )
        else:
            format_freq_xlabels(
                ax_spec,
                plot_axis_label=False,
                plot_tick_label=True,
                major_tick_spacing=100.0,
            )

        if i_row == 0:
            format_norm_psd_ylabels(ax_spec, major_tick_spacing=10.0)
        else:
            format_norm_psd_ylabels(
                ax_spec,
                plot_axis_label=False,
                major_tick_spacing=10.0,
            )

    fig.suptitle(
        "Hub event waveforms and spectra",
        fontsize=16,
        fontweight="bold",
        y=0.92,
    )
    axes[0, 1].legend(
        handles=legend_handles,
        loc="lower left",
        ncol=1,
        fontsize=11,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.3",
    )

    save_figure(fig, "aguadv_event_fourier_spec.png")


if __name__ == "__main__":
    main()
