"""
Plot normalized Z-component waveforms and Nuttall-window Fourier spectra for the
smallest and largest events in a classified event group.

Events are ranked by log_magnitude from log_magnitudes_group{N}_{suffix}.csv.
Waveforms are loaded at each event's first_onset_station around first_onset.
Waveform and spectrum panels are arranged horizontally (1x2) with both events
overlaid. Smallest traces use a lighter blue; largest traces use a darker blue.
"""

from argparse import ArgumentParser
from os.path import isfile, join

from scipy.signal.windows import nuttall

from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots
from numpy import amax, arange
from pandas import Timedelta, read_csv, to_datetime

from utils_basic import (
    DETECTION_DIR as dirpath_event,
    GEO_COMPONENTS as components,
    SAMPLING_RATE as sample_rate,
    geo_channel2component,
    get_freq_limits_string,
    power2db,
)
from utils_cc import get_repeating_snippet_suffix
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_plot import format_freq_xlabels, format_norm_psd_ylabels, save_figure
from utils_spec import get_fft_psd
from utils_sta_lta import get_sta_lta_suffix

VERTICAL_COMPONENT = "Z"
SMALL_COLOR = "skyblue"
LARGE_COLOR = "royalblue"
LINEWIDTH_WAVEFORM = 2.0
LINEWIDTH_SPECTRUM = 2.0


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


def load_station_waveforms_between(starttime, endtime, station):
    """Load unfiltered 3-component waveforms for an absolute time window."""
    stream = read_and_process_windowed_geo_waveforms(
        starttime,
        endtime=endtime,
        stations=[station],
        components=components,
        filter=False,
        normalize=True,
    )
    if stream is None or len(stream) == 0:
        raise ValueError(f"No waveform data for station {station}.")
    return stream_station_to_waveform_dict(stream, station)


def load_station_waveforms_at_onset(onset_time, station, buffer_before_sec, buffer_after_sec):
    """Load unfiltered 3-component waveforms for the window around first_onset."""
    starttime_plot = onset_time - Timedelta(seconds=buffer_before_sec)
    endtime_plot = onset_time + Timedelta(seconds=buffer_after_sec)
    waveform_dict = load_station_waveforms_between(starttime_plot, endtime_plot, station)
    return waveform_dict, buffer_before_sec + buffer_after_sec


def compute_normalized_psd_db(waveform):
    """Nuttall-window PSD normalized to peak power, returned in dB."""
    freqax, psd = get_fft_psd(waveform, sample_rate, window=nuttall)
    psd_peak = amax(psd)
    if psd_peak > 0:
        psd = psd / psd_peak
    return freqax, power2db(psd)


def normalize_waveform_for_display(waveform):
    if amax(abs(waveform)) > 0:
        return waveform / amax(abs(waveform))
    return waveform


def plot_z_waveform(
    ax, waveform, window_sec, color, label, timeax=None, linewidth=LINEWIDTH_WAVEFORM,
):
    """Plot normalized Z component vs time (seconds from window start)."""
    if timeax is None:
        timeax = arange(len(waveform)) / sample_rate
    ax.plot(
        timeax,
        normalize_waveform_for_display(waveform),
        color=color,
        linewidth=linewidth,
        label=label,
    )
    ax.set_xlim(0.0, window_sec)
    ax.set_ylim(-1.0, 1.0)


def plot_z_spectrum(
    ax, waveform, color, label, min_freq_plot, max_freq_plot, linewidth=LINEWIDTH_SPECTRUM,
):
    """Plot Nuttall-window Z-component PSD in dB, normalized to its peak."""
    freqax, psd_db = compute_normalized_psd_db(waveform)
    ax.plot(
        freqax,
        psd_db,
        color=color,
        linewidth=linewidth,
        label=label,
    )
    ax.set_xlim(min_freq_plot, max_freq_plot)
    ax.grid(True, linestyle=":", which="both", axis="x")


def load_log_magnitudes(group_label, suffix_group):
    """Load log-magnitude CSV for one event group."""
    filename = f"log_magnitudes_group{group_label:d}_{suffix_group}.csv"
    filepath = join(dirpath_event, filename)
    if not isfile(filepath):
        raise FileNotFoundError(f"Log magnitudes file not found: {filepath}")
    mag_df = read_csv(filepath)
    mag_df["first_onset"] = to_datetime(mag_df["first_onset"], format="ISO8601")
    return mag_df


def select_small_and_large_events(mag_df):
    """Return (small_row, large_row) Series for min/max log_magnitude."""
    valid_df = mag_df.dropna(subset=["log_magnitude"])
    if len(valid_df) < 2:
        raise ValueError(
            f"Need at least 2 events with valid log_magnitude; got {len(valid_df)}."
        )
    small_row = valid_df.loc[valid_df["log_magnitude"].idxmin()]
    large_row = valid_df.loc[valid_df["log_magnitude"].idxmax()]
    if int(small_row["event_id"]) == int(large_row["event_id"]):
        raise ValueError("Smallest and largest events resolve to the same event_id.")
    return small_row, large_row


def main():
    parser = ArgumentParser(
        description=(
            "Plot smallest and largest event Z waveforms and Fourier spectra "
            "for one event group (overlaid, 1x2 layout)"
        ),
    )
    parser.add_argument("--group_label", type=int, required=True, help="Event group label")
    parser.add_argument("--figwidth", type=float, default=14.0)
    parser.add_argument("--figheight", type=float, default=4.5)
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
    parser.add_argument(
        "--figname",
        type=str,
        default=None,
        help="Output figure filename (default: auto from group label and suffix)",
    )
    args = parser.parse_args()

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

    print("--------------------------------")
    print("Plotting small and large event waveforms and spectra...")
    print("--------------------------------")
    print(f"Group label: {args.group_label}")
    print(f"Suffix group: {suffix_group}")

    mag_df = load_log_magnitudes(args.group_label, suffix_group)
    small_row, large_row = select_small_and_large_events(mag_df)

    id_small = int(small_row["event_id"])
    id_large = int(large_row["event_id"])
    station_small = small_row["first_onset_station"]
    station_large = large_row["first_onset_station"]
    log_mag_small = float(small_row["log_magnitude"])
    log_mag_large = float(large_row["log_magnitude"])
    onset_small = small_row["first_onset"]
    onset_large = large_row["first_onset"]

    print(
        f"Smallest event: id={id_small}, station={station_small}, "
        f"log_magnitude={log_mag_small:.3f}"
    )
    print(
        f"Largest event: id={id_large}, station={station_large}, "
        f"log_magnitude={log_mag_large:.3f}"
    )

    print(f"Loading smallest event waveforms at {station_small}...")
    small_waveform_dict, _ = load_station_waveforms_at_onset(
        onset_small,
        station_small,
        args.buffer_before_sec,
        args.buffer_after_sec,
    )

    print(f"Loading largest event waveforms at {station_large}...")
    large_waveform_dict, _ = load_station_waveforms_at_onset(
        onset_large,
        station_large,
        args.buffer_before_sec,
        args.buffer_after_sec,
    )

    label_small = f"Smallest (id={id_small}, {station_small}, logM={log_mag_small:.2f})"
    label_large = f"Largest (id={id_large}, {station_large}, logM={log_mag_large:.2f})"

    print("Generating the 1x2 figure...")
    fig, axes = subplots(
        1,
        2,
        figsize=(args.figwidth, args.figheight),
        gridspec_kw={"wspace": 0.15},
    )
    ax_wf = axes[0]
    ax_spec = axes[1]

    plot_z_waveform(
        ax_wf,
        small_waveform_dict[VERTICAL_COMPONENT],
        window_sec,
        SMALL_COLOR,
        label_small,
    )
    plot_z_waveform(
        ax_wf,
        large_waveform_dict[VERTICAL_COMPONENT],
        window_sec,
        LARGE_COLOR,
        label_large,
    )

    plot_z_spectrum(
        ax_spec,
        small_waveform_dict[VERTICAL_COMPONENT],
        SMALL_COLOR,
        label_small,
        min_freq_plot,
        max_freq_plot,
    )
    plot_z_spectrum(
        ax_spec,
        large_waveform_dict[VERTICAL_COMPONENT],
        LARGE_COLOR,
        label_large,
        min_freq_plot,
        max_freq_plot,
    )

    ax_spec.set_ylim(args.min_db, args.max_db)

    ax_wf.text(
        -0.08,
        1.05,
        "(a)",
        fontsize=14,
        fontweight="bold",
        transform=ax_wf.transAxes,
        ha="right",
        va="top",
    )
    ax_spec.text(
        -0.08,
        1.05,
        "(b)",
        fontsize=14,
        fontweight="bold",
        transform=ax_spec.transAxes,
        ha="right",
        va="top",
    )

    ax_wf.set_xlabel("Time (s)", fontsize=12)
    ax_wf.set_ylabel("Normalized amplitude", fontsize=12)
    format_freq_xlabels(
        ax_spec,
        plot_axis_label=True,
        plot_tick_label=True,
        major_tick_spacing=100.0,
    )
    format_norm_psd_ylabels(ax_spec, major_tick_spacing=10.0)

    legend_handles = [
        Line2D([0], [0], color=SMALL_COLOR, linewidth=LINEWIDTH_WAVEFORM, label=label_small),
        Line2D([0], [0], color=LARGE_COLOR, linewidth=LINEWIDTH_WAVEFORM, label=label_large),
    ]
    ax_spec.legend(
        handles=legend_handles,
        loc="lower left",
        ncol=1,
        fontsize=10,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.3",
    )

    fig.suptitle(
        f"Smallest vs largest event waveforms and spectra, Group {args.group_label:d}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    if args.figname is not None:
        figname = args.figname
    else:
        figname = (
            f"small_n_large_event_waveform_n_spec_group{args.group_label:d}_{suffix}.png"
        )
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
