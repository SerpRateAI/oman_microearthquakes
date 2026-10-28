"""
Plot normalized Z-component waveforms and Nuttall-window Fourier spectra for the
hub event and one aligned satellite event selected by discordant rank in a group.

Waveform and spectrum panels are arranged horizontally. Hub traces use a darker
blue; satellite traces use a lighter blue.
"""

from argparse import ArgumentParser
from os.path import join

from scipy.signal.windows import nuttall

from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots
from numpy import amax, arange
from pandas import Timedelta, read_csv, read_json, to_datetime

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
HUB_COLOR = "royalblue"
SATELLITE_COLOR = "skyblue"
LINEWIDTH_WAVEFORM = 2.0
LINEWIDTH_SPECTRUM = 2.0
LINEWIDTH_WINDOW_BAR = 2.0
WINDOW_CAP_HALF_HEIGHT = 0.05
WINDOW_BAR_COLOR = "crimson"
SHORT_WINDOW_Y = 0.75
LONG_WINDOW_Y = -0.75


def convert_station_times(station_dict):
    return {
        station: to_datetime(times, utc=True, format="ISO8601")
        for station, times in station_dict.items()
    }


def load_alignment_df(group_label, suffix_group):
    filename = f"event_alignments_group{group_label:d}_{suffix_group}.csv"
    alignment_df = read_csv(join(dirpath_event, filename))
    alignment_df["aligned_time"] = to_datetime(alignment_df["aligned_time"], format="ISO8601")
    alignment_df["hub"] = alignment_df["hub"].astype(bool)
    return alignment_df


def load_event_df(group_label, suffix_group):
    filename = f"grouped_events_group{group_label:d}_{suffix_group}.jsonl"
    event_df = read_json(join(dirpath_event, filename), lines=True)
    event_df["per_station_times"] = event_df["per_station_times"].apply(convert_station_times)
    return event_df


def load_discordant_scores(group_label, suffix_group):
    filename = f"discordant_scores_group{group_label:d}_{suffix_group}.csv"
    filepath = join(dirpath_event, filename)
    discordant_df = read_csv(filepath)
    return discordant_df.sort_values("discordant_score", ascending=False)


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


def load_station_waveforms_at_arrival(arrival_time, station, buffer_before_sec, buffer_after_sec):
    """Load unfiltered 3-component waveforms for the window centered on arrival_time."""
    starttime_plot = arrival_time - Timedelta(seconds=buffer_before_sec)
    endtime_plot = arrival_time + Timedelta(seconds=buffer_after_sec)
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


def get_hub_short_long_windows(hub_dict, station):
    """Return hub short and long window times used for discordant-score computation."""
    t0_hub, t1_hub = hub_dict["per_station_times"][station]
    dur_short = t1_hub - t0_hub
    dur_long = 2 * dur_short
    return t0_hub, t1_hub, t0_hub, t0_hub + dur_long


def window_times_to_plot_x(t_start, t_end, plot_starttime):
    return (
        (t_start - plot_starttime).total_seconds(),
        (t_end - plot_starttime).total_seconds(),
    )


def plot_window_bar_with_caps(ax, x0, x1, y_center):
    """Draw a horizontal crimson bar with short vertical end caps."""
    ax.hlines(
        y_center,
        x0,
        x1,
        colors=WINDOW_BAR_COLOR,
        linewidth=LINEWIDTH_WINDOW_BAR,
        zorder=3,
    )
    ax.vlines(
        [x0, x1],
        y_center - WINDOW_CAP_HALF_HEIGHT,
        y_center + WINDOW_CAP_HALF_HEIGHT,
        colors=WINDOW_BAR_COLOR,
        linewidth=LINEWIDTH_WINDOW_BAR,
        zorder=3,
    )


def plot_discordant_window_bars(ax, t0_short, t1_short, t0_long, t1_long, plot_starttime):
    """Mark short and long discordant-score windows as capped crimson bars."""
    x0_short, x1_short = window_times_to_plot_x(t0_short, t1_short, plot_starttime)
    x0_long, x1_long = window_times_to_plot_x(t0_long, t1_long, plot_starttime)
    plot_window_bar_with_caps(ax, x0_short, x1_short, SHORT_WINDOW_Y)
    plot_window_bar_with_caps(ax, x0_long, x1_long, LONG_WINDOW_Y)


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


def select_satellite_id(discordant_df, id_satellite, rank_discordant):
    if id_satellite is not None:
        return int(id_satellite)
    if rank_discordant < 1 or rank_discordant > len(discordant_df):
        raise ValueError(
            f"rank_discordant must be between 1 and {len(discordant_df)}; got {rank_discordant}."
        )
    return int(discordant_df.iloc[rank_discordant - 1]["id_satellite"])


def main():
    parser = ArgumentParser(
        description=(
            "Plot hub and discordant satellite Z waveforms and Fourier spectra "
            "for one event group"
        ),
    )
    parser.add_argument("--group_label", type=int, required=True, help="Event group label")
    parser.add_argument(
        "--id_satellite",
        type=int,
        default=None,
        help="Plot a specific satellite event (overrides rank_discordant)",
    )
    parser.add_argument(
        "--rank_discordant",
        type=int,
        default=1,
        help="Discordant rank of the satellite event to plot (1 = most discordant)",
    )
    parser.add_argument(
        "--station",
        type=str,
        default=None,
        help="Station to plot (default: hub aligned_station)",
    )
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
    parser.add_argument(
        "--plot_duration_sec",
        type=float,
        default=0.25,
        help="Total duration of waveform panels in seconds",
    )
    parser.add_argument(
        "--max_freq_plot",
        type=float,
        default=None,
        help="Maximum frequency for spectrum panels (default: Nyquist)",
    )
    parser.add_argument("--min_db", type=float, default=-45.0, help="Spectrum y-axis minimum (dB)")
    parser.add_argument("--max_db", type=float, default=5.0, help="Spectrum y-axis maximum (dB)")
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
    window_sec = args.plot_duration_sec
    buffer_after_sec = max(0.0, args.plot_duration_sec - args.buffer_before_sec)

    print("--------------------------------")
    print("Plotting discordant event waveforms and spectra...")
    print("--------------------------------")
    print(f"Group label: {args.group_label}")
    print(f"Suffix group: {suffix_group}")

    alignment_df = load_alignment_df(args.group_label, suffix_group)
    event_df = load_event_df(args.group_label, suffix_group)
    discordant_df = load_discordant_scores(args.group_label, suffix_group)

    hub_row = alignment_df.loc[alignment_df["hub"]].iloc[0]
    id_hub = int(hub_row["id"])
    station = args.station if args.station is not None else hub_row["aligned_station"]
    hub_aligned_time = hub_row["aligned_time"]
    hub_dict = event_df.loc[event_df["event_id"] == id_hub].iloc[0].to_dict()

    id_satellite = select_satellite_id(discordant_df, args.id_satellite, args.rank_discordant)

    print(f"Hub event ID: {id_hub}")
    print(f"Station: {station}")
    print(f"Satellite event ID: {id_satellite}")
    if args.id_satellite is None:
        print(f"Discordant rank: {args.rank_discordant}")

    print(f"Loading hub event waveforms at {station}...")
    hub_waveform_dict, _ = load_station_waveforms_at_arrival(
        hub_aligned_time,
        station,
        args.buffer_before_sec,
        buffer_after_sec,
    )

    print("Generating the 1×2 figure...")
    fig, axes = subplots(
        1,
        2,
        figsize=(args.figwidth, args.figheight),
        sharex="col",
        gridspec_kw={"wspace": 0.15},
    )
    ax_wf = axes[0]
    ax_spec = axes[1]

    legend_handles = [
        Line2D([0], [0], color=HUB_COLOR, linewidth=LINEWIDTH_WAVEFORM, label="Hub"),
        Line2D([0], [0], color=SATELLITE_COLOR, linewidth=LINEWIDTH_WAVEFORM, label="Satellite"),
    ]

    sat_rows = alignment_df.loc[alignment_df["id"] == id_satellite]
    if sat_rows.empty:
        raise ValueError(f"Satellite {id_satellite} not found in alignments.")

    sat_row = sat_rows.iloc[0]
    sat_aligned_time = sat_row["aligned_time"]
    discordant_score = discordant_df.loc[
        discordant_df["id_satellite"] == id_satellite, "discordant_score"
    ].iloc[0]

    sat_waveform_dict, _ = load_station_waveforms_at_arrival(
        sat_aligned_time,
        station,
        args.buffer_before_sec,
        buffer_after_sec,
    )

    plot_z_waveform(
        ax_wf,
        hub_waveform_dict[VERTICAL_COMPONENT],
        window_sec,
        HUB_COLOR,
        "Hub",
    )
    plot_z_waveform(
        ax_wf,
        sat_waveform_dict[VERTICAL_COMPONENT],
        window_sec,
        SATELLITE_COLOR,
        "Satellite",
    )

    if station in hub_dict.get("per_station_times", {}):
        t0_short, t1_short, t0_long, t1_long = get_hub_short_long_windows(hub_dict, station)
        plot_starttime = hub_aligned_time - Timedelta(seconds=args.buffer_before_sec)
        plot_discordant_window_bars(
            ax_wf, t0_short, t1_short, t0_long, t1_long, plot_starttime,
        )
    else:
        print(f"Warning: no per-station times for {station}; skipping window markers.")

    plot_z_spectrum(
        ax_spec,
        hub_waveform_dict[VERTICAL_COMPONENT],
        HUB_COLOR,
        "Hub",
        min_freq_plot,
        max_freq_plot,
    )
    plot_z_spectrum(
        ax_spec,
        sat_waveform_dict[VERTICAL_COMPONENT],
        SATELLITE_COLOR,
        "Satellite",
        min_freq_plot,
        max_freq_plot,
    )

    ax_spec.set_ylim(args.min_db, args.max_db)
    ax_wf.set_title(
        (
            f"Group {args.group_label:d}, Hub {id_hub} vs Satellite {id_satellite} "
        ),
        fontsize=12,
        fontweight="bold",
    )

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
    ax_wf.text(
        0.02,
        0.98,
        station,
        fontsize=12,
        fontweight="bold",
        transform=ax_wf.transAxes,
        ha="left",
        va="top",
    )

    ax_wf.set_xlabel("Time (s)", fontsize=12)
    format_freq_xlabels(
        ax_spec,
        plot_axis_label=True,
        plot_tick_label=True,
        major_tick_spacing=100.0,
    )
    format_norm_psd_ylabels(ax_spec, major_tick_spacing=10.0)

    fig.suptitle(
        f"Discordant event waveforms and spectra, Group {args.group_label:d}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    ax_spec.legend(
        handles=legend_handles,
        loc="lower left",
        ncol=1,
        fontsize=11,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.3",
    )

    if args.id_satellite is None:
        figname = (
            f"discordant_event_waveforms_n_spectra_group{args.group_label:d}_"
            f"rank{args.rank_discordant:d}_{suffix}.png"
        )
    else:
        figname = (
            f"discordant_event_waveforms_n_spectra_group{args.group_label:d}_"
            f"sat{id_satellite:d}_{suffix}.png"
        )
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
