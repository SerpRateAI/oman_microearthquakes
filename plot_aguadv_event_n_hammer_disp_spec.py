"""
Plot unfiltered Fourier displacement power spectra of Group 18 and 19 hub-event
stacks at A04 (with Hammer 100129) and the Group 20 hub-event stack at B04
in a 3×3 figure (AGU Advances).
"""

from argparse import ArgumentParser
from os.path import join
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots
from numpy import amax
from obspy import read
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
    format_norm_psd_ylabels,
    get_geo_component_color,
    save_figure,
)
from utils_snuffler import read_time_windows
from utils_spec import get_fft_psd, vel2disp
from utils_sta_lta import get_sta_lta_suffix

HAMMER_ID = "100129"
HAMMER_STATION = "A04"
GROUP_ROWS = [
    {"label": 18, "station": "A04", "plot_hammer": True},
    {"label": 19, "station": "A04", "plot_hammer": True},
    {"label": 20, "station": "B04", "plot_hammer": False},
]

LINEWIDTH_EVENT = 3.0
LINEWIDTH_HAMMER = 3.0
LINESTYLE_EVENT = "-"
LINESTYLE_HAMMER = "--"
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


def load_hammer_arrival_time(pick_filepath, station):
    """Return the manual P-wave arrival time (pick window center) for the given station."""
    arrival_df = read_time_windows(pick_filepath, phase_marker=True)
    station_df = arrival_df[arrival_df["station"] == station]
    if station_df.empty:
        raise ValueError(f"No pick for station {station} in {pick_filepath}")
    row = station_df.iloc[0]
    return row["starttime"] + (row["endtime"] - row["starttime"]) / 2


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


def load_raw_hub_stack_waveforms(group_label, station, suffix_group):
    """Load unfiltered hub-event stack traces for one station from MiniSEED."""
    filename = f"hub_event_waveform_stack_raw_group{group_label:d}_{suffix_group}.mseed"
    filepath = join(dirpath_event, filename)
    if not Path(filepath).is_file():
        raise FileNotFoundError(
            f"Raw hub-event waveform stack not found: {filepath}. "
            f"Run compute_hub_event_waveform_stacks_raw.py --group_label {group_label:d}."
        )
    stream = read(filepath)
    return stream_station_to_waveform_dict(stream, station)


def load_station_waveforms_at_arrival(arrival_time, station, buffer_before_sec, buffer_after_sec):
    """Load unfiltered, unnormalized 3-component velocity waveforms around arrival_time."""
    starttime_plot = arrival_time - Timedelta(seconds=buffer_before_sec)
    endtime_plot = arrival_time + Timedelta(seconds=buffer_after_sec)
    stream = read_and_process_windowed_geo_waveforms(
        starttime_plot,
        endtime=endtime_plot,
        stations=[station],
        components=components,
        filter=False,
        normalize=False,
    )
    if stream is None or len(stream) == 0:
        raise ValueError(f"No waveform data for station {station}.")
    return stream_station_to_waveform_dict(stream, station)


def compute_normalized_disp_psd_db(velocity, min_freq_plot, max_freq_plot):
    """Hann-windowed Fourier displacement PSD, peak-normalized in the plot band, in dB."""
    displacement = vel2disp(velocity, sample_rate)
    freqax, psd = get_fft_psd(displacement, sample_rate)
    in_band = (freqax >= min_freq_plot) & (freqax <= max_freq_plot)
    freqax = freqax[in_band]
    psd = psd[in_band]
    psd_peak = amax(psd)
    if psd_peak > 0:
        psd = psd / psd_peak
    return freqax, power2db(psd)


def main():
    parser = ArgumentParser(
        description=(
            "Plot unfiltered Fourier displacement power spectra of Group 18 and 19 "
            "hub-event stacks at A04 (with hammer) and the Group 20 hub-event stack "
            "at B04 (AGU Advances)"
        ),
    )
    parser.add_argument("--hammer_id", type=str, default=HAMMER_ID, help="Hammer shot ID")
    parser.add_argument(
        "--hammer_pick_file",
        type=str,
        default=None,
        help="Snuffler pick file for the hammer shot (default: PICK_DIR/hammer_{hammer_id}.mkr)",
    )
    parser.add_argument("--figwidth", type=float, default=15.0)
    parser.add_argument("--figheight", type=float, default=15.0)
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
    parser.add_argument("--min_freq_plot", type=float, default=10.0)
    parser.add_argument(
        "--max_freq_plot",
        type=float,
        default=None,
        help="Maximum frequency for spectrum panels (default: Nyquist)",
    )
    parser.add_argument("--min_db", type=float, default=-45.0, help="Spectrum y-axis minimum (dB)")
    parser.add_argument("--max_db", type=float, default=5.0, help="Spectrum y-axis maximum (dB)")
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

    max_freq_plot = args.max_freq_plot if args.max_freq_plot is not None else sample_rate / 2.0

    print(f"Loading hammer shot {args.hammer_id} waveforms at {HAMMER_STATION}...")
    hammer_arrival_time = load_hammer_arrival_time(hammer_pick_file, HAMMER_STATION)
    hammer_waveform_dict = load_station_waveforms_at_arrival(
        hammer_arrival_time,
        HAMMER_STATION,
        args.buffer_before_sec,
        args.buffer_after_sec,
    )

    hub_waveform_dicts = {}
    for group_row in GROUP_ROWS:
        group_label = group_row["label"]
        station = group_row["station"]
        id_hub, _ = load_hub_alignment(group_label, suffix_group)
        print(f"Group {group_label:d} hub event ID: {id_hub}")
        print(f"Loading raw hub-event stack for group {group_label:d} at {station}...")
        hub_waveform_dicts[group_label] = load_raw_hub_stack_waveforms(
            group_label, station, suffix_group,
        )

    print("Computing displacement PSDs...")
    fig, axes = subplots(
        len(GROUP_ROWS),
        len(components),
        figsize=(args.figwidth, args.figheight),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.18, "wspace": 0.12},
    )

    legend_handles = [
        Line2D(
            [0], [0],
            color="black",
            linewidth=LINEWIDTH_EVENT,
            linestyle=LINESTYLE_EVENT,
            label="Event",
        ),
        Line2D(
            [0], [0],
            color="black",
            linewidth=LINEWIDTH_HAMMER,
            linestyle=LINESTYLE_HAMMER,
            label="Hammer",
        ),
    ]

    for i_row, group_row in enumerate(GROUP_ROWS):
        group_label = group_row["label"]
        station = group_row["station"]
        plot_hammer = group_row["plot_hammer"]
        hub_waveform_dict = hub_waveform_dicts[group_label]

        for i_col, component in enumerate(components):
            ax = axes[i_row, i_col]
            color = get_geo_component_color(component)

            freqax_event, psd_event_db = compute_normalized_disp_psd_db(
                hub_waveform_dict[component], args.min_freq_plot, max_freq_plot,
            )

            ax.plot(
                freqax_event,
                psd_event_db,
                color=color,
                linewidth=LINEWIDTH_EVENT,
                linestyle=LINESTYLE_EVENT,
            )
            if plot_hammer:
                freqax_hammer, psd_hammer_db = compute_normalized_disp_psd_db(
                    hammer_waveform_dict[component], args.min_freq_plot, max_freq_plot,
                )
                ax.plot(
                    freqax_hammer,
                    psd_hammer_db,
                    color=HAMMER_SPECTRUM_COLOR,
                    linewidth=LINEWIDTH_HAMMER,
                    linestyle=LINESTYLE_HAMMER,
                )

            ax.set_xscale("log")
            ax.set_xlim(args.min_freq_plot, max_freq_plot)
            ax.set_ylim(args.min_db, args.max_db)
            ax.grid(True, linestyle="--", alpha=0.5, which="major")
            ax.grid(True, linestyle=":", alpha=0.5, which="minor")

            if i_row == 0:
                ax.set_title(component2label(component), fontsize=12, fontweight="bold")
            if i_row == len(GROUP_ROWS) - 1:
                ax.set_xlabel("Frequency (Hz)", fontsize=12)
            else:
                ax.tick_params(axis="x", labelbottom=False)

            if i_col > 0:
                ax.tick_params(axis="y", labelleft=False)

        axes[i_row, 0].text(
            -0.08,
            1.05,
            f"({chr(97 + i_row)})",
            fontsize=14,
            fontweight="bold",
            transform=axes[i_row, 0].transAxes,
            ha="right",
            va="top",
        )
        axes[i_row, 0].text(
            0.03,
            0.97,
            f"Group {group_label:d}, {station}",
            transform=axes[i_row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            zorder=4,
        )

    for i_row in range(len(GROUP_ROWS)):
        format_norm_psd_ylabels(axes[i_row, 0])

    axes[0, 0].legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=11,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.3",
    )

    fig.suptitle(
        f"Hub event waveform stacks vs hammer displacement spectra",
        fontsize=14,
        fontweight="bold",
        y=0.92,
    )
    fig.subplots_adjust(left=0.12)

    print("Saving figure to aguadv_event_n_hammer_disp_spec.png...")
    save_figure(fig, "aguadv_event_n_hammer_disp_spec.png")


if __name__ == "__main__":
    main()
