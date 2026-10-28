"""
Plot unfiltered multi-taper displacement power spectra of Group 18 and 19 hub
events at A04 (with Hammer 100129) and the Group 20 hub event at B04, each with
an equal-length pre-event noise window, in a 3×3 figure (AGU Advances).
"""

from argparse import ArgumentParser
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots
from numpy import amax
from pandas import Timedelta

from utils_basic import (
    GEO_COMPONENTS as components,
    SAMPLING_RATE as sample_rate,
    get_freq_limits_string,
)
from utils_cc import get_repeating_snippet_suffix
from utils_plot import (
    component2label,
    format_norm_psd_ylabels,
    get_geo_component_color,
    save_figure,
)
from utils_sta_lta import get_sta_lta_suffix

from plot_event_n_hammer_disp_spec import (
    HAMMER_ID,
    hammer_pick_filepath,
    load_hammer_arrival_time,
    load_hub_alignment,
    load_station_waveforms_at_arrival,
)
from plot_hub_event_n_noise_disp_spec import (
    LINESTYLE_EVENT,
    LINESTYLE_NOISE,
    LINEWIDTH_EVENT,
    LINEWIDTH_NOISE,
    NOISE_SPECTRUM_COLOR,
    compute_disp_psd,
    load_station_waveforms_between,
    plot_spectrum_with_uncertainty,
)

HAMMER_STATION = "A04"
LINEWIDTH_HAMMER = 3.0
LINESTYLE_HAMMER = "--"
HAMMER_SPECTRUM_COLOR = "black"
GROUP_ROWS = [
    {"label": 18, "station": "A04", "plot_hammer": True},
    {"label": 19, "station": "A04", "plot_hammer": True},
    {"label": 20, "station": "B04", "plot_hammer": False},
]


def main():
    parser = ArgumentParser(
        description=(
            "Plot unfiltered multi-taper displacement power spectra of Group 18 "
            "and 19 hub events at A04 (with hammer) and the Group 20 hub event "
            "at B04, each with an equal-length pre-event noise window (AGU Advances)"
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
    parser.add_argument("--nw", type=int, default=3, help="Time-bandwidth product")
    parser.add_argument("--fill_alpha", type=float, default=0.2, help="Uncertainty-band opacity")
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
    nw = args.nw
    fill_alpha = args.fill_alpha
    min_freq_plot = args.min_freq_plot
    max_freq_plot_arg = args.max_freq_plot
    min_db = args.min_db
    max_db = args.max_db

    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(
        sta_window_sec, lta_window_sec, thr_on, thr_off,
    )
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=min_cc, min_num_similar=min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

    max_freq_plot = max_freq_plot_arg if max_freq_plot_arg is not None else sample_rate / 2.0
    window_sec = buffer_before_sec + buffer_after_sec

    hammer_pick_file = (
        Path(hammer_pick_file_arg)
        if hammer_pick_file_arg is not None
        else hammer_pick_filepath(hammer_id)
    )
    print(f"Loading hammer shot {hammer_id} waveforms at {HAMMER_STATION}...")
    hammer_arrival_time = load_hammer_arrival_time(hammer_pick_file, HAMMER_STATION)
    hammer_waveform_dict = load_station_waveforms_at_arrival(
        hammer_arrival_time,
        HAMMER_STATION,
        buffer_before_sec,
        buffer_after_sec,
    )

    hub_waveform_dicts = {}
    noise_waveform_dicts = {}
    for group_row in GROUP_ROWS:
        group_label = group_row["label"]
        station = group_row["station"]
        print(f"Loading hub event for group {group_label:d} at {station}...")
        id_hub, hub_aligned_time = load_hub_alignment(group_label, suffix_group)
        print(f"Group {group_label:d} hub event ID: {id_hub}")
        hub_waveform_dicts[group_label] = load_station_waveforms_at_arrival(
            hub_aligned_time,
            station,
            buffer_before_sec,
            buffer_after_sec,
        )
        event_start = hub_aligned_time - Timedelta(seconds=buffer_before_sec)
        noise_end = event_start
        noise_start = noise_end - Timedelta(seconds=window_sec)
        print(f"Loading pre-event noise at {station} from {noise_start} to {noise_end}...")
        noise_waveform_dicts[group_label] = load_station_waveforms_between(
            noise_start, noise_end, station,
        )

    print("Computing multi-taper displacement PSDs...")
    fig, axes = subplots(
        len(GROUP_ROWS),
        len(components),
        figsize=(figwidth, figheight),
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
            color=HAMMER_SPECTRUM_COLOR,
            linewidth=LINEWIDTH_HAMMER,
            linestyle=LINESTYLE_HAMMER,
            label="Hammer",
        ),
        Line2D(
            [0], [0],
            color=NOISE_SPECTRUM_COLOR,
            linewidth=LINEWIDTH_NOISE,
            linestyle=LINESTYLE_NOISE,
            label="Noise",
        ),
    ]

    for i_row, group_row in enumerate(GROUP_ROWS):
        group_label = group_row["label"]
        station = group_row["station"]
        plot_hammer = group_row["plot_hammer"]
        hub_waveform_dict = hub_waveform_dicts[group_label]
        noise_waveform_dict = noise_waveform_dicts[group_label]

        for i_col, component in enumerate(components):
            ax = axes[i_row, i_col]
            color = get_geo_component_color(component)

            freqax_event, psd_event, psd_event_lo, psd_event_hi = compute_disp_psd(
                hub_waveform_dict[component], min_freq_plot, max_freq_plot, nw,
            )
            freqax_noise, psd_noise, psd_noise_lo, psd_noise_hi = compute_disp_psd(
                noise_waveform_dict[component], min_freq_plot, max_freq_plot, nw,
            )
            psd_peak = amax(psd_event)
            if psd_peak > 0:
                psd_event = psd_event / psd_peak
                psd_event_lo = psd_event_lo / psd_peak
                psd_event_hi = psd_event_hi / psd_peak
                psd_noise = psd_noise / psd_peak
                psd_noise_lo = psd_noise_lo / psd_peak
                psd_noise_hi = psd_noise_hi / psd_peak

            plot_spectrum_with_uncertainty(
                ax,
                freqax_event,
                psd_event,
                psd_event_lo,
                psd_event_hi,
                color,
                LINEWIDTH_EVENT,
                LINESTYLE_EVENT,
                fill_alpha,
            )
            plot_spectrum_with_uncertainty(
                ax,
                freqax_noise,
                psd_noise,
                psd_noise_lo,
                psd_noise_hi,
                NOISE_SPECTRUM_COLOR,
                LINEWIDTH_NOISE,
                LINESTYLE_NOISE,
                fill_alpha,
            )
            if plot_hammer:
                freqax_hammer, psd_hammer, psd_hammer_lo, psd_hammer_hi = compute_disp_psd(
                    hammer_waveform_dict[component], min_freq_plot, max_freq_plot, nw,
                )
                hammer_peak = amax(psd_hammer)
                if hammer_peak > 0:
                    psd_hammer = psd_hammer / hammer_peak
                    psd_hammer_lo = psd_hammer_lo / hammer_peak
                    psd_hammer_hi = psd_hammer_hi / hammer_peak
                plot_spectrum_with_uncertainty(
                    ax,
                    freqax_hammer,
                    psd_hammer,
                    psd_hammer_lo,
                    psd_hammer_hi,
                    HAMMER_SPECTRUM_COLOR,
                    LINEWIDTH_HAMMER,
                    LINESTYLE_HAMMER,
                    fill_alpha,
                )

            ax.set_xscale("log")
            ax.set_xlim(min_freq_plot, max_freq_plot)
            ax.set_ylim(min_db, max_db)
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
        "Multi-taper displacement spectra",
        fontsize=16,
        fontweight="bold",
        y=0.92,
    )
    fig.subplots_adjust(left=0.12)

    print("Saving figure to aguadv_event_mt_spec.png...")
    save_figure(fig, "aguadv_event_mt_spec.png")


if __name__ == "__main__":
    main()
