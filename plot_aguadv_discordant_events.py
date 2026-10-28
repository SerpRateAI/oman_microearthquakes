"""
Plot discordant hub/satellite Z waveforms and Nuttall-window Fourier spectra for
groups 18 and 19 at A04 (Subarray A) and group 20 at B04 (Subarray B) in a 3×2
figure (AGU Advances).
"""

from argparse import ArgumentParser

from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots
from pandas import Timedelta

from utils_basic import SAMPLING_RATE as sample_rate, get_freq_limits_string
from utils_cc import get_repeating_snippet_suffix
from utils_plot import (
    format_freq_xlabels,
    format_norm_amp_ylabels,
    format_norm_psd_ylabels,
    save_figure,
)
from utils_sta_lta import get_sta_lta_suffix

from plot_discordant_event_waveforms_n_spectra import (
    HUB_COLOR,
    LINEWIDTH_SPECTRUM,
    LINEWIDTH_WAVEFORM,
    SATELLITE_COLOR,
    VERTICAL_COMPONENT,
    get_hub_short_long_windows,
    load_alignment_df,
    load_discordant_scores,
    load_event_df,
    load_station_waveforms_at_arrival,
    plot_discordant_window_bars,
    plot_z_spectrum,
    plot_z_waveform,
    select_satellite_id,
)

GROUP_ROWS = [
    {"label": 18, "station": "A04"},
    {"label": 19, "station": "A04"},
    {"label": 20, "station": "B04"},
]


def plot_discordant_group_row(
    ax_wf,
    ax_spec,
    group_label,
    station,
    suffix_group,
    rank_discordant,
    buffer_before_sec,
    buffer_after_sec,
    min_freq_plot,
    max_freq_plot,
    min_db,
    max_db,
    linewidth_waveform,
    linewidth_spectrum,
):
    """
    Plot hub vs discordant-satellite waveforms and spectra for one group row.

    Returns (id_hub, id_satellite, discordant_score) for the row title.
    """
    alignment_df = load_alignment_df(group_label, suffix_group)
    event_df = load_event_df(group_label, suffix_group)
    discordant_df = load_discordant_scores(group_label, suffix_group)

    hub_row = alignment_df.loc[alignment_df["hub"]].iloc[0]
    id_hub = int(hub_row["id"])
    hub_aligned_time = hub_row["aligned_time"]
    hub_dict = event_df.loc[event_df["event_id"] == id_hub].iloc[0].to_dict()

    id_satellite = select_satellite_id(discordant_df, None, rank_discordant)
    sat_rows = alignment_df.loc[alignment_df["id"] == id_satellite]
    if sat_rows.empty:
        raise ValueError(f"Satellite {id_satellite} not found in alignments for group {group_label}.")

    sat_row = sat_rows.iloc[0]
    sat_aligned_time = sat_row["aligned_time"]
    discordant_score = float(
        discordant_df.loc[discordant_df["id_satellite"] == id_satellite, "discordant_score"].iloc[0]
    )

    window_sec = buffer_before_sec + buffer_after_sec

    hub_waveform_dict, _ = load_station_waveforms_at_arrival(
        hub_aligned_time, station, buffer_before_sec, buffer_after_sec,
    )
    sat_waveform_dict, _ = load_station_waveforms_at_arrival(
        sat_aligned_time, station, buffer_before_sec, buffer_after_sec,
    )

    plot_z_waveform(
        ax_wf,
        hub_waveform_dict[VERTICAL_COMPONENT],
        window_sec,
        HUB_COLOR,
        "Hub",
        linewidth=linewidth_waveform,
    )
    plot_z_waveform(
        ax_wf,
        sat_waveform_dict[VERTICAL_COMPONENT],
        window_sec,
        SATELLITE_COLOR,
        "Satellite",
        linewidth=linewidth_waveform,
    )

    if station in hub_dict.get("per_station_times", {}):
        t0_short, t1_short, t0_long, t1_long = get_hub_short_long_windows(hub_dict, station)
        plot_starttime = hub_aligned_time - Timedelta(seconds=buffer_before_sec)
        plot_discordant_window_bars(
            ax_wf, t0_short, t1_short, t0_long, t1_long, plot_starttime,
        )

    plot_z_spectrum(
        ax_spec,
        hub_waveform_dict[VERTICAL_COMPONENT],
        HUB_COLOR,
        "Hub",
        min_freq_plot,
        max_freq_plot,
        linewidth=linewidth_spectrum,
    )
    plot_z_spectrum(
        ax_spec,
        sat_waveform_dict[VERTICAL_COMPONENT],
        SATELLITE_COLOR,
        "Satellite",
        min_freq_plot,
        max_freq_plot,
        linewidth=linewidth_spectrum,
    )
    ax_spec.set_ylim(min_db, max_db)

    return id_hub, id_satellite, discordant_score


def main():
    parser = ArgumentParser(
        description=(
            "Plot discordant hub/satellite Z waveforms and Nuttall-window Fourier "
            "spectra for groups 18 and 19 (A04) and group 20 (B04) (AGU Advances)"
        ),
    )
    parser.add_argument("--figwidth", type=float, default=14.0)
    parser.add_argument("--figheight", type=float, default=12.0)
    parser.add_argument(
        "--rank_discordant",
        type=int,
        default=1,
        help="Discordant rank of the satellite event to plot in each group (1 = most discordant)",
    )
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
    parser.add_argument("--linewidth_waveform", type=float, default=LINEWIDTH_WAVEFORM)
    parser.add_argument("--linewidth_spectrum", type=float, default=LINEWIDTH_SPECTRUM)
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
            [0], [0], color=HUB_COLOR, linewidth=args.linewidth_spectrum, label="Hub",
        ),
        Line2D(
            [0], [0],
            color=SATELLITE_COLOR,
            linewidth=args.linewidth_spectrum,
            label="Satellite",
        ),
    ]

    for i_row, group_row in enumerate(GROUP_ROWS):
        group_label = group_row["label"]
        station = group_row["station"]
        ax_wf = axes[i_row, 0]
        ax_spec = axes[i_row, 1]

        print(f"Processing group {group_label:d} at {station} (rank {args.rank_discordant})...")
        try:
            id_hub, id_satellite, discordant_score = plot_discordant_group_row(
                ax_wf,
                ax_spec,
                group_label,
                station,
                suffix_group,
                args.rank_discordant,
                args.buffer_before_sec,
                args.buffer_after_sec,
                min_freq_plot,
                max_freq_plot,
                args.min_db,
                args.max_db,
                args.linewidth_waveform,
                args.linewidth_spectrum,
            )
        except Exception as exc:
            print(f"Warning: skipping group {group_label:d}: {exc}")
            ax_wf.set_visible(False)
            ax_spec.set_visible(False)
            continue

        ax_wf.set_title(
            (
                f"Group {group_label:d}, Hub {id_hub} vs Satellite {id_satellite} "
            ),
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
            format_norm_amp_ylabels(ax_wf, major_tick_spacing=0.5)
            format_norm_psd_ylabels(ax_spec, major_tick_spacing=10.0)
        else:
            format_norm_amp_ylabels(
                ax_wf,
                plot_axis_label=False,
                major_tick_spacing=0.5,
            )
            format_norm_psd_ylabels(
                ax_spec,
                plot_axis_label=False,
                major_tick_spacing=10.0,
            )

    fig.suptitle(
        "Discordant event waveforms and spectra",
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

    save_figure(fig, "aguadv_discordant_events.png")


if __name__ == "__main__":
    main()
