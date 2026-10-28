"""
Plot the aligned vertical-component waveforms of the most popular repeating detection
and its matches at a geophone station (left), and the unfiltered multi-taper
displacement power spectrum of the most popular detection (right) (AGU Advances).
"""

from argparse import ArgumentParser
from os.path import join

from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots
from matplotlib.ticker import AutoLocator, ScalarFormatter
from numpy import amax
from pandas import Timedelta

from utils_basic import (
    ROOTDIR_GEO as dirpath_waveform,
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

from plot_event_n_hammer_disp_spec import load_station_waveforms_at_arrival
from plot_geophone_most_popular_detection_waveform import (
    VERTICAL_COMPONENT,
    find_most_popular_detection,
    plot_detection_waveforms_z,
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


def main():
    parser = ArgumentParser(
        description=(
            "Plot the aligned waveforms of the most popular repeating detection at a geophone "
            "station and its multi-taper displacement spectrum (AGU Advances)"
        ),
    )
    parser.add_argument("--station", type=str, default="B13")
    parser.add_argument("--min_cc", type=float, default=0.85)
    parser.add_argument("--min_num_similar", type=int, default=10)
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--thr_on", type=float, default=4.0)
    parser.add_argument("--thr_off", type=float, default=1.0)
    parser.add_argument("--buffer_before_sec", type=float, default=0.05)
    parser.add_argument("--buffer_after_sec", type=float, default=0.1)
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--nw", type=int, default=3, help="Time-bandwidth product")
    parser.add_argument("--fill_alpha", type=float, default=0.2, help="Uncertainty-band opacity")
    parser.add_argument("--min_freq_plot", type=float, default=10.0)
    parser.add_argument(
        "--max_freq_plot",
        type=float,
        default=None,
        help="Maximum frequency for the spectrum panel (default: Nyquist)",
    )
    parser.add_argument("--min_db", type=float, default=-45.0, help="Spectrum y-axis minimum (dB)")
    parser.add_argument("--max_db", type=float, default=5.0, help="Spectrum y-axis maximum (dB)")
    parser.add_argument("--figwidth", type=float, default=15.0)
    parser.add_argument("--figheight", type=float, default=6.0)
    args = parser.parse_args()

    station = args.station
    min_cc = args.min_cc
    min_num_similar = args.min_num_similar
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    thr_on = args.thr_on
    thr_off = args.thr_off
    buffer_before_sec = args.buffer_before_sec
    buffer_after_sec = args.buffer_after_sec
    scale_factor = args.scale_factor
    nw = args.nw
    fill_alpha = args.fill_alpha
    min_freq_plot = args.min_freq_plot
    max_freq_plot_arg = args.max_freq_plot
    min_db = args.min_db
    max_db = args.max_db
    figwidth = args.figwidth
    figheight = args.figheight

    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    suffix_repeating = get_repeating_snippet_suffix(min_cc=min_cc, min_num_similar=min_num_similar)
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating}"

    max_freq_plot = max_freq_plot_arg if max_freq_plot_arg is not None else sample_rate / 2.0
    window_sec = buffer_before_sec + buffer_after_sec
    component = VERTICAL_COMPONENT
    component_label = component2label(component)

    snippet_hub, aligned_starttimes, num_matches = find_most_popular_detection(station, suffix, min_cc)
    template_starttime = snippet_hub.starttime

    fig, axes = subplots(1, 2, figsize=(figwidth, figheight), gridspec_kw={"wspace": 0.15})

    # Waveform panel
    print("Plotting the waveforms...")
    ax_wave = axes[0]
    waveform_filepath = join(dirpath_waveform, f"preprocessed_data_{suffix_freq}.h5")
    plot_detection_waveforms_z(
        ax_wave,
        station,
        waveform_filepath,
        template_starttime,
        aligned_starttimes,
        buffer_before_sec,
        buffer_after_sec,
        scale_factor,
        linewidth_template=1.5,
    )
    ax_wave.yaxis.set_major_locator(AutoLocator())
    ax_wave.yaxis.set_major_formatter(ScalarFormatter())
    ax_wave.set_ylabel("Normalized amplitude", fontsize=12)
    ax_wave.set_title(
        f"{station}, {num_matches + 1} detections, {component_label}",
        fontsize=12,
        fontweight="bold",
    )

    # Spectrum panel
    print("Computing the multi-taper displacement PSDs of the most popular detection and noise...")
    ax_spec = axes[1]
    event_waveform_dict = load_station_waveforms_at_arrival(
        template_starttime,
        station,
        buffer_before_sec,
        buffer_after_sec,
    )
    event_start = template_starttime - Timedelta(seconds=buffer_before_sec)
    noise_end = event_start
    noise_start = noise_end - Timedelta(seconds=window_sec)
    print(f"Loading pre-event noise at {station} from {noise_start} to {noise_end}...")
    noise_waveform_dict = load_station_waveforms_between(noise_start, noise_end, station)

    freqax_event, psd_event, psd_event_lo, psd_event_hi = compute_disp_psd(
        event_waveform_dict[component], min_freq_plot, max_freq_plot, nw,
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
        ax_spec,
        freqax_event,
        psd_event,
        psd_event_lo,
        psd_event_hi,
        get_geo_component_color(component),
        LINEWIDTH_EVENT,
        LINESTYLE_EVENT,
        fill_alpha,
    )
    plot_spectrum_with_uncertainty(
        ax_spec,
        freqax_noise,
        psd_noise,
        psd_noise_lo,
        psd_noise_hi,
        NOISE_SPECTRUM_COLOR,
        LINEWIDTH_NOISE,
        LINESTYLE_NOISE,
        fill_alpha,
    )
    ax_spec.set_xscale("log")
    ax_spec.set_xlim(min_freq_plot, max_freq_plot)
    ax_spec.set_ylim(min_db, max_db)
    ax_spec.grid(True, linestyle="--", alpha=0.5, which="major")
    ax_spec.grid(True, linestyle=":", alpha=0.5, which="minor")
    ax_spec.set_xlabel("Frequency (Hz)", fontsize=12)
    format_norm_psd_ylabels(ax_spec)
    ax_spec.set_title("Displacement Spectra", fontsize=12, fontweight="bold")

    legend_handles = [
        Line2D(
            [0], [0],
            color=get_geo_component_color(component),
            linewidth=LINEWIDTH_EVENT,
            linestyle=LINESTYLE_EVENT,
            label="Event",
        ),
        Line2D(
            [0], [0],
            color=NOISE_SPECTRUM_COLOR,
            linewidth=LINEWIDTH_NOISE,
            linestyle=LINESTYLE_NOISE,
            label="Noise",
        ),
    ]
    ax_spec.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=11,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.3",
    )

    for i_ax, ax in enumerate(axes):
        ax.text(
            -0.07,
            1.05,
            f"({chr(97 + i_ax)})",
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )

    figname = f"aguadv_repeating_detection_waveform_n_spec_{station}.png"
    print(f"Saving figure to {figname}...")
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
