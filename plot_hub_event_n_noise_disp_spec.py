"""
Plot unfiltered multi-taper displacement power spectra of a group hub event and
an equal-length pre-event noise window at one station, one panel per component.
"""

from argparse import ArgumentParser

from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots
from numpy import amax
from pandas import Timedelta
from scipy.signal.windows import dpss

from utils_basic import (
    GEO_COMPONENTS as components,
    SAMPLING_RATE as sample_rate,
    get_freq_limits_string,
    power2db,
)
from utils_cc import get_repeating_snippet_suffix
from utils_mt import mt_autospec
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_plot import (
    component2label,
    format_norm_psd_ylabels,
    get_geo_component_color,
    save_figure,
)
from utils_spec import vel2disp
from utils_sta_lta import get_sta_lta_suffix

from plot_event_n_hammer_disp_spec import (
    load_hub_alignment,
    load_station_waveforms_at_arrival,
    stream_station_to_waveform_dict,
)

DEFAULT_GROUP_LABEL = 18
DEFAULT_STATION = "A04"

LINEWIDTH_EVENT = 3.0
LINEWIDTH_NOISE = 3.0
LINESTYLE_EVENT = "-"
LINESTYLE_NOISE = "--"
NOISE_SPECTRUM_COLOR = "gray"


def load_station_waveforms_between(starttime, endtime, station):
    """Load unfiltered, unnormalized 3-component velocity waveforms in [starttime, endtime]."""
    stream = read_and_process_windowed_geo_waveforms(
        starttime,
        endtime=endtime,
        stations=[station],
        components=components,
        filter=False,
        normalize=False,
    )
    if stream is None or len(stream) == 0:
        raise ValueError(f"No waveform data for station {station}.")
    return stream_station_to_waveform_dict(stream, station)


def compute_disp_psd(velocity, min_freq_plot, max_freq_plot, nw):
    """Multi-taper displacement PSD (mean, lo, hi) restricted to the plot band."""
    displacement = vel2disp(velocity, sample_rate)
    num_pts = len(displacement)
    taper_mat, ratio_vec = dpss(num_pts, nw, 2 * nw - 1, return_ratios=True)
    aspec_param = mt_autospec(displacement, taper_mat, ratio_vec, sample_rate, verbose=False)
    freqax = aspec_param.freqax
    in_band = (freqax >= min_freq_plot) & (freqax <= max_freq_plot)
    return (
        freqax[in_band],
        aspec_param.aspec[in_band],
        aspec_param.aspec_lo[in_band],
        aspec_param.aspec_hi[in_band],
    )


def plot_spectrum_with_uncertainty(ax, freqax, psd, psd_lo, psd_hi, color, linewidth, linestyle, fill_alpha):
    """Plot a mean spectrum with a same-color transparent uncertainty band."""
    ax.fill_between(
        freqax,
        power2db(psd_lo),
        power2db(psd_hi),
        color=color,
        alpha=fill_alpha,
        linewidth=0,
    )
    ax.plot(
        freqax,
        power2db(psd),
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )


def main():
    parser = ArgumentParser(
        description=(
            "Plot unfiltered multi-taper displacement power spectra of a group hub "
            "event and an equal-length pre-event noise window at one station"
        ),
    )
    parser.add_argument(
        "--group_label",
        type=int,
        default=DEFAULT_GROUP_LABEL,
        help="Event group label",
    )
    parser.add_argument("--station", type=str, default=DEFAULT_STATION, help="Station to plot")
    parser.add_argument("--figwidth", type=float, default=15.0)
    parser.add_argument("--figheight", type=float, default=5.0)
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
    group_label = args.group_label
    station = args.station
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

    print(f"Loading hub event for group {group_label:d} at {station}...")
    id_hub, hub_aligned_time = load_hub_alignment(group_label, suffix_group)
    print(f"Hub event ID: {id_hub}")
    hub_waveform_dict = load_station_waveforms_at_arrival(
        hub_aligned_time,
        station,
        buffer_before_sec,
        buffer_after_sec,
    )

    event_start = hub_aligned_time - Timedelta(seconds=buffer_before_sec)
    noise_end = event_start
    noise_start = noise_end - Timedelta(seconds=window_sec)
    print(f"Loading pre-event noise at {station} from {noise_start} to {noise_end}...")
    noise_waveform_dict = load_station_waveforms_between(noise_start, noise_end, station)

    print("Computing multi-taper displacement PSDs...")
    fig, axes = subplots(
        1,
        len(components),
        figsize=(figwidth, figheight),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.12},
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
            color=NOISE_SPECTRUM_COLOR,
            linewidth=LINEWIDTH_NOISE,
            linestyle=LINESTYLE_NOISE,
            label="Noise",
        ),
    ]

    for i, component in enumerate(components):
        ax = axes[i]
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

        ax.set_xscale("log")
        ax.set_xlim(min_freq_plot, max_freq_plot)
        ax.set_ylim(min_db, max_db)
        ax.set_title(component2label(component), fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5, which="major")
        ax.grid(True, linestyle=":", alpha=0.5, which="minor")

        if i > 0:
            ax.tick_params(axis="y", labelleft=False)

    format_norm_psd_ylabels(axes[0])

    axes[0].legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=11,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.3",
    )

    fig.suptitle(
        f"Group {group_label:d}, Hub {id_hub} vs pre-event noise, {station}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    figname = f"hub_event_n_noise_disp_spec_group{group_label:d}_{station}.png"
    print(f"Saving figure to {figname}...")
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
