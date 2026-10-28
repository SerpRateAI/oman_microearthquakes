"""
Plot the vertical-component waveform of the repeating STA/LTA detection with the most
similar signals at a geophone station, overlaid with all its matches aligned to it.
"""

from argparse import ArgumentParser
from os.path import join
from pathlib import Path

from matplotlib.pyplot import subplots
from numpy import amax, arange, argmax, load, where
from pandas import Timedelta

from utils_basic import (
    DETECTION_DIR as dirpath_det,
    ROOTDIR_GEO as dirpath_waveform,
    SAMPLING_RATE as sample_rate,
    get_freq_limits_string,
)
from utils_cc import get_repeating_snippet_suffix
from utils_cont_waveform import load_waveform_slice
from utils_plot import component2label, get_geo_component_color, save_figure
from utils_sta_lta import Snippets, get_sta_lta_suffix

VERTICAL_COMPONENT = "Z"


def find_most_popular_detection(station, suffix, min_cc):
    """Return the template snippet, the aligned start times of its matches, and the number of matches."""
    print(f"Loading the repeating-detection data for station {station}...")
    corr_mat = load(Path(dirpath_det) / f"correlation_matrix_repeating_{suffix}_{station}.npy")
    lag_mat = load(Path(dirpath_det) / f"lag_matrix_repeating_{suffix}_{station}.npy")
    snippets = Snippets.from_hdf(Path(dirpath_det) / f"snippets_repeating_{suffix}_{station}.h5")
    print(f"Number of repeating detections: {len(snippets)}")

    num_similar = (corr_mat > min_cc).sum(axis=1) - 1
    i_hub = int(argmax(num_similar))
    indices_match = where(corr_mat[i_hub] > min_cc)[0]
    indices_match = indices_match[indices_match != i_hub]
    snippet_hub = snippets[i_hub]
    num_matches = len(indices_match)
    print(f"Most popular detection: {snippet_hub.id} at {snippet_hub.starttime} with {num_matches} matches")

    aligned_starttimes = []
    for i_match in indices_match:
        lag = lag_mat[i_match, i_hub]
        aligned_starttimes.append(snippets[int(i_match)].starttime - Timedelta(seconds=lag / sample_rate))

    return snippet_hub, aligned_starttimes, num_matches


def plot_detection_waveforms_z(
    ax,
    station,
    waveform_filepath,
    template_starttime,
    aligned_starttimes,
    buffer_before_sec,
    buffer_after_sec,
    scale_factor,
    linewidth_template=1.0,
):
    """Overlay the aligned matches in gray and the template in the vertical-component color."""
    component = VERTICAL_COMPONENT
    color_hub = get_geo_component_color(component)

    traces = [(starttime, False) for starttime in aligned_starttimes] + [(template_starttime, True)]
    for aligned_starttime, hub in traces:
        if hub:
            color = color_hub
            linewidth = linewidth_template
            zorder = 2
        else:
            color = "lightgray"
            linewidth = 0.2
            zorder = 0

        starttime_plot = aligned_starttime - Timedelta(seconds=buffer_before_sec)
        endtime_plot = aligned_starttime + Timedelta(seconds=buffer_after_sec)
        try:
            waveform_dict, _ = load_waveform_slice(
                waveform_filepath,
                station,
                starttime_plot,
                endtime=endtime_plot,
                normalize=True,
            )
        except Exception as e:
            print(f"Error loading waveform slice starting at {starttime_plot}: {e}")
            continue

        waveform = waveform_dict[component]
        timeax = arange(len(waveform)) / sample_rate
        waveform_plot = waveform / amax(abs(waveform)) * scale_factor
        ax.plot(timeax, waveform_plot, color=color, linewidth=linewidth, zorder=zorder)

    ax.set_xlim(0.0, buffer_before_sec + buffer_after_sec)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel("Time (s)", fontsize=12)


def main():
    parser = ArgumentParser(
        description="Plot the detection with the most similar signals at a geophone station and all its matches",
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
    parser.add_argument("--scale_factor", type=float, default=0.7)
    parser.add_argument("--figwidth", type=float, default=10.0)
    parser.add_argument("--figheight", type=float, default=5.0)
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
    figwidth = args.figwidth
    figheight = args.figheight

    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    suffix_repeating = get_repeating_snippet_suffix(min_cc=min_cc, min_num_similar=min_num_similar)
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating}"

    snippet_hub, aligned_starttimes, num_matches = find_most_popular_detection(station, suffix, min_cc)

    waveform_filepath = join(dirpath_waveform, f"preprocessed_data_{suffix_freq}.h5")
    component_label = component2label(VERTICAL_COMPONENT)

    fig, ax = subplots(1, 1, figsize=(figwidth, figheight))
    plot_detection_waveforms_z(
        ax,
        station,
        waveform_filepath,
        snippet_hub.starttime,
        aligned_starttimes,
        buffer_before_sec,
        buffer_after_sec,
        scale_factor,
    )
    ax.set_title(
        f"{station}, {num_matches + 1} detections, {component_label}",
        fontsize=12,
        fontweight="bold",
    )

    save_figure(fig, f"geophone_most_popular_detection_waveform_{station}_{suffix}.png")


if __name__ == "__main__":
    main()
