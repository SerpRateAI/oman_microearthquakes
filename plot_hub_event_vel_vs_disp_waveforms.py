"""
Plot the unfiltered vertical-component velocity and displacement seismograms of
a group hub event recorded at its first-onset station, arranged vertically.
"""

from argparse import ArgumentParser
from os.path import join

from matplotlib.pyplot import subplots
from numpy import arange
from pandas import read_csv, to_datetime
from scipy.signal import detrend

from utils_basic import (
    DETECTION_DIR as dirpath_event,
    SAMPLING_RATE as sample_rate,
    get_freq_limits_string,
)
from utils_cc import get_repeating_snippet_suffix
from utils_plot import component2label, get_geo_component_color, save_figure
from utils_spec import vel2disp
from utils_sta_lta import get_sta_lta_suffix

from plot_event_n_hammer_disp_spec import load_station_waveforms_at_arrival

DEFAULT_GROUP_LABEL = 18
COMPONENT = "Z"

LINEWIDTH_WAVEFORM = 1.5
LINEWIDTH_ONSET = 1.0


def load_hub_alignment_with_station(group_label, suffix_group):
    """Return hub event id, aligned_time, and first-onset station for a group."""
    filename = f"event_alignments_group{group_label:d}_{suffix_group}.csv"
    alignment_df = read_csv(join(dirpath_event, filename))
    alignment_df["aligned_time"] = to_datetime(alignment_df["aligned_time"], format="ISO8601")
    alignment_df["hub"] = alignment_df["hub"].astype(bool)
    hub_row = alignment_df.loc[alignment_df["hub"]].iloc[0]
    return int(hub_row["id"]), hub_row["aligned_time"], hub_row["aligned_station"]


def main():
    parser = ArgumentParser(
        description=(
            "Plot the vertical-component velocity and displacement seismograms of a "
            "group hub event at its first-onset station"
        ),
    )
    parser.add_argument(
        "--group_label",
        type=int,
        default=DEFAULT_GROUP_LABEL,
        help="Event group label",
    )
    parser.add_argument("--figwidth", type=float, default=10.0)
    parser.add_argument("--figheight", type=float, default=8.0)
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
    args = parser.parse_args()
    group_label = args.group_label
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

    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(
        sta_window_sec, lta_window_sec, thr_on, thr_off,
    )
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=min_cc, min_num_similar=min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

    print(f"Loading hub event alignment for group {group_label:d}...")
    id_hub, hub_aligned_time, station = load_hub_alignment_with_station(group_label, suffix_group)
    print(f"Hub event ID: {id_hub}")
    print(f"First-onset station: {station}")

    print(f"Loading unfiltered velocity waveforms at {station}...")
    waveform_dict = load_station_waveforms_at_arrival(
        hub_aligned_time,
        station,
        buffer_before_sec,
        buffer_after_sec,
    )
    velocity = waveform_dict[COMPONENT]

    print("Integrating velocity to displacement...")
    displacement = detrend(vel2disp(velocity, sample_rate), type="linear")

    timeax = arange(len(velocity)) / sample_rate - buffer_before_sec
    color = get_geo_component_color(COMPONENT)

    fig, axes = subplots(
        2,
        1,
        figsize=(figwidth, figheight),
        sharex=True,
        gridspec_kw={"hspace": 0.1},
    )

    panels = [
        (axes[0], velocity, "Velocity (nm s$^{-1}$)"),
        (axes[1], displacement, "Displacement (nm)"),
    ]
    for ax, waveform, ylabel in panels:
        ax.plot(timeax, waveform, color=color, linewidth=LINEWIDTH_WAVEFORM)
        ax.axvline(0.0, color="crimson", linestyle="--", linewidth=LINEWIDTH_ONSET)
        ax.set_xlim(timeax[0], timeax[-1])
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].text(
        0.01,
        0.97,
        component2label(COMPONENT),
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )
    axes[1].set_xlabel("Time (s)", fontsize=12)

    fig.suptitle(
        f"Group {group_label:d}, Hub {id_hub}, {station}",
        fontsize=14,
        fontweight="bold",
        y=0.94,
    )

    figname = f"hub_event_vel_vs_disp_waveforms_group{group_label:d}_{station}.png"
    print(f"Saving figure to {figname}...")
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
